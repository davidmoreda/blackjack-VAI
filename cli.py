"""
Modo terminal — sin servidor ni puertos.
Abre la camara, detecta cartas con YOLO y muestra el resultado en una
ventana OpenCV. Toda la logica de juego y conteo funciona igual que la API.

Controles de teclado (juego):
  S            — nueva partida
  R            — nueva ronda
  ESPACIO      — stand del jugador actual
  Z            — DESHACER ultima carta detectada
  H            — abrir navegador de historial (modo borrar carta concreta)
  Q            — salir

En el navegador de historial (H):
  ARRIBA/ABAJO — moverse entre cartas
  X / DELETE   — eliminar la carta seleccionada
  H / ESC      — cerrar navegador

Settings (P):
  ARRIBA/ABAJO     — navegar opciones
  IZQUIERDA/DERECHA — cambiar valor
  P / ESC          — cerrar panel (guarda en config.yaml)
"""

import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import yaml

from vision.capture import CameraCapture
from vision.detector import CardDetector
from vision.zones import ZoneManager, draw_zones, ZONE_COLORS
from game.state_machine import BlackjackStateMachine, GameState
from game.counter import CardCounter


# ── Helpers de dibujo ─────────────────────────────────────────────────────────

def _put(frame, text, pos, scale=0.6, color=(220, 220, 220), thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ── Zonas dinamicas segun num_players ────────────────────────────────────────

def build_zones(num_players: int) -> dict:
    """
    Genera dealer arriba (40% altura) + N columnas iguales para los jugadores
    en el 60% inferior. Ignora lo que haya en config.yaml para evitar
    desalineamientos cuando se cambia el numero de jugadores.
    """
    n = max(1, int(num_players))
    zones = {"dealer": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.4}}
    width = 1.0 / n
    for i in range(n):
        zones[f"player_{i+1}"] = {
            "x": i * width, "y": 0.4, "w": width, "h": 0.6,
        }
    return zones


# ── Debouncer de detecciones ─────────────────────────────────────────────────

class CardDebouncer:
    """
    Una carta debe detectarse de forma continua durante `threshold` segundos
    antes de confirmarse. Filtra falsos positivos y oscilaciones del modelo.

    Entrada en cada tick: pares (zone, label) actualmente visibles.
    Salida: pares (zone, label) que acaban de superar el umbral.

    Una carta confirmada no vuelve a emitirse hasta que desaparece y reaparece.
    """

    def __init__(self, threshold_s: float = 1.0):
        self.threshold = threshold_s
        self.pending: Dict[Tuple[str, str], float] = {}
        self.committed: Set[Tuple[str, str]] = set()

    def tick(self, visible: Iterable[Tuple[str, str]], now: float) -> List[Tuple[str, str]]:
        visible_set = set(visible)
        # Olvidar candidatos / confirmados que ya no se ven
        for k in list(self.pending):
            if k not in visible_set:
                del self.pending[k]
        self.committed &= visible_set

        new_commits: List[Tuple[str, str]] = []
        for key in visible_set:
            if key in self.committed:
                continue
            t0 = self.pending.get(key)
            if t0 is None:
                self.pending[key] = now
            elif now - t0 >= self.threshold:
                new_commits.append(key)
                self.committed.add(key)
                self.pending.pop(key, None)
        return new_commits

    def progress(self, key: Tuple[str, str], now: float) -> float:
        if key in self.committed:
            return 1.0
        t0 = self.pending.get(key)
        if t0 is None:
            return 0.0
        return min(1.0, (now - t0) / self.threshold)

    def reset(self):
        self.pending.clear()
        self.committed.clear()


# ── Panel de Settings ─────────────────────────────────────────────────────────

COUNTING_SYSTEMS = ["hilo", "ko", "omega2"]
NUM_DECKS_OPTIONS = [1, 2, 4, 6, 8]
NUM_PLAYERS_OPTIONS = list(range(1, 8))
CONFIDENCE_OPTIONS = [round(x * 0.05, 2) for x in range(2, 20)]  # 0.10 … 0.95
CAMERA_OPTIONS = list(range(6))


class SettingsPanel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.visible = False
        self.selected = 0
        self._items = [
            ("Camara",           "camera",    "index",           CAMERA_OPTIONS),
            ("Jugadores",        "game",      "num_players",     NUM_PLAYERS_OPTIONS),
            ("Mazos",            "game",      "num_decks",       NUM_DECKS_OPTIONS),
            ("Sistema conteo",   "game",      "counting_system", COUNTING_SYSTEMS),
            ("Confianza YOLO",   "detection", "confidence",      CONFIDENCE_OPTIONS),
        ]

    def _current_value(self, item):
        _, section, key, _ = item
        return self.cfg[section][key]

    def _option_index(self, item):
        _, section, key, options = item
        try:
            return options.index(self.cfg[section][key])
        except ValueError:
            return 0

    def move(self, direction: int):
        self.selected = (self.selected + direction) % len(self._items)

    def change(self, direction: int):
        item = self._items[self.selected]
        _, section, key, options = item
        idx = (self._option_index(item) + direction) % len(options)
        self.cfg[section][key] = options[idx]

    def save(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False, allow_unicode=True)

    def draw(self, frame):
        if not self.visible:
            return frame
        h, w = frame.shape[:2]
        panel_w, panel_h = 440, 40 + len(self._items) * 38 + 40
        x0 = (w - panel_w) // 2
        y0 = (h - panel_h) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 2)

        _put(frame, "CONFIGURACION  (</> cambiar, P cerrar)",
             (x0 + 12, y0 + 26), 0.52, (200, 200, 200))

        for i, item in enumerate(self._items):
            label, _, _, _ = item
            val = self._current_value(item)
            iy = y0 + 52 + i * 38
            if i == self.selected:
                cv2.rectangle(frame, (x0 + 6, iy - 18),
                              (x0 + panel_w - 6, iy + 14), (50, 50, 90), -1)
            color = (0, 220, 255) if i == self.selected else (180, 180, 180)
            _put(frame, label, (x0 + 16, iy), 0.58, color)

            val_str = str(val)
            (tw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
            val_x = x0 + panel_w - tw - 30
            val_color = (40, 220, 40) if i == self.selected else (160, 220, 160)
            _put(frame, val_str, (val_x, iy), 0.62, val_color, 2)
            if i == self.selected:
                _put(frame, "<", (val_x - 20, iy), 0.62, (200, 200, 200))
                _put(frame, ">", (val_x + tw + 6, iy), 0.62, (200, 200, 200))

        note = "Los cambios se guardan en config.yaml al cerrar"
        _put(frame, note, (x0 + 12, y0 + panel_h - 12), 0.4, (120, 120, 120))
        return frame


# ── Sidebar de historial ─────────────────────────────────────────────────────

SIDEBAR_W = 200
ROW_H = 22
HEADER_H = 38
ZONE_TAGS = {
    "dealer":   ("D", (60,  60,  220)),
    "player_1": ("P1", (40, 200, 40)),
    "player_2": ("P2", (220, 150, 0)),
    "player_3": ("P3", (220, 0, 220)),
    "player_4": ("P4", (200, 200, 0)),
    "player_5": ("P5", (0, 200, 200)),
    "player_6": ("P6", (200, 100, 100)),
    "player_7": ("P7", (160, 160, 160)),
}


def draw_sidebar(frame, sm, hist_open: bool, hist_idx: int, scroll: int):
    h, w = frame.shape[:2]
    x0 = w - SIDEBAR_W
    y0 = 90

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    cv2.rectangle(frame, (x0, y0), (w, h), (60, 60, 60), 1)

    title = "HISTORIAL" if not hist_open else "HISTORIAL  [borrar]"
    title_color = (220, 220, 220) if not hist_open else (40, 220, 220)
    _put(frame, title, (x0 + 10, y0 + 22), 0.55, title_color, 2)
    cv2.line(frame, (x0 + 8, y0 + HEADER_H - 6),
             (w - 8, y0 + HEADER_H - 6), (60, 60, 60), 1)

    visible_rows = (h - y0 - HEADER_H - 8) // ROW_H
    history = sm.history
    n = len(history)

    start = scroll
    if hist_open and n > 0:
        if hist_idx < start:
            start = hist_idx
        elif hist_idx >= start + visible_rows:
            start = hist_idx - visible_rows + 1
    end = min(start + visible_rows, n)

    if n == 0:
        _put(frame, "(sin cartas)", (x0 + 12, y0 + HEADER_H + 18),
             0.42, (130, 130, 130))
        return frame, start

    for row, i in enumerate(range(start, end)):
        zone, card = history[i]
        ry = y0 + HEADER_H + row * ROW_H + 4
        if hist_open and i == hist_idx:
            cv2.rectangle(frame, (x0 + 4, ry - 14),
                          (w - 4, ry + 6), (60, 60, 110), -1)
            cv2.rectangle(frame, (x0 + 4, ry - 14),
                          (w - 4, ry + 6), (40, 220, 220), 1)

        tag, color = ZONE_TAGS.get(zone, ("?", (180, 180, 180)))
        _put(frame, f"{i+1:>3}", (x0 + 8, ry), 0.42, (140, 140, 140))
        _put(frame, tag, (x0 + 42, ry), 0.45, color, 2)

        short = card.split()[0] if " " in card else card
        suit  = card.split()[1][0] if " " in card else ""
        _put(frame, f"{short}{suit}", (x0 + 80, ry), 0.5, (230, 230, 230))

    if n > visible_rows:
        _put(frame, f"{end}/{n}", (w - 50, h - 8), 0.4, (120, 120, 120))

    if hist_open:
        _put(frame, "X borrar  ESC salir",
             (x0 + 8, h - 10), 0.4, (180, 180, 180))
    else:
        _put(frame, "Z deshacer  H navegar",
             (x0 + 8, h - 10), 0.38, (130, 130, 130))

    return frame, start


# ── HUD principal ─────────────────────────────────────────────────────────────

ACTION_COLORS = {
    "HIT":    (40,  220, 40),
    "STAND":  (80,  80,  255),
    "DOUBLE": (0,   200, 255),
    "SPLIT":  (0,   180, 255),
    "WAIT":   (160, 160, 160),
}


def _player_status(pdata: dict, dealer_has_card: bool) -> Tuple[str, Tuple[int, int, int]]:
    """Devuelve (texto, color) que SIEMPRE da informacion sobre el jugador."""
    cards = pdata.get("cards") or []
    if pdata.get("blackjack"):
        return "BLACKJACK!", (40, 220, 220)
    if pdata.get("bust"):
        return f"BUST ({pdata.get('total','?')})", (40, 40, 255)
    if pdata.get("stood"):
        return f"PLANTADO ({pdata.get('total','?')})", (180, 180, 180)
    sug = pdata.get("suggestion")
    if sug:
        return sug, ACTION_COLORS.get(sug, (200, 200, 200))
    if not cards:
        return "esperando cartas", (140, 140, 140)
    if len(cards) < 2:
        return f"{len(cards)}/2 cartas", (160, 160, 160)
    if not dealer_has_card:
        return "esperando upcard dealer", (160, 160, 160)
    return "...", (160, 160, 160)


def draw_hud(frame, sm: BlackjackStateMachine, settings_open: bool, hist_open: bool):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    state_colors = {
        GameState.WAITING:     (160, 160, 160),
        GameState.DEALING:     (0,   200, 255),
        GameState.PLAYER_TURN: (40,  220, 40),
        GameState.DEALER_TURN: (0,   150, 255),
        GameState.ROUND_END:   (80,  80,  255),
    }
    sc = state_colors.get(sm.state, (200, 200, 200))
    _put(frame, f"ESTADO: {sm.state.value.upper()}", (10, 22), 0.65, sc, 2)

    cnt = sm.counter.state()
    deck_st = sm.deck.state()
    _put(frame, f"RC: {cnt['running_count']:+d}  TC: {cnt['true_count']:+.1f}  "
                f"Vistas: {cnt['cards_seen']}  Restantes: {deck_st['total_remaining']}",
         (10, 48), 0.5, (200, 200, 80))

    extras = []
    if hist_open:    extras.append("[HIST]")
    if settings_open: extras.append("[CONFIG]")
    suffix = "  ".join(extras)
    controls = "[S]Partida [R]Ronda [SPACE]Stand [Z]Undo [H]Hist [P]Cfg [Q]Salir"
    _put(frame, f"{controls}  {suffix}", (10, 78), 0.43, (150, 150, 150))

    # Sugerencias (siempre visibles, una linea por jugador)
    snapshot = sm.to_dict()
    dealer_has_card = bool(snapshot["dealer"]["cards"])

    x_sug = w - SIDEBAR_W - 340
    _put(frame, "RECOMENDACION:", (x_sug, 20), 0.5, (200, 200, 200))

    y = 42
    for pid, pdata in snapshot["players"].items():
        text, col = _player_status(pdata, dealer_has_card)
        evs = pdata.get("evs")
        line = f"{pid.replace('_',' ').upper()}: {text}"
        if evs and pdata.get("suggestion"):
            ev_main = evs.get(pdata["suggestion"])
            if ev_main is not None:
                line += f"  ({ev_main:+.2f})"
        _put(frame, line, (x_sug, y), 0.55, col, 2)

        if evs:
            ev_str = "  ".join(f"{k[0]}{v:+.2f}" for k, v in evs.items())
            _put(frame, ev_str, (x_sug + 8, y + 16), 0.4, (180, 180, 180))
            y += 36
        else:
            y += 22
    return frame


# ── Bucle principal ──────────────────────────────────────────────────────────

def _rebuild(cfg, cfg_file) -> tuple:
    from game.ev_calculator import clear_caches
    clear_caches()
    det_cfg  = cfg["detection"]
    game_cfg = cfg["game"]
    detector = CardDetector(
        model_path=det_cfg["model_path"],
        confidence=det_cfg["confidence"],
        iou=det_cfg["iou"],
    )
    cam = CameraCapture(cfg)
    counter = CardCounter(
        system=game_cfg["counting_system"],
        num_decks=game_cfg["num_decks"],
    )
    sm = BlackjackStateMachine(
        num_players=game_cfg["num_players"],
        counter=counter,
        num_decks=game_cfg["num_decks"],
    )
    # Zonas dinamicas: ignoramos cfg["zones"] y las regeneramos a partir de num_players
    zone_mgr = ZoneManager(build_zones(game_cfg["num_players"]))
    return detector, cam, sm, zone_mgr


def run(config_path: str = "config.yaml"):
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"[ERROR] No se encontro config.yaml en: {cfg_file.resolve()}")
        sys.exit(1)

    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    print("[CLI] Cargando modelo YOLO y camara...")
    detector, cam, sm, zone_mgr = _rebuild(cfg, cfg_file)
    panel = SettingsPanel(cfg)
    debouncer = CardDebouncer(threshold_s=1.0)

    fps_target = cfg["detection"].get("inference_fps", 15)
    frame_interval = 1.0 / fps_target
    last_inference = 0.0
    last_detections: List = []  # cache de la ultima inferencia para repintar bboxes

    cv2.namedWindow("Blackjack VAI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Blackjack VAI", 1280, 720)
    print("[CLI] Listo.  P=config  H=historial  Z=undo  Q=salir")

    KEY_UP    = 2490368
    KEY_DOWN  = 2621440
    KEY_LEFT  = 2424832
    KEY_RIGHT = 2555904
    KEY_DELETE = 3014656
    KEY_ESC   = 27

    hist_open = False
    hist_idx = 0
    hist_scroll = 0

    while True:
        try:
            frame = cam.read()
        except RuntimeError as e:
            print(f"[WARN] {e}")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        now = time.monotonic()

        if now - last_inference >= frame_interval:
            last_inference = now
            last_detections = detector.detect(frame)

            # Construir lista de (zone, label) actualmente visibles
            visible: List[Tuple[str, str]] = []
            for det in last_detections:
                zone = zone_mgr.get_zone(det.center[0], det.center[1], w, h)
                if zone:
                    visible.append((zone, det.label))

            # Solo confirmamos cartas si estamos en una fase activa del juego
            if sm.state in (GameState.DEALING, GameState.PLAYER_TURN, GameState.DEALER_TURN):
                for zone, label in debouncer.tick(visible, now):
                    sm.add_card(zone, label)
            else:
                debouncer.reset()

        # Repintar bboxes (siempre, incluso entre inferencias) usando el ultimo resultado
        for det in last_detections:
            x1, y1, x2, y2 = det.bbox
            zone = zone_mgr.get_zone(det.center[0], det.center[1], w, h)
            base_color = ZONE_COLORS.get(zone, (200, 200, 200)) if zone else (200, 200, 200)

            key = (zone, det.label) if zone else None
            prog = debouncer.progress(key, now) if key else 1.0

            if prog < 1.0:
                # Pendiente: bbox mas tenue + barra de progreso bajo el bbox
                color = tuple(int(c * 0.5) for c in base_color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                bar_w = int((x2 - x1) * prog)
                cv2.rectangle(frame, (x1, y2 + 3), (x2, y2 + 7), (40, 40, 40), -1)
                cv2.rectangle(frame, (x1, y2 + 3), (x1 + bar_w, y2 + 7), (0, 220, 220), -1)
                _put(frame, f"{det.label} {det.confidence:.2f}",
                     (x1, max(y1 - 6, 12)), 0.42, (180, 180, 180))
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 2)
                _put(frame, f"{det.label} {det.confidence:.2f}",
                     (x1, max(y1 - 6, 12)), 0.45, base_color)

        frame = draw_zones(frame, zone_mgr, sm)
        frame = draw_hud(frame, sm, panel.visible, hist_open)

        n = len(sm.history)
        if n == 0:
            hist_open = False
            hist_idx = 0
        else:
            hist_idx = max(0, min(hist_idx, n - 1))

        frame, hist_scroll = draw_sidebar(frame, sm, hist_open, hist_idx, hist_scroll)
        frame = panel.draw(frame)
        cv2.imshow("Blackjack VAI", frame)

        key = cv2.waitKeyEx(1)
        key_char = key & 0xFF

        if panel.visible:
            if key_char in (ord('p'), KEY_ESC):
                panel.save(cfg_file)
                panel.visible = False
                print("[CLI] Config guardada. Reiniciando camara y juego...")
                cam.release()
                detector, cam, sm, zone_mgr = _rebuild(cfg, cfg_file)
                debouncer.reset()
                last_detections = []
                fps_target = cfg["detection"].get("inference_fps", 15)
                frame_interval = 1.0 / fps_target
                hist_idx = 0
                hist_scroll = 0
                hist_open = False
                print("[CLI] Listo.")
            elif key == KEY_UP:    panel.move(-1)
            elif key == KEY_DOWN:  panel.move(1)
            elif key == KEY_LEFT:  panel.change(-1)
            elif key == KEY_RIGHT: panel.change(1)
            elif key_char == ord('q'): break
            continue

        if hist_open:
            if key == KEY_UP:
                hist_idx = max(0, hist_idx - 1)
            elif key == KEY_DOWN:
                hist_idx = min(len(sm.history) - 1, hist_idx + 1)
            elif key_char == ord('x') or key == KEY_DELETE:
                if sm.history:
                    zone, card = sm.history[hist_idx]
                    if sm.remove_card(hist_idx):
                        # Limpiar el debouncer para que la carta pueda reconfirmarse
                        debouncer.committed.discard((zone, card))
                        debouncer.pending.pop((zone, card), None)
                        print(f"[CLI] Carta eliminada: {card} ({zone})")
                        if hist_idx >= len(sm.history):
                            hist_idx = max(0, len(sm.history) - 1)
            elif key_char == ord('h') or key == KEY_ESC:
                hist_open = False
            elif key_char == ord('q'):
                break
            continue

        if key_char == ord('q'):
            break
        elif key_char == ord('p'):
            panel.visible = True
        elif key_char == ord('s'):
            sm.start_game()
            debouncer.reset()
            hist_idx = 0; hist_scroll = 0
            print("[CLI] Nueva partida")
        elif key_char == ord('r'):
            sm.reset_round()
            debouncer.reset()
            print("[CLI] Nueva ronda")
        elif key_char == ord(' '):
            cp = sm.current_player
            if cp:
                sm.player_stand(cp)
                print(f"[CLI] {cp} plantado")
            else:
                print("[CLI] No hay jugador activo")
        elif key_char == ord('z'):
            if sm.history:
                zone, card = sm.history[-1]
                if sm.undo_last():
                    debouncer.committed.discard((zone, card))
                    debouncer.pending.pop((zone, card), None)
                    print(f"[CLI] Ultima carta deshecha: {card}")
            else:
                print("[CLI] No hay cartas que deshacer")
        elif key_char == ord('h'):
            if sm.history:
                hist_open = True
                hist_idx = len(sm.history) - 1
            else:
                print("[CLI] Historial vacio")

    cam.release()
    cv2.destroyAllWindows()
    print("[CLI] Sesion terminada.")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    run()
