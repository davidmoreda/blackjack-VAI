"""
WebSocket: emite el estado del juego en tiempo real al frontend.
El loop de vision corre en un hilo separado y publica via asyncio.Queue.

Esta version usa la misma logica que cli.py:
  - DeckTracker para EV dinamico (via state machine)
  - CardDebouncer (1s) para confirmar cartas
  - Zonas dinamicas segun num_players
"""

import asyncio
import json
import threading
import time
from typing import Optional, Set

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from game.counter import CardCounter
from game.state_machine import BlackjackStateMachine, GameState
from vision.debouncer import CardDebouncer
from vision.zones import ZoneManager, build_zones

ws_router = APIRouter()
_clients: Set[WebSocket] = set()
_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
_event_loop: Optional[asyncio.AbstractEventLoop] = None

_latest_frame: Optional[bytes] = None
_frame_lock = threading.Lock()

_game_sm: Optional[BlackjackStateMachine] = None
_debouncer: CardDebouncer = CardDebouncer(threshold_s=1.0)
_zone_mgr: Optional[ZoneManager] = None
_config: dict = {}
_config_lock = threading.Lock()


def init(config: dict):
    """Inicializa el estado global del juego con la configuracion dada."""
    global _config
    with _config_lock:
        _config = config
    gcfg = config.get("game", {})
    _build_game(
        num_players=gcfg.get("num_players", 1),
        num_decks=gcfg.get("num_decks", 6),
        counting_system=gcfg.get("counting_system", "hilo"),
    )


def _build_game(num_players: int, num_decks: int, counting_system: str):
    global _game_sm, _zone_mgr
    counter = CardCounter(system=counting_system, num_decks=num_decks)
    _game_sm = BlackjackStateMachine(
        num_players=num_players,
        counter=counter,
        num_decks=num_decks,
    )
    _zone_mgr = ZoneManager(build_zones(num_players))
    _debouncer.reset()


def reconfigure(num_players: int, num_decks: int, counting_system: str) -> dict:
    """
    Aplica una nueva configuracion de partida (jugadores, mazos, sistema).
    Resetea por completo el juego y las zonas — pensado para invocarse
    antes de empezar a jugar.
    """
    from game.ev_calculator import clear_caches
    clear_caches()
    with _config_lock:
        gcfg = _config.setdefault("game", {})
        gcfg["num_players"]     = num_players
        gcfg["num_decks"]       = num_decks
        gcfg["counting_system"] = counting_system
    _build_game(num_players, num_decks, counting_system)
    return {
        "num_players": num_players,
        "num_decks": num_decks,
        "counting_system": counting_system,
    }


def get_config() -> dict:
    with _config_lock:
        gcfg = _config.get("game", {})
        return {
            "num_players":     gcfg.get("num_players", 1),
            "num_decks":       gcfg.get("num_decks", 6),
            "counting_system": gcfg.get("counting_system", "hilo"),
        }


def get_game_sm() -> Optional[BlackjackStateMachine]:
    return _game_sm


def get_debouncer() -> CardDebouncer:
    return _debouncer


def get_latest_frame() -> Optional[bytes]:
    with _frame_lock:
        return _latest_frame


def _set_latest_frame(data: bytes):
    global _latest_frame
    with _frame_lock:
        _latest_frame = data


async def broadcast(data: dict):
    global _clients
    msg = json.dumps(data)
    dead = set()
    for client in _clients.copy():
        try:
            await client.send_text(msg)
        except Exception:
            dead.add(client)
    _clients -= dead


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _event_loop
    await websocket.accept()
    _clients.add(websocket)
    _event_loop = asyncio.get_event_loop()
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.discard(websocket)


async def vision_dispatch_loop():
    while True:
        try:
            state = _queue.get_nowait()
            await broadcast(state)
        except asyncio.QueueEmpty:
            pass
        await asyncio.sleep(1 / 15)


def _draw_detections(frame: np.ndarray, detections, debouncer: CardDebouncer,
                     zone_mgr, now: float) -> np.ndarray:
    """Dibuja bboxes (color tenue + barra de progreso si aun no confirmadas)."""
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        zone = zone_mgr.get_zone(det.center[0], det.center[1], w, h) if zone_mgr else None
        base_color = _label_color(det.label)
        key = (zone, det.label) if zone else None
        prog = debouncer.progress(key, now) if key else 1.0

        if det.mask is not None:
            frame[det.mask] = (
                frame[det.mask] * 0.55 + np.array(base_color) * 0.45
            ).astype(np.uint8)

        if prog < 1.0:
            color = tuple(int(c * 0.5) for c in base_color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            bar_w = int((x2 - x1) * prog)
            cv2.rectangle(frame, (x1, y2 + 3), (x2, y2 + 7), (40, 40, 40), -1)
            cv2.rectangle(frame, (x1, y2 + 3), (x1 + bar_w, y2 + 7), (0, 220, 220), -1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 2)

        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 4, y1), base_color, -1)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def _label_color(label: str) -> tuple:
    if "Spades" in label or "Clubs" in label or "Hearts" in label:
        return (60, 60, 220)
    if "Diamonds" in label:
        return (220, 60, 60)
    if label in ("Joker", "Cards", "card_back"):
        return (180, 60, 220)
    return (0, 220, 100)


def start_vision_thread():
    t = threading.Thread(target=_vision_loop, daemon=True)
    t.start()


def _vision_loop():
    """Loop de vision: captura, infiere, asigna a zonas, debounce y emite.
    Lee `_zone_mgr` y `_game_sm` globales en cada iteracion para que la
    reconfiguracion en caliente tome efecto sin reiniciar el thread."""
    from vision.capture import CameraCapture
    from vision.detector import CardDetector
    from vision.zones import draw_zones

    det_cfg  = _config.get("detection", {})
    inference_fps = det_cfg.get("inference_fps", 15)
    frame_interval = 1.0 / inference_fps

    cam = CameraCapture(_config)

    detector = None
    try:
        detector = CardDetector(
            model_path=det_cfg.get("model_path", "models/best.pt"),
            confidence=det_cfg.get("confidence", 0.35),
            iou=det_cfg.get("iou", 0.45),
        )
    except Exception as e:
        print(f"[vision] Modelo no disponible ({e}), streaming sin inferencia")

    last_inference = 0.0
    last_detections = []

    while True:
        try:
            frame = cam.read()
        except RuntimeError:
            time.sleep(0.1)
            continue

        # Snapshot de globals (se pueden cambiar via reconfigure)
        zone_mgr = _zone_mgr
        sm       = _game_sm

        now = time.monotonic()
        h, w = frame.shape[:2]

        if detector and (now - last_inference) >= frame_interval:
            last_inference = now
            last_detections = detector.detect(frame)

            if sm and zone_mgr:
                visible = []
                for det in last_detections:
                    zone = zone_mgr.get_zone(det.center[0], det.center[1], w, h)
                    if zone:
                        visible.append((zone, det.label))

                if sm.state in (GameState.DEALING, GameState.PLAYER_TURN,
                                GameState.DEALER_TURN):
                    for zone, label in _debouncer.tick(visible, now):
                        sm.add_card(zone, label)
                else:
                    _debouncer.reset()

        if sm and zone_mgr:
            frame = draw_zones(frame, zone_mgr, sm)
        if last_detections and zone_mgr:
            frame = _draw_detections(frame, last_detections, _debouncer, zone_mgr, now)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            _set_latest_frame(bytes(buf))

        if sm and _event_loop and not _queue.full():
            state = sm.to_dict()
            asyncio.run_coroutine_threadsafe(_queue.put(state), _event_loop)

        if now - last_inference < frame_interval * 0.5:
            time.sleep(0.01)
