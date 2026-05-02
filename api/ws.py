"""
WebSocket: emite el estado del juego en tiempo real al frontend.
El loop de vision corre en un hilo separado y publica via asyncio.Queue.
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
from game.state_machine import BlackjackStateMachine

ws_router = APIRouter()
_clients: Set[WebSocket] = set()
_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Frame JPEG mas reciente para streaming MJPEG (bytes)
_latest_frame: Optional[bytes] = None
_frame_lock = threading.Lock()

# Instancia global compartida del juego
_game_sm: Optional[BlackjackStateMachine] = None
_config: dict = {}


def init(config: dict):
    """Inicializa el estado global del juego con la configuracion dada."""
    global _game_sm, _config
    _config = config
    gcfg = config.get("game", {})
    counter = CardCounter(
        system=gcfg.get("counting_system", "hilo"),
        num_decks=gcfg.get("num_decks", 6),
    )
    _game_sm = BlackjackStateMachine(
        num_players=gcfg.get("num_players", 1),
        counter=counter,
    )


def get_game_sm() -> Optional[BlackjackStateMachine]:
    return _game_sm


def get_latest_frame() -> Optional[bytes]:
    with _frame_lock:
        return _latest_frame


def _set_latest_frame(data: bytes):
    global _latest_frame
    with _frame_lock:
        _latest_frame = data


async def broadcast(data: dict):
    """Envia el estado a todos los clientes conectados."""
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
    """Tarea asyncio: consume estados del hilo de vision y los emite por WebSocket."""
    while True:
        try:
            state = _queue.get_nowait()
            await broadcast(state)
        except asyncio.QueueEmpty:
            pass
        await asyncio.sleep(1 / 15)


def _draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    """Dibuja mascaras de segmentacion, bboxes y etiquetas sobre el frame."""
    overlay = frame.copy()

    for det in detections:
        color = _label_color(det.label)

        # Mascara de segmentacion semitransparente
        if det.mask is not None:
            overlay[det.mask] = (
                overlay[det.mask] * 0.45 + np.array(color) * 0.55
            ).astype(np.uint8)

        # Bounding box
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Etiqueta con fondo
        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return overlay


def _label_color(label: str) -> tuple:
    """Color distinto por palo para distinguir cartas visualmente."""
    if "Spades" in label or "Clubs" in label:
        return (60, 60, 220)    # azul
    if "Hearts" in label:
        return (60, 60, 220)    # azul (corazones)
    if "Diamonds" in label:
        return (220, 60, 60)    # rojo
    if label in ("Joker", "Cards", "card_back"):
        return (180, 60, 220)   # morado
    return (0, 220, 100)        # verde por defecto


def start_vision_thread():
    """Arranca el hilo de captura de camara e inferencia YOLO."""
    t = threading.Thread(target=_vision_loop, daemon=True)
    t.start()


def _vision_loop():
    """
    Loop principal de vision. Captura frames, infiere con YOLO,
    asigna cartas a zonas y emite via WebSocket.
    """
    from vision.capture import CameraCapture
    from vision.detector import CardDetector
    from vision.zones import ZoneManager, draw_zones

    det_cfg = _config.get("detection", {})
    inference_fps = det_cfg.get("inference_fps", 15)
    frame_interval = 1.0 / inference_fps

    cam = CameraCapture(_config)

    detector = None
    model_path = det_cfg.get("model_path", "models/best.pt")
    try:
        detector = CardDetector(
            model_path=model_path,
            confidence=det_cfg.get("confidence", 0.35),
            iou=det_cfg.get("iou", 0.45),
        )
    except Exception as e:
        print(f"[vision] Modelo no disponible ({e}), streaming sin inferencia")

    # Zonas de la mesa
    zone_mgr = None
    zones_cfg = _config.get("zones")
    if zones_cfg:
        zone_mgr = ZoneManager(zones_cfg)

    last_inference = 0.0

    # track_id -> {'zone': str, 'label': str, 'frames': int}
    track_state: dict = {}
    registered_ids: set = set()
    CONFIRM_FRAMES = 4
    detections = []
    last_round_seen: set = set()

    while True:
        try:
            frame = cam.read()
        except RuntimeError:
            time.sleep(0.1)
            continue

        now = time.time()

        if detector and _game_sm and _game_sm.game_active and (now - last_inference) >= frame_interval:
            last_inference = now

            # Detectar reset de ronda
            current_seen = _game_sm._seen_this_round
            if len(current_seen) < len(last_round_seen):
                registered_ids.clear()
                track_state.clear()
                detections = []
            last_round_seen = set(current_seen)

            detections = detector.detect(frame)
            h, w = frame.shape[:2]
            active_ids = {det.track_id for det in detections if det.track_id != -1}

            for det in detections:
                tid = det.track_id
                if tid == -1:
                    continue

                # Determinar zona por posicion del centro
                cx, cy = det.center
                zone = zone_mgr.get_zone(cx, cy, w, h) if zone_mgr else "player_1"
                if zone is None:
                    zone = "player_1"

                prev = track_state.get(tid)
                if prev is None or prev['zone'] != zone:
                    # Nueva deteccion o cambio de zona — reiniciar contador
                    track_state[tid] = {'zone': zone, 'label': det.label, 'frames': 1}
                else:
                    prev['frames'] += 1
                    if prev['frames'] == CONFIRM_FRAMES and tid not in registered_ids:
                        registered_ids.add(tid)
                        _game_sm.add_card(zone, det.label)

            # Limpiar ids que desaparecieron
            for tid in set(track_state) - active_ids:
                track_state.pop(tid, None)

        # Overlay de zonas — siempre visible para ayudar a encuadrar la camara
        if zone_mgr and _game_sm:
            frame = draw_zones(frame, zone_mgr, _game_sm)

        # Bboxes de las detecciones encima
        if detections:
            frame = _draw_detections(frame, detections)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            _set_latest_frame(bytes(buf))

        if _game_sm and _event_loop and not _queue.full():
            state = _game_sm.to_dict()
            asyncio.run_coroutine_threadsafe(_queue.put(state), _event_loop)

        if now - last_inference < frame_interval * 0.5:
            time.sleep(0.01)
