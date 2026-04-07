"""
WebSocket: emite el estado del juego en tiempo real al frontend.
El loop de vision corre en un hilo separado y publica via asyncio.Queue.
"""

import asyncio
import json
import threading
from typing import Set

import yaml
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()
_clients: Set[WebSocket] = set()
_queue: asyncio.Queue = asyncio.Queue()


async def broadcast(data: dict):
    """Envia el estado a todos los clientes conectados."""
    msg = json.dumps(data)
    dead = set()
    for ws in _clients.copy():
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _clients -= dead


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            # Mantiene la conexion viva esperando mensajes del cliente
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.discard(websocket)


async def _vision_loop():
    """
    Loop principal de vision. Se ejecuta como tarea asyncio.
    Lee frames, detecta cartas, actualiza el estado y emite via WebSocket.
    TODO: integrar con vision.capture y vision.detector una vez entrenado el modelo.
    """
    import time
    while True:
        if not _queue.empty():
            state = await _queue.get()
            await broadcast(state)
        await asyncio.sleep(1 / 15)  # ~15 fps


def start_vision_thread(game_sm, config: dict):
    """
    Arranca el hilo de captura de camara.
    game_sm: instancia de BlackjackStateMachine
    """
    import cv2
    from vision.capture import CameraCapture
    from vision.detector import CardDetector
    from vision.zone_manager import ZoneManager, Zone

    # TODO: cargar zonas desde config o zona_setup guardada
    zones = []  # Se definen con zone_setup.py

    def _run():
        cam = CameraCapture(config)
        # detector = CardDetector(config["detection"]["model_path"])
        while True:
            frame = cam.read()
            # detections = detector.detect(frame)
            # zone_map = ZoneManager(zones).assign(detections)
            # ... actualizar game_sm con nuevas cartas
            state = game_sm.to_dict()
            asyncio.run_coroutine_threadsafe(_queue.put(state), asyncio.get_event_loop())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
