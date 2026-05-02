"""
Punto de entrada del servidor FastAPI.
Uso: python -m uvicorn api.main:app --reload
     o: python api/main.py
"""

import asyncio
from pathlib import Path

import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.ws import ws_router, init, start_vision_thread, get_latest_frame, vision_dispatch_loop

app = FastAPI(title="Blackjack VAI")
app.include_router(router)
app.include_router(ws_router)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.on_event("startup")
async def startup():
    config_path = Path("config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    init(cfg)
    start_vision_thread()
    asyncio.create_task(vision_dispatch_loop())


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


@app.get("/video_feed")
async def video_feed():
    """Streaming MJPEG de la camara con bboxes dibujados."""
    async def generate():
        blank = None
        while True:
            frame = get_latest_frame()
            if frame is None:
                # Frame en negro mientras no hay camara
                if blank is None:
                    import cv2, numpy as np
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "Camara no disponible", (120, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    _, buf = cv2.imencode(".jpg", img)
                    blank = bytes(buf)
                frame = blank
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(1 / 30)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    api_cfg = cfg.get("api", {})
    uvicorn.run("api.main:app", host=api_cfg.get("host", "0.0.0.0"),
                port=api_cfg.get("port", 8000), reload=False)
