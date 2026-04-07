"""
Punto de entrada del servidor FastAPI.
Uso: python api/main.py
     o: uvicorn api.main:app --reload
"""

import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router
from api.ws import ws_router

app = FastAPI(title="Blackjack VAI")
app.include_router(router)
app.include_router(ws_router)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    api_cfg = cfg.get("api", {})
    uvicorn.run("api.main:app", host=api_cfg.get("host", "0.0.0.0"),
                port=api_cfg.get("port", 8000), reload=True)
