"""REST endpoints para control del juego."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api")


def _sm():
    from api.ws import get_game_sm
    sm = get_game_sm()
    if sm is None:
        raise HTTPException(status_code=503, detail="Game not initialized")
    return sm


class StandRequest(BaseModel):
    player_id: str


class ConfigRequest(BaseModel):
    num_decks: int = 6
    num_players: int = 2
    counting_system: str = "hilo"


@router.post("/start")
async def start_game():
    """Inicia una nueva partida — resetea todo e inicia el conteo."""
    _sm().start_game()
    return {"ok": True}


@router.post("/stand")
async def stand(req: StandRequest):
    """El jugador se planta."""
    _sm().player_stand(req.player_id)
    return {"ok": True}


@router.post("/reset")
async def reset():
    """Nueva ronda — limpia manos pero conserva el running count del zapato."""
    _sm().reset_round()
    return {"ok": True}


@router.get("/state")
async def get_state():
    """Estado actual del juego."""
    return _sm().to_dict()


@router.post("/config")
async def configure(req: ConfigRequest):
    """Reconfigurar numero de mazos, jugadores y sistema de conteo."""
    return {"ok": True, "config": req.dict()}
