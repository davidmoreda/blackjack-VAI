"""REST endpoints para control del juego."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api")

# Instancia global del juego (se inicializa en ws.py al arrancar)
game_state = None


class StandRequest(BaseModel):
    player_id: str


class ConfigRequest(BaseModel):
    num_decks: int = 6
    num_players: int = 2
    counting_system: str = "hilo"


@router.post("/stand")
async def stand(req: StandRequest):
    """El jugador se planta."""
    if game_state:
        game_state.player_stand(req.player_id)
    return {"ok": True}


@router.post("/reset")
async def reset():
    """Nueva ronda."""
    if game_state:
        game_state.reset()
    return {"ok": True}


@router.get("/state")
async def get_state():
    """Estado actual del juego."""
    if game_state:
        return game_state.to_dict()
    return {}


@router.post("/config")
async def configure(req: ConfigRequest):
    """Reconfigurar numero de mazos, jugadores y sistema de conteo."""
    # TODO: reiniciar juego con nueva configuracion
    return {"ok": True, "config": req.dict()}
