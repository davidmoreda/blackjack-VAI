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


def _debouncer():
    from api.ws import get_debouncer
    return get_debouncer()


class StandRequest(BaseModel):
    player_id: str


class ConfigRequest(BaseModel):
    num_decks: int = 6
    num_players: int = 1
    counting_system: str = "hilo"


COUNTING_SYSTEMS = {"hilo", "ko", "omega2"}
NUM_DECKS_OPTIONS = {1, 2, 4, 6, 8}
NUM_PLAYERS_RANGE = range(1, 8)


@router.post("/start")
async def start_game():
    """Inicia una nueva partida — resetea contador, marcador y zapato."""
    _sm().start_game()
    _debouncer().reset()
    return {"ok": True}


@router.post("/stand")
async def stand(req: StandRequest):
    _sm().player_stand(req.player_id)
    return {"ok": True}


@router.post("/reset")
async def reset():
    """Nueva ronda — evalua marcador y limpia manos. Conserva el zapato."""
    _sm().reset_round()
    _debouncer().reset()
    return {"ok": True}


@router.post("/undo")
async def undo_last():
    """Deshace la ultima carta detectada. Refunde contador y zapato."""
    sm = _sm()
    if not sm.history:
        raise HTTPException(status_code=400, detail="No hay cartas que deshacer")
    zone, card = sm.history[-1]
    sm.undo_last()
    _debouncer().forget((zone, card))
    return {"ok": True, "removed": {"zone": zone, "card": card}}


@router.post("/remove/{idx}")
async def remove_card(idx: int):
    """Elimina la carta en la posicion `idx` del historial."""
    sm = _sm()
    if idx < 0 or idx >= len(sm.history):
        raise HTTPException(status_code=400, detail="Indice fuera de rango")
    zone, card = sm.history[idx]
    sm.remove_card(idx)
    _debouncer().forget((zone, card))
    return {"ok": True, "removed": {"zone": zone, "card": card}}


@router.get("/state")
async def get_state():
    return _sm().to_dict()


@router.get("/config")
async def get_current_config():
    """Devuelve la configuracion actual de la partida."""
    from api.ws import get_config
    return get_config()


@router.post("/config")
async def configure(req: ConfigRequest):
    """
    Reconstruye la partida con la nueva configuracion (jugadores / mazos /
    sistema). Resetea el marcador, las manos y el zapato.
    """
    if req.counting_system not in COUNTING_SYSTEMS:
        raise HTTPException(400, f"sistema desconocido: {req.counting_system}")
    if req.num_decks not in NUM_DECKS_OPTIONS:
        raise HTTPException(400, f"num_decks debe estar en {sorted(NUM_DECKS_OPTIONS)}")
    if req.num_players not in NUM_PLAYERS_RANGE:
        raise HTTPException(400, f"num_players entre 1 y 7")

    from api.ws import reconfigure
    applied = reconfigure(
        num_players=req.num_players,
        num_decks=req.num_decks,
        counting_system=req.counting_system,
    )
    return {"ok": True, "config": applied}
