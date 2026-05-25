"""
Maquina de estados del juego. Controla turnos y transiciones.
Estados: WAITING | DEALING | PLAYER_TURN | DEALER_TURN | ROUND_END
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from game.engine import hand_total, is_bust, is_blackjack, dealer_should_hit
from game.counter import CardCounter
from game.deck_tracker import DeckTracker, label_to_value
from game.strategy import suggest_action, suggest_action_with_ev

# Estas clases pueden aparecer varias veces en la misma sesion/ronda
REPEATABLE_LABELS: Set[str] = {"Joker", "joker", "Cards", "card_back"}


class GameState(str, Enum):
    WAITING = "waiting"
    DEALING = "dealing"
    PLAYER_TURN = "player_turn"
    DEALER_TURN = "dealer_turn"
    ROUND_END = "round_end"


@dataclass
class PlayerHand:
    player_id: str
    cards: List[str] = field(default_factory=list)
    stood: bool = False

    @property
    def total(self):
        return hand_total(self.cards)

    @property
    def bust(self):
        return is_bust(self.cards)

    @property
    def blackjack(self):
        return is_blackjack(self.cards)


class BlackjackStateMachine:
    def __init__(self, num_players: int, counter: CardCounter, num_decks: int = 6):
        self.num_players = num_players
        self.counter = counter
        self.deck = DeckTracker(num_decks=num_decks)
        self.state = GameState.WAITING
        self.game_active = False
        self.dealer_hand = PlayerHand("dealer")
        self.player_hands: Dict[str, PlayerHand] = {
            f"player_{i+1}": PlayerHand(f"player_{i+1}")
            for i in range(num_players)
        }
        self.current_player_idx = 0
        self.player_ids = list(self.player_hands.keys())
        # Cartas ya registradas esta ronda (evita contar la misma carta varias veces por frame)
        self._seen_this_round: Set[str] = set()
        # Historial cronologico (zone, card) — incluye cartas de rondas previas del mismo zapato
        self.history: List[Tuple[str, str]] = []
        # Sugerencias cacheadas — se invalidan cada vez que cambia el estado.
        # Evita recalcular EV (caro) en cada frame del bucle de render.
        self._suggestions: Dict[str, Tuple[Optional[str], Optional[dict]]] = {}
        self._suggestions_dirty = True
        # Marcador de la partida actual: rondas ganadas por dealer/jugadores
        # y empates por jugador. Se resetea con `start_game`, NO con reset_round.
        self.wins: Dict[str, int] = {"dealer": 0, **{pid: 0 for pid in self.player_ids}}
        self.pushes: Dict[str, int] = {pid: 0 for pid in self.player_ids}
        self.rounds_played: int = 0

    @property
    def current_player(self) -> Optional[str]:
        if self.state == GameState.PLAYER_TURN and self.current_player_idx < len(self.player_ids):
            return self.player_ids[self.current_player_idx]
        return None

    def start_game(self):
        """Inicia una nueva partida completa — resetea contador, zapato, historial y marcador."""
        self.game_active = True
        self.state = GameState.DEALING
        self.counter.reset()
        self.deck.reset()
        self._seen_this_round = set()
        self.history = []
        self.dealer_hand = PlayerHand("dealer")
        self.player_hands = {
            f"player_{i+1}": PlayerHand(f"player_{i+1}")
            for i in range(self.num_players)
        }
        self.current_player_idx = 0
        self._suggestions = {}
        self._suggestions_dirty = True
        # Reset del marcador (partida nueva)
        self.wins = {"dealer": 0, **{pid: 0 for pid in self.player_ids}}
        self.pushes = {pid: 0 for pid in self.player_ids}
        self.rounds_played = 0

    def add_card(self, zone: str, card: str):
        """
        Registra una carta nueva detectada en una zona.
        Si la carta ya fue registrada esta ronda y no es REPEATABLE, la ignora.
        """
        if card not in REPEATABLE_LABELS and card in self._seen_this_round:
            return  # ya contada, no duplicar
        self._seen_this_round.add(card)
        self.counter.register(card)
        self.deck.remove(card)
        self.history.append((zone, card))
        if zone == "dealer":
            if card not in self.dealer_hand.cards:
                self.dealer_hand.cards.append(card)
        elif zone in self.player_hands:
            if card not in self.player_hands[zone].cards:
                self.player_hands[zone].cards.append(card)
        self._suggestions_dirty = True

    def remove_card(self, index: int) -> bool:
        """
        Elimina una carta del historial (por mala deteccion). Refunde el contador
        y devuelve la carta al zapato. `index` es la posicion en `self.history`.
        """
        if index < 0 or index >= len(self.history):
            return False
        zone, card = self.history.pop(index)

        # Devolver al zapato
        self.deck.restore(card)

        # Revertir contador (delta opuesto)
        v = label_to_value(card)
        if v is not None:
            delta = self.counter.system.get(v, 0)
            self.counter.running_count -= delta
            self.counter.cards_seen = max(0, self.counter.cards_seen - 1)

        # Quitar de la mano si sigue ahi
        if zone == "dealer" and card in self.dealer_hand.cards:
            self.dealer_hand.cards.remove(card)
        elif zone in self.player_hands and card in self.player_hands[zone].cards:
            self.player_hands[zone].cards.remove(card)

        # Permitir que vuelva a registrarse si reaparece
        self._seen_this_round.discard(card)
        self._suggestions_dirty = True
        return True

    def undo_last(self) -> bool:
        """Atajo: borra la ultima carta registrada."""
        if not self.history:
            return False
        return self.remove_card(len(self.history) - 1)

    def player_stand(self, player_id: str):
        if player_id in self.player_hands:
            self.player_hands[player_id].stood = True
            self._advance_turn()
            self._suggestions_dirty = True

    def _advance_turn(self):
        self.current_player_idx += 1
        if self.current_player_idx >= len(self.player_ids):
            self.state = GameState.DEALER_TURN

    def _finalize_round(self) -> bool:
        """
        Evalua la ronda actual y suma al marcador. Solo cuenta si hay datos
        suficientes (dealer y al menos un jugador con >=2 cartas).
        Devuelve True si se sumo algo.
        """
        if len(self.dealer_hand.cards) < 2:
            return False
        any_scored = False
        d_total = self.dealer_hand.total
        d_bust  = self.dealer_hand.bust
        d_bj    = self.dealer_hand.blackjack

        for pid, hand in self.player_hands.items():
            if len(hand.cards) < 2:
                continue
            any_scored = True
            if hand.bust:
                self.wins["dealer"] += 1
            elif d_bust:
                self.wins[pid] += 1
            elif hand.blackjack and not d_bj:
                self.wins[pid] += 1
            elif d_bj and not hand.blackjack:
                self.wins["dealer"] += 1
            elif hand.total > d_total:
                self.wins[pid] += 1
            elif hand.total < d_total:
                self.wins["dealer"] += 1
            else:
                self.pushes[pid] += 1
        if any_scored:
            self.rounds_played += 1
        return any_scored

    def reset_round(self):
        """Nueva ronda — evalua marcador, limpia manos. Conserva el zapato."""
        self._finalize_round()
        self.state = GameState.DEALING
        self._seen_this_round = set()
        self.dealer_hand = PlayerHand("dealer")
        self.player_hands = {
            f"player_{i+1}": PlayerHand(f"player_{i+1}")
            for i in range(self.num_players)
        }
        self.current_player_idx = 0
        self._suggestions = {}
        self._suggestions_dirty = True

    def reset(self):
        """Alias de reset_round para compatibilidad con routes.py."""
        self.reset_round()

    def _refresh_suggestions(self):
        """Recalcula sugerencias y EVs para todos los jugadores. Se llama solo
        cuando algo del estado cambia, no cada frame."""
        self._suggestions = {}
        dealer_upcard = self.dealer_hand.cards[0] if self.dealer_hand.cards else None
        for pid, hand in self.player_hands.items():
            if hand.bust or hand.stood or len(hand.cards) < 2:
                self._suggestions[pid] = (None, None)
            else:
                self._suggestions[pid] = suggest_action_with_ev(
                    hand.cards, dealer_upcard, self.deck
                )
        self._suggestions_dirty = False

    def to_dict(self) -> dict:
        if self._suggestions_dirty:
            self._refresh_suggestions()
        players = {}
        for pid, hand in self.player_hands.items():
            sug, evs = self._suggestions.get(pid, (None, None))
            players[pid] = {
                "cards": hand.cards,
                "total": hand.total,
                "bust": hand.bust,
                "blackjack": hand.blackjack,
                "stood": hand.stood,
                "suggestion": sug,
                "evs": evs,
            }
        return {
            "state": self.state.value,
            "game_active": self.game_active,
            "current_player": self.current_player,
            "config": {
                "num_players": len(self.player_ids),
                "num_decks": self.deck.num_decks,
            },
            "dealer": {
                "cards": self.dealer_hand.cards,
                "total": self.dealer_hand.total,
            },
            "players": players,
            "counting": self.counter.state(),
            "deck": self.deck.state(),
            "history": [{"zone": z, "card": c} for z, c in self.history],
            "seen_cards": sorted(self._seen_this_round),
            "score": {
                "dealer": self.wins["dealer"],
                "players": {pid: self.wins[pid] for pid in self.player_ids},
                "pushes": dict(self.pushes),
                "rounds": self.rounds_played,
            },
        }
