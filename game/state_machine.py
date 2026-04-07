"""
Maquina de estados del juego. Controla turnos y transiciones.
Estados: WAITING | DEALING | PLAYER_TURN | DEALER_TURN | ROUND_END
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from game.engine import hand_total, is_bust, is_blackjack, dealer_should_hit
from game.counter import CardCounter
from game.strategy import suggest_action


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
    def __init__(self, num_players: int, counter: CardCounter):
        self.num_players = num_players
        self.counter = counter
        self.state = GameState.WAITING
        self.dealer_hand = PlayerHand("dealer")
        self.player_hands: Dict[str, PlayerHand] = {
            f"player_{i+1}": PlayerHand(f"player_{i+1}")
            for i in range(num_players)
        }
        self.current_player_idx = 0
        self.player_ids = list(self.player_hands.keys())

    @property
    def current_player(self) -> Optional[str]:
        if self.state == GameState.PLAYER_TURN and self.current_player_idx < len(self.player_ids):
            return self.player_ids[self.current_player_idx]
        return None

    def add_card(self, zone: str, card: str):
        """Registra una carta nueva detectada en una zona."""
        self.counter.register(card)
        if zone == "dealer":
            self.dealer_hand.cards.append(card)
        elif zone in self.player_hands:
            self.player_hands[zone].cards.append(card)

    def player_stand(self, player_id: str):
        if player_id in self.player_hands:
            self.player_hands[player_id].stood = True
            self._advance_turn()

    def _advance_turn(self):
        self.current_player_idx += 1
        if self.current_player_idx >= len(self.player_ids):
            self.state = GameState.DEALER_TURN

    def reset(self):
        self.state = GameState.DEALING
        self.dealer_hand = PlayerHand("dealer")
        self.player_hands = {
            f"player_{i+1}": PlayerHand(f"player_{i+1}")
            for i in range(self.num_players)
        }
        self.current_player_idx = 0

    def to_dict(self) -> dict:
        dealer_upcard = self.dealer_hand.cards[0] if self.dealer_hand.cards else None
        players = {}
        for pid, hand in self.player_hands.items():
            suggestion = (
                suggest_action(hand.cards, dealer_upcard)
                if dealer_upcard and not hand.bust and not hand.stood
                else None
            )
            players[pid] = {
                "cards": hand.cards,
                "total": hand.total,
                "bust": hand.bust,
                "blackjack": hand.blackjack,
                "stood": hand.stood,
                "suggestion": suggestion,
            }
        return {
            "state": self.state.value,
            "current_player": self.current_player,
            "dealer": {
                "cards": self.dealer_hand.cards,
                "total": self.dealer_hand.total,
            },
            "players": players,
            "counting": self.counter.state(),
        }
