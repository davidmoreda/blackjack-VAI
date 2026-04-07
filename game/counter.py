"""
Sistemas de conteo de cartas.
Soporta Hi-Lo, KO y Omega II.
"""

from typing import Dict, List

# running_count delta por carta para cada sistema
# Clave: valor de carta (sin palo). Valor: delta del conteo.
SYSTEMS: Dict[str, Dict[str, int]] = {
    "hilo": {
        "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
        "7": 0, "8": 0, "9": 0,
        "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
    },
    "ko": {
        "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1,
        "8": 0, "9": 0,
        "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
    },
    "omega2": {
        "2": 1, "3": 1, "4": 2, "5": 2, "6": 2, "7": 1,
        "8": 0, "9": -1,
        "10": -2, "J": -2, "Q": -2, "K": -2, "A": 0,
    },
}


class CardCounter:
    def __init__(self, system: str = "hilo", num_decks: int = 6):
        if system not in SYSTEMS:
            raise ValueError(f"Sistema desconocido: {system}. Opciones: {list(SYSTEMS)}")
        self.system = SYSTEMS[system]
        self.num_decks = num_decks
        self.running_count = 0
        self.cards_seen = 0
        self.total_cards = num_decks * 52

    def register(self, label: str):
        """Registra una carta vista (label tipo 'AS', 'KH')."""
        if label == "card_back":
            return
        value = label[:-1]
        delta = self.system.get(value, 0)
        self.running_count += delta
        self.cards_seen += 1

    @property
    def decks_remaining(self) -> float:
        remaining = max(self.total_cards - self.cards_seen, 1)
        return remaining / 52

    @property
    def true_count(self) -> float:
        """True count = running count / mazos restantes."""
        return self.running_count / self.decks_remaining

    def reset(self):
        self.running_count = 0
        self.cards_seen = 0

    def state(self) -> dict:
        return {
            "running_count": self.running_count,
            "true_count": round(self.true_count, 2),
            "cards_seen": self.cards_seen,
            "decks_remaining": round(self.decks_remaining, 1),
        }
