"""
Mantiene la composicion exacta del zapato: cuantas cartas de cada valor quedan.
Permite registrar cartas vistas y deshacer (devolver) cartas mal detectadas.
"""

from typing import Dict, Iterable, List

# Valores en formato de label sin palo
VALUES: List[str] = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

# Etiquetas que no son cartas reales del zapato (no afectan al conteo)
NON_CARD_LABELS = {"card_back", "Cards", "Joker", "joker"}


def label_to_value(label: str) -> str | None:
    """'A Spades' -> 'A'   '10 Clubs' -> '10'   'card_back' -> None."""
    if label in NON_CARD_LABELS:
        return None
    val = label.split()[0] if " " in label else label
    return val if val in VALUES else None


class DeckTracker:
    """
    Composicion residual del zapato.
    `remaining[v]` = cuantas cartas con valor `v` quedan en el zapato.
    """

    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.remaining: Dict[str, int] = {}
        self.reset()

    def reset(self):
        # 4 cartas por valor por mazo, salvo 10/J/Q/K que son 4 cada una (16 figuras)
        self.remaining = {v: 4 * self.num_decks for v in VALUES}

    def total(self) -> int:
        return sum(self.remaining.values())

    def remove(self, label: str) -> bool:
        """Marca una carta como vista. Devuelve True si se descontaron cartas."""
        v = label_to_value(label)
        if v is None or self.remaining.get(v, 0) <= 0:
            return False
        self.remaining[v] -= 1
        return True

    def restore(self, label: str) -> bool:
        """Devuelve una carta al zapato (al borrarla por error). True si se aplico."""
        v = label_to_value(label)
        if v is None:
            return False
        # No exceder el numero original
        max_per_value = 4 * self.num_decks
        if self.remaining[v] >= max_per_value:
            return False
        self.remaining[v] += 1
        return True

    def probability(self, value: str) -> float:
        """Probabilidad de sacar una carta de valor `value` ahora mismo."""
        tot = self.total()
        if tot <= 0:
            return 0.0
        return self.remaining.get(value, 0) / tot

    def value_probabilities(self) -> Dict[int, float]:
        """
        Probabilidades agregadas por VALOR NUMERICO (1-10), util para EV.
        El As cuenta como 1 aqui — la logica de blandas la maneja el EV.
        Los 10/J/Q/K se agregan en la clave 10.
        """
        tot = self.total()
        if tot <= 0:
            return {}
        probs: Dict[int, float] = {n: 0.0 for n in range(1, 11)}
        for v, count in self.remaining.items():
            p = count / tot
            if v == "A":
                probs[1] += p
            elif v in ("10", "J", "Q", "K"):
                probs[10] += p
            else:
                probs[int(v)] += p
        return probs

    def snapshot(self) -> Dict[str, int]:
        return dict(self.remaining)

    def state(self) -> dict:
        return {
            "num_decks": self.num_decks,
            "total_remaining": self.total(),
            "by_value": self.snapshot(),
        }
