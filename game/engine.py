"""Motor de reglas del Blackjack."""

from typing import List

# Valor numerico de cada carta (palo ignorado para el calculo de mano)
CARD_VALUES = {
    "A": 11, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6,  "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10,
}


def card_value(label: str) -> int:
    """Devuelve el valor numerico de una carta (label tipo 'AS', 'KH', '10D')."""
    value = label[:-1] if label != "card_back" else "0"
    return CARD_VALUES.get(value, 0)


def hand_total(cards: List[str]) -> int:
    """Calcula el total de una mano. Los ases valen 11 o 1 segun convenga."""
    total = 0
    aces = 0
    for card in cards:
        v = card_value(card)
        if card[:-1] == "A":
            aces += 1
        total += v
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def is_bust(cards: List[str]) -> bool:
    return hand_total(cards) > 21


def is_blackjack(cards: List[str]) -> bool:
    return len(cards) == 2 and hand_total(cards) == 21


def dealer_should_hit(cards: List[str]) -> bool:
    """El dealer pide carta si tiene menos de 17 (regla estandar)."""
    return hand_total(cards) < 17
