"""
Basic Strategy: devuelve la accion optima para una mano dada.
Basado en la tabla estandar de Basic Strategy para blackjack.
Acciones: HIT | STAND | DOUBLE | SPLIT
"""

from game.engine import hand_total


def suggest_action(player_cards: list, dealer_upcard: str) -> str:
    """
    Devuelve la accion recomendada por Basic Strategy.
    dealer_upcard: label de la carta visible del dealer (e.g. 'KH', '6D').
    """
    if len(player_cards) < 2:
        return "WAIT"

    dealer_val = _dealer_value(dealer_upcard)
    total = hand_total(player_cards)
    soft = _is_soft(player_cards)
    pair = _is_pair(player_cards)

    if pair:
        return _pair_strategy(player_cards[0], dealer_val)
    if soft:
        return _soft_strategy(total, dealer_val)
    return _hard_strategy(total, dealer_val)


def _dealer_value(label: str) -> int:
    val = label[:-1]
    mapping = {"A": 11, "J": 10, "Q": 10, "K": 10}
    return mapping.get(val, int(val) if val.isdigit() else 10)


def _is_soft(cards: list) -> bool:
    total_no_ace = sum(10 if c[:-1] in ("10", "J", "Q", "K") else int(c[:-1]) if c[:-1].isdigit() else 0 for c in cards)
    aces = sum(1 for c in cards if c[:-1] == "A")
    return aces > 0 and total_no_ace + 11 + (aces - 1) <= 21


def _is_pair(cards: list) -> bool:
    return len(cards) == 2 and cards[0][:-1] == cards[1][:-1]


def _hard_strategy(total: int, dealer: int) -> str:
    if total >= 17:
        return "STAND"
    if total >= 13 and dealer <= 6:
        return "STAND"
    if total == 12 and 4 <= dealer <= 6:
        return "STAND"
    if total == 11:
        return "DOUBLE"
    if total == 10 and dealer <= 9:
        return "DOUBLE"
    if total == 9 and 3 <= dealer <= 6:
        return "DOUBLE"
    return "HIT"


def _soft_strategy(total: int, dealer: int) -> str:
    if total >= 19:
        return "STAND"
    if total == 18:
        return "STAND" if 2 <= dealer <= 8 else "HIT"
    if total in (17, 18) and 3 <= dealer <= 6:
        return "DOUBLE"
    return "HIT"


def _pair_strategy(card: str, dealer: int) -> str:
    val = card[:-1]
    always_split = {"A", "8"}
    never_split = {"10", "J", "Q", "K", "5"}
    if val in always_split:
        return "SPLIT"
    if val in never_split:
        return "STAND" if val != "5" else "DOUBLE"
    if val in {"2", "3", "7"} and dealer <= 7:
        return "SPLIT"
    if val == "4" and 5 <= dealer <= 6:
        return "SPLIT"
    if val == "6" and dealer <= 6:
        return "SPLIT"
    if val == "9" and dealer not in (7, 10, 11):
        return "SPLIT"
    return "HIT"
