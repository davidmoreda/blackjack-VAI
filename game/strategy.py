"""
Basic Strategy: accion optima para cada mano.
Acepta labels YOLO: 'A Spades', '10 Hearts', 'K Clubs', etc.
Acciones: HIT | STAND | DOUBLE | SPLIT | WAIT

Tambien expone `suggest_action_with_ev()` que usa el calculador EV dinamico
si se le pasa un DeckTracker — devuelve la accion con mayor valor esperado
en lugar de leer de la tabla estatica.
"""

from typing import Tuple

from game.engine import hand_total


def _val(label: str) -> str:
    """Extrae el valor de un label YOLO: 'K Hearts' -> 'K', '10 Clubs' -> '10'."""
    return label.split()[0] if " " in label else label


def _dealer_value(label: str) -> int:
    v = _val(label)
    mapping = {"A": 11, "J": 10, "Q": 10, "K": 10}
    try:
        return mapping.get(v, int(v))
    except ValueError:
        return 10


def _is_soft(cards: list) -> bool:
    vals = [_val(c) for c in cards]
    total_no_ace = 0
    aces = 0
    for v in vals:
        if v == "A":
            aces += 1
        elif v in ("J", "Q", "K"):
            total_no_ace += 10
        else:
            try:
                total_no_ace += int(v)
            except ValueError:
                pass
    return aces > 0 and total_no_ace + 11 + (aces - 1) <= 21


def _is_pair(cards: list) -> bool:
    return len(cards) == 2 and _val(cards[0]) == _val(cards[1])


def suggest_action(player_cards: list, dealer_upcard: str | None) -> str:
    if len(player_cards) < 2:
        return "WAIT"
    if dealer_upcard is None:
        # Sin carta del dealer usamos dealer=10 (peor caso más frecuente)
        dealer_upcard = "10 Spades"

    dealer = _dealer_value(dealer_upcard)
    total  = hand_total(player_cards)
    soft   = _is_soft(player_cards)
    pair   = _is_pair(player_cards)

    if pair:
        return _pair_strategy(_val(player_cards[0]), dealer)
    if soft:
        return _soft_strategy(total, dealer)
    return _hard_strategy(total, dealer)


def _hard_strategy(total: int, dealer: int) -> str:
    if total >= 17:                          return "STAND"
    if total >= 13 and dealer <= 6:          return "STAND"
    if total == 12 and 4 <= dealer <= 6:     return "STAND"
    if total == 11:                          return "DOUBLE"
    if total == 10 and dealer <= 9:          return "DOUBLE"
    if total == 9 and 3 <= dealer <= 6:      return "DOUBLE"
    return "HIT"


def _soft_strategy(total: int, dealer: int) -> str:
    if total >= 19:                          return "STAND"
    if total == 18 and 2 <= dealer <= 8:     return "STAND"
    if total in (17, 18) and 3 <= dealer <= 6: return "DOUBLE"
    return "HIT"


def _pair_strategy(val: str, dealer: int) -> str:
    if val in ("A", "8"):                    return "SPLIT"
    if val in ("10", "J", "Q", "K"):         return "STAND"
    if val == "5":                           return "DOUBLE" if dealer <= 9 else "HIT"
    if val in ("2", "3", "7") and dealer <= 7: return "SPLIT"
    if val == "4" and 5 <= dealer <= 6:      return "SPLIT"
    if val == "6" and dealer <= 6:           return "SPLIT"
    if val == "9" and dealer not in (7, 10, 11): return "SPLIT"
    return "HIT"


def suggest_action_with_ev(
    player_cards: list,
    dealer_upcard: str | None,
    deck_tracker=None,
) -> Tuple[str, dict | None]:
    """
    Devuelve (accion_recomendada, dict_EVs).

    Si `deck_tracker` esta presente, calcula EV dinamico real basado en la
    composicion residual del zapato. En caso contrario, usa la tabla basica
    y devuelve evs=None.

    El EV se devuelve como dict {"STAND": 0.12, "HIT": -0.04, "DOUBLE": ...}.
    """
    if len(player_cards) < 2:
        return "WAIT", None

    # Las parejas se siguen rigiendo por la tabla porque el EV de SPLIT
    # requiere modelar dos manos independientes (ampliable en el futuro).
    if _is_pair(player_cards):
        return suggest_action(player_cards, dealer_upcard), None

    if deck_tracker is None:
        return suggest_action(player_cards, dealer_upcard), None

    from game.ev_calculator import evaluate, evaluate_unknown_dealer, best_action

    if dealer_upcard is None:
        evs = evaluate_unknown_dealer(player_cards, deck_tracker, can_double=True)
    else:
        evs = evaluate(player_cards, dealer_upcard, deck_tracker, can_double=True)

    if not evs:
        return suggest_action(player_cards, dealer_upcard), None
    return best_action(evs), evs
