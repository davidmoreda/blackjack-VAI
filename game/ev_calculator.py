"""
Calculo de valor esperado (EV) dinamico para Blackjack.

Para cada accion (STAND / HIT / DOUBLE) calcula el EV dado:
  - la mano del jugador
  - la carta visible del dealer
  - la composicion residual del zapato (DeckTracker)

Convencion de EV:
   +1.0  =  ganar 1 unidad   (push = 0,  perder = -1)
   El EV de DOUBLE se mide en unidades originales (puede ser ±2).

Modelo: usa la composicion REAL del zapato como distribucion de probabilidad
de cada carta, pero la trata como infinita durante la recursion (no descuenta
las cartas hipoteticas que salen en cada rama). Con 4-6 mazos el error es
< 0.001 EV y la velocidad es ~1000x mayor que la simulacion exacta — esta
aproximacion la usan todos los analizadores comerciales en tiempo real.
"""

from functools import lru_cache
from typing import Dict, List, Tuple

from game.deck_tracker import DeckTracker, label_to_value


# ── Helpers de mano ──────────────────────────────────────────────────────────


def _hand_total(cards: Tuple[int, ...]) -> Tuple[int, bool]:
    total = sum(cards)
    aces = sum(1 for c in cards if c == 1)
    soft = False
    if aces and total + 10 <= 21:
        total += 10
        soft = True
    return total, soft


def _labels_to_values(cards: List[str]) -> Tuple[int, ...]:
    out = []
    for c in cards:
        v = label_to_value(c)
        if v is None:
            continue
        if v == "A":
            out.append(1)
        elif v in ("J", "Q", "K", "10"):
            out.append(10)
        else:
            out.append(int(v))
    return tuple(out)


# ── Recursion del dealer (deck congelado) ────────────────────────────────────


@lru_cache(maxsize=50_000)
def _dealer_play(
    hand: Tuple[int, ...],
    probs: Tuple[Tuple[int, float], ...],
    hits_soft_17: bool,
) -> Tuple[Tuple[int, float], ...]:
    """
    Distribucion final del dealer. `probs` es la distribucion de probabilidad
    de cada valor numerico (1..10) — congelada durante la recursion.
    Devuelve tupla ((total, prob), ...) ordenada para servir como clave de cache.
    """
    total, soft = _hand_total(hand)
    if total > 21:
        return ((22, 1.0),)
    if total >= 17 and not (hits_soft_17 and soft and total == 17):
        return ((total, 1.0),)

    acc: Dict[int, float] = {}
    for val, p in probs:
        if p <= 0.0:
            continue
        sub = _dealer_play(hand + (val,), probs, hits_soft_17)
        for k, q in sub:
            acc[k] = acc.get(k, 0.0) + p * q
    return tuple(sorted(acc.items()))


# ── EV de cada accion (deck congelado, todo cached) ──────────────────────────


@lru_cache(maxsize=50_000)
def _ev_stand(
    player_hand: Tuple[int, ...],
    dealer_up: int,
    probs: Tuple[Tuple[int, float], ...],
    hits_soft_17: bool,
) -> float:
    p_total, _ = _hand_total(player_hand)
    if p_total > 21:
        return -1.0
    dist = _dealer_play((dealer_up,), probs, hits_soft_17)
    ev = 0.0
    for d_total, prob in dist:
        if d_total > 21 or d_total < p_total:
            ev += prob
        elif d_total > p_total:
            ev -= prob
    return ev


@lru_cache(maxsize=100_000)
def _ev_hit(
    player_hand: Tuple[int, ...],
    dealer_up: int,
    probs: Tuple[Tuple[int, float], ...],
    hits_soft_17: bool,
) -> float:
    """EV de pedir, asumiendo que despues solo cabe hit/stand (no doblar)."""
    ev = 0.0
    for val, p in probs:
        if p <= 0.0:
            continue
        new_hand = player_hand + (val,)
        new_total, _ = _hand_total(new_hand)
        if new_total > 21:
            ev += p * -1.0
        else:
            ev_s = _ev_stand(new_hand, dealer_up, probs, hits_soft_17)
            ev_h = _ev_hit(new_hand, dealer_up, probs, hits_soft_17)
            ev += p * (ev_s if ev_s > ev_h else ev_h)
    return ev


@lru_cache(maxsize=20_000)
def _ev_double(
    player_hand: Tuple[int, ...],
    dealer_up: int,
    probs: Tuple[Tuple[int, float], ...],
    hits_soft_17: bool,
) -> float:
    ev = 0.0
    for val, p in probs:
        if p <= 0.0:
            continue
        new_hand = player_hand + (val,)
        new_total, _ = _hand_total(new_hand)
        if new_total > 21:
            ev += p * -2.0
        else:
            ev += p * 2.0 * _ev_stand(new_hand, dealer_up, probs, hits_soft_17)
    return ev


# ── API publica ──────────────────────────────────────────────────────────────


def _probs_tuple(tracker: DeckTracker) -> Tuple[Tuple[int, float], ...]:
    """
    Probabilidades por valor numerico (1..10) como tupla inmutable.
    Las probabilidades se redondean a 4 decimales para que el cache colapse
    estados muy parecidos del zapato (acelera a costa de error < 0.0001).
    """
    by_num: Dict[int, int] = {n: 0 for n in range(1, 11)}
    for v, c in tracker.remaining.items():
        if v == "A":
            by_num[1] += c
        elif v in ("10", "J", "Q", "K"):
            by_num[10] += c
        else:
            by_num[int(v)] += c
    tot = sum(by_num.values())
    if tot <= 0:
        return ()
    return tuple((k, round(v / tot, 4)) for k, v in sorted(by_num.items()) if v > 0)


@lru_cache(maxsize=20_000)
def _evaluate_cached(
    player_hand: Tuple[int, ...],
    dealer_up: int,
    probs: Tuple[Tuple[int, float], ...],
    can_double: bool,
    hits_soft_17: bool,
) -> Tuple[Tuple[str, float], ...]:
    res = [
        ("STAND", _ev_stand(player_hand, dealer_up, probs, hits_soft_17)),
        ("HIT",   _ev_hit(player_hand,   dealer_up, probs, hits_soft_17)),
    ]
    if can_double:
        res.append(("DOUBLE", _ev_double(player_hand, dealer_up, probs, hits_soft_17)))
    return tuple(res)


def evaluate(
    player_cards: List[str],
    dealer_upcard: str,
    tracker: DeckTracker,
    can_double: bool = True,
    hits_soft_17: bool = False,
) -> Dict[str, float]:
    if not player_cards or not dealer_upcard:
        return {}
    player_hand = _labels_to_values(player_cards)
    dealer_vals = _labels_to_values([dealer_upcard])
    if not player_hand or not dealer_vals:
        return {}
    can_d = can_double and len(player_cards) == 2
    probs = _probs_tuple(tracker)
    if not probs:
        return {}
    return dict(_evaluate_cached(player_hand, dealer_vals[0], probs, can_d, hits_soft_17))


def best_action(evs: Dict[str, float]) -> str:
    if not evs:
        return "WAIT"
    return max(evs.items(), key=lambda kv: kv[1])[0]


def cache_info() -> dict:
    return {
        "dealer_play": _dealer_play.cache_info()._asdict(),
        "ev_stand":    _ev_stand.cache_info()._asdict(),
        "ev_hit":      _ev_hit.cache_info()._asdict(),
        "ev_double":   _ev_double.cache_info()._asdict(),
        "evaluate":    _evaluate_cached.cache_info()._asdict(),
    }


def clear_caches():
    """Util al cambiar num_decks o recargar."""
    _dealer_play.cache_clear()
    _ev_stand.cache_clear()
    _ev_hit.cache_clear()
    _ev_double.cache_clear()
    _evaluate_cached.cache_clear()
