"""
Debouncer de detecciones: una carta debe verse de forma continua durante
`threshold_s` segundos antes de considerarse confirmada. Filtra falsos
positivos y oscilaciones del modelo.

Compartido entre cli.py y api/ws.py para que ambas vias se comporten igual.
"""

from typing import Dict, Iterable, List, Set, Tuple


class CardDebouncer:
    def __init__(self, threshold_s: float = 1.0):
        self.threshold = threshold_s
        self.pending: Dict[Tuple[str, str], float] = {}
        self.committed: Set[Tuple[str, str]] = set()

    def tick(
        self,
        visible: Iterable[Tuple[str, str]],
        now: float,
    ) -> List[Tuple[str, str]]:
        """Aplica un tick. Devuelve lista de (zone, label) recien confirmadas."""
        visible_set = set(visible)
        for k in list(self.pending):
            if k not in visible_set:
                del self.pending[k]
        self.committed &= visible_set

        new_commits: List[Tuple[str, str]] = []
        for key in visible_set:
            if key in self.committed:
                continue
            t0 = self.pending.get(key)
            if t0 is None:
                self.pending[key] = now
            elif now - t0 >= self.threshold:
                new_commits.append(key)
                self.committed.add(key)
                self.pending.pop(key, None)
        return new_commits

    def progress(self, key: Tuple[str, str], now: float) -> float:
        if key in self.committed:
            return 1.0
        t0 = self.pending.get(key)
        if t0 is None:
            return 0.0
        return min(1.0, (now - t0) / self.threshold)

    def forget(self, key: Tuple[str, str]):
        """Olvida una entrada (al borrar carta del historial, p.ej.)."""
        self.committed.discard(key)
        self.pending.pop(key, None)

    def reset(self):
        self.pending.clear()
        self.committed.clear()
