"""
Gestiona las zonas de jugadores en la mesa y asigna cartas detectadas a cada mano.
Las zonas se definen manualmente via zone_setup.py y se guardan en config.yaml.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vision.detector import Detection


@dataclass
class Zone:
    name: str           # "dealer", "player_1", "player_2", ...
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def point_in_zone(cx: int, cy: int, zone: Zone) -> bool:
    x1, y1, x2, y2 = zone.bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


class ZoneManager:
    def __init__(self, zones: List[Zone]):
        self.zones = zones

    def assign(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Asigna cada deteccion a la zona que contiene su centro."""
        result: Dict[str, List[Detection]] = {z.name: [] for z in self.zones}
        result["unassigned"] = []
        for det in detections:
            cx, cy = det.center
            assigned = False
            for zone in self.zones:
                if point_in_zone(cx, cy, zone):
                    result[zone.name].append(det)
                    assigned = True
                    break
            if not assigned:
                result["unassigned"].append(det)
        return result
