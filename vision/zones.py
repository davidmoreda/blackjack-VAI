"""Gestión de zonas de la mesa — asigna detecciones YOLO a dealer/jugadores."""

import cv2
import numpy as np
from typing import Optional


def build_zones(num_players: int) -> dict:
    """
    Genera dealer arriba (40% altura, ancho completo) + N columnas iguales para
    los jugadores en el 60% inferior. Util para que cambiar `num_players` en
    settings actualice la geometria sin tocar config.yaml.
    """
    n = max(1, int(num_players))
    zones = {"dealer": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.4}}
    width = 1.0 / n
    for i in range(n):
        zones[f"player_{i+1}"] = {
            "x": i * width, "y": 0.4, "w": width, "h": 0.6,
        }
    return zones


class ZoneManager:
    def __init__(self, zones_cfg: dict):
        # zones_cfg: {zone_name: {x, y, w, h}} en fracciones [0,1] del frame
        self.zones = zones_cfg  # orden importa: se comprueba en secuencia

    def get_zone(self, cx: int, cy: int, frame_w: int, frame_h: int) -> Optional[str]:
        """Devuelve el nombre de zona que contiene el punto (cx, cy)."""
        for name, z in self.zones.items():
            x0 = z['x'] * frame_w
            y0 = z['y'] * frame_h
            x1 = (z['x'] + z['w']) * frame_w
            y1 = (z['y'] + z['h']) * frame_h
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return name
        return None

    def get_rect(self, zone_name: str, frame_w: int, frame_h: int):
        """Devuelve (x0, y0, x1, y1) en píxeles."""
        z = self.zones[zone_name]
        x0 = int(z['x'] * frame_w)
        y0 = int(z['y'] * frame_h)
        x1 = int((z['x'] + z['w']) * frame_w)
        y1 = int((z['y'] + z['h']) * frame_h)
        return x0, y0, x1, y1


# Colores BGR por zona
ZONE_COLORS = {
    'dealer':   (60,  60,  200),
    'player_1': (30,  160, 30),
    'player_2': (200, 130, 0),
    'player_3': (180, 0,   180),
}


def draw_zones(frame: np.ndarray, zone_mgr: ZoneManager, game_sm) -> np.ndarray:
    """
    Dibuja sobre el frame:
    - Rectángulo semitransparente por zona
    - Nombre de zona
    - Total de la mano (grande, centrado)
    - BUST en rojo si se pasa de 21
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for zone_name, z_cfg in zone_mgr.zones.items():
        x0, y0, x1, y1 = zone_mgr.get_rect(zone_name, w, h)
        color = ZONE_COLORS.get(zone_name, (120, 120, 120))

        # Obtener mano correspondiente
        if zone_name == 'dealer':
            hand = game_sm.dealer_hand
        else:
            hand = game_sm.player_hands.get(zone_name)

        if hand is None:
            total, bust, bj = 0, False, False
        else:
            total = hand.total
            bust  = hand.bust
            bj    = getattr(hand, 'blackjack', False)

        if bust:
            fill_color = (0, 0, 140)
        elif total == 21:
            fill_color = (0, 100, 120)   # dorado oscuro
        else:
            fill_color = (color[0]//4, color[1]//4, color[2]//4)

        cv2.rectangle(overlay, (x0, y0), (x1, y1), fill_color, -1)

    # Mezclar overlay con el frame original
    frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

    # Bordes, etiquetas y totales encima de la mezcla
    import time as _time
    blink = int(_time.time() * 3) % 2 == 0   # parpadeo ~3Hz para alertas

    for zone_name in zone_mgr.zones:
        x0, y0, x1, y1 = zone_mgr.get_rect(zone_name, w, h)
        color = ZONE_COLORS.get(zone_name, (120, 120, 120))

        if zone_name == 'dealer':
            hand = game_sm.dealer_hand
        else:
            hand = game_sm.player_hands.get(zone_name)

        if hand is None:
            total, bust, bj = 0, False, False
        else:
            total = hand.total
            bust  = hand.bust
            bj    = getattr(hand, 'blackjack', False)

        # Borde: rojo parpadeo=bust, dorado=21, normal=color zona
        if bust:
            border_color = (0, 0, 255) if blink else (0, 0, 120)
            border_thick = 3
        elif total == 21:
            border_color = (0, 215, 255) if blink else (0, 140, 180)  # amarillo/dorado
            border_thick = 3
        else:
            border_color = color
            border_thick = 2

        cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, border_thick)

        # Nombre de zona
        label = zone_name.replace('_', ' ').upper()
        cv2.putText(frame, label, (x0 + 8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        if total > 0:
            if bust:
                t_text  = f"BUST!  {total}"
                t_color = (40, 40, 255) if blink else (120, 120, 255)
                t_scale = 1.5
            elif total == 21:
                t_text  = "BLACKJACK!" if bj else "21 !"
                t_color = (0, 215, 255) if blink else (0, 160, 200)
                t_scale = 1.4 if bj else 1.6
            else:
                t_text  = str(total)
                t_color = (40, 255, 40)
                t_scale = 1.6

            t_thick = 3
            (tw, th), _ = cv2.getTextSize(t_text, cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_thick)
            tx = x0 + (x1 - x0 - tw) // 2
            ty = y0 + (y1 - y0 + th) // 2
            cv2.putText(frame, t_text, (tx + 2, ty + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, t_scale, (0, 0, 0), t_thick + 2, cv2.LINE_AA)
            cv2.putText(frame, t_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, t_scale, t_color, t_thick, cv2.LINE_AA)

    return frame
