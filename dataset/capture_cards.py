"""
Captura de imagenes para el dataset de cartas de blackjack.

Modos:
  sequential  Recorre las 52 cartas en orden. Avanza automaticamente
              cuando alcanzas el objetivo de capturas por carta.
  manual      Tu eliges que carta capturar en cada momento.

Controles:
  ESPACIO     Capturar imagen
  A           Activar/desactivar captura automatica (cada 1s)
  N           Siguiente carta (modo sequential)
  P           Carta anterior (modo sequential)
  Q           Salir
"""

import argparse
import os
import time
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Definicion de las 53 clases
# ---------------------------------------------------------------------------
SUITS = ["S", "H", "D", "C"]       # Spades, Hearts, Diamonds, Clubs
VALUES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
CARDS = [f"{v}{s}" for s in SUITS for v in VALUES] + ["card_back"]

SUIT_NAMES = {"S": "Spades", "H": "Hearts", "D": "Diamonds", "C": "Clubs"}
VALUE_NAMES = {
    "A": "Ace", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "10": "10", "J": "Jack", "Q": "Queen", "K": "King",
}

# Colores por palo
SUIT_COLORS = {
    "S": (50, 50, 50),
    "H": (0, 0, 220),
    "D": (0, 0, 220),
    "C": (50, 50, 50),
}


def card_display_name(card: str) -> str:
    if card == "card_back":
        return "Card Back (boca abajo)"
    suit = card[-1]
    value = card[:-1]
    return f"{VALUE_NAMES.get(value, value)} of {SUIT_NAMES.get(suit, suit)}"


def count_existing(output_dir: Path, card: str) -> int:
    card_dir = output_dir / card
    if not card_dir.exists():
        return 0
    return len(list(card_dir.glob("*.jpg")))


def save_image(frame, output_dir: Path, card: str) -> str:
    card_dir = output_dir / card
    card_dir.mkdir(parents=True, exist_ok=True)
    existing = count_existing(output_dir, card)
    filename = card_dir / f"{card}_{existing + 1:04d}.jpg"
    cv2.imwrite(str(filename), frame)
    return str(filename)


def draw_overlay(frame, card: str, count: int, target: int, auto_mode: bool, mode: str, card_index: int, total: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Barra superior
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Nombre de la carta
    color = SUIT_COLORS.get(card[-1], (200, 200, 200)) if card != "card_back" else (150, 150, 150)
    cv2.putText(frame, card_display_name(card), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Clase / indice
    cv2.putText(frame, f"[{card}]  {card_index + 1}/{total}", (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Barra de progreso
    bar_x, bar_y, bar_w, bar_h = w - 220, 15, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * min(count / target, 1.0))
    bar_color = (0, 200, 80) if count >= target else (0, 140, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{count}/{target}", (bar_x + bar_w + 8, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Indicadores inferiores
    bottom = h - 10
    controls = "[ESPACIO] Capturar  [A] Auto  [N] Sig.  [P] Ant.  [Q] Salir"
    cv2.putText(frame, controls, (10, bottom),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    if auto_mode:
        cv2.putText(frame, "AUTO ON", (w - 110, bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2)

    if count >= target:
        cv2.putText(frame, "COMPLETA", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 100), 3)

    return frame


def pick_card_manual(cards: list, output_dir: Path, target: int) -> str:
    """Muestra menu de seleccion de carta en terminal."""
    os.system("clear" if os.name == "posix" else "cls")
    print("\n=== SELECCION DE CARTA ===\n")
    for i, c in enumerate(cards):
        count = count_existing(output_dir, c)
        status = "[OK]" if count >= target else f"{count:3d}/{target}"
        print(f"  {i:2d}. {c:8s}  {card_display_name(c):28s}  {status}")
    print(f"\n  {len(cards)}. card_back")
    print("\nEscribe el numero (o nombre) de la carta: ", end="")
    raw = input().strip()
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(cards):
            return cards[idx]
        if idx == len(cards):
            return "card_back"
    elif raw.upper() in cards:
        return raw.upper()
    print("Seleccion invalida, usando la primera.")
    return cards[0]


def run(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Error: no se puede abrir la camara {args.camera}")
        return

    cards = CARDS.copy()
    card_index = 0
    auto_capture = False
    last_auto_time = 0.0

    if args.mode == "manual":
        current_card = pick_card_manual(cards, output_dir, args.target)
        card_index = cards.index(current_card) if current_card in cards else 0
    else:
        # En secuencial, empieza por la primera carta sin completar
        for i, c in enumerate(cards):
            if count_existing(output_dir, c) < args.target:
                card_index = i
                break

    print("\nIniciando captura. Pulsa Q para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame de la camara.")
            break

        current_card = cards[card_index]
        count = count_existing(output_dir, current_card)

        # Captura automatica
        if auto_capture and (time.time() - last_auto_time) >= 1.0:
            if count < args.target:
                save_image(frame, output_dir, current_card)
                count += 1
                last_auto_time = time.time()
                print(f"  Auto-captura: {current_card} [{count}/{args.target}]")
            else:
                # Avance automatico al siguiente en modo sequential
                if args.mode == "sequential":
                    next_idx = find_next_incomplete(cards, card_index, output_dir, args.target)
                    if next_idx is None:
                        print("\nDataset completo! Todas las cartas tienen suficientes capturas.")
                        break
                    card_index = next_idx
                    print(f"\n>>> Avanzando a: {cards[card_index]}")

        display = draw_overlay(
            frame.copy(), current_card, count, args.target,
            auto_capture, args.mode, card_index, len(cards)
        )
        cv2.imshow("Captura Dataset Blackjack", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord(" "):  # ESPACIO — captura manual
            path = save_image(frame, output_dir, current_card)
            count += 1
            print(f"  Guardada: {os.path.basename(path)}  [{count}/{args.target}]")
            if count >= args.target and args.mode == "sequential":
                next_idx = find_next_incomplete(cards, card_index, output_dir, args.target)
                if next_idx is None:
                    print("\nDataset completo!")
                    break
                card_index = next_idx
                print(f"\n>>> Avanzando a: {cards[card_index]}")

        elif key == ord("a") or key == ord("A"):
            auto_capture = not auto_capture
            last_auto_time = time.time()
            print(f"  Captura automatica: {'ON' if auto_capture else 'OFF'}")

        elif key == ord("n") or key == ord("N"):
            card_index = (card_index + 1) % len(cards)
            print(f"  Carta: {cards[card_index]}")

        elif key == ord("p") or key == ord("P"):
            card_index = (card_index - 1) % len(cards)
            print(f"  Carta: {cards[card_index]}")

        elif key == ord("m") or key == ord("M"):
            # Abrir menu de seleccion manual
            cap.release()
            cv2.destroyAllWindows()
            current_card = pick_card_manual(cards, output_dir, args.target)
            card_index = cards.index(current_card) if current_card in cards else card_index
            cap = cv2.VideoCapture(args.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap.release()
    cv2.destroyAllWindows()
    print_summary(cards, output_dir, args.target)


def find_next_incomplete(cards: list, current_idx: int, output_dir: Path, target: int):
    for offset in range(1, len(cards)):
        idx = (current_idx + offset) % len(cards)
        if count_existing(output_dir, cards[idx]) < target:
            return idx
    return None


def print_summary(cards: list, output_dir: Path, target: int):
    print("\n=== RESUMEN DEL DATASET ===")
    complete = 0
    for card in cards:
        count = count_existing(output_dir, card)
        status = "OK" if count >= target else f"FALTA {target - count}"
        if count >= target:
            complete += 1
        else:
            print(f"  {card:8s}  {count:3d}/{target}  {status}")
    print(f"\n  Completas: {complete}/{len(cards)}")
    total = sum(count_existing(output_dir, c) for c in cards)
    print(f"  Total imagenes: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura dataset de cartas para blackjack")
    parser.add_argument("--mode", choices=["sequential", "manual"], default="sequential",
                        help="Modo de captura (default: sequential)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Indice de camara (default: 0)")
    parser.add_argument("--output", default="dataset/raw",
                        help="Carpeta de salida (default: dataset/raw)")
    parser.add_argument("--target", type=int, default=50,
                        help="Capturas objetivo por carta (default: 50)")
    args = parser.parse_args()
    run(args)
