"""
Muestra estadisticas del dataset capturado.
Uso: python dataset/stats.py [--output dataset/raw] [--target 50]
"""

import argparse
from pathlib import Path

SUITS = ["S", "H", "D", "C"]
VALUES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
CARDS = [f"{v}{s}" for s in SUITS for v in VALUES] + ["card_back"]


def main(output_dir: Path, target: int):
    print(f"\n{'='*55}")
    print(f"  Dataset: {output_dir}")
    print(f"{'='*55}")
    print(f"  {'Carta':<10} {'Capturas':>8}  Progreso")
    print(f"  {'-'*45}")

    total = 0
    complete = 0
    missing = []

    for card in CARDS:
        card_dir = output_dir / card
        count = len(list(card_dir.glob("*.jpg"))) if card_dir.exists() else 0
        total += count
        filled = int(20 * min(count / target, 1.0))
        bar = "[" + "#" * filled + "." * (20 - filled) + "]"
        status = "OK" if count >= target else ""
        print(f"  {card:<10} {count:>5}/{target}  {bar} {status}")
        if count >= target:
            complete += 1
        else:
            missing.append((card, target - count))

    print(f"\n  Total imagenes  : {total}")
    print(f"  Clases completas: {complete}/{len(CARDS)}")

    if missing:
        print(f"\n  Clases incompletas ({len(missing)}):")
        for card, needed in sorted(missing, key=lambda x: -x[1]):
            print(f"    {card:<10} faltan {needed}")
    else:
        print("\n  Dataset completo!")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/raw")
    parser.add_argument("--target", type=int, default=50)
    args = parser.parse_args()
    main(Path(args.output), args.target)
