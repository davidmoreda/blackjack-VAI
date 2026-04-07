"""
Entrena YOLOv8 con el dataset exportado desde Roboflow.
Uso: python dataset/train.py --data dataset/roboflow/data.yaml --epochs 100
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main(args):
    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="cards_yolov8",
        project="models",
        exist_ok=True,
    )
    print(f"\nModelo guardado en: {results.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ruta al data.yaml de Roboflow")
    parser.add_argument("--weights", default="yolov8s.pt", help="Pesos base (yolov8n/s/m)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    main(args)
