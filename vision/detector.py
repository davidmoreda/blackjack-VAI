"""Inferencia YOLOv8 sobre frames de la camara."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    label: str      # e.g. "AS", "KH", "card_back"
    confidence: float
    bbox: tuple     # (x1, y1, x2, y2) en pixeles
    center: tuple   # (cx, cy)


class CardDetector:
    def __init__(self, model_path: str, confidence: float = 0.7, iou: float = 0.45):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.confidence, iou=self.iou, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self.model.names[int(box.cls)]
                detections.append(Detection(
                    label=label,
                    confidence=float(box.conf),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                ))
        return detections
