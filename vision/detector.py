"""Inferencia Ultralytics + ByteTrack sobre frames de la camara."""

import cv2
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from ultralytics import RTDETR, YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple           # (x1, y1, x2, y2) en pixeles del frame original
    center: tuple         # (cx, cy)
    track_id: int = -1    # ID estable del tracker (-1 si no hay tracking)
    mask: Optional[np.ndarray] = field(default=None, repr=False)


class CardDetector:
    def __init__(
        self,
        model_path: str,
        backend: str = "yolo",
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 1280,
    ):
        backend_key = backend.lower()
        model_cls = {
            "yolo": YOLO,
            "rtdetr": RTDETR,
        }.get(backend_key)
        if model_cls is None:
            raise ValueError(f"Backend no soportado: {backend!r}. Usa 'yolo' o 'rtdetr'.")

        self.model = model_cls(model_path)
        self.backend = backend_key
        self.confidence = confidence
        self.iou = iou
        self.imgsz = int(imgsz)

    @staticmethod
    def _enhance(frame: np.ndarray) -> np.ndarray:
        """Sharpening suave + normalización de contraste para cámaras de baja calidad."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Usa model.track() con ByteTrack para mantener IDs estables entre frames.
        El tracker predice posicion con filtro de Kalman cuando el detector falla
        un frame, eliminando el parpadeo sin anadir ghosting artificial.
        persist=True mantiene el estado del tracker entre llamadas.
        """
        frame = self._enhance(frame)
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            persist=True,          # estado del tracker persiste entre frames
            tracker="bytetrack.yaml",
            verbose=False,
        )

        detections = []
        for r in results:
            h, w = frame.shape[:2]
            masks_data = r.masks

            if r.boxes is None:
                continue

            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self.model.names[int(box.cls)]
                track_id = int(box.id[0]) if box.id is not None else -1

                mask = None
                if masks_data is not None and i < len(masks_data.data):
                    m = masks_data.data[i].cpu().numpy()
                    mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5

                detections.append(Detection(
                    label=label,
                    confidence=float(box.conf),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    track_id=track_id,
                    mask=mask,
                ))
        return detections
