"""Captura de frames desde webcam."""

import cv2
import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class CameraCapture:
    def __init__(self, config: dict):
        cfg = config["camera"]
        self.cap = cv2.VideoCapture(cfg["index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        self.cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se puede leer de la camara")
        return frame

    def release(self):
        self.cap.release()
