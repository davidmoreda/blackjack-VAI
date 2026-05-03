"""Captura de frames desde webcam o stream MJPEG HTTP."""

import threading
import cv2
import numpy as np
import requests


class CameraCapture:
    def __init__(self, config: dict):
        cfg = config["camera"]
        source = cfg["index"]  # int o URL string

        self.is_stream = isinstance(source, str) and source.startswith("http")

        if self.is_stream:
            self._stream_url = source
            self._frame = None
            self._lock = threading.Lock()
            self._thread = threading.Thread(target=self._read_mjpeg_loop, daemon=True)
            self._thread.start()
        else:
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
            self.cap.set(cv2.CAP_PROP_FPS,          cfg["fps"])
            if not self.cap.isOpened():
                raise RuntimeError(f"No se puede abrir la camara: {source}")

    def _read_mjpeg_loop(self):
        """Lee el stream MJPEG via requests y extrae frames JPEG."""
        import time
        while True:
            try:
                r = requests.get(self._stream_url, stream=True, timeout=5)
                buf = b""
                for chunk in r.iter_content(chunk_size=4096):
                    buf += chunk
                    # Buscar el marcador de inicio y fin JPEG
                    start = buf.find(b'\xff\xd8')
                    end   = buf.find(b'\xff\xd9')
                    if start != -1 and end != -1 and end > start:
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
            except Exception as e:
                print(f"[capture] Error en stream HTTP: {e} — reintentando en 2s")
                time.sleep(2)

    def read(self):
        if self.is_stream:
            with self._lock:
                frame = self._frame
            if frame is None:
                raise RuntimeError("Sin frames del stream todavia")
            return frame.copy()
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("No se puede leer frame de la camara")
            return frame

    def release(self):
        if not self.is_stream:
            self.cap.release()
