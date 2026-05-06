"""
Kalman filter por track_id para suavizar el vector normal entre frames.

La normal calculada por Depth Anything tiene ruido frame a frame.
Este módulo mantiene un KalmanFilter independiente por carta (track_id)
y devuelve la estimación suavizada en cada frame.

Estado:  [Nx, Ny, Nz]  (vector normal en [-1, 1])
Medida:  [Nx, Ny, Nz]  (normal raw de Depth Anything)
"""

import cv2
import numpy as np
from typing import Dict, Optional


class NormalTracker:
    """
    Mantiene un KalmanFilter por track_id.
    Limpia automáticamente los filtros de cartas que desaparecen.
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 5e-2,
    ):
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._filters: Dict[int, cv2.KalmanFilter] = {}
        self._last_seen: Dict[int, int] = {}   # track_id → frame counter
        self._frame_count = 0

    def update(self, track_id: int, measured_normal: np.ndarray) -> np.ndarray:
        """
        Actualiza el filtro con la normal medida y devuelve la estimación suavizada.
        measured_normal: array (3,) con (Nx, Ny, Nz) en [-1, 1]
        """
        if track_id not in self._filters:
            self._filters[track_id] = self._make_filter(measured_normal)

        kf = self._filters[track_id]
        self._last_seen[track_id] = self._frame_count

        measurement = measured_normal.reshape(3, 1).astype(np.float32)
        kf.correct(measurement)
        predicted = kf.predict()

        smoothed = predicted[:3, 0].astype(np.float64)
        norm = np.linalg.norm(smoothed)
        return smoothed / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

    def predict(self, track_id: int) -> Optional[np.ndarray]:
        """
        Devuelve la predicción del filtro sin nueva medida (para frames sin depth).
        Retorna None si el track_id es desconocido.
        """
        if track_id not in self._filters:
            return None
        predicted = self._filters[track_id].predict()
        smoothed = predicted[:3, 0].astype(np.float64)
        norm = np.linalg.norm(smoothed)
        return smoothed / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

    def tick(self, active_track_ids: set, max_missing_frames: int = 30):
        """Llamar una vez por frame para limpiar tracks inactivos."""
        self._frame_count += 1
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._frame_count - last) > max_missing_frames
            and tid not in active_track_ids
        ]
        for tid in stale:
            del self._filters[tid]
            del self._last_seen[tid]

    # ------------------------------------------------------------------

    def _make_filter(self, initial_normal: np.ndarray) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(3, 3)   # 3 estados, 3 medidas

        # Transición: identidad (la normal cambia poco entre frames)
        kf.transitionMatrix = np.eye(3, dtype=np.float32)

        # Observación: identidad (medimos directamente Nx,Ny,Nz)
        kf.measurementMatrix = np.eye(3, dtype=np.float32)

        # Covarianzas
        kf.processNoiseCov  = np.eye(3, dtype=np.float32) * self._process_noise
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * self._measurement_noise
        kf.errorCovPost     = np.eye(3, dtype=np.float32)

        # Estado inicial = primera medida
        kf.statePost = initial_normal.reshape(3, 1).astype(np.float32)
        return kf
