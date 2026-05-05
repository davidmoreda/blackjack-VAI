"""
Estimación de normales de superficie por carta usando Depth Anything V2 (HuggingFace).

Flujo:
  frame + detecciones YOLO → depth map por carta → normal map (XYZ) + ejes 3D sobre carta
"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple

from .detector import Detection

DEPTH_ANYTHING_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

# Colores BGR para los ejes (estándar: X=rojo, Y=verde, Z=azul)
COLOR_X = (0, 0, 255)
COLOR_Y = (0, 255, 0)
COLOR_Z = (255, 0, 0)


class DepthNormalEstimator:
    def __init__(self, variant: str = "small", device: Optional[str] = None):
        from transformers import pipeline as hf_pipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = DEPTH_ANYTHING_MODELS[variant]

        print(f"[DepthNormals] Cargando {model_id} en {self.device} ...")
        self.pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if self.device == "cuda" else -1,
        )
        print("[DepthNormals] Modelo listo.")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process_detections(self, frame: np.ndarray, detections: List[Detection]) -> List[dict]:
        """
        Para cada detección con máscara devuelve:
          - detection, depth_crop, normal_map, normal_masked
          - avg_normal: vector normal medio (Nx, Ny, Nz) normalizado en [-1,1]
        """
        results = []
        for det in detections:
            if det.mask is None:
                continue
            result = self._process_one(frame, det)
            if result is not None:
                results.append(result)
        return results

    def process_full_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        depth = self._infer_depth(frame)
        normals = self._depth_to_normals(depth)
        return depth, normals

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_one(self, frame: np.ndarray, det: Detection) -> Optional[dict]:
        x1, y1, x2, y2 = det.bbox
        pad = 4
        h_frame, w_frame = frame.shape[:2]
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w_frame, x2 + pad)
        y2c = min(h_frame, y2 + pad)

        if x2c - x1c < 16 or y2c - y1c < 16:
            return None

        crop_bgr = frame[y1c:y2c, x1c:x2c]
        depth_crop = self._infer_depth(crop_bgr)
        normal_map = self._depth_to_normals(depth_crop)       # uint8 RGB, rango [0,255]

        mask_crop = det.mask[y1c:y2c, x1c:x2c].astype(np.uint8)
        mask_crop = cv2.resize(
            mask_crop, (normal_map.shape[1], normal_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        normal_masked = normal_map * mask_crop[:, :, np.newaxis]

        # Normal media en el área de la carta: de [0,255] → [-1,1]
        pixels = normal_map[mask_crop > 0].astype(np.float32)
        if len(pixels) == 0:
            avg_normal = np.array([0.0, 0.0, 1.0])
        else:
            avg_raw = pixels.mean(axis=0) / 127.5 - 1.0   # (Nx, Ny, Nz)
            norm = np.linalg.norm(avg_raw)
            avg_normal = avg_raw / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

        return {
            "detection":    det,
            "bbox_crop":    (x1c, y1c, x2c, y2c),
            "depth_crop":   depth_crop,
            "normal_map":   normal_map,
            "normal_masked": normal_masked,
            "avg_normal":   avg_normal,   # (Nx, Ny, Nz) en [-1,1]
        }

    def _infer_depth(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        output = self.pipe(Image.fromarray(rgb))
        depth = np.array(output["depth"], dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        return depth

    @staticmethod
    def _depth_to_normals(depth: np.ndarray) -> np.ndarray:
        dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
        n = np.stack([-dzdx, -dzdy, np.ones_like(depth)], axis=-1)
        norm = np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-8)
        n = n / norm
        return ((n + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Visualización
# ------------------------------------------------------------------

def draw_normals_overlay(
    frame: np.ndarray,
    results: List[dict],
    alpha: float = 0.55,
    axis_length: int = 60,
) -> np.ndarray:
    """
    Sobre cada carta detectada dibuja:
      1. Blend del normal map en color (RGB → superficie curva/plana)
      2. Sistema de ejes XYZ en el centro de la carta:
           Rojo  → X (derecha en el plano de la carta)
           Verde → Y (abajo  en el plano de la carta)
           Azul  → Z (normal al plano, hacia la cámara)
    """
    out = frame.copy()

    for r in results:
        det = r["detection"]
        x1, y1, x2, y2 = det.bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ── 1. Normal map blend ────────────────────────────────────────
        normal_bgr = cv2.cvtColor(r["normal_masked"], cv2.COLOR_RGB2BGR)
        normal_resized = cv2.resize(normal_bgr, (w, h))

        roi = out[y1:y2, x1:x2]
        mask_bool = cv2.resize(
            det.mask[y1:y2, x1:x2].astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        roi[mask_bool] = cv2.addWeighted(roi, 1 - alpha, normal_resized, alpha, 0)[mask_bool]

        # ── 2. Ejes XYZ ────────────────────────────────────────────────
        N = r["avg_normal"]          # (Nx, Ny, Nz) en [-1,1]
        _draw_axes(out, cx, cy, N, axis_length)

        # ── 3. Etiqueta ────────────────────────────────────────────────
        label = f"{det.label}  N=({N[0]:.2f},{N[1]:.2f},{N[2]:.2f})"
        cv2.putText(out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def _draw_axes(img: np.ndarray, cx: int, cy: int, normal: np.ndarray, length: int):
    """
    Dibuja ejes X (rojo), Y (verde), Z (azul) en (cx, cy).

    X e Y son tangentes ortogonales al plano de la carta.
    Z es la normal media calculada por Depth Anything.

    La proyección es ortográfica simple: los componentes x,y del vector
    se usan directamente como offset en píxeles (escalados por length).
    """
    Nx, Ny, Nz = normal

    # Eje X: tangente horizontal al plano → perpendicular a N en el plano XZ
    # Calculamos un vector tangente usando cross product N × (0,1,0)
    up = np.array([0.0, 1.0, 0.0])
    Xaxis = np.cross(normal, up)
    xnorm = np.linalg.norm(Xaxis)
    if xnorm < 1e-6:
        Xaxis = np.array([1.0, 0.0, 0.0])
    else:
        Xaxis /= xnorm

    # Eje Y: tangente vertical → perpendicular a N y a Xaxis
    Yaxis = np.cross(normal, Xaxis)
    ynorm = np.linalg.norm(Yaxis)
    if ynorm > 1e-6:
        Yaxis /= ynorm

    def endpoint(axis):
        # Proyección ortográfica: x→columna, y→fila, z ignorado
        return (
            int(cx + axis[0] * length),
            int(cy + axis[1] * length),
        )

    origin = (cx, cy)
    tip_x = endpoint(Xaxis)
    tip_y = endpoint(Yaxis)
    # Z proyectado: Nx,Ny del vector normal dan la dirección en pantalla
    tip_z = (int(cx + Nx * length), int(cy + Ny * length))

    thick = 2
    tip_size = 8

    cv2.arrowedLine(img, origin, tip_x, COLOR_X, thick, tipLength=tip_size / length)
    cv2.arrowedLine(img, origin, tip_y, COLOR_Y, thick, tipLength=tip_size / length)
    cv2.arrowedLine(img, origin, tip_z, COLOR_Z, thick, tipLength=tip_size / length)

    # Letras
    offset = 6
    cv2.putText(img, "X", (tip_x[0] + offset, tip_x[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_X, 2)
    cv2.putText(img, "Y", (tip_y[0] + offset, tip_y[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_Y, 2)
    cv2.putText(img, "Z", (tip_z[0] + offset, tip_z[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_Z, 2)

    # Punto de origen
    cv2.circle(img, origin, 4, (255, 255, 255), -1)
