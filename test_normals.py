"""
Prueba rápida: Depth Anything V2 → normal maps sobre imagen estática o webcam.

Uso:
  # Con una imagen:
  python test_normals.py --image foto_cartas.jpg

  # Con la webcam en tiempo real:
  python test_normals.py --webcam

  # Con YOLO + normals por carta:
  python test_normals.py --image foto_cartas.jpg --yolo models/best.pt
"""

import argparse
import cv2
import numpy as np
import sys

from vision.depth_normals import DepthNormalEstimator, draw_normals_overlay


def run_image(path: str, estimator: DepthNormalEstimator, yolo_path: str | None):
    frame = cv2.imread(path)
    if frame is None:
        print(f"No se pudo leer: {path}")
        sys.exit(1)

    if yolo_path:
        from vision.detector import CardDetector
        detector = CardDetector(yolo_path)
        detections = detector.detect(frame)
        print(f"  Cartas detectadas: {len(detections)}")
        results = estimator.process_detections(frame, detections)
        out = draw_normals_overlay(frame, results, alpha=0.65)
        for r in results:
            label = r["detection"].label
            print(f"  Normal map carta '{label}': shape={r['normal_map'].shape}")
            # Guardar normal map individual por carta
            out_card = cv2.cvtColor(r["normal_map"], cv2.COLOR_RGB2BGR)
            card_path = f"normal_{label.replace(' ', '_')}.jpg"
            cv2.imwrite(card_path, out_card)
            print(f"  Guardado: {card_path}")
    else:
        depth, normals = estimator.process_full_frame(frame)
        normals_bgr = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)
        depth_vis = (depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        out = np.hstack([frame, depth_color, normals_bgr])

    out_path = "normals_result.jpg"
    cv2.imwrite(out_path, out)
    print(f"\nResultado guardado en: {out_path}")


def run_webcam(estimator: DepthNormalEstimator, yolo_path: str | None):
    """Captura N frames de la webcam y guarda el resultado en disco (WSL2 no tiene display)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la webcam.")
        sys.exit(1)

    detector = None
    if yolo_path:
        from vision.detector import CardDetector
        detector = CardDetector(yolo_path)

    print("Capturando frames... Ctrl+C para parar.")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if detector:
                detections = detector.detect(frame)
                results = estimator.process_detections(frame, detections)
                out = draw_normals_overlay(frame, results, alpha=0.65)
            else:
                depth, normals = estimator.process_full_frame(frame)
                normals_bgr = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)
                depth_vis = (depth * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                out = np.hstack([frame, depth_color, normals_bgr])

            out_path = f"normals_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(out_path, out)
            print(f"  Guardado: {out_path}")
            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    print(f"\nTotal frames guardados: {frame_idx}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Ruta a imagen de cartas")
    group.add_argument("--webcam", action="store_true", help="Usar webcam en tiempo real")
    parser.add_argument("--yolo", default=None, help="Ruta al modelo YOLO (opcional)")
    parser.add_argument(
        "--variant", default="small", choices=["small", "base", "large"],
        help="Variante de Depth Anything V2 (default: small)"
    )
    args = parser.parse_args()

    estimator = DepthNormalEstimator(variant=args.variant)

    if args.image:
        run_image(args.image, estimator, args.yolo)
    else:
        run_webcam(estimator, args.yolo)


if __name__ == "__main__":
    main()
