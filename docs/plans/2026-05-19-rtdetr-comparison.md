# Comparacion RT-DETR-L vs YOLOv8m-seg

**Fecha:** 2026-05-19
**Proyecto:** Vision Artificial Avanzada - Master IA

---

## Contexto

El sistema BlackjackVAI usa actualmente `YOLOv8m-seg` para detectar cartas y
alimentar el pipeline de zonas, debouncer, maquina de estados y conteo. La app
solo consume bounding boxes; las mascaras sirven como ayuda visual en el overlay.

Para la comparacion academica se introduce un segundo detector real-time:
`RT-DETR-L`, cargado con la API `ultralytics.RTDETR`. Esto permite contrastar una
familia CNN/YOLO contra un detector Transformer end-to-end sobre el mismo dataset
BlackjackVAI V4.

## Decision

| Aspecto | YOLOv8m-seg V4 | RT-DETR-L v1 |
|----------|----------------|--------------|
| Familia | CNN real-time | Transformer real-time |
| API | `ultralytics.YOLO` | `ultralytics.RTDETR` |
| Tarea | Deteccion + segmentacion | Deteccion |
| Salida usada por la app | `boxes` | `boxes` |
| Modelo | `models/best.pt` | `models/best_rtdetr_v1.pt` |
| Experimento MLflow | `blackjackvai-v4` | `blackjackvai-rtdetr-v1` |

RT-DETR se elige porque mantiene la misma libreria, el mismo formato de dataset,
la misma interfaz `.predict()` / `.track()` y una narrativa de comparacion clara:
CNN vs Transformer en deteccion real-time de 54 clases propias.

## Arquitectura

Se anaden dos notebooks y un backend configurable:

```
RTDETR_blackjack.ipynb       # entrenamiento RT-DETR-L
compare_models.ipynb        # validacion y benchmark lado a lado
vision/detector.py          # CardDetector con backend yolo | rtdetr
config.yaml                 # detection.backend + detection.imgsz
models/best_rtdetr_v1.pt    # pesos RT-DETR entrenados
```

El resto del pipeline no cambia. `CardDetector.detect()` sigue devolviendo una
lista de `Detection` con `label`, `confidence`, `bbox`, `center`, `track_id` y
`mask`. En RT-DETR, `mask` sera `None`, caso que el codigo ya soporta.

## Entrenamiento RT-DETR

`RTDETR_blackjack.ipynb` clona la estructura del notebook V4 y cambia:

- `MLFLOW_EXPERIMENT = "blackjackvai-rtdetr-v1"`
- `MODEL_BASE = "rtdetr-l.pt"`
- `RUN_NAME = "rtdetr_l_v1"`
- `from ultralytics import RTDETR`
- entrenamiento sin `task="segment"`, `overlap_mask` ni `mask_ratio`
- evaluacion solo con `m.box.*`
- export estable a `models/best_rtdetr_v1.pt`

Hiperparametros iniciales:

| Parametro | Valor |
|-----------|-------|
| `epochs` | 100 |
| `imgsz` | 640 |
| `batch` | 8 |
| `lr0` | 0.0001 |
| `lrf` | 0.01 |
| `patience` | 20 |
| `amp` | True |
| `cos_lr` | True |

## Comparacion

`compare_models.ipynb` carga `models/best.pt` y `models/best_rtdetr_v1.pt` y
mide ambos modelos sobre el mismo split `val`:

- `mAP50(box)` y `mAP50-95(box)`
- precision y recall
- latencia media, p50, p95 y FPS
- parametros, FLOPs y tamano en disco
- VRAM pico durante inferencia
- grids cualitativos lado a lado

Los outputs se guardan en `comparison_runs/`:

```
comparison_summary.csv
comparison_summary.md
comparison_metrics.png
qualitative_grid.png
hard_cases_grid.png
```

## Riesgos

| Riesgo | Mitigacion |
|--------|------------|
| RT-DETR no converge con `lr0=1e-4` | probar `5e-5` y `2e-4`, registrando cada tirada en MLflow |
| `batch=8` no cabe en 8 GB VRAM | bajar a `batch=4` |
| augmentations agresivas perjudican a RT-DETR | repetir con augmentations conservadoras de V3 |
| comparacion injusta por preprocesado | documentar que Ultralytics preserva aspect ratio en YOLO, pero no igual en RT-DETR |
| falta `best_rtdetr_v1.pt` | no activar `backend: rtdetr` en produccion hasta exportar pesos |
| `imgsz` suboptimo por backend | usar 1280 para YOLO en webcam y 640 como punto inicial para RT-DETR |

## Estado actual

- Notebook RT-DETR creado.
- Notebook de comparacion creado.
- Produccion soporta `backend: yolo | rtdetr`, con `yolo` por defecto.
- Quedan pendientes entrenamiento, validacion, MLflow y resultados finales.

## Referencias

- Ultralytics RT-DETR: https://docs.ultralytics.com/models/rtdetr/
- Ultralytics train settings: https://docs.ultralytics.com/modes/train/
- Ultralytics configuration: https://docs.ultralytics.com/usage/cfg/
