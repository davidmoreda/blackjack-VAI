# Plan — Segundo modelo para comparación con YOLOv8m-seg

**Fecha:** 2026-05-19
**Autor:** Javier Cerón
**Objetivo:** Entrenar un segundo detector sobre el mismo dataset (BlackjackVAI V4)
para comparar arquitectónica, métrica y operativamente contra el modelo actual
YOLOv8m-seg (mAP50(box) = 0.954).

---

## 1. Decisión — modelo elegido: **RT-DETR-L**

**Real-Time Detection Transformer** (Baidu, 2023), variante *Large* — `rtdetr-l.pt`.

| Aspecto | YOLOv8m-seg (actual) | RT-DETR-L (nuevo) |
|---|---|---|
| Familia | CNN (anchor-based, NMS) | **Transformer** (anchor-free, sin NMS) |
| Backbone | CSPDarknet | HGNetv2 + Transformer encoder/decoder |
| Parámetros | ~27 M | ~32 M |
| Tarea entrenada | Detección + segmentación | Detección |
| API | `ultralytics.YOLO` | `ultralytics.RTDETR` |
| Postproceso | NMS | Hungarian matching (sin NMS) |
| Real-time en GPU | ✅ | ✅ (paper bate a YOLOv8 en COCO a igual latencia) |

### Por qué RT-DETR y no otra cosa

Considerados y descartados:

- **Faster R-CNN / Mask R-CNN (Detectron2)** — dos etapas, no real-time, ecosistema distinto (Detectron2 vs ultralytics), narrativa débil (“obvio que es más lento”).
- **DETR original** — requiere ~500 epochs para converger, inviable en RTX 4070 con 3.2k imágenes.
- **EfficientDet** — TF/PyTorch viejo, fricción de integración alta.
- **YOLOv9 / v10 / v11 / v12** — misma familia, comparación poco interesante para un máster.
- **DINO / DINO-DETR** — demasiado pesado para 8 GB VRAM, entreno muy lento.

**RT-DETR-L gana porque:**

1. **Narrativa académica fuerte:** “CNN vs Transformer en detección real-time sobre dataset propio de 54 clases.” Es exactamente el tipo de comparación que un trabajo de Visión Artificial Avanzada espera.
2. **Misma librería** (`ultralytics ≥ 8.0`): mismo formato de dataset (txt + data.yaml), mismos argumentos de entreno, mismo callback de MLflow, mismo `.predict()` / `.track()` con ByteTrack. Reutilizamos infraestructura.
3. **Drop-in en producción:** `vision/detector.py` sólo necesita cambiar la clase importada — el resto del pipeline (zonas, debouncer, máquina de estados) no se toca.
4. **Comparable en tamaño:** YOLOv8m ≈ 27 M params, RT-DETR-L ≈ 32 M params, batch=8 cabe en RTX 4070 (8 GB) sin problemas con AMP.
5. **Comparable en propósito:** ambos están diseñados explícitamente para detección real-time. Comparas dos enfoques que compiten en la misma liga.

### Alcance de la comparación

La aplicación sólo consume **bboxes** (las máscaras de YOLO-seg son decorativas en el overlay). Por tanto comparamos en **modo detección puro**:

- **YOLOv8m-seg V4** ya reporta métricas de bbox (`box.map50`, `box.map`). Usamos esas.
- **RT-DETR-L** entrena sólo detección. Comparamos `box.map50` y `box.map` 1:1.

No hace falta reentrenar YOLO en modo detección puro — el head de bbox se entrena junto al de seg y `model.val()` ya da las métricas de box independientemente.

---

## 2. Arquitectura del trabajo

```
blackjack-VAI/
├── RTDETR_blackjack.ipynb         ← NUEVO: notebook de entreno RT-DETR
├── compare_models.ipynb              ← NUEVO: notebook de comparación lado a lado
├── docs/plans/
│   └── 2026-05-19-rtdetr-comparison.md   ← NUEVO: documento de diseño
├── models/
│   ├── best.pt                       ← (existente, symlink a YOLO v4)
│   └── best_rtdetr_v1.pt             ← NUEVO: pesos RT-DETR entrenados
├── vision/
│   └── detector.py                   ← MODIFICAR: soporte multi-backend
├── config.yaml                       ← MODIFICAR: campo `detection.backend`
└── README.md                         ← MODIFICAR: sección de comparación
```

---

## 3. Fases del trabajo

### Fase 1 — Notebook de entrenamiento RT-DETR (`RTDETR_blackjack.ipynb`)

Clon estructural del notebook V4 actual, con estos cambios concretos:

| Celda V4 | Cambio para RT-DETR |
|---|---|
| Config | `MLFLOW_EXPERIMENT = "blackjackvai-rtdetr-v1"`, `MODEL_BASE = "rtdetr-l.pt"` (descarga automática), `RUN_NAME = "rtdetr_l_v1"` |
| Dataset | **Sin cambios** — mismo Roboflow V4, mismo `data.yaml`, mismas 54 clases |
| Labels fix | **Sin cambios** — RT-DETR ignora los polígonos y usa el bbox encerrado |
| Letterbox | **Sin cambios** — RT-DETR también usa letterbox |
| MLflow | **Sin cambios** — `yolo_settings.update({"mlflow": True})` funciona idéntico |
| Entreno | `from ultralytics import RTDETR; model = RTDETR("rtdetr-l.pt")`. **Quitar** `task="segment"`, `overlap_mask`, `mask_ratio`. Mantener augmentations (RT-DETR las soporta vía ultralytics). |
| Eval | `m = model.val(...)` — sólo `m.box.map50` y `m.box.map`, no `m.seg.*` |
| Webcam test | `from ultralytics import RTDETR` y `imgsz=640` (RT-DETR está pensado para 640; subir a 1280 sólo si la métrica lo justifica) |

**Hiperparámetros sugeridos (primera tirada):**

```python
train_kwargs = dict(
    data="BlackjackVAI-4/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,             # mismo que YOLO V4
    device=0,
    workers=4,
    seed=42,
    amp=True,
    cos_lr=True,
    lr0=0.0001,          # RT-DETR usa LR más bajo que YOLO (AdamW)
    lrf=0.01,
    warmup_epochs=2000,  # RT-DETR usa warmup en pasos, no epochs — ajusta
    patience=20,
    save_period=10,
    project="training_runs/runs",
    name="rtdetr_l_v1",
    # Augmentations — espejan V4 para fairness
    hsv_h=0.02, hsv_s=0.90, hsv_v=0.60,
    degrees=20.0, translate=0.20, scale=0.85, shear=8.0, perspective=0.001,
    flipud=0.25, fliplr=0.50,
    mosaic=1.0, mixup=0.25, copy_paste=0.60, erasing=0.50,
    close_mosaic=20,
)
```

**Nota sobre `lr0` y `warmup_epochs`:** RT-DETR es sensible al LR (usa AdamW por defecto). Si la primera tirada no converge, bajar a `lr0=0.00005`. Documentar la decisión en MLflow.

**Pesos finales:** copiar a [models/best_rtdetr_v1.pt](models/best_rtdetr_v1.pt) con timestamp, igual que se hace en la celda 13 de V4.

### Fase 2 — Notebook de comparación (`compare_models.ipynb`)

Carga ambos modelos y los pasa por el **mismo split de validación** del V4 (644 imágenes). Mide:

| Métrica | Cómo |
|---|---|
| `mAP50(box)` | `model.val(data=..., split="val", conf=0.001, iou=0.6)` |
| `mAP50-95(box)` | mismo `.val()` |
| `precision` / `recall` | `m.box.mp`, `m.box.mr` |
| Latencia inferencia (ms/img) | `model.predict(..., verbose=False)` con `time.perf_counter()` sobre 200 imgs de val, descartar primeras 20 (warm-up GPU) |
| FPS efectivo | 1000 / latencia media |
| Parámetros | `sum(p.numel() for p in model.model.parameters())` |
| FLOPs | `ultralytics.utils.torch_utils.get_flops(model.model, imgsz=640)` |
| Tamaño en disco (.pt MB) | `os.path.getsize(path) / 1e6` |
| Memoria VRAM en inferencia | `torch.cuda.max_memory_allocated()` |
| Matriz de confusión por clase | `confusion_matrix.png` que ya genera `.val()` |

**Análisis cualitativo:**

- Grid 3×3 con la misma imagen procesada por ambos modelos (lado a lado).
- 5 casos difíciles seleccionados a mano: carta lejana, carta ocluida, dos cartas solapadas, iluminación lateral, carta rotada >30°.
- Para cada caso: ¿qué detectó cada uno? ¿con qué confianza?

**Output:** tabla resumen en markdown + figura `comparison_metrics.png` con barras agrupadas (mAP, FPS, params).

### Fase 3 — Integración en producción (multi-backend en `detector.py`)

Refactor mínimo para que la app pueda usar cualquiera de los dos modelos según `config.yaml`:

```yaml
# config.yaml — añadir campo backend
detection:
  backend: yolo          # yolo | rtdetr
  model_path: models/best.pt
  # ... resto igual
```

```python
# vision/detector.py — cambio mínimo
from ultralytics import YOLO, RTDETR

class CardDetector:
    def __init__(self, model_path: str, backend: str = "yolo",
                 confidence: float = 0.35, iou: float = 0.45):
        cls = {"yolo": YOLO, "rtdetr": RTDETR}[backend.lower()]
        self.model = cls(model_path)
        self.backend = backend
        self.confidence = confidence
        self.iou = iou
```

El método `.detect()` no cambia: ambas APIs exponen `.track()` con ByteTrack y devuelven `r.boxes` con el mismo schema. **Único matiz:** RT-DETR no produce máscaras → `r.masks` será `None`. El código actual ya maneja ese caso (`if masks_data is not None`), así que no hay que tocar nada más.

Tests de humo:

1. `python cli.py` con `backend: yolo` → comportamiento idéntico al actual.
2. `python cli.py` con `backend: rtdetr` y `model_path: models/best_rtdetr_v1.pt` → debe detectar y trackear con IDs estables.
3. Comparar FPS de ambos sobre el mismo stream de webcam (cronometrar 60 s).

### Fase 4 — Documentación

- **`docs/plans/2026-05-19-rtdetr-comparison.md`** — documento de diseño (1–2 páginas) explicando la decisión, idéntico en estilo a [2026-04-07-blackjack-cv-design.md](docs/plans/2026-04-07-blackjack-cv-design.md).
- **`README.md`** — añadir sección **“Comparación de modelos”** debajo de “Entrenamiento del modelo”, con tabla de resultados y un párrafo de conclusiones.
- **MLflow:** asegurarse de que existen los experimentos `blackjackvai-v4` y `blackjackvai-rtdetr-v1` en el mismo `mlflow.db` para que la comparación sea trivial en la UI.

---

## 4. Métricas que reportaremos (tabla final)

| Métrica | YOLOv8m-seg V4 | RT-DETR-L v1 |
|---|---|---|
| mAP50 (box) | 0.954 (referencia) | _por medir_ |
| mAP50-95 (box) | 0.948 (referencia) | _por medir_ |
| Precision | _de results.csv_ | _por medir_ |
| Recall | _de results.csv_ | _por medir_ |
| Params (M) | ~27 | ~32 |
| FLOPs @ 640 (G) | _por medir_ | _por medir_ |
| Latencia inferencia GPU (ms) | _por medir_ | _por medir_ |
| FPS @ 640 | _por medir_ | _por medir_ |
| FPS @ 1280 | _por medir_ | _por medir_ |
| Tamaño .pt (MB) | 55 | _por medir_ |
| VRAM inferencia (MB) | _por medir_ | _por medir_ |
| Epochs hasta best | _de results.csv_ | _por medir_ |
| Tiempo total entreno (h) | _de MLflow_ | _por medir_ |

---

## 5. Riesgos y mitigación

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| RT-DETR no converge con `lr0=0.0001` | Media | Probar `5e-5` y `2e-4`, dejar la primera tirada corta (20 epochs) como prueba antes de lanzar 100 |
| Batch=8 OOM en 8 GB VRAM | Baja-Media | Bajar a `batch=4` y `accumulate=2` para mantener batch efectivo |
| Etiquetas de polígono confunden a RT-DETR | Muy baja | Ultralytics ya las colapsa a bbox; verificar con una imagen de muestra en la celda 5 del notebook |
| Augmentations agresivas perjudican a RT-DETR (más sensible que YOLO) | Media | Si la primera tirada sale baja, hacer una segunda con augmentations conservadoras (las de V3) y comparar |
| Comparación injusta por configuraciones distintas | Media | Documentar **todo** en MLflow: misma seed, mismo split, mismo `data.yaml`, mismo `imgsz` |
| Roboflow API key expirada o dataset movido | Baja | El dataset ya está en disco (`BlackjackVAI-4/`), no hace falta volver a descargar |

---

## 6. Checklist de ejecución

- [ ] **F1.1** Crear `RTDETR_blackjack.ipynb` (copia + modificaciones del V4)
- [ ] **F1.2** Verificar descarga `rtdetr-l.pt` desde ultralytics
- [ ] **F1.3** Lanzar primera tirada corta (20 epochs) — sanity check de convergencia
- [ ] **F1.4** Lanzar entreno completo (100 epochs, patience 20)
- [ ] **F1.5** `model.val()` sobre split `val` → métricas finales
- [ ] **F1.6** Copiar `best.pt` → `models/best_rtdetr_v1.pt` con timestamp
- [ ] **F1.7** Subir artefactos a MLflow (`blackjackvai-rtdetr-v1`)
- [ ] **F2.1** Crear `compare_models.ipynb`
- [ ] **F2.2** Validar ambos modelos sobre mismo split
- [ ] **F2.3** Medir latencia, FPS, FLOPs, VRAM
- [ ] **F2.4** Análisis cualitativo: 5 casos difíciles lado a lado
- [ ] **F2.5** Exportar figuras (`comparison_metrics.png`, grid lado a lado)
- [ ] **F3.1** Modificar [vision/detector.py](vision/detector.py) — soporte multi-backend
- [ ] **F3.2** Añadir `detection.backend` a [config.yaml](config.yaml)
- [ ] **F3.3** Smoke test: `python cli.py` con cada backend
- [ ] **F3.4** Smoke test: medir FPS real sobre stream webcam de 60 s
- [ ] **F4.1** Crear `docs/plans/2026-05-19-rtdetr-comparison.md`
- [ ] **F4.2** Añadir sección “Comparación de modelos” a [README.md](README.md)
- [ ] **F4.3** Commit final con `git log` que documente cada fase

---

## 7. Definición de “hecho” (Definition of Done)

El trabajo se considera completo cuando:

1. `models/best_rtdetr_v1.pt` existe y es cargable.
2. `compare_models.ipynb` ejecuta de arriba a abajo sin error y produce la tabla final.
3. La app web (`uvicorn api.main:app`) funciona indistintamente con `backend: yolo` o `backend: rtdetr` configurado en `config.yaml`.
4. README contiene la tabla de comparación y un párrafo de 5–10 líneas con la conclusión (cuál ganó, en qué métrica, y por qué).
5. MLflow muestra los dos experimentos en paralelo con métricas comparables.

---

## 8. Siguiente paso inmediato

Cuando confirmes este plan, el primer movimiento es **crear `RTDETR_blackjack.ipynb`** clonando estructuralmente [YOLO_blackjack_v4.ipynb](YOLO_blackjack_v4.ipynb) y aplicando los cambios listados en la Fase 1. No tocamos producción ni dataset hasta que el modelo esté entrenado y validado.
