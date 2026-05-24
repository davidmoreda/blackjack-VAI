# Blackjack VAI

Sistema de seguimiento de blackjack en tiempo real mediante visión artificial. Detecta cartas con YOLOv8m-seg + ByteTrack, sigue el estado del juego, lleva conteo de cartas y muestra probabilidades, valor esperado (EV) y sugerencias óptimas en una interfaz web.

---

## Capacidades

- **Detección y clasificación** de 54 clases de cartas con YOLOv8m-seg + ByteTrack
- **Seguimiento del juego** automático: mano del jugador, mano del crupier, valor de cada mano
- **Conteo de cartas** con Hi-Lo, KO y Omega II
- **EV dinámico** (STAND / HIT / DOUBLE) calculado sobre la composición real del zapato
- **Sugerencia óptima** de jugada en tiempo real
- **Interfaz web** vía FastAPI + WebSockets

---

## Arquitectura

```
blackjack-VAI/
├── api/
│   ├── main.py            # FastAPI: startup, /video_feed MJPEG, sirve frontend
│   ├── routes.py          # REST: /api/start|stand|reset|undo|remove|state|config
│   └── ws.py              # WebSocket /ws: vision thread, debounce, broadcast
├── frontend/
│   ├── index.html         # UI — panel lateral + feed de cámara
│   └── panel.js           # WebSocket client, scoreboard, EV, historial
├── game/
│   ├── engine.py          # hand_total, is_bust, is_blackjack, dealer_should_hit
│   ├── counter.py         # CardCounter: Hi-Lo / KO / Omega II
│   ├── deck_tracker.py    # DeckTracker: composición exacta del zapato
│   ├── ev_calculator.py   # EV dinámico con lru_cache
│   ├── strategy.py        # Estrategia básica + suggest_action_with_ev
│   └── state_machine.py   # BlackjackStateMachine: turnos, historial, marcador
├── vision/
│   ├── capture.py         # CameraCapture: webcam local o stream HTTP
│   ├── detector.py        # CardDetector: YOLO + ByteTrack, devuelve Detection[]
│   ├── debouncer.py       # CardDebouncer: confirma carta tras 1 s de presencia
│   ├── zones.py           # ZoneManager, build_zones, draw_zones
│   └── zone_manager.py    # Alias / helpers de zona
├── models/
│   └── best.pt            # Modelo entrenado (no incluido en repo)
├── cli.py                 # Modo terminal: ventana OpenCV, sin servidor
├── cam_server_windows.py  # Servidor MJPEG para pasar webcam de Windows a WSL
├── config.yaml            # Configuración global
├── Dockerfile
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── YOLO_blackjack_v2.ipynb
├── YOLO_blackjack_v3.ipynb
└── YOLO_blackjack_v4.ipynb
```

---

## Fases del proyecto

| Fase | Estado | Descripción |
|------|--------|-------------|
| 1. Dataset | ✅ | Creación, etiquetado y limpieza en Roboflow (V3 → V4) |
| 2. Entrenamiento | 🔄 En curso | YOLOv8m-seg local con RTX 4070 + MLflow (v4 en progreso) |
| 3. API | ✅ | FastAPI + WebSockets + endpoints REST |
| 4. Lógica de juego | ✅ | Máquina de estados, conteo, EV dinámico, estrategia básica |
| 5. Frontend | ✅ | Interfaz web en tiempo real con panel lateral |
| 6. Integración | ✅ | Sistema completo webcam → detección → sugerencia |

---

## Historial de entrenamiento

### Dataset

| Versión | Imágenes train | Notas |
|---------|---------------|-------|
| V1 | ~10 506 | Grayscale + Tiling 2×2 → incompatible con webcam color en tiempo real |
| V2 | ~3 741 | Sin augmentations → no se usó para entrenar |
| V3 | ~3 741 | Sin augmentations Roboflow, augmentations YOLO estándar |
| **V4** | **2 574** | Letterbox (sin distorsión), augmentations agresivos para detección a distancia |
| **V5** | **7 722** | Dataset V4 triplicado con albumentations: blur, JPEG degradation, CLAHE |

### Modelos

| Modelo | Epochs | Best epoch | Mask mAP50-95 | Notas |
|--------|--------|------------|--------------|-------|
| V3 | 100 | — | 0.954 | Baseline funcional, funciona bien de cerca |
| V4 | 120 | 89 | **0.984** | Fine-tune desde V3. Mejor en distancia y ángulos |
| **V5** | 71* | 51 | **0.980** | Fine-tune desde V4 sobre dataset 3× degradado. Más robusto en condiciones reales de webcam |

*V5 paró por early stopping (patience=20).

### Configuración de entrenamiento

| Parámetro | V3 | V4 | V5 |
|-----------|----|----|-----|
| Modelo base | `yolov8m-seg.pt` | `best_v3.pt` | `best_v4.pt` |
| Epochs | 100 | 120 | 80 |
| Batch | 8 | 8 | 8 |
| lr0 | auto | auto | 0.0005 |
| close_mosaic | 10 | 20 | 15 |
| copy_paste | 0.20 | 0.50 | 0.60 |
| scale | 0.30 | 0.70 | 0.85 |
| GPU | RTX 4070 Laptop 8 GB | idem | idem |
| Tracking | MLflow | MLflow | MLflow |

### Lanzar entrenamiento

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Arrancar el servidor
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Abre http://localhost:8000
```

### Seguimiento con MLflow

La base de datos MLflow está en el propio repositorio (`mlflow.db`, excluida de git).

```bash
python cli.py
```

Controles de teclado:

| Tecla | Acción |
|-------|--------|
| `S` | Nueva partida |
| `R` | Nueva ronda |
| `ESPACIO` | Stand del jugador activo |
| `Z` | Deshacer última carta |
| `H` | Abrir navegador de historial |
| `P` | Abrir panel de configuración |
| `Q` | Salir |

En el navegador de historial (`H`):

| Tecla | Acción |
|-------|--------|
| `↑ / ↓` | Navegar entre cartas |
| `X` / `DEL` | Eliminar carta seleccionada |
| `H` / `ESC` | Cerrar navegador |

En el panel de configuración (`P`):

| Tecla | Acción |
|-------|--------|
| `↑ / ↓` | Navegar opciones |
| `← / →` | Cambiar valor |
| `P` / `ESC` | Guardar y cerrar |

### Opción C — Docker

```bash
git clone <url-del-repo>
cd blackjack-VAI
```

Git LFS descargará automáticamente el modelo `models/best.pt` durante el clone.

---

### 3. Elegir modo de ejecución

---

### Opción A — Linux con webcam USB (más simple)

La webcam está conectada directamente al host Linux.

**1. Editar `config.yaml`**

```yaml
camera:
  index: 0   # índice de /dev/video0
```

**2. Descomentar el dispositivo en `docker-compose.yml`**

```yaml
devices:
  - /dev/video0:/dev/video0
```

**3. Levantar**

```bash
docker compose up --build
# Abre http://localhost:8000
```

---

### Opción B — WSL2 + Docker + cámara de Windows (recomendado en Windows)

WSL2 no tiene acceso directo a la webcam USB. Se expone la cámara desde Windows como stream MJPEG y Docker la consume por HTTP.

**1. En Windows** (PowerShell o CMD, **no** WSL)

```powershell
uv run --with flask --with opencv-python python cam_server_windows.py
```

Salida esperada:
```
Cámara 0 abierta: 1280x720
Stream disponible en: http://0.0.0.0:5050/video
```

**2. `config.yaml`** — ya configurado por defecto

```yaml
camera:
  index: "http://host.docker.internal:5050/video"
```

`host.docker.internal` es resuelto automáticamente al host Windows por Docker. Está declarado en `docker-compose.yml` via `extra_hosts`.

**3. Levantar Docker desde WSL**

```bash
docker compose up --build
# Abre http://localhost:8000
```

**Verificar conectividad de la cámara**

```bash
docker compose exec blackjack-vai python -c \
  "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:5050/health').read())"
# Respuesta esperada: b'{"cam":0,"height":720,"status":"ok","width":1280}'
```

---

### Opción C — Windows con Docker Desktop

Igual que la Opción B. Docker Desktop expone `host.docker.internal` de forma nativa, no es necesario configurar nada adicional.

---

### Opción D — Sin Docker (desarrollo local, Linux/WSL)

```bash
# 1. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instalar PyTorch con soporte CUDA (si tienes GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Sin GPU (solo CPU):
# pip install torch torchvision

# 3. Instalar el resto de dependencias
pip install lapx>=0.5.2
pip install -r requirements.txt

# 4. Arrancar
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Abre http://localhost:8000
```

---

### Opción E — Modo terminal (sin servidor web)

```bash
python cli.py
```

| Tecla | Acción |
|-------|--------|
| `S` | Nueva partida |
| `R` | Nueva ronda |
| `ESPACIO` | Stand del jugador activo |
| `Z` | Deshacer última carta |
| `H` | Historial de cartas |
| `P` | Panel de configuración |
| `Q` | Salir |

---

### Actualizar el modelo sin reconstruir la imagen

```bash
# Copiar nuevo best.pt al directorio models/
cp training_runs/runs/yolo8m_seg_v5/weights/best.pt models/best.pt

# Inyectar en el contenedor en ejecución y reiniciar
docker compose cp models/best.pt blackjack-vai:/app/models/best.pt
docker compose restart blackjack-vai
```

---

## Configuración

`config.yaml` se monta como volumen en Docker — los cambios se aplican reiniciando el contenedor, sin rebuild.

```yaml
api:
  host: 0.0.0.0
  port: 8000

camera:
  index: 0                  # int para webcam local, URL para stream HTTP
  fps: 30
  width: 1280
  height: 720

detection:
  model_path: models/best.pt
  confidence: 0.55          # umbral de confianza YOLO
  iou: 0.45
  inference_fps: 15         # inferencias YOLO por segundo

game:
  num_players: 1            # 1–7
  num_decks: 1              # 1, 2, 4, 6 u 8
  counting_system: hilo     # hilo | ko | omega2

zones:
  dealer:   { x: 0.0, y: 0.0, w: 1.0, h: 0.4 }
  player_1: { x: 0.0, y: 0.4, w: 0.5, h: 0.6 }
  player_2: { x: 0.5, y: 0.4, w: 0.5, h: 0.6 }
```

---

## API REST

Base URL: `http://localhost:8000`

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Interfaz web |
| `GET` | `/video_feed` | Stream MJPEG con bboxes y máscaras |
| `WebSocket` | `/ws` | Estado del juego en tiempo real |
| `POST` | `/api/start` | Nueva partida (resetea todo) |
| `POST` | `/api/reset` | Nueva ronda (conserva el zapato) |
| `POST` | `/api/stand` | Jugador se planta `{"player_id": "player_1"}` |
| `POST` | `/api/undo` | Deshace la última carta detectada |
| `POST` | `/api/remove/{idx}` | Elimina carta por índice del historial |
| `GET` | `/api/state` | Estado completo del juego |
| `GET` | `/api/config` | Configuración actual |
| `POST` | `/api/config` | Reconfigura jugadores, mazos y sistema de conteo |

---

## Pipeline de detección

```
Webcam (30 fps)
  └─ CameraCapture.read()
       └─ CardDetector.detect()         ← YOLOv8m-seg + ByteTrack (15 fps)
            └─ ZoneManager.get_zone()   ← asigna dealer / player_N
                 └─ CardDebouncer.tick() ← confirma tras 1 s de presencia
                      └─ BlackjackStateMachine.add_card()
                           └─ CardCounter.register()   ← Hi-Lo / KO / Omega II
                           └─ DeckTracker.remove()     ← composición del zapato
                           └─ suggest_action_with_ev() ← EV dinámico
```

### CardDebouncer

Evita falsos positivos: una carta debe estar presente en la zona asignada durante **1 segundo continuo** antes de registrarse. Si desaparece antes, el contador se resetea. La barra de progreso bajo cada bbox muestra el tiempo acumulado.

### EV dinámico

El calculador (`ev_calculator.py`) computa el valor esperado real de cada acción usando la composición residual del zapato:

- Usa `lru_cache` con hasta 100 000 entradas por función
- La distribución del zapato se redondea a 4 decimales para colapsar estados similares (~1000× más rápido que simulación exacta, error < 0.001 EV)
- El dealer sigue las reglas estándar (se planta en 17 duro, configurable soft-17)
- SPLIT aún usa la tabla de estrategia básica (EV de split requiere modelar dos manos independientes)

---

## Zonas de la mesa

Las zonas se generan dinámicamente según `num_players`:

- **Dealer**: franja superior (40 % del alto, ancho completo)
- **Player 1…N**: franja inferior (60 % del alto) dividida en N columnas iguales

El overlay en el feed de cámara muestra:
- Rectángulo semitransparente por zona con color identificativo
- Total de la mano centrado en grande
- `BUST!` en rojo parpadeante / `BLACKJACK!` en dorado / `21 !` cuando corresponde
- Bboxes YOLO con máscara de segmentación y barra de progreso del debouncer

---

## config.yaml

```yaml
api:
  host: 0.0.0.0
  port: 8080

camera:
  index: 0          # índice de webcam o URL HTTP para WSL
  fps: 30
  width: 1280
  height: 720

detection:
  model_path: models/best.pt
  confidence: 0.55
  iou: 0.45
  inference_fps: 15  # inferencias YOLO por segundo

game:
  num_players: 1
  num_decks: 1
  counting_system: hilo  # hilo | ko | omega2
```

---

## Sistemas de conteo soportados

| Sistema | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10/J/Q/K | A |
|---------|---|---|---|---|---|---|---|---|----------|---|
| Hi-Lo | +1 | +1 | +1 | +1 | +1 | 0 | 0 | 0 | −1 | −1 |
| KO | +1 | +1 | +1 | +1 | +1 | +1 | 0 | 0 | −1 | −1 |
| Omega II | +1 | +1 | +2 | +2 | +2 | +1 | 0 | −1 | −2 | 0 |

El **true count** = running count / mazos restantes en el zapato.

---

## Clases del dataset

54 clases en total:

| Palo | Símbolo | Clases |
|------|---------|--------|
| Spades (picas) | S | A Spades … K Spades |
| Hearts (corazones) | H | A Hearts … K Hearts |
| Diamonds (diamantes) | D | A Diamonds … K Diamonds |
| Clubs (tréboles) | C | A Clubs … K Clubs |
| Especiales | — | card_back, Joker |

---

## Entrenamiento del modelo

### Historial del dataset

| Versión | Imágenes | Preprocesado | Augmentations | Resultado |
|---------|----------|--------------|---------------|-----------|
| **V1** | ~10 506 | Auto-Orient, Tiling 2×2, **Grayscale**, Contrast adaptativo | Múltiples (Roboflow) | ❌ No detecta en webcam — grayscale vs color en tiempo real |
| **V2** | ~3 741 | Auto-Orient, Resize 640×640 | Ninguna | ❌ No se usó para entrenar |
| **V3** | ~3 741 | Auto-Orient, Resize 640×640 | **Ninguna** (las hace YOLO) | ✅ mAP50=0.954 — funciona bien de cerca |
| **V4** | ~3 741+ | Auto-Orient, **Letterbox** (sin distorsión) | Augmentation agresivo para distancia | 🔄 En entrenamiento |

Errores en V1:
1. **Grayscale** → el modelo aprendió imágenes en gris pero la webcam envía color
2. **Tiling 2×2** → el modelo vio recortes de cartas, no frames completos
3. **imgsz mismatch** → entrenado a 960 px, inferencia a 640 px
4. **Bug de ruta** → el notebook descargaba V2 pero usaba la ruta de V1

### V3 — configuración

- **Preprocessing:** Auto-Orient + Resize 640×640 (Stretch)
- **Augmentations:** YOLO — scale=0.30, mosaic=1.0, flipud=0.15, fliplr=0.50, mixup=0.15, copy_paste=0.20
- **Split:** 80 % train / 10 % val / 10 % test
- **Resultado:** mAP50(box)=0.954, mAP50-95(box)=0.948

### V4 — configuración (producción)

- **Preprocessing:** Auto-Orient + **Letterbox** (padding gris, sin distorsión de cartas)
- **Split:** **90 % train / 10 % val** (test fusionado en train para maximizar datos de producción)
- **Augmentations agresivas para detección a distancia:**

| Parámetro | V3 | V4 | Efecto |
|-----------|----|----|--------|
| `scale` | 0.30 | **0.70** | Cartas al 30% del tamaño → simula cámara lejana |
| `copy_paste` | 0.20 | **0.50** | Pega cartas sobre fondos variados |
| `erasing` | — | **0.40** | Oclusión por manos/fichas |
| `perspective` | — | **0.0008** | Ángulo de webcam sobre la mesa |
| `shear` | 2.0 | **6.0** | Perspectiva lateral |
| `degrees` | 10 | **15** | Más rotación |
| `close_mosaic` | 10 | **15** | Más epochs con mosaic |

- **Inferencia:** `imgsz=1280` (resolución nativa webcam) para detección óptima a distancia

### Configuración de entrenamiento local

| Parámetro | V3 | V4 |
|-----------|----|----|
| GPU | RTX 4070 Laptop (8 GB VRAM) | RTX 4070 Laptop (8 GB VRAM) |
| Modelo base | `yolov8m-seg.pt` | `yolov8m-seg.pt` |
| Epochs | 100 (patience 20) | 120 (patience 25) |
| imgsz (train) | 640 | 640 |
| imgsz (infer) | 640 | **1280** |
| Batch | 8 | 8 |
| Tracking | MLflow | MLflow (mismo DB) |

### Lanzar entrenamiento V4

```bash
# Activar el entorno con MLflow y ultralytics
source "/home/dmore/code/Máster IA/01.-MASTER COURSES/12.- MLOPS Y AI BI/mlops_env/.venv/bin/activate"

jupyter notebook YOLO_blackjack_v4.ipynb
```

### Reanudar entrenamiento interrumpido

```python
RESUME = True  # valor por defecto en ambos notebooks
```

### Seguimiento con MLflow

```bash
cd "/home/dmore/code/Máster IA/01.-MASTER COURSES/12.- MLOPS Y AI BI/mlops_env"
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001
# http://localhost:5001
```

Los experimentos `blackjackvai-v3` y `blackjackvai-v4` están en el mismo `mlflow.db`.  
Métricas por epoch: `box_loss`, `seg_loss`, `cls_loss`, `mAP50`, `mAP50-95`.

### Estructura tras el entrenamiento

```
training_runs/runs/yolo8m_seg_v4/
├── weights/
│   ├── best.pt
│   ├── last.pt
│   └── epoch10.pt
├── results.csv
└── *.png

models/
├── best_v3_YYYYMMDD_HHMM.pt
├── best_v4_YYYYMMDD_HHMM.pt
└── best.pt  →  best_v4_...pt  (symlink al último)
```

---

## Comparación de modelos — YOLOv8m-seg vs RT-DETR-L

Para validar la elección de arquitectura se entrenó un segundo detector, **RT-DETR-L** (Real-Time Detection Transformer, Baidu 2023), sobre el mismo dataset y se comparó cabeza a cabeza con el modelo de producción.

### Por qué RT-DETR como modelo alternativo

| Aspecto | YOLOv8m-seg | RT-DETR-L |
|---|---|---|
| Familia | CNN (anchor-based, NMS) | **Transformer** (anchor-free, sin NMS) |
| Backbone | CSPDarknet | HGNetv2 + Transformer encoder/decoder |
| Postproceso | NMS | Hungarian matching |
| Parámetros | ~27 M | ~32 M |
| API ultralytics | `YOLO` | `RTDETR` |

La narrativa académica es la comparación entre el paradigma convolucional dominante (YOLO) y el paradigma Transformer emergente (RT-DETR) en detección real-time sobre un dataset propio de 54 clases.

### Metodología de matching

Para que la comparación fuera científicamente válida, el RT-DETR se entrenó replicando **el régimen de entrenamiento** de YOLO V4:

| Tipo | Variables | Valor |
|---|---|---|
| **Igualadas** | epochs, patience, batch, imgsz, seed, warmup_epochs, close_mosaic, augmentations | 120, 25, 8, 640, 42, 3, 20, idénticas |
| **Específicas por arquitectura** | optimizer, lr0 | YOLO: SGD lr0=0.001 / RT-DETR: AdamW lr0=0.0001 |

El optimizer y el learning rate no se igualan porque cada arquitectura tiene su régimen óptimo (es la práctica estándar en papers de comparación).

### Resultados — calidad de detección

Evaluado sobre el mismo split `val` (644 imágenes, BlackjackVAI V4), `conf=0.001`, `iou=0.6`:

| Métrica | YOLOv8m-seg V5 | RT-DETR-L matched | Δ |
|---|---:|---:|---:|
| **mAP50(box)** | **0.9862** | 0.9804 | −0.58 pt |
| **mAP50-95(box)** | **0.9802** | 0.9158 | **−6.44 pt** |
| Precision | **0.9797** | 0.9688 | −1.09 pt |
| Recall | **0.9839** | 0.9752 | −0.87 pt |

### Resultados — coste e inferencia

Benchmark sobre RTX 4070 Laptop, imgsz=640, 200 imágenes con 20 de warm-up:

| Métrica | YOLOv8m-seg V5 | RT-DETR-L matched |
|---|---:|---:|
| Parámetros | 27.25 M | 32.09 M |
| Tamaño .pt en disco | 54.9 MB | 66.4 MB |
| Latencia media | 61.2 ms | 74.4 ms |
| Latencia p95 | 64.7 ms | 85.5 ms |
| **FPS efectivo** | **16.3** | 13.4 |
| VRAM pico | 673 MB | 663 MB |

### Conclusión

**YOLOv8m-seg V5 supera a RT-DETR-L en todos los ejes relevantes** bajo configuración de entrenamiento equivalente:

1. **Calidad de localización (mAP50-95):** YOLO produce bboxes más ajustados (+6.44 pt). RT-DETR detecta correctamente las cartas pero su loss GIoU genera cajas más laxas. Para detección bruta (mAP50) la diferencia es marginal (~0.6 pt), ambos al techo del dataset.

2. **Velocidad:** YOLO es ~21% más rápido (16.3 vs 13.4 FPS). Para el objetivo real-time (15 FPS de inferencia configurados en producción), YOLO va sobrado; RT-DETR queda al límite.

3. **Coste de despliegue:** YOLO ocupa 12 MB menos en disco, usa 5 M parámetros menos y tiene latencia p95 21 ms más baja.

4. **Decisión:** YOLOv8m-seg permanece como modelo de producción. RT-DETR queda como baseline alternativo entrenado y disponible (`models/best_rtdetr_matched.pt`) para experimentación futura.

### Reproducibilidad

- Notebook de entreno RT-DETR: [RTDETR_blackjack_match_yolo.ipynb](RTDETR_blackjack_match_yolo.ipynb) (lanzable en Colab con GPU T4, ~2.7 h)
- Notebook de comparación: [compare_models.ipynb](compare_models.ipynb)
- Outputs: [comparison_runs/](comparison_runs/) — `comparison_summary.csv`, `comparison_metrics.png`, `qualitative_grid.png`, matrices de confusión y curvas P/R/F1 por modelo
- Tracking MLflow: experimentos `blackjackvai-v4`, `blackjackvai-rtdetr-matched` en `mlflow.db`

---

## Licencia

Proyecto académico — Máster en Inteligencia Artificial.
