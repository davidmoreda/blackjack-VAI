# Blackjack VAI

Sistema de seguimiento de blackjack en tiempo real mediante visión artificial. Apunta la cámara a la mesa y el sistema detecta las cartas automáticamente, lleva el estado del juego, cuenta las cartas y sugiere la jugada óptima con el valor esperado (EV) en tiempo real.

**Tecnologías principales:** YOLOv8m-seg · ByteTrack · FastAPI · WebSockets · OpenCV

**Funcionalidades:**
- Detección y clasificación de 54 clases de cartas
- Seguimiento automático del juego: turnos, totales de mano, estado de la ronda
- Conteo de cartas con Hi-Lo, KO y Omega II
- EV dinámico (HIT / STAND / DOUBLE) calculado sobre la composición real del zapato
- Sugerencia óptima de jugada en tiempo real
- Interfaz web + modo terminal

---

## Estructura del proyecto

```
blackjack-VAI/
│
├── api/                        # Backend FastAPI + WebSockets
│   ├── main.py                 # Startup, /video_feed MJPEG, sirve frontend en /
│   ├── routes.py               # REST: /api/start|stand|reset|undo|remove|state|config
│   ├── ws.py                   # WebSocket /ws: hilo de visión, debounce, broadcast
│   └── __init__.py
│
├── game/                       # Lógica completa de blackjack
│   ├── engine.py               # hand_total, is_bust, is_blackjack, dealer_should_hit
│   ├── counter.py              # CardCounter: Hi-Lo / KO / Omega II
│   ├── deck_tracker.py         # DeckTracker: composición exacta del la baraja
│   ├── ev_calculator.py        # EV dinámico con lru_cache (~100k entradas)
│   ├── strategy.py             # Estrategia básica + suggest_action_with_ev
│   └── state_machine.py        # BlackjackStateMachine: FSM, turnos, historial, marcador
│
├── vision/                     # Pipeline de visión artificial
│   ├── capture.py              # CameraCapture: webcam local o stream HTTP MJPEG
│   ├── detector.py             # CardDetector: YOLOv8/RT-DETR + ByteTrack → Detection[]
│   ├── debouncer.py            # CardDebouncer: confirma carta tras 1s de presencia
│   ├── zones.py                # ZoneManager, build_zones, draw_zones con overlay
│   ├── depth_normals.py        # DepthNormalEstimator: Depth Anything V2 (opcional)
│   └── normal_tracker.py       # Filtro Kalman para normales 3D (auxiliar)
│
├── frontend/                   # Interfaz web (sin frameworks)
│   ├── index.html              # UI tema casino: panel lateral + feed de cámara
│   └── panel.js                # WebSocket listener, scoreboard, EV, historial
│
├── models/                     # Pesos del modelo (Git LFS)
│   └── best.pt                 # YOLOv8m-seg V5 — mAP50-95=0.980 (53 MB)
│
├── notebooks/
│   ├── training/               # Notebooks de entrenamiento por versión
│   │   ├── YOLO_blackjack_v2.ipynb
│   │   ├── YOLO_blackjack_v3.ipynb
│   │   ├── YOLO_blackjack_v4.ipynb
│   │   ├── YOLO_blackjack_v5.ipynb
│   │   ├── YOLO_blackjack_v6.ipynb
│   │   └── RTDETR_blackjack_match_yolo.ipynb
│   └── analysis/               # Análisis y comparativas
│       └── compare_models.ipynb
│
├── docs/                       # Documentación y assets
│   ├── analysis/               # Resultados comparativa YOLOv8m vs RT-DETR
│   │   ├── comparison_metrics.png
│   │   ├── comparison_summary.csv
│   │   ├── comparison_summary.md
│   │   ├── qualitative_grid.png
│   │   └── val/                # Curvas P/R/F1 y matrices de confusión por modelo
│   ├── plans/                  # Documentos de arquitectura y diseño
│   ├── presentation_assets/    # Presentación del proyecto (.pptx)
│   └── lanzador.md             # Guía de ejecución paso a paso para Windows
│
├── tests/                      # Tests del proyecto
│
├── cli.py                      # Punto de entrada CLI (OpenCV, sin servidor web)
├── cam_server_windows.py       # Servidor MJPEG para exponer webcam Windows a WSL/Docker
├── config.yaml                 # Configuración global (cámara, detección, juego, zonas)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt            # Dependencias de producción
└── requirements-train.txt      # Dependencias adicionales para entrenamiento
```

---

## Lanzamiento

### Requisitos previos

```bash
pip install -r requirements.txt
```

**GPU NVIDIA (recomendado):** instala PyTorch con CUDA antes del paso anterior.

Para tarjetas Ampere/Ada (RTX 30xx/40xx):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Para tarjetas Blackwell (RTX 50xx):
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verifica que la GPU se detecta:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

### Opción A — Servidor web (recomendado)

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Abre `http://localhost:8080` en el navegador.

---

### Opción B — Modo terminal (sin navegador)

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
| `H` | Navegador de historial |
| `P` | Panel de configuración |
| `Q` | Salir |

En el navegador de historial (`H`):

| Tecla | Acción |
|-------|--------|
| `↑ / ↓` | Navegar entre cartas |
| `X` / `DEL` | Eliminar carta seleccionada |
| `H` / `ESC` | Cerrar |

---

### Opción C — Docker

```bash
docker compose up --build
```

Abre `http://localhost:8080`.

Si la webcam está en Windows (no en el contenedor), primero lanza el servidor de cámara en una terminal aparte:

```bash
python cam_server_windows.py
```

Y configura `config.yaml`:
```yaml
camera:
  index: "http://host.docker.internal:5050/video"
```

---

### Opción D — WSL2 + Docker + webcam Windows

**1. En PowerShell (Windows):**
```powershell
python cam_server_windows.py
```

**2. En WSL:**
```bash
docker compose up --build
# http://localhost:8080
```

---

## Configuración

`config.yaml` controla todo el sistema. Los cambios se aplican reiniciando el servidor (sin rebuild si usas Docker con volumen).

```yaml
api:
  host: 0.0.0.0
  port: 8080

camera:
  index: 0              # int para webcam local, URL string para stream HTTP
  fps: 30
  width: 1280
  height: 720

detection:
  backend: yolo         # yolo | rtdetr
  model_path: models/best.pt
  confidence: 0.55
  iou: 0.45
  imgsz: 1280
  inference_fps: 15

depth_anything:
  enabled: false        # true activa normales 3D (requiere 'transformers')
  variant: small        # small | base | large
  fps: 5

game:
  num_players: 1        # 1–7
  num_decks: 1          # 1, 2, 4, 6 u 8
  counting_system: hilo # hilo | ko | omega2

zones:
  dealer:   { x: 0.0, y: 0.0, w: 1.0, h: 0.4 }
  player_1: { x: 0.0, y: 0.4, w: 0.5, h: 0.6 }
  player_2: { x: 0.5, y: 0.4, w: 0.5, h: 0.6 }
```

---

## API REST

Base URL: `http://localhost:8080`

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
       └─ CardDetector.detect()          ← YOLOv8m-seg + ByteTrack (15 fps)
            └─ ZoneManager.get_zone()    ← asigna dealer / player_N
                 └─ CardDebouncer.tick() ← confirma tras 1s de presencia
                      └─ BlackjackStateMachine.add_card()
                           ├─ CardCounter.register()    ← Hi-Lo / KO / Omega II
                           ├─ DeckTracker.remove()      ← composición del zapato
                           └─ suggest_action_with_ev()  ← EV dinámico
```

El **CardDebouncer** evita falsos positivos: una carta debe estar presente en la zona asignada durante 1 segundo continuo antes de registrarse. La barra de progreso bajo cada bbox muestra el tiempo acumulado.

El **EV dinámico** usa `lru_cache` con distribución del zapato redondeada a 4 decimales (~1000× más rápido que simulación exacta, error < 0.001 EV).

---

## Sistemas de conteo

| Sistema | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10/J/Q/K | A |
|---------|---|---|---|---|---|---|---|---|----------|---|
| Hi-Lo | +1 | +1 | +1 | +1 | +1 | 0 | 0 | 0 | −1 | −1 |
| KO | +1 | +1 | +1 | +1 | +1 | +1 | 0 | 0 | −1 | −1 |
| Omega II | +1 | +1 | +2 | +2 | +2 | +1 | 0 | −1 | −2 | 0 |

**True count** = running count / mazos restantes en el zapato.

---

## Historial de entrenamiento

### Dataset

| Versión | Imágenes train | Notas |
|---------|----------------|-------|
| V1 | ~10 506 | Grayscale + Tiling 2×2 → incompatible con webcam color |
| V2 | ~3 741 | Sin augmentations → no se usó para entrenar |
| V3 | ~3 741 | Augmentations estándar YOLO — mAP50=0.954 |
| V4 | ~3 741 | Letterbox + augmentations agresivos para distancia |
| **V5** | **7 722** | Dataset V4 triplicado con albumentations (blur, JPEG, CLAHE) |

### Modelos

| Modelo | Epochs | Mask mAP50-95 | Notas |
|--------|--------|---------------|-------|
| V3 | 100 | 0.954 | Baseline funcional |
| V4 | 120 | 0.984 | Fine-tune desde V3 |
| **V5** | 71* | **0.980** | Fine-tune desde V4, más robusto en webcam real |

*Paró por early stopping (patience=20).

---

## Comparación YOLOv8m-seg vs RT-DETR-L

Evaluado sobre el mismo split val (644 imágenes), `conf=0.001`, `iou=0.6`:

| Métrica | YOLOv8m-seg V5 | RT-DETR-L matched | Δ |
|---------|---------------:|------------------:|--:|
| mAP50(box) | **0.9862** | 0.9804 | −0.58 pt |
| mAP50-95(box) | **0.9802** | 0.9158 | −6.44 pt |
| FPS efectivo | **16.3** | 13.4 | −21% |
| Tamaño .pt | **54.9 MB** | 66.4 MB | +12 MB |

YOLOv8m-seg supera a RT-DETR-L en calidad de localización, velocidad y coste de despliegue bajo el mismo régimen de entrenamiento. Permanece como modelo de producción.

Notebooks: [`notebooks/training/RTDETR_blackjack_match_yolo.ipynb`](notebooks/training/RTDETR_blackjack_match_yolo.ipynb) · [`notebooks/analysis/compare_models.ipynb`](notebooks/analysis/compare_models.ipynb)  
Resultados: [`docs/analysis/`](docs/analysis/)

---

## Licencia

Proyecto académico — Máster en Inteligencia Artificial.
