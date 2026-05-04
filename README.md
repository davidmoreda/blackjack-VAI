# Blackjack VAI — Vision Artificial Avanzada

Sistema de seguimiento de blackjack en tiempo real mediante visión artificial. Detecta cartas con YOLOv8m-seg + ByteTrack, sigue el estado del juego, lleva conteo de cartas y muestra probabilidades, valor esperado (EV) y sugerencias óptimas en una interfaz web.

## Objetivo

Construir un sistema completo que, mediante una webcam situada sobre la mesa de blackjack, sea capaz de:

1. **Detectar y clasificar** cada carta visible (54 clases) con YOLOv8m-seg + ByteTrack
2. **Seguir el estado del juego** automáticamente (mano del jugador, mano del crupier, valor de mano)
3. **Contar cartas** con los sistemas Hi-Lo, KO y Omega II
4. **Calcular el EV dinámico** (STAND / HIT / DOUBLE) a partir de la composición real del zapato
5. **Sugerir la jugada óptima** según EV dinámico y estrategia básica como fallback
6. **Mostrar todo en una interfaz web** en tiempo real vía FastAPI + WebSockets

---

## Arquitectura del proyecto

```
blackjack-VAI/
├── api/
│   ├── main.py            # FastAPI: startup, /video_feed MJPEG, sirve frontend
│   ├── routes.py          # REST: /api/start|stand|reset|undo|remove|state|config
│   └── ws.py              # WebSocket /ws: vision thread, debounce, broadcast
├── frontend/
│   ├── index.html         # UI casino — panel lateral + feed de cámara
│   └── panel.js           # WebSocket client, scoreboard, EV, historial
├── game/
│   ├── engine.py          # hand_total, is_bust, is_blackjack, dealer_should_hit
│   ├── counter.py         # CardCounter: Hi-Lo / KO / Omega II
│   ├── deck_tracker.py    # DeckTracker: composición exacta del zapato
│   ├── ev_calculator.py   # EV dinámico con lru_cache (STAND/HIT/DOUBLE)
│   ├── strategy.py        # Basic strategy + suggest_action_with_ev
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

## Instalación y uso

### Opción A — Sin Docker (desarrollo local)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Arrancar el servidor
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Abre http://localhost:8000
```

### Opción B — Modo terminal (sin servidor)

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
# 1. Construir imagen
docker compose build

# 2. Levantar en segundo plano
docker compose up -d

# 3. Abrir en el navegador
#    http://localhost:8000

# Ver logs
docker compose logs -f

# Parar
docker compose down
```

Si no tienes GPU o NVIDIA Container Toolkit, comenta el bloque `deploy` en `docker-compose.yml`.

### Opción D — WSL + Docker + cámara de Windows (recomendado)

WSL2 no accede directamente a la webcam USB del host. La solución es un servidor de streaming en Windows que Docker consume por HTTP.

**Paso 1 — Lanzar el servidor de cámara en Windows** (PowerShell/CMD, no WSL)

```powershell
pip install flask opencv-python
python cam_server_windows.py
```

Salida esperada:
```
Cámara 1 abierta: 1280x720
Stream disponible en: http://0.0.0.0:5050/video
```

**Paso 2 — Configurar `config.yaml`**

```yaml
camera:
  index: "http://host.docker.internal:5050/video"
```

`host.docker.internal` es el nombre DNS que Docker resuelve automáticamente al host Windows. Está declarado en `docker-compose.yml` vía `extra_hosts`.

**Paso 3 — Levantar Docker desde WSL**

```bash
docker compose up --build
# Abre http://localhost:8000
```

**Verificar conectividad cámara desde el contenedor**

```bash
docker compose exec blackjack-vai python -c \
  "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:5050/health').read())"
# Respuesta esperada: b'{"cam":1,"height":720,"status":"ok","width":1280}'
```

---

## API REST

Base URL: `http://localhost:8000`

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Interfaz web (index.html) |
| `GET` | `/video_feed` | Stream MJPEG con bboxes dibujados |
| `WebSocket` | `/ws` | Estado del juego en tiempo real |
| `POST` | `/api/start` | Nueva partida (resetea todo) |
| `POST` | `/api/reset` | Nueva ronda (conserva el zapato) |
| `POST` | `/api/stand` | Jugador se planta (`{"player_id": "player_1"}`) |
| `POST` | `/api/undo` | Deshace la última carta detectada |
| `POST` | `/api/remove/{idx}` | Elimina carta por índice del historial |
| `GET` | `/api/state` | Estado completo del juego |
| `GET` | `/api/config` | Configuración actual |
| `POST` | `/api/config` | Reconfigura jugadores, mazos y sistema de conteo |

### Configuración vía API

```json
POST /api/config
{
  "num_players": 1,
  "num_decks": 6,
  "counting_system": "hilo"
}
```

- `num_players`: 1–7
- `num_decks`: 1, 2, 4, 6 u 8
- `counting_system`: `"hilo"`, `"ko"` o `"omega2"`

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

## Licencia

Proyecto académico — Máster en Inteligencia Artificial.
