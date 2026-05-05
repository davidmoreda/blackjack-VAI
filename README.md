# Blackjack VAI

Sistema de seguimiento de blackjack en tiempo real mediante visión artificial. Detecta y clasifica cartas con YOLOv8m-seg + ByteTrack, gestiona el estado de la partida y calcula probabilidades, EV dinámico y sugerencias óptimas en una interfaz web.

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
│   ├── detector.py        # CardDetector: YOLO + ByteTrack → Detection[]
│   ├── debouncer.py       # CardDebouncer: confirma carta tras 1s de presencia
│   └── zones.py           # ZoneManager: asigna dealer / player_N por coordenadas
├── models/
│   └── best.pt            # Modelo entrenado (Git LFS)
├── cli.py                 # Modo terminal: ventana OpenCV sin servidor web
├── cam_server_windows.py  # Servidor MJPEG para exponer webcam de Windows a WSL/Docker
├── config.yaml            # Configuración global
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Fases del proyecto

| Fase | Estado | Descripción |
|------|--------|-------------|
| Dataset | ✅ | Creación, etiquetado y limpieza en Roboflow (V3 → V4) |
| Entrenamiento | ✅ | YOLOv8m-seg con RTX 4070 Laptop + MLflow — V3, V4, V5 completados |
| API | ✅ | FastAPI + WebSockets + endpoints REST |
| Lógica de juego | ✅ | Máquina de estados, conteo, EV dinámico, estrategia básica |
| Frontend | ✅ | Interfaz web en tiempo real con panel lateral |
| Integración | ✅ | Sistema completo: webcam → detección → sugerencia |

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
# 1. Crear y activar entorno virtual (si no lo tienes ya)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lapx>=0.5.2
pip install -r requirements.txt
pip install jupyter

# 3. Abrir el notebook correspondiente
jupyter notebook YOLO_blackjack_v5.ipynb
```

### Seguimiento con MLflow

La base de datos MLflow está en el propio repositorio (`mlflow.db`, excluida de git).

```bash
# Desde la raíz del proyecto
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001
# http://localhost:5001
```

---

## Instalación y uso

### 1. Requisitos previos del sistema

Instala las siguientes herramientas si aún no las tienes:

**Git y Git LFS** (necesario para descargar el modelo `best.pt`):
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git git-lfs -y
git lfs install
```

**Docker y Docker Compose** (para ejecutar la app en contenedor):
```bash
# Ubuntu/Debian
sudo apt install docker.io docker-compose-plugin -y
sudo usermod -aG docker $USER   # permite usar docker sin sudo (reinicia sesión)
```
> En Windows instala [Docker Desktop](https://www.docker.com/products/docker-desktop/).

**Python 3.10+** (solo necesario para las opciones sin Docker):
```bash
sudo apt install python3 python3-pip python3-venv -y
```

**NVIDIA Container Toolkit** (opcional — solo si tienes GPU NVIDIA y quieres usarla en Docker):
```bash
# Guía oficial: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
sudo apt install nvidia-container-toolkit -y
sudo systemctl restart docker
```

---

### 2. Clonar el repositorio

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
pip install flask opencv-python
python cam_server_windows.py
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

## Sistemas de conteo

| Sistema | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10/J/Q/K | A |
|---------|---|---|---|---|---|---|---|---|----------|---|
| Hi-Lo | +1 | +1 | +1 | +1 | +1 | 0 | 0 | 0 | −1 | −1 |
| KO | +1 | +1 | +1 | +1 | +1 | +1 | 0 | 0 | −1 | −1 |
| Omega II | +1 | +1 | +2 | +2 | +2 | +1 | 0 | −1 | −2 | 0 |

**True count** = running count / mazos restantes en el zapato.

---

## Licencia

Proyecto académico — Máster en Inteligencia Artificial.
