# Blackjack VAI — Vision Artificial Avanzada

Sistema de seguimiento de blackjack en tiempo real mediante visión artificial. Detecta cartas con YOLOv8, sigue el juego, lleva conteo de cartas y muestra probabilidades y sugerencias óptimas en una interfaz web.

## Objetivo

Construir un sistema completo que, mediante una webcam situada sobre la mesa de blackjack, sea capaz de:

1. **Detectar y clasificar** cada carta visible (52 clases + reverso) con YOLOv8 segmentación
2. **Seguir el estado del juego** automáticamente (mano del jugador, mano del crupier, valor de mano)
3. **Contar cartas** con los sistemas Hi-Lo, KO y Omega II
4. **Sugerir la jugada óptima** (pedir, plantarse, doblar, dividir) según estrategia básica y el conteo
5. **Mostrar todo en una interfaz web** en tiempo real vía FastAPI + WebSockets

---

## Arquitectura del proyecto

```
blackjack-VAI/
├── YOLO_blackjack_v3.ipynb   # Notebook de entrenamiento local (RTX 4070+)
├── vision/                    # Captura, detección YOLOv8, tracking, zonas
├── game/                      # Reglas blackjack, máquina de estados, conteo
├── api/                       # FastAPI + WebSockets
├── frontend/                  # Interfaz web (HTML/JS)
├── models/                    # Modelos entrenados (.pt)
│   └── best.pt                # Enlace al mejor modelo entrenado
├── training_runs/             # Checkpoints y logs del entrenamiento
├── docs/plans/                # Documentos de diseño
└── config.yaml                # Configuración global
```

---

## Estado actual — Fase de entrenamiento

### Historial del dataset

El dataset se creó y etiquetó en **Roboflow** (proyecto `BlackjackVAI`, workspace `javiers-workspace-q8mnr`).

| Versión | Imágenes | Preprocesado | Augmentations | Resultado |
|---------|----------|--------------|---------------|-----------|
| **V1** | ~10 506 | Auto-Orient, Tiling 2×2, **Grayscale**, Contrast adaptativo | Múltiples (Roboflow) | ❌ No detecta en webcam — grayscale vs color en tiempo real |
| **V2** | ~3 741 | Auto-Orient, Resize 640×640 | Ninguna | ❌ No se usó para entrenar |
| **V3** | ~3 741 | Auto-Orient, Resize 640×640 | **Ninguna** (las hace YOLO) | ✅ Versión correcta |

**Errores detectados en V1 que causaban 0 detecciones en tiempo real:**
1. **Grayscale en preprocesado** → el modelo aprendió imágenes en gris pero la webcam envía color
2. **Tiling 2×2** → el modelo vio recortes de cartas, no frames completos
3. **imgsz mismatch** → entrenado a 960px, inferencia a 640px
4. **Bug de ruta** → el notebook descargaba V2 pero usaba la ruta de V1

### V3 — configuración correcta (Roboflow)

Al generar la versión V3 en Roboflow:
- **Preprocessing:** Auto-Orient + Resize 640×640 (Stretch) — solo esto
- **Augmentations:** ninguna — YOLO aplica sus propias en tiempo de entrenamiento
- **Split:** 80 % train / 10 % val / 10 % test (estratificado automático)

### Entrenamiento — modelo V1 (Colab, referencia)

Entrenado en Google Colab con H100 sobre el dataset V1 (incorrecto):
- Modelo: `yolov8m-seg`
- Epochs: 100 | imgsz: 960 | batch: 8
- **Val mAP50 (box): 0.878** — buen resultado en validación (también en gris)
- **Fallo en tiempo real** por los errores de dataset descritos arriba

### Entrenamiento — V3 (local, en curso)

Notebook: `YOLO_blackjack_v3.ipynb`

| Parámetro | Valor |
|-----------|-------|
| GPU | RTX 4070 Laptop (8 GB VRAM) |
| Modelo base | `yolov8m-seg.pt` |
| Epochs | 100 (patience 20) |
| imgsz | 640 |
| Batch | 8 |
| Tracking | MLflow (`mlflow ui`) |
| Checkpoints | Cada 10 epochs + best/last siempre |
| Resume | Automático si se interrumpe |

---

## Instalación y uso

### Opción A — Docker (recomendado)

La forma más sencilla de levantar el sistema. No hay que instalar nada en el host salvo Docker, y la GPU y la webcam se pasan directamente al contenedor.

#### Requisitos previos

| Requisito | Para qué |
|-----------|----------|
| [Docker Engine](https://docs.docker.com/engine/install/) + [Docker Compose v2](https://docs.docker.com/compose/install/) | Construir y levantar el contenedor |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Dar acceso a la GPU desde el contenedor |
| Webcam conectada como `/dev/video0` | Captura de video |
| `models/best.pt` entrenado | Inferencia YOLO |

#### Cómo está construido el Docker

**`Dockerfile`** — etapa única con tres bloques:

1. **Base `python:3.11-slim`** — imagen Debian mínima con Python.  
   No se usa la imagen oficial de PyTorch (`pytorch/pytorch`) porque pesa ~7 GB; tampoco Alpine porque OpenCV necesita glibc.

2. **Dependencias de sistema** (`libgl1`, `libglib2.0-0`, etc.) — OpenCV necesita estas librerías en tiempo de ejecución para codificar/decodificar imágenes, aunque no haya pantalla.

3. **PyTorch con CUDA 12.1** — se instala desde el índice oficial de PyTorch antes que el resto de `requirements.txt` para que pip resuelva las dependencias correctamente. Si el host no tiene GPU, PyTorch cae a CPU automáticamente sin errores.

**`docker-compose.yml`** — orquesta el contenedor con:
- `devices: /dev/video0` → pasa la webcam al contenedor
- `deploy.resources.reservations` → pasa todas las GPUs NVIDIA
- `volumes: ./models` y `./config.yaml` montados como lectura → el modelo y la configuración se actualizan sin reconstruir la imagen

**`.dockerignore`** — excluye el dataset (~cientos de MB), los runs de entrenamiento y los `.pt` del contexto de build. El modelo se monta como volumen, no se copia a la imagen.

#### Levantar

```bash
# 1. Construir la imagen (solo la primera vez o tras cambiar código)
docker compose build

# 2. Levantar el contenedor en segundo plano
docker compose up -d

# 3. Abrir en el navegador
#    http://localhost:8000

# Ver logs en tiempo real
docker compose logs -f

# Parar
docker compose down
```

#### Si no tienes GPU o no tienes NVIDIA Container Toolkit

Comenta el bloque `deploy` en `docker-compose.yml`:

```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

El modelo correrá en CPU — más lento pero funcional.

#### Si tu webcam no es `/dev/video0`

```bash
# Listar cámaras disponibles
ls /dev/video*
```

Edita `docker-compose.yml` y cambia la línea de `devices` al índice correcto.  
También actualiza `config.yaml` → `camera.index`.

---

### Opción B — Sin Docker (desarrollo local)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Arrancar el servidor
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# Abre http://localhost:8000
```

---

### Opción C — WSL + cámara de Windows (`cam_server_windows.py`)

WSL2 no accede directamente a la webcam USB del host. La solución es levantar un servidor de streaming en Windows y que WSL consuma el vídeo por HTTP.

#### Paso 1 — Instalar dependencias en Windows (Python nativo, **no** WSL)

```cmd
pip install flask opencv-python
```

#### Paso 2 — Lanzar el servidor de cámara en Windows

Abre un terminal de Windows (CMD o PowerShell, **no** WSL):

```cmd
python cam_server_windows.py
```

Verás algo como:
```
Cámara 0 abierta: 1280x720
Stream disponible en: http://0.0.0.0:5050/video
Desde WSL usa:        http://10.255.255.254:5050/video
```

> Si tu webcam no es la cámara 0, edita `CAM_INDEX = 1` (u otro índice) en `cam_server_windows.py`.

#### Paso 3 — Configurar WSL para usar la cámara de Windows

Edita `config.yaml` y asegúrate de que la URL apunta al servidor de Windows:

```yaml
camera:
  index: 'http://10.255.255.254:5050/video'
```

La IP `10.255.255.254` es la IP del host Windows vista desde WSL2 (añadida automáticamente por WSL al fichero `/etc/hosts` como `host.docker.internal` o equivalente). Si no funciona, localiza la IP real con:

```bash
# Desde WSL:
cat /etc/resolv.conf | grep nameserver
```

#### Paso 4 — Arrancar el sistema desde WSL

```bash
# Instalar dependencias (solo la primera vez)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Arrancar la API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# Abre en el navegador: http://localhost:8000
```

#### Verificar que el stream funciona antes de arrancar

```bash
curl -I http://10.255.255.254:5050/health
# Debe devolver: {"status": "ok", "cam": 0, ...}
```

---

### Entrenar el modelo

#### Requisitos previos

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install mlflow
```

#### Opción 1 — Entrenar desde Windows (Jupyter local)

1. Instala Python 3.10+ y las dependencias anteriores en Windows.
2. Abre `YOLO_blackjack_v3.ipynb` con Jupyter (o VS Code):
   ```cmd
   jupyter notebook YOLO_blackjack_v3.ipynb
   ```
3. En la celda de configuración ajusta:
   ```python
   ROBOFLOW_API_KEY = "tu_api_key"
   ROBOFLOW_VERSION = 3
   ```
4. Ejecuta todas las celdas. El entreno puede durar varias horas.

#### Opción 2 — Entrenar desde WSL (recomendado si tienes GPU con drivers CUDA en Windows)

WSL hereda la GPU de Windows a través de los drivers de CUDA para WSL2.

```bash
# Desde WSL, en la carpeta del proyecto:
jupyter notebook YOLO_blackjack_v3.ipynb
# o ejecutar el script directamente:
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
mlflow.set_experiment('blackjack-yolo')
# el resto del entreno está en el notebook
"
```

#### Seguimiento con MLflow

El notebook loguea automáticamente todos los experimentos. Para ver los resultados:

```bash
# En WSL o Windows (donde hayas entrenado):
mlflow ui --port 5000 --backend-store-uri ./mlruns
# Abre en el navegador: http://localhost:5000
```

Métricas registradas por epoch:
- `box_loss`, `seg_loss`, `cls_loss`
- `mAP50`, `mAP50-95`
- Hiperparámetros: modelo, epochs, imgsz, batch, augmentations
- Artefactos: `best.pt`, curvas de entrenamiento, confusion matrix

#### Reanudar un entreno interrumpido

```python
RESUME = True  # ya es el valor por defecto en el notebook
```

Simplemente vuelve a ejecutar todas las celdas — detectará `last.pt` automáticamente.

### Estructura de carpetas tras el entrenamiento

```
training_runs/runs/yolo8m_seg_v3/
├── weights/
│   ├── best.pt          # mejor checkpoint por mAP
│   ├── last.pt          # último epoch (para resume)
│   └── epoch10.pt       # checkpoints intermedios
├── results.csv          # métricas por epoch
└── *.png                # curvas y confusion matrix

models/
├── best_v3_YYYYMMDD_HHMM.pt   # copia fechada del best
└── best.pt                     # enlace simbólico al último best
```

---

## Fases del proyecto

| Fase | Estado | Descripción |
|------|--------|-------------|
| 1. Dataset | ✅ | Creación, etiquetado y limpieza en Roboflow (V3) |
| 2. Entrenamiento | 🔄 En curso | YOLOv8m-seg local con RTX 4070 + MLflow |
| 3. API | ⏳ | FastAPI + WebSockets para streaming de detecciones |
| 4. Lógica de juego | ⏳ | Máquina de estados, conteo de cartas, estrategia básica |
| 5. Frontend | ⏳ | Interfaz web en tiempo real |
| 6. Integración | ⏳ | Sistema completo webcam → detección → sugerencia |

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

## Sistemas de conteo soportados

| Sistema | Descripción |
|---------|-------------|
| Hi-Lo | +1 (2-6), 0 (7-9), -1 (10-A) |
| KO (Knock-Out) | Como Hi-Lo pero 7 cuenta como +1 |
| Omega II | Sistema multinivel más preciso |

---

## Tracking con MLflow

El entrenamiento loggea automáticamente:
- **Hiperparámetros**: modelo, epochs, imgsz, batch, augmentations
- **Métricas por epoch**: box_loss, seg_loss, mAP50, mAP50-95
- **Artefactos**: best.pt, curvas de entrenamiento, confusion matrix, data.yaml

```bash
mlflow ui --port 5000
```

---

## Licencia

Proyecto académico — Máster en Inteligencia Artificial.
