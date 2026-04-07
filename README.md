# Blackjack VAI — Vision Artificial Avanzada

Sistema de seguimiento de blackjack en tiempo real mediante vision artificial. Detecta cartas con YOLOv8, sigue el juego, lleva conteo de cartas y muestra probabilidades y sugerencias optimas en una interfaz web.

## Requisitos

- Python 3.10+
- Webcam (posicion cenital sobre la mesa)
- [Roboflow](https://roboflow.com) cuenta gratuita para etiquetado

```bash
pip install -r requirements.txt
```

## Estructura del proyecto

```
blackjack-VAI/
├── vision/          # Captura, deteccion YOLOv8, tracking, zonas
├── game/            # Reglas blackjack, maquina de estados, conteo
├── api/             # FastAPI + WebSockets
├── frontend/        # Interfaz web (HTML/JS)
├── dataset/         # Scripts de captura y organizacion del dataset
├── models/          # Modelos entrenados (.pt)
├── docs/plans/      # Documentos de diseno
└── config.yaml      # Configuracion global
```

---

## Fase 1 — Crear el dataset de cartas

### 1.1 Capturar imagenes con la webcam

Hay dos modos disponibles:

**Modo secuencial** (recomendado para empezar): recorre las 52 cartas en orden automaticamente.

```bash
python dataset/capture_cards.py --mode sequential
```

**Modo manual**: elige tu la carta a capturar en cada momento.

```bash
python dataset/capture_cards.py --mode manual
```

**Opciones adicionales:**

| Opcion | Descripcion | Default |
|--------|-------------|---------|
| `--camera` | Indice de camara | `0` |
| `--output` | Carpeta de salida | `dataset/raw` |
| `--target` | Capturas por carta | `50` |
| `--mode` | `sequential` o `manual` | `sequential` |

**Controles durante la captura:**

| Tecla | Accion |
|-------|--------|
| `ESPACIO` | Capturar imagen |
| `A` | Captura automatica (cada 1s) |
| `N` | Siguiente carta (modo secuencial) |
| `P` | Carta anterior (modo secuencial) |
| `Q` | Salir |

### 1.2 Revisar el dataset capturado

```bash
python dataset/stats.py
```

Muestra cuantas imagenes tienes por carta y cuales necesitan mas capturas.

### 1.3 Subir a Roboflow para etiquetar

1. Crea un proyecto en [roboflow.com](https://roboflow.com) → Object Detection
2. Sube la carpeta `dataset/raw/`
3. Etiqueta cada carta (clase: `AS`, `2H`, `KC`, `card_back`, etc.)
4. Aplica augmentations: rotacion ±15°, brillo ±25%, blur leve
5. Exporta en formato **YOLOv8** y descarga en `dataset/roboflow/`

---

## Fase 2 — Entrenar el modelo

```bash
python dataset/train.py --data dataset/roboflow/data.yaml --epochs 100
```

El modelo entrenado se guarda en `models/cards_yolov8.pt`.

---

## Fase 3 — Configurar la mesa

Edita `config.yaml` para ajustar:
- Numero de jugadores (1-7)
- Numero de mazos (1, 2, 4, 6 u 8)
- Sistema de conteo (hilo, ko, omega2)

```bash
python vision/zone_setup.py
```

Abre una ventana donde defines las zonas de cada jugador haciendo click.

---

## Fase 4 — Iniciar el sistema

```bash
python api/main.py
```

Abre el navegador en `http://localhost:8000` para ver la interfaz.

---

## Clases del dataset

53 clases en total:

| Palo | Simbolo | Ejemplo de clases |
|------|---------|-------------------|
| Spades (picas) | S | AS, 2S, 3S ... KS |
| Hearts (corazones) | H | AH, 2H, 3H ... KH |
| Diamonds (diamantes) | D | AD, 2D, 3D ... KD |
| Clubs (treboles) | C | AC, 2C, 3C ... KC |
| Boca abajo | — | card_back |

Valores: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K

---

## Sistemas de conteo soportados

| Sistema | Descripcion |
|---------|-------------|
| Hi-Lo | +1 (2-6), 0 (7-9), -1 (10-A) |
| KO (Knock-Out) | Como Hi-Lo pero 7 cuenta como +1 |
| Omega II | Sistema multinivel mas preciso |

---

## Licencia

Proyecto academico — Master en Inteligencia Artificial.
