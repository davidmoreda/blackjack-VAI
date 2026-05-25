# Estructura Presentación — BlackjackVAI
> Puntos 1 y 2: Explicación del juego/proyecto + Dataset

---

## BLOQUE 1 — EXPLICACIÓN DEL JUEGO Y PROYECTO

---

### Diapositiva 1 — Portada

**Título:** BlackjackVAI
**Subtítulo:** Sistema de visión por computador para detección y análisis en tiempo real de partidas de blackjack
**Elementos visuales:**
- Fondo estilo tapete de casino (verde oscuro #0b1a0e, detalles dorados)
- Logo o icono de cartas sobre mesa
- Nombres del equipo + asignatura

**Notas del ponente:**
> Presentamos BlackjackVAI, un sistema completo de visión artificial que detecta cartas en una mesa de blackjack en tiempo real usando una webcam cenital y sugiere la jugada óptima calculando el Valor Esperado sobre la composición real del zapato.

---

### Diapositiva 2 — ¿Qué es el Blackjack?

**Título:** El juego: Blackjack

**Contenido (columna izquierda):**
- Objetivo: acercarse a 21 sin pasarse, superando al croupier
- Carta con valor numérico (2–10), figuras valen 10, As vale 1 u 11
- Acciones: **HIT** (pedir carta) · **STAND** (plantarse) · **DOUBLE** (doblar apuesta) · **SPLIT** (separar pareja)
- El croupier sigue reglas fijas: pide carta hasta llegar a 17+

**Contenido (columna derecha — diagrama de flujo simplificado):**
```
Reparto inicial (2 cartas cada jugador)
       ↓
¿Jugador ≤ 21?  →  NO → Bust (pierde)
       ↓ SÍ
Turno del croupier
       ↓
Comparación → Gana quien más se acerque a 21
```

**Elementos visuales:**
- Ilustración/foto de una mano de blackjack con cartas visibles
- Tabla de valores de cartas

**Notas del ponente:**
> La detección debe ser precisa: un error de clasificación cambia completamente el consejo de jugada.

---

### Diapositiva 3 — Pipeline de visión artificial

**Título:** De cámara a jugada en tiempo real

**Diseño: 4 bloques grandes en horizontal con flechas entre ellos**

```
[Cámara]  →  [Detección]  →  [Tracking]  →  [Decisión]
```

**Bloque 1 — Cámara**
- Webcam cenital sobre la mesa
- Captura continua del estado de la partida

**Bloque 2 — Detección (YOLOv8m-seg)**
- Localiza cada carta en el frame
- Clasifica entre 54 clases posibles
- Genera una máscara de segmentación por carta

**Bloque 3 — Tracking (ByteTrack)**
- Mantiene la identidad de cada carta entre frames
- Evita registrar la misma carta múltiples veces

**Bloque 4 — Decisión**
- Asigna cada carta a su zona (dealer / jugador)
- Calcula el Valor Esperado de cada acción posible
- Recomienda HIT / STAND / DOUBLE

**Elementos visuales:**
- `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` — detecciones reales con máscaras y clases

**Notas del ponente:**
> Todo el pipeline corre a 15 fps. El tracking es lo que permite que mover una carta no la registre dos veces. La decisión final no es una tabla fija — se recalcula en cada frame con las cartas que quedan en el zapato.

---

## BLOQUE 2 — DATASET

---

### Diapositiva 6 — El Problema del Dataset

**Título:** El reto: conseguir datos de calidad

**Contenido:**
- **54 clases** → necesitamos ejemplos de cada carta en múltiples condiciones
- Variabilidad real: ángulo cenital, luz de habitación, cartas parcialmente tapadas, movimiento
- Opciones consideradas:
  - ❌ Fotografiar manualmente las 54 clases → tedioso, poco escalable
  - ❌ Datasets genéricos de cartas → no adaptados a vista cenital
  - ✅ **Roboflow** → plataforma de datasets de CV con anotaciones ya hechas
  - ✅ **SAM (Segment Anything Model)** → explorado para segmentación automática

**Elementos visuales:**
- Capturas de ejemplo del dataset (val_batch0_labels.jpg de yolov8m_seg_v5)
- `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg`

**Notas del ponente:**
> El problema principal no fue encontrar imágenes, sino encontrar imágenes con las anotaciones correctas para 54 clases y en condiciones similares a nuestro caso de uso real.

---

### Diapositiva 7 — Roboflow: Fuente Principal del Dataset

**Título:** Dataset base: Roboflow

**Contenido (izquierda):**
- Plataforma online de datasets de Computer Vision
- Dataset de cartas de póker/blackjack con **54 clases anotadas**:
  - As, 2–10, J, Q, K × 4 palos (♠♥♦♣)
  - `card_back` (reverso)
  - `joker`
- Anotaciones en formato **YOLO** (bounding boxes)
- Versión base: ~3.741 imágenes de entrenamiento

**Contenido (derecha — tabla evolución):**
| Versión | Imágenes | Estado |
|---------|----------|--------|
| V1 | ~10.506 | ❌ Descartada (escala de grises + tiling) |
| V2 | ~3.741 | ❌ Sin augmentations |
| V3 | ~3.741 | ✅ Baseline funcional |
| V4 | ~3.741 | ✅ Augmentations agresivas |
| **V5** | **~7.722** | ✅ **Producción** (triplicado con Albumentations) |

**Elementos visuales:**
- Logo de Roboflow
- Ejemplo de anotación con bounding box sobre carta
- `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg`

**Notas del ponente:**
> La V1 tenía 10k imágenes pero era en escala de grises con técnica de tiling, incompatible con nuestro caso de uso (cámara en color). Partimos de cero con V2/V3 usando el mismo dataset pero procesado correctamente.

---

### Diapositiva 8 — Por qué descartamos V1 y V2

**Título:** Lecciones aprendidas: V1 y V2

**Diseño: 2 tarjetas de "problema → solución"**

**Tarjeta 1 — V1 (Descartada)**
- **Problema:** Dataset en escala de grises + técnica de *image tiling* (partir imágenes grandes en tiles)
- **Consecuencia:** El modelo entrenado no funcionaba con la webcam en color
- **Lección:** El dominio de entrenamiento debe coincidir con el de inferencia

**Tarjeta 2 — V2 (No entrenada)**
- **Problema:** Dataset sin augmentations de ningún tipo
- **Consecuencia:** Se esperaba overfitting y poca robustez a variaciones de luz/posición
- **Decisión:** Saltamos directamente a V3 con augmentations estándar

**Elementos visuales:**
- Icono ❌ en V1 y V2
- Flecha hacia V3 ✅

**Notas del ponente:**
> Estas decisiones nos ahorraron tiempo de entrenamiento. Identificar rápidamente que V1 era incompatible con el uso real fue un aprendizaje clave del proyecto.

---

### Diapositiva 9 — SAM: Exploración de Segmentación Alternativa

**Título:** Exploración adicional: Segment Anything Model (SAM)

**Contenido:**
- **SAM** (Meta AI, 2023): modelo de segmentación zero-shot capaz de segmentar cualquier objeto
- Explorado como alternativa para generar **máscaras de segmentación automáticas** sobre cartas
- Ventaja potencial: no necesitar anotaciones manuales → pipeline semi-automático de etiquetado

**Resultado:**
| Aspecto | SAM | Roboflow |
|---------|-----|---------|
| Anotación | Semi-automática (prompts) | Manual/pre-anotada |
| Calidad máscaras | Alta (a nivel píxel) | Bounding boxes |
| Clases | Sin etiquetas de clase | 54 clases etiquetadas |
| Integración YOLO | Requiere post-procesado | Nativa |
| **Decisión** | ❌ No usado en producción | ✅ Usado |

**Por qué no se usó:**
- SAM segmenta bien pero **no clasifica** → hay que añadir la clase a mano igualmente
- El dataset de Roboflow ya tenía las 54 clases etiquetadas correctamente
- No aportaba ventaja suficiente para justificar el pipeline adicional

**Elementos visuales:**
- Logo de Meta/SAM
- Ejemplo visual de segmentación SAM (máscara colorizada sobre carta)
- `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` (referencia de cómo queda la segmentación con YOLO)

**Notas del ponente:**
> SAM es una herramienta muy potente pero en nuestro caso no añadía valor porque el problema principal era la clasificación de las 54 clases, no la segmentación en sí. Roboflow lo resolvía más directamente.

---

### Diapositiva 10 — El Dataset Final: Evolución y Estructura

**Título:** Dataset final: de 3.741 a 7.722 imágenes

**Diseño: timeline horizontal con 3 hitos**

**Hito 1 — V3 (Baseline)**
- 3.741 imágenes de entrenamiento
- Augmentations estándar YOLO: rotaciones, flips, brillo, contraste
- mAP50: **0.954**

**Hito 2 — V4 (Augmentations agresivas)**
- 3.741 imágenes (mismo dataset)
- Letterbox preprocessing + augmentations agresivas para robustez a distancia
- Fine-tune desde V3
- mAP50: **0.984**

**Hito 3 — V5 (Producción)**
- **7.722 imágenes** (V4 triplicado con Albumentations)
- Técnicas de Albumentations aplicadas:
  - **Gaussian Blur** → simula cartas fuera de foco
  - **JPEG Compression** → simula artefactos de cámara
  - **CLAHE** → simula variaciones de iluminación real
- Fine-tune desde V4
- mAP50: **0.980** · mAP50-95: **0.980**

**Elementos visuales:**
- `docs/presentation_assets/01_evolucion_modelos.png` — gráfica de evolución de modelos
- Iconos de cada técnica de augmentation

**Notas del ponente:**
> La V5 no subió el mAP50 respecto a V4 (0.980 vs 0.984) pero es significativamente más robusta en condiciones reales. El mAP en validación es solo una parte de la historia; la robustez a webcam real es lo que importa en producción.

---

### Diapositiva 11 — Visualización del Dataset

**Título:** ¿Cómo son los datos?

**Diseño: mosaico de imágenes 2×2 o 3×2**

**Imágenes a usar:**
1. `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` — batch de validación con anotaciones ground truth
2. `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` — mismas imágenes con predicciones del modelo
3. `docs/analysis/val/yolov8m_seg_v5/val_batch1_labels.jpg` — segunda muestra de validación
4. `docs/analysis/qualitative_grid.png` — grid cualitativo de comparación de modelos

**Pie de imagen:**
- "Ground truth" (anotaciones reales) vs "Predicciones del modelo"
- Destacar que las máscaras de segmentación permiten identificar la carta incluso parcialmente tapada

**Notas del ponente:**
> Podemos ver que el dataset incluye cartas en distintas posiciones, solapadas, con diferentes iluminaciones. Esto es lo que hace que el modelo sea robusto a condiciones reales.

---

*— Fin de los Bloques 1 y 2 —*

---

## ÍNDICE DE RECURSOS GRÁFICOS DISPONIBLES

### Para el Bloque 1 (Juego y Proyecto)
| Slide | Recurso sugerido | Ruta |
|-------|-----------------|------|
| 5 (Arquitectura) | — | Diagrama a crear |

### Para el Bloque 2 (Dataset)
| Slide | Recurso sugerido | Ruta |
|-------|-----------------|------|
| 6 | Batch con labels GT | `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` |
| 7 | Batch con labels GT | `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` |
| 9 | Predicciones modelo | `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` |
| 10 | Evolución modelos | `docs/presentation_assets/01_evolucion_modelos.png` |
| 11 | Grid cualitativo | `docs/analysis/qualitative_grid.png` |
| 11 | Batch labels | `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` |
| 11 | Batch preds | `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` |
