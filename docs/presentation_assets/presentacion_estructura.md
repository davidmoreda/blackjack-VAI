# Estructura Presentación — BlackjackVAI

---

## BLOQUE 1 — EXPLICACIÓN DEL JUEGO Y PROYECTO

---

### Diapositiva 1 — Portada

**Título:** BlackjackVAI
**Subtítulo:** Sistema de visión artificial para blackjack en tiempo real
**Autores:** Carlos Díaz · David Moreda · Javier Cerón
**Contexto:** Máster en IA — Visión Artificial Avanzada — 2026
**Elementos visuales:**
- Fondo estilo tapete de casino (verde oscuro #0b1a0e, detalles dorados)
- Logo o icono de cartas sobre mesa

**Notas del ponente:**
> Presentamos BlackjackVAI, un sistema completo de visión artificial que detecta cartas en una mesa de blackjack en tiempo real usando una webcam cenital y sugiere la jugada óptima calculando el Valor Esperado sobre la composición real del zapato.

---

### Diapositiva 2 — El proyecto

**Título:** El proyecto

**Contenido — qué hace el sistema (bullet list):**
- Reconoce **54 clases** (52 cartas + dorso + Joker)
- Sigue turnos automáticamente (jugador → dealer → fin de ronda)
- Lleva el marcador (rondas ganadas / perdidas / empates)
- Sugiere la **jugada óptima** en cada turno mediante Valor Esperado (EV)

**Contenido — qué es el Blackjack (columna o bloque secundario):**
- Objetivo: acercarse a 21 sin pasarse, superando al croupier
- Acciones del jugador: **HIT** · **STAND** · **DOUBLE** · **SPLIT**
- El dealer sigue reglas fijas: pide carta hasta llegar a 17+
- Un error de clasificación cambia completamente el consejo de jugada → la precisión del modelo es crítica

**Foto:** `docs/presentation_assets/Foto_cartas.png`

**Notas del ponente:**
> Esta slide ancla el proyecto: sabemos qué hace el sistema y por qué la precisión importa. Un 10 de corazones clasificado como J cambia el total de la mano y por tanto la recomendación.

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

### Diapositiva 4 — El dataset: visión general

**Título:** El dataset: punto de partida

**Contenido (columna izquierda — qué tenemos):**
- Fuente: **Roboflow** — plataforma de datasets de CV con anotaciones ya hechas
- **54 clases**: As, 2–10, J, Q, K × 4 palos + reverso + joker
- Anotaciones en formato YOLO (bounding boxes)
- Split estándar: train / valid / test

**Contenido (columna derecha — los problemas generales):**
- Las imágenes son de **estudio controlado**, no de mesa real con webcam
- Iluminación uniforme, fondo neutro → el modelo no generaliza bien a condiciones reales
- Solo bounding boxes → sin información de forma (las cartas se solapan)
- Necesitamos iterar el dataset para acercarlo a nuestro caso de uso real

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg`

**Notas del ponente:**
> El dataset base existe y está bien anotado, pero hay un gap importante entre las condiciones del dataset y las condiciones reales de una partida con webcam en una habitación. Todo el proceso de evolución del dataset busca cerrar ese gap.

---

### Diapositiva 5 — Intento previo: SAM para segmentación automática

**Título:** Antes de Roboflow: probamos SAM

**Contenido (columna izquierda — qué es y qué hicimos):**
- **SAM** (Segment Anything Model, Meta AI): segmenta cualquier objeto dado un punto o bounding box como prompt, sin entrenamiento específico
- Idea: usarlo para generar **máscaras de segmentación automáticas** sobre un dataset de cartas de Kaggle (2.757 imágenes con anotaciones Pascal VOC)
- Pipeline implementado:
  1. Parsear bounding boxes del XML de Kaggle
  2. Usar el **centro de cada bbox como punto prompt** para SAM
  3. SAM genera 3 máscaras candidatas → elegimos la de **mayor área** (la carta completa)
  4. Convertir a formato COCO JSON → subir a Roboflow
- Resultado: **2.757 imágenes procesadas, 0 errores**, ZIP listo para Roboflow

**Contenido (columna derecha — por qué no lo usamos):**
- SAM segmenta bien, **pero no clasifica** → las etiquetas de clase hay que añadirlas igualmente
- El dataset de Kaggle tiene clases en formato distinto (`AS`, `10C`) que hay que mapear manualmente
- El dataset de Roboflow ya tenía las 54 clases correctamente etiquetadas y en formato YOLO nativo
- Coste de integración > beneficio obtenido

**Foto:** screenshot del notebook mostrando el overlay de máscara SAM sobre una carta (punto prompt en rojo + máscara cyan)

**Notas del ponente:**
> Fue un experimento valioso: aprendimos cómo funciona SAM con point prompts y construimos un pipeline completo de anotación semi-automática. Pero al final Roboflow nos daba lo mismo con menos fricción y ya con las clases correctas.

---

### Diapositiva 6 — Dataset V1: primera toma de contacto

**Título:** V1 — Primera versión: ~10.500 imágenes

**Lo que trae:**
- Dataset grande (~10.506 imágenes) descargado directamente de Roboflow
- Muchas imágenes → aparentemente un buen punto de partida

**El problema que genera:**
- Las imágenes están en **escala de grises** y con técnica de *image tiling* (imágenes grandes partidas en tiles)
- El modelo entrenado **no funciona con nuestra webcam en color**
- Dominio de entrenamiento ≠ dominio de inferencia → fallo total

**Foto:** ejemplo de imagen del dataset V1 (escala de grises, aspecto de tile)

**Notas del ponente:**
> Este es el error clásico de no revisar el dataset antes de entrenar. 10k imágenes inútiles porque el dominio no coincide. Nos costó un ciclo de entrenamiento completo descubrirlo.

---

### Diapositiva 7 — Dataset V2: reset y vuelta a empezar

**Título:** V2 — Reset: ~3.700 imágenes en color

**Lo que trae:**
- Nuevo dataset limpio desde Roboflow: **~3.741 imágenes en color**
- Imágenes completas (sin tiling), condiciones más variadas
- Compatible con webcam en color

**El problema que genera:**
- **Sin augmentations de ningún tipo**
- El modelo vería siempre las cartas en las mismas condiciones → overfitting esperado
- Poca robustez a variaciones de luz, rotación o distancia
- Decisión: no entrenar con V2, pasar directamente a V3 con augmentations

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` *(dataset en color, estructura limpia)*

**Notas del ponente:**
> V2 es el dataset correcto pero sin tratar. Sabíamos que sin augmentations el modelo iba a memorizar en lugar de generalizar, así que no perdimos tiempo entrenando.

---

### Diapositiva 8 — Dataset V3: baseline funcional

**Título:** V3 — Primer modelo funcional: augmentations estándar

**Lo que trae:**
- Mismas ~3.741 imágenes que V2
- **Augmentations estándar de YOLO**: rotaciones, flips, cambios de brillo y contraste
- Primer modelo entrenado que realmente funciona → **mAP50: 0.954**

**El problema que genera:**
- El modelo funciona bien en validación, pero **falla con cartas a distancia** (webcam cenital real)
- Las augmentations estándar no cubren variaciones de escala extremas ni desenfoque
- La cámara real produce imágenes con menos contraste y algo de blur que el dataset no contempla

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch1_labels.jpg`

**Notas del ponente:**
> mAP50 de 0.954 es un resultado sólido en papel, pero al probarlo en la mesa real los errores eran evidentes con cartas alejadas o bajo iluminación de habitación. Hay que acercar el dataset a las condiciones reales.

---

### Diapositiva 9 — Dataset V4: augmentations agresivas

**Título:** V4 — Augmentations agresivas para el caso real

**Lo que trae:**
- Mismo dataset base (~3.741 imágenes)
- **Augmentations agresivas** orientadas a distancia y condiciones reales:
  - Letterbox preprocessing (mantiene aspecto al redimensionar)
  - Escala variable (simula cartas más lejos o más cerca)
  - Blur moderado, variaciones de contraste fuertes
- Fine-tune desde V3 (aprovecha lo aprendido)
- **mAP50: 0.984** — mejor resultado hasta ahora

**El problema que genera:**
- El dataset sigue siendo de tamaño limitado (~3.7k imágenes)
- En condiciones muy adversas (luz tenue, cámara de baja calidad) sigue habiendo errores
- Necesitamos más diversidad de datos, no solo más augmentations en tiempo de entrenamiento

**Foto:** `docs/analysis/val/yolov8m_seg_v6/val_batch0_pred.jpg`

**Notas del ponente:**
> V4 sube el mAP50 a 0.984, el mejor en validación de toda la serie. Pero el cuello de botella ya no es el modelo sino la cantidad y variedad de datos de entrenamiento.

---

### Diapositiva 10 — Dataset V5: triplicar los datos con Albumentations

**Título:** V5 — Dataset de producción: 7.722 imágenes

**Lo que trae:**
- Dataset V4 **triplicado offline** con la librería **Albumentations**:
  - **Gaussian Blur** → simula cartas fuera de foco o cámara de baja resolución
  - **JPEG Compression** → simula artefactos de cámara barata
  - **CLAHE** → simula variaciones de iluminación real (contraluz, sombras)
- Resultado: **7.722 imágenes** de entrenamiento (vs 3.741 anteriores)
- Fine-tune desde V4
- **mAP50: 0.980 · mAP50-95: 0.980**

**El resultado:**
- mAP50 baja ligeramente en validación (0.984 → 0.980) pero la **robustez en webcam real mejora significativamente**
- El modelo aguanta mejor condiciones adversas de iluminación y cámara
- Este es el modelo que va a producción

**Foto:** `docs/presentation_assets/01_evolucion_modelos.png` *(gráfica de evolución de mAP a lo largo de versiones)*

**Notas del ponente:**
> La ligera caída en mAP50 de V5 respecto a V4 no es un problema — es el modelo siendo más honesto. V4 estaba algo sobreajustado al dominio del dataset; V5 generaliza mejor aunque la métrica de validación sea un poco inferior.

---

---

## BLOQUE 3 — EL JUEGO Y LA APLICACIÓN WEB

---

### Diapositiva 11 — Cómo evitamos los falsos positivos

**Título:** El problema: una carta no es una detección

**Contenido (izquierda — el problema):**
- El modelo detecta cartas en cada frame: 15 veces por segundo
- Una carta que pasa por encima de una zona se detectaría y registraría instantáneamente
- Una misma carta puede aparecer en múltiples frames con IDs distintos si no hay tracking
- Resultado sin control: el juego se llenaría de cartas fantasma

**Contenido (derecha — las tres capas de protección):**

**1. ByteTrack — identidad estable entre frames**
- Asigna un ID único a cada carta y lo mantiene aunque la carta se mueva
- Si la misma carta aparece en los frames 1, 2 y 3 → siempre es el mismo ID
- Evita contar la misma carta varias veces por movimiento

**2. Zonas configurables**
- La mesa está dividida en regiones: dealer, jugador 1, jugador 2
- Una carta solo se considera si está dentro de una zona reconocida
- Las zonas se definen en coordenadas normalizadas en `config.yaml`

**3. Debouncer — 1 segundo de presencia continua**
- Una carta no se registra al detectarse: debe permanecer en la misma zona durante **≥ 1 segundo**
- Si sale antes de ese tiempo → se descarta (carta en tránsito, mano que pasa)
- La barra de progreso bajo cada bbox en la UI muestra el tiempo acumulado

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` *(detecciones con bboxes y máscaras — ilustra el tracking visual)*

**Notas del ponente:**
> Estas tres capas trabajan en orden: ByteTrack garantiza que es la misma carta, las zonas garantizan que está en el sitio correcto, y el debouncer garantiza que lleva suficiente tiempo para ser una carta real de la mano y no una que alguien está moviendo.

---

### Diapositiva 12 — Lógica del juego y recomendación por EV

**Título:** Del estado de la mesa a la jugada óptima

**Contenido (izquierda — la máquina de estados):**
- El juego sigue una **máquina de estados finita**:
  ```
  WAITING → DEALING → PLAYER_TURN → DEALER_TURN → ROUND_END
  ```
- Cada carta confirmada actualiza el estado: a quién le toca, cuánto suma cada mano
- El `DeckTracker` descuenta cada carta vista del zapato → sabe exactamente qué queda

**Contenido (derecha — el cálculo de EV):**
- Para cada turno del jugador se calculan 3 EVs:
  - **EV STAND**: simula todos los desenlaces posibles del dealer ponderados por probabilidad
  - **EV HIT**: recursivo — evalúa cada carta que puede llegar y elige el mejor sub-estado
  - **EV DOUBLE**: igual que HIT pero apuesta ×2, solo una carta más
- Si el dealer aún no ha puesto carta: el EV se promedia sobre todos los valores posibles del zapato
- Se recomienda la acción con **mayor EV**
- Convención: **+1.0** = ganar una unidad · **0** = empate · **−1.0** = perder

**Ejemplo concreto:**
| Situación | EV STAND | EV HIT | EV DOUBLE | Recomendación |
|-----------|----------|--------|-----------|---------------|
| Mano 16 vs dealer 10 | −0.54 | −0.48 | — | **HIT** |
| Mano 11 vs dealer 6 | +0.18 | +0.34 | +0.58 | **DOUBLE** |

**Foto:** `docs/presentation_assets/02_metricas_finales.png`

**Notas del ponente:**
> El EV no es una tabla fija — se recalcula en cada frame con las cartas que quedan en el zapato. Conforme avanza la partida y salen cartas, las probabilidades cambian y la recomendación puede cambiar también. Es matemáticamente óptimo dado el estado conocido del zapato.

---

### Diapositiva 13 — La aplicación web

**Título:** Interfaz web en tiempo real

**Contenido (columna izquierda — qué ve el usuario):**
- Interfaz web con temática de casino (tapete verde, detalles dorados)
- Feed de vídeo en vivo con las detecciones superpuestas (bboxes + máscaras + barra de debounce)
- Panel lateral con:
  - Marcador de la partida (dealer / jugador / empates)
  - Recomendación de acción con los 3 EVs en tiempo real
  - Mesa: cartas de cada jugador con totales
  - Historial de cartas con opción de eliminar manualmente
  - Controles: empezar partida, plantarse, nueva ronda, deshacer

**Contenido (columna derecha — cómo se despliega):**
- Backend: **FastAPI + Uvicorn** (ASGI, asíncrono)
- Comunicación: **WebSocket** a 15 fps para el estado del juego + **MJPEG** a 30 fps para el vídeo
- Aceleración: **PyTorch + CUDA** (GPU NVIDIA)
- Contenedores: **Docker + NVIDIA Container Toolkit**
- Compatible con Windows nativo y WSL2

```
Windows:
  python cam_server_windows.py   ← expone la webcam
  uvicorn api.main:app           ← servidor web

WSL / Linux:
  docker compose up --build      ← todo en un contenedor
```

**Foto:** captura de pantalla de la interfaz web en funcionamiento *(hacer captura manual de `http://localhost:8080`)*

**Notas del ponente:**
> El WebSocket actualiza el panel lateral 15 veces por segundo — el usuario ve la recomendación cambiar en tiempo real según lo que detecta la cámara. El MJPEG es el stream de vídeo con las detecciones dibujadas encima.

---

---

## BLOQUE 4 — CIERRE

---

### Diapositiva 14 — Limitaciones y trabajos futuros

**Título:** Limitaciones actuales y trabajos futuros

**Contenido (izquierda — dónde el sistema todavía sufre):**
- Cartas dobladas / sucias / manchadas no se han probado sistemáticamente
- Zonas fijas en mesa — no detecta jugadores arbitrarios, asume layout pre-configurado en `config.yaml`
- No re-identificación entre rondas — una carta que sale y vuelve al zapato cuenta como dos

**Contenido (derecha — próximas iteraciones):**
- **SPLIT con EV real** — modelar dos manos independientes en `ev_calculator.py` (actualmente usa tabla estática)
- **Fine-tuning RT-DETR** — lr0 específico para objetos pequeños + más epochs para cerrar el gap con YOLOv8m-seg
- Detección de layout automático — inferir zonas de la mesa sin configuración manual

**Foto:** `docs/presentation_assets/04_comparativa_inferencia.png`

**Notas del ponente:**
> Las limitaciones más importantes son las de dominio — la cámara fija y el layout conocido. En un casino real las manos se mezclan, las cartas se doblan y los jugadores se mueven. Ese es el salto que habría que dar en una siguiente versión.

---

### Diapositiva 15 — Conclusiones

**Título:** Conclusiones

**Contenido — lo que hemos demostrado:**
- **Pipeline completo end-to-end** — dataset → entrenamiento → comparativa → app funcional en producción
- **Modelo robusto** — YOLOv8m-seg V6 con mAP50-95 = 0.98 sobre 54 clases
- **Comparativa rigurosa CNN vs Transformer** — misma configuración de entrenamiento, métrica objetiva → YOLO gana en precisión y velocidad
- **App real-time funcional** — WebSocket a 15 fps, recomendación por EV en tiempo real
- **Reproducibilidad** — MLflow con 5 runs trazados, notebooks ejecutables

**Foto:** `docs/presentation_assets/09_v6_results_official.png`

**Notas del ponente:**
> Lo más valioso del proyecto no es solo el mAP — es haber cerrado el ciclo completo: datos reales, entrenamiento con criterio, comparativa honesta y sistema desplegado que funciona con una webcam. Eso es lo que diferencia un experimento de un producto.

---

*— Fin de los Bloques 1, 2, 3 y 4 —*

---

## ÍNDICE DE RECURSOS GRÁFICOS DISPONIBLES

| Slide | Recurso | Ruta |
|-------|---------|------|
| 4 | Batch con labels GT | `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` |
| 5 | Screenshot SAM notebook | captura manual del notebook |
| 6 | Tiling explicación | `docs/presentation_assets/v1_tiling_explicacion.png` |
| 7 | Batch labels color | `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` |
| 8 | Batch labels V3 | `docs/analysis/val/yolov8m_seg_v5/val_batch1_labels.jpg` |
| 9 | Predicciones V4/V6 | `docs/analysis/val/yolov8m_seg_v6/val_batch0_pred.jpg` |
| 10 | Evolución modelos | `docs/presentation_assets/01_evolucion_modelos.png` |
| 11 | Detecciones con tracking | `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` |
| 12 | Métricas finales | `docs/presentation_assets/02_metricas_finales.png` |
| 13 | Captura interfaz web | **captura manual de localhost:8080** |
| 14 | Comparativa inferencia | `docs/presentation_assets/04_comparativa_inferencia.png` |
| 15 | Resultados oficiales V6 | `docs/presentation_assets/09_v6_results_official.png` |
