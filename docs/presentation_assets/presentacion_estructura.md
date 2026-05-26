# Estructura Presentación — BlackjackVAI
> Modo historia: cada diapositiva narra una decisión, un problema encontrado y cómo se resolvió.

---

## BLOQUE 1 — INTRODUCCIÓN

---

### Diapositiva 1 — Portada

**Título:** BlackjackVAI
**Subtítulo:** Sistema de visión artificial para blackjack en tiempo real
**Autores:** Carlos Díaz · David Moreda · Javier Cerón
**Contexto:** Máster en IA — Visión Artificial Avanzada — 2026

**Foto:** `docs/presentation_assets/Foto_cartas.png`

**Notas del ponente:**
> Abrimos con la pregunta que nos hicimos al empezar el proyecto: ¿podemos poner una cámara encima de una mesa de blackjack y que el ordenador juegue mejor que nosotros?

---

### Diapositiva 2 — El juego: Blackjack

**Título:** Blackjack

**Historia:**
Antes de hablar de visión artificial, necesitamos entender qué estamos resolviendo. El blackjack parece simple — llegar a 21 — pero la complejidad está en las decisiones: ¿pido carta con un 16 si el dealer tiene un 10? ¿Doblo con un 11? Estas decisiones tienen una respuesta matemáticamente óptima que depende de qué cartas quedan en el mazo. Eso es exactamente lo que nuestro sistema va a calcular en tiempo real.

**Contenido:**
- Objetivo: acercarse a 21 sin pasarse, superando al croupier
- Valores: 2–10 numérico · figuras = 10 · As = 1 u 11
- Acciones: **HIT** · **STAND** · **DOUBLE** · **SPLIT**
- El dealer sigue reglas fijas: pide carta hasta llegar a 17+

**Foto:** `docs/presentation_assets/Foto_cartas.png`

**Notas del ponente:**
> La clave de esta slide es que la detección debe ser precisa: un 10 clasificado como J no cambia el total, pero un 6 clasificado como 9 sí cambia completamente el consejo de jugada.

---

### Diapositiva 3 — El proyecto

**Título:** El proyecto

**Historia:**
La idea era construir un sistema completo, no solo un clasificador de cartas. Queríamos que el sistema entendiese el estado de la partida, llevase la cuenta de quién tiene qué, y en cada momento dijese qué hacer. Todo eso con una webcam normal encima de la mesa.

**Contenido:**
- Reconoce **54 clases** (52 cartas + dorso + Joker)
- Sigue turnos automáticamente: jugador → dealer → fin de ronda
- Lleva el marcador: rondas ganadas / perdidas / empates
- Sugiere la **jugada óptima** en cada turno mediante Valor Esperado (EV)

**Foto:** `docs/presentation_assets/Foto_cartas.png`

**Notas del ponente:**
> Lo que diferencia este proyecto de un clasificador de imágenes es el sistema completo: visión + lógica de juego + interfaz. Cada pieza depende de la anterior.

---

### Diapositiva 4 — Arquitectura del sistema

**Título:** Arquitectura del sistema

**Historia:**
Desde el principio separamos el sistema en capas para que cada parte pudiese evolucionar de forma independiente. Si cambiamos el modelo de detección, la lógica del juego no se toca. Si cambiamos las reglas del EV, la cámara no se toca.

**Las cuatro capas:**

```
[CÁMARA]      Webcam sobre la mesa — captura continua del estado de la partida
     ↓
[DETECCIÓN]   YOLOv8m-seg — detecta y clasifica cada carta, genera máscara de segmentación
     ↓
[TRACKING]    ByteTrack — mantiene la identidad de cada carta entre frames,
              evita registrar la misma carta múltiples veces
     ↓
[JUEGO]       Almacena jugadas, calcula EV, recomienda acciones — interfaz web
```

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg`

**Notas del ponente:**
> Esta separación en capas fue una decisión de diseño clave. Nos permitió iterar el dataset y el modelo sin tocar nada del motor de juego, y viceversa.

---

## BLOQUE 2 — EL DATASET

---

### Diapositiva 5 — El punto de partida: qué necesitamos

**Título:** Dataset — el punto de partida

**Historia:**
Lo primero que nos preguntamos fue: ¿de dónde sacamos datos? Necesitábamos imágenes de cartas anotadas con 54 clases, en condiciones similares a una mesa real con webcam. Fotografiar todo a mano era inviable. La solución fue Roboflow — una plataforma de datasets de visión artificial donde encontramos un dataset de cartas ya etiquetado con el split hecho.

**Contenido:**
- **54 clases**: 52 cartas + dorso + Joker
- Alta variabilidad: ángulo, luz, cartas parcialmente cubiertas
- Fuente: **Roboflow** — dataset ya etiquetado, split 80% train / 20% val
- Anotaciones en formato YOLO

**El problema de fondo que ya intuíamos:**
Las imágenes son de estudio controlado, no de mesa real. Hay un gap entre el dominio de entrenamiento y el de inferencia que vamos a tener que cerrar versión a versión.

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg`

**Notas del ponente:**
> Roboflow nos dio una base sólida para empezar rápido. El reto real no era conseguir imágenes, era conseguir que el modelo entrenado con esas imágenes funcionase con nuestra webcam en condiciones reales.

---

### Diapositiva 6 — Intento previo: SAM para segmentación automática

**Título:** SAM para segmentación automática

**Historia:**
Antes de centrarnos en Roboflow, exploramos si podíamos generar nuestros propios datos de segmentación automáticamente. Encontramos un dataset de Kaggle con 2.757 imágenes de cartas etiquetadas con bounding boxes, pero sin máscaras de segmentación. La idea: usar SAM (Segment Anything Model de Meta) para generar los contornos automáticamente y así tener un dataset de segmentación sin anotar a mano.

**Cómo funcionó:**
1. Parseamos los bounding boxes del XML de Kaggle
2. Usamos el **centro de cada bbox como point prompt** para SAM
3. SAM genera 3 máscaras candidatas → elegimos la de **mayor área** (la carta completa)
4. La **clase la tomamos del XML de Kaggle** — SAM solo aporta el contorno, no sabe qué carta es
5. Exportamos a formato COCO JSON → ZIP listo para subir a Roboflow
- Resultado: **2.757 imágenes procesadas, 0 errores**

**Por qué no lo usamos al final:**
El dataset de Roboflow ya tenía las 54 clases correctamente etiquetadas en formato YOLO nativo. SAM añadía el contorno pero el coste de integración era mayor que el beneficio frente a lo que ya teníamos.

**Foto:** captura del notebook mostrando overlay SAM — máscara cyan + punto prompt rojo sobre carta

**Notas del ponente:**
> Fue un experimento valioso: aprendimos cómo funciona SAM con point prompts y construimos un pipeline de anotación semi-automática completo. Pero al final Roboflow resolvía el mismo problema con menos fricción.

---

### Diapositiva 7 — V1 y V2: aprender a base de errores

**Título:** Dataset V1 y V2 — los primeros intentos

**Historia V1:**
Empezamos con el dataset más grande que encontramos: ~10.500 imágenes. Entrenamos, y el modelo no funcionaba con nuestra webcam. Al investigar descubrimos el problema: las imágenes estaban en **escala de grises** y habían sido generadas con *image tiling* — partir fotos de alta resolución en tiles más pequeños. El modelo había aprendido a ver fragmentos de cartas en gris, no cartas completas en color. Tuvimos que descartarlo entero.

**Historia V2:**
Volvemos a Roboflow y descargamos un dataset diferente: 3.741 imágenes en color, completas, sin tiling. Pero esta vez no entrenamos — al revisar el dataset vimos que no tenía ninguna augmentation aplicada. Con tan poco dato y sin augmentations, el modelo iba a memorizar las condiciones exactas del dataset y fallar en cualquier variación real. Pasamos directamente a V3.

**Contenido:**
| | V1 | V2 |
|---|---|---|
| Imágenes | ~10.500 | ~3.741 |
| Problema | Escala de grises + tiling | Sin augmentations |
| Decisión | ❌ Descartada | ❌ No entrenada |

**Foto:** `docs/presentation_assets/v1_tiling_explicacion.png`

**Notas del ponente:**
> V1 es el error clásico de no revisar el dataset antes de entrenar. V2 es la decisión correcta de no perder tiempo entrenando algo que ya sabíamos que iba a fallar.

---

### Diapositiva 8 — V3 y V4: acercándonos al caso real

**Título:** Dataset V3 y V4 — augmentations para el mundo real

**Historia V3:**
Con V2 como base añadimos las augmentations estándar de YOLO: rotaciones, flips, cambios de brillo y contraste. Entrenamos 100 epochs y llegamos a **mAP50 = 0.954** — el primer modelo que realmente funcionaba. Pero al probarlo en la mesa real, las cartas alejadas de la cámara se detectaban mal. El modelo había aprendido cartas a una distancia fija, no a distancias variables.

**Historia V4:**
El problema estaba claro: necesitábamos que el modelo viese cartas a distintas escalas durante el entrenamiento. Añadimos augmentations agresivas orientadas a distancia — variaciones de escala fuertes, letterbox preprocessing — y arrancamos V4 como fine-tune desde V3. El mAP50 subió a **0.984**, el mejor resultado hasta ahora. Pero el dataset seguía siendo el mismo de 3.741 imágenes — poca diversidad para casos muy adversos.

**Contenido:**
| | V3 | V4 |
|---|---|---|
| Base | 3.741 imágenes | 3.741 imágenes |
| Nuevo | Augmentations YOLO estándar | Augmentations agresivas de escala |
| Entrenamiento | Desde cero, 100 epochs | Fine-tune desde V3, 120 epochs |
| mAP50 | 0.954 | 0.984 |
| Problema | Falla con cartas lejanas | Dataset aún pequeño |

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch1_labels.jpg`

**Notas del ponente:**
> V3 a V4 es el salto de "funciona en validación" a "funciona un poco mejor en el mundo real". El problema que queda ya no es el modelo sino la cantidad y variedad de los datos.

---

### Diapositiva 9 — V5 y V6: el dataset de producción

**Título:** Dataset V5 y V6 — triplicar los datos con Albumentations

**Historia:**
El cuello de botella ya no era el modelo ni las augmentations en entrenamiento — era que 3.741 imágenes son pocas para cubrir toda la variabilidad real. La solución: triplicar el dataset *offline* usando **Albumentations**, generando versiones sintéticas de cada imagen que simulen las condiciones más difíciles de una webcam real.

**Las tres transformaciones clave:**
- **Gaussian Blur** → simula cámara de baja resolución o cartas fuera de foco
- **JPEG Compression** → simula artefactos de cámara barata o stream de vídeo
- **CLAHE** → simula variaciones de iluminación real: contraluz, sombras, luz tenue

Con esto pasamos de 3.741 a **7.722 imágenes** de entrenamiento.

**V5 vs V6:**
- **V5** — fine-tune desde V4 (aprovecha lo aprendido): mAP50 = 0.980
- **V6** — entrenamiento desde cero con el dataset triplicado: **mAP50 = 0.986, mAP50-95 = 0.980** → modelo de producción

El mAP50 de V5 baja ligeramente respecto a V4 (0.980 vs 0.984) pero la robustez en webcam real mejora significativamente. V6 cierra ese gap arrancando desde cero con más datos.

**Foto:** `docs/presentation_assets/01_evolucion_modelos.png`

**Notas del ponente:**
> La caída de mAP50 en V5 respecto a V4 no es un problema — es el modelo siendo más honesto. V4 estaba algo sobreajustado al dominio del dataset. V6 es el que está corriendo en la app.

---

## BLOQUE 3 — DETECCIÓN, JUEGO Y WEB

---

### Diapositiva 10 — Cómo evitamos los falsos positivos

**Título:** Detección y tracking — cómo evitamos los falsos positivos

**Historia:**
El modelo detecta a 15 fps. Sin ningún control, una carta que alguien mueve por la mesa se registraría docenas de veces. Una carta en tránsito entre zonas contaminaría el estado del juego. Necesitábamos tres capas de protección antes de que una detección se convirtiera en una carta registrada.

**Las tres capas:**

**1. ByteTrack + filtro de Kalman — identidad estable**
Asigna un ID único a cada carta y lo mantiene aunque YOLO falle un frame o la carta se mueva. Elimina el parpadeo sin crear cartas fantasma.

**2. Zonas configurables**
La mesa está dividida en regiones: dealer, jugador 1, jugador 2. Solo se considera una detección si está dentro de una zona reconocida. Configurables en `config.yaml`.

**3. Debouncer — 1 segundo de presencia continua**
Una carta no se registra al detectarse: debe permanecer en la misma zona durante **≥ 1 segundo**. Si sale antes → se descarta. La barra de progreso bajo cada bbox en la UI muestra el tiempo acumulado.

**Resultado:** cada carta se registra exactamente una vez por mano, independientemente de los FPS.

**Foto:** `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg`

**Notas del ponente:**
> El debouncer es la pieza más importante de robustez del sistema. Sin él, cualquier movimiento de mano sobre la mesa contaminaría el estado del juego. Con él, solo cuenta lo que realmente se está jugando.

---

### Diapositiva 11 — Lógica del juego y recomendación por EV

**Título:** Juego y web — de la detección a la jugada óptima

**Historia:**
Una vez que una carta se confirma, entra en la máquina de estados del juego. El sistema sabe en qué fase está la ronda, de quién es el turno, y cuánto suma cada mano. Con esa información y sabiendo exactamente qué cartas quedan en el zapato, calcula el Valor Esperado de cada acción posible.

**Cómo funciona el EV:**
El programa cuenta todas las cartas que han aparecido y sabe qué queda en el zapato. Para cada turno calcula:
- **EV STAND**: simula todos los desenlaces posibles del dealer ponderados por probabilidad
- **EV HIT**: recursivo — evalúa cada carta que puede llegar y elige el mejor sub-estado
- **EV DOUBLE**: igual que HIT pero apuesta ×2, solo una carta más

Se recomienda la acción con mayor EV. Si el dealer aún no tiene carta, el EV se promedia sobre todos los valores posibles del zapato.

**Ejemplo:**
| Situación | EV STAND | EV HIT | Recomendación |
|-----------|----------|--------|---------------|
| Mano 16 vs dealer 10 | −0.54 | −0.48 | **HIT** |
| Mano 11 vs dealer 6 | +0.18 | +0.34 | **DOUBLE** |

**Foto:** `docs/presentation_assets/02_metricas_finales.png`

**Notas del ponente:**
> El EV no es una tabla fija — se recalcula en cada frame con las cartas que quedan. Conforme avanza la partida y salen cartas, las probabilidades cambian y la recomendación puede cambiar también.

---

### Diapositiva 12 — La aplicación web

**Título:** La aplicación web

**Historia:**
Montamos toda la lógica detrás de una interfaz web con FastAPI y Uvicorn. El usuario abre el navegador, apunta la cámara a la mesa y el sistema hace el resto: detecta las cartas, sigue la partida y muestra la recomendación actualizada en tiempo real.

**Qué ve el usuario:**
- Feed de vídeo en vivo con detecciones superpuestas (bboxes + máscaras + barra de debounce)
- Marcador de la partida y estado de cada mano
- Recomendación de acción con los 3 EVs en tiempo real
- Historial de cartas con opción de corrección manual

**Cómo se despliega:**
- Backend: **FastAPI + Uvicorn** — WebSocket a 15 fps para el estado + MJPEG a 30 fps para el vídeo
- Aceleración: **PyTorch + CUDA**
- Contenedores: **Docker + NVIDIA Container Toolkit**

```
Windows:
  python cam_server_windows.py   ← expone la webcam
  uvicorn api.main:app           ← servidor web

WSL / Docker:
  docker compose up --build
```

**Foto:** captura de pantalla de la interfaz web en funcionamiento *(captura manual de localhost:8080)*

**Notas del ponente:**
> El WebSocket actualiza el panel 15 veces por segundo. El usuario ve la recomendación cambiar en tiempo real según lo que detecta la cámara — no hay que pulsar nada.

---

## BLOQUE 4 — MODELOS Y COMPARATIVA

---

### Diapositiva 13 — ¿Por qué un segundo modelo? RT-DETR

**Título:** ¿Por qué un segundo modelo? RT-DETR

**Historia:**
Teníamos YOLOv8m-seg funcionando bien, pero queríamos validar que era realmente la mejor opción y no solo la más cómoda. Decidimos entrenar RT-DETR-L — un transformer para detección en tiempo real — con exactamente la misma configuración: mismo dataset, misma resolución, mismo régimen de entrenamiento. Así la comparativa sería honesta.

**Por qué RT-DETR y no otra YOLO:**
RT-DETR es un paradigma diferente — transformer vs CNN — lo que hace la comparativa más interesante académicamente. Comparar dos YOLOs hubiera sido comparar versiones del mismo enfoque.

**Foto:** `docs/presentation_assets/04_comparativa_inferencia.png`

**Notas del ponente:**
> La clave es "misma config". Si hubiéramos optimizado RT-DETR por separado podría haber ganado o perdido por razones ajenas al modelo. Así sabemos que las diferencias son del modelo, no del setup.

---

### Diapositiva 14 — Comparativa de modelos

**Título:** Comparativa final — calidad + coste

**Historia:**
Los resultados hablan solos. Todos los modelos llegan al techo en mAP50, pero las diferencias aparecen en mAP50-95 (calidad de localización) y en FPS (velocidad de inferencia). YOLOv8m-seg V6 gana en ambas dimensiones.

**Foto:** `docs/presentation_assets/03_modelo_produccion_v6.png`

**Notas del ponente:**
> mAP50 es fácil de saturar con 54 clases bien separadas visualmente. mAP50-95 es la métrica que de verdad mide la calidad de las máscaras de segmentación — ahí es donde V6 se separa de RT-DETR.

---

### Diapositiva 15 — Modelo de producción: V6

**Título:** Modelo de producción — YOLOv8m-seg V6

**Historia:**
V6 es el resultado de todo el proceso: dataset triplicado, entrenamiento desde cero, early stopping. Es el modelo que está corriendo en la app. Las curvas de loss muestran convergencia limpia sin overfitting — train y val siguen la misma tendencia hasta el final.

**Foto:** `docs/presentation_assets/09_v6_results_official.png`

**Notas del ponente:**
> V6 entrenado desde cero con el dataset triplicado supera a V5 fine-tuneado. El dataset más grande compensa el coste de no partir de pesos preentrenados.

---

## BLOQUE 5 — CIERRE

---

### Diapositiva 16 — ¿Qué hemos aportado?

**Título:** ¿Qué hemos aportado?

**Historia:**
Más allá de los números, lo que hemos construido tiene valor por las decisiones que tomamos y por cerrar el ciclo completo.

**Contenido:**
- **Dataset propio de segmentación** — máscaras generadas con SAM sobre un dataset que solo tenía bounding boxes
- **Augmentations para cámara real** — blur, ruido JPEG y low-light → ×3 datos de entrenamiento sin recolectar más imágenes
- **Comparativa contra otra arquitectura** — RT-DETR-L (transformer), no otra YOLO — con la misma configuración para que sea objetiva
- **Modelo elegido por métricas** — V6 gana en mAP, latencia y tamaño de modelo

**Foto:** `docs/presentation_assets/08_v6_results_summary.png`

**Notas del ponente:**
> Lo más valioso es el proceso, no el número final. Cualquiera puede entrenar YOLOv8 con un dataset de Roboflow. Lo diferencial es haber iterado el dataset, comparado con un paradigma distinto y desplegado un sistema que funciona de verdad.

---

### Diapositiva 17 — Limitaciones y trabajos futuros

**Título:** Limitaciones y trabajos futuros

**Dónde el sistema todavía sufre:**
- Cartas dobladas / sucias / manchadas — no probadas sistemáticamente
- Zonas fijas — no detecta jugadores arbitrarios, asume layout pre-configurado
- No re-identificación — una carta que sale y vuelve al zapato cuenta como dos

**Próximas iteraciones:**
- **SPLIT con EV real** — modelar dos manos independientes en `ev_calculator.py` (actualmente usa tabla estática)
- **Fine-tuning RT-DETR** — lr0 específico para objetos pequeños + más epochs para cerrar el gap con V6
- Detección automática del layout de la mesa

**Foto:** `docs/presentation_assets/04_comparativa_inferencia.png`

**Notas del ponente:**
> Las limitaciones más importantes son las de dominio — layout fijo, cartas en buen estado. En una partida real los jugadores se mueven y las cartas se desgastan. Ese es el siguiente salto.

---

### Diapositiva 18 — Conclusiones

**Título:** Conclusiones

**Lo que hemos demostrado:**
- **Pipeline completo end-to-end** — dataset → entrenamiento → comparativa → app funcional en producción
- **Modelo robusto** — YOLOv8m-seg V6 con mAP50-95 = 0.98 sobre 54 clases
- **Comparativa rigurosa CNN vs Transformer** — misma configuración, métrica objetiva → YOLO gana en precisión y velocidad
- **App real-time funcional** — WebSocket a 15 fps, recomendación por EV en tiempo real
- **Reproducibilidad** — MLflow con 5 runs trazados, notebooks ejecutables

**Foto:** `docs/presentation_assets/09_v6_results_official.png`

**Notas del ponente:**
> Lo que diferencia este proyecto de un experimento académico es que funciona: cámara, mesa, cartas, recomendación. El ciclo está cerrado.

---

*— Fin —*

---

## ÍNDICE DE RECURSOS GRÁFICOS

| Slide | Recurso | Ruta |
|-------|---------|------|
| 1 | Foto cartas reales | `docs/presentation_assets/Foto_cartas.png` |
| 2 | Foto cartas reales | `docs/presentation_assets/Foto_cartas.png` |
| 3 | Foto cartas reales | `docs/presentation_assets/Foto_cartas.png` |
| 4 | Detecciones con máscaras | `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` |
| 5 | Batch con labels GT | `docs/analysis/val/yolov8m_seg_v5/val_batch0_labels.jpg` |
| 6 | Captura notebook SAM | captura manual del notebook |
| 7 | Tiling explicación | `docs/presentation_assets/v1_tiling_explicacion.png` |
| 8 | Batch labels V3 | `docs/analysis/val/yolov8m_seg_v5/val_batch1_labels.jpg` |
| 9 | Evolución modelos | `docs/presentation_assets/01_evolucion_modelos.png` |
| 10 | Detecciones con tracking | `docs/analysis/val/yolov8m_seg_v5/val_batch0_pred.jpg` |
| 11 | Métricas finales | `docs/presentation_assets/02_metricas_finales.png` |
| 12 | Captura interfaz web | **captura manual de localhost:8080** |
| 13 | Comparativa inferencia | `docs/presentation_assets/04_comparativa_inferencia.png` |
| 14 | Modelo producción V6 | `docs/presentation_assets/03_modelo_produccion_v6.png` |
| 15 | Resultados oficiales V6 | `docs/presentation_assets/09_v6_results_official.png` |
| 16 | Resumen resultados | `docs/presentation_assets/08_v6_results_summary.png` |
| 17 | Comparativa inferencia | `docs/presentation_assets/04_comparativa_inferencia.png` |
| 18 | Resultados oficiales V6 | `docs/presentation_assets/09_v6_results_official.png` |
