# Blackjack VAI — Documento de Diseno

**Fecha:** 2026-04-07
**Proyecto:** Vision Artificial Avanzada — Master IA

---

## Contexto

Sistema de seguimiento de blackjack en tiempo real mediante vision artificial. Una webcam en posicion cenital captura la mesa. El sistema detecta cartas, sigue el juego, lleva conteo y muestra probabilidades y sugerencias optimas en una interfaz web.

## Requisitos funcionales

- Configuracion de numero de jugadores (1-7) y mazos (1, 2, 4, 6, 8)
- Deteccion de cartas con YOLOv8 (dataset propio etiquetado en Roboflow)
- Deteccion de acciones: HIT automatico por aparicion de carta nueva, STAND/DOUBLE/SPLIT por UI
- Conteo de cartas: Hi-Lo, KO, Omega II (seleccionable)
- Probabilidades de proxima carta en tiempo real
- Sugerencia de accion optima por Basic Strategy
- Interfaz web con feed de camara + overlay + panel lateral

## Arquitectura — Enfoque B (modular monolito)

```
Webcam → vision/capture.py → vision/detector.py (YOLOv8)
       → vision/zone_manager.py (asignacion carta→jugador)
       → game/state_machine.py (motor de reglas + turnos)
       → game/counter.py (conteo)
       → game/strategy.py (Basic Strategy)
       → api/ws.py (WebSocket ~15fps)
       → frontend/ (HTML + JS)
```

## Fases de desarrollo

### Fase 1 — Dataset y modelo
- Captura con `dataset/capture_cards.py` (modos: sequential / manual)
- 50+ imagenes por clase, 53 clases (52 cartas + card_back)
- Etiquetado en Roboflow con augmentations
- Entrenamiento YOLOv8s con `dataset/train.py`

### Fase 2 — Deteccion y zonas
- Definicion de zonas de mesa (zone_setup.py)
- Asignacion de cartas a manos via ZoneManager
- Tracking entre frames

### Fase 3 — Logica de juego
- Motor de reglas Blackjack (engine.py)
- Maquina de estados por turno (state_machine.py)
- Conteo y True Count (counter.py)
- Basic Strategy (strategy.py)

### Fase 4 — Interfaz web
- FastAPI + WebSockets
- Frontend con overlay en tiempo real
- Panel: conteo, probabilidades, sugerencias, acciones

## Decisiones de diseno

| Decision | Eleccion | Razon |
|----------|----------|-------|
| Deteccion HIT | Aparicion de nueva carta en zona | Mas robusto que gestos |
| STAND/DOUBLE | Botones en UI | Evita falsos positivos |
| Streaming | WebSockets | Latencia minima para tiempo real |
| Vista camara | Cenital | Sin distorsion de perspectiva |
| Modelo base | YOLOv8s | Balance velocidad/precision |
