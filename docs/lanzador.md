# Lanzador — Blackjack VAI

## Requisitos previos

- Python/uv instalado
- Webcam conectada (índice 0 por defecto)
- `config.yaml` con `camera.index: "http://localhost:5050/video"`

---

## Paso 1 — Servidor de cámara

Abre una terminal y ejecuta:

```powershell
uv run python cam_server_windows.py
```

Verifica que arranca correctamente:

```
Cámara 0 abierta: 1280x720
Stream disponible en: http://0.0.0.0:5050/video
```

Comprobación opcional:

```powershell
curl http://localhost:5050/health
# → {"cam":0,"height":720,"status":"ok","width":1280}
```

---

## Paso 2 — API principal

Abre **otra terminal** y ejecuta:

```powershell
uv run uvicorn api.main:app --port 8080 --reload
```

Espera a ver:

```
Application startup complete.
Uvicorn running on http://127.0.0.1:8080
```

---

## Paso 3 — Abrir en el navegador

```
http://localhost:8080
```

---

## Para parar

- `Ctrl+C` en la terminal de uvicorn
- `Ctrl+C` en la terminal de cam_server_windows.py

---

## Notas

- El `--reload` hace que uvicorn recargue automáticamente al editar código.
- `cam_server_windows.py` usa `CAM_INDEX = 0`. Si tu webcam está en otro índice, edita esa línea.
- El `config.yaml` debe tener `index: "http://localhost:5050/video"` (no `host.docker.internal`, eso solo funciona dentro de Docker).
