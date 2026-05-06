# ─────────────────────────────────────────────────────────────────
# Imagen base: Python 3.11 slim (Debian Bookworm)
# Se elige slim en lugar de alpine porque OpenCV necesita glibc,
# y en lugar de la imagen pytorch oficial porque la controlamos
# mejor y es más ligera (~200 MB frente a ~7 GB).
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Dependencias de sistema que OpenCV necesita en tiempo de ejecución.
# libgl1          → rendering de imágenes (cv2.imshow, imencode)
# libglib2.0-0    → hilo de GLib que usa OpenCV internamente
# libsm6 / libxext6 / libxrender1 → dependencias de libGL en Debian
# Todas en un solo RUN para minimizar capas en la imagen final.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencias Python ────────────────────────────────────────────
# Se copia requirements.txt antes que el código para aprovechar
# la caché de capas de Docker: si solo cambia código fuente,
# pip install no se vuelve a ejecutar.
COPY requirements.txt .

RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir lapx>=0.5.2

RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente ──────────────────────────────────────────────────
# Se copian solo los módulos de la aplicación, no el dataset,
# los runs de entrenamiento ni los notebooks (ver .dockerignore).
COPY api/          ./api/
COPY game/         ./game/
COPY vision/       ./vision/
COPY frontend/     ./frontend/
COPY tests/        ./tests/
COPY config.yaml   .

# Puerto en el que escucha uvicorn (debe coincidir con docker-compose)
EXPOSE 8000

# reload=False porque en producción/contenedor no hay watch de ficheros.
# --workers 1 porque el estado del juego es en memoria y no es
# seguro compartirlo entre procesos.
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
