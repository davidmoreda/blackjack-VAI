"""
Servidor de webcam para Windows.
Ejecutar en Windows (NO en WSL) con Python de Windows:

    python cam_server_windows.py

Desde WSL conéctate a:
    http://10.255.255.254:5050/video

Requisitos Windows:
    pip install flask opencv-python
"""

import cv2
from flask import Flask, Response

# ── Configuración ──────────────────────────────
CAM_INDEX = 0       # 1 = cámara por defecto en Windows
WIDTH     = 1280
HEIGHT    = 720
PORT      = 5050
# ──────────────────────────────────────────────

app = Flask(__name__)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"No se puede abrir la cámara {CAM_INDEX}")

print(f"Cámara {CAM_INDEX} abierta: {int(cap.get(3))}x{int(cap.get(4))}")
print(f"Stream disponible en: http://0.0.0.0:{PORT}/video")
print(f"Desde WSL usa:        http://10.255.255.254:{PORT}/video")


def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buf.tobytes()
            + b'\r\n'
        )


@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return {'status': 'ok', 'cam': CAM_INDEX, 'width': WIDTH, 'height': HEIGHT}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, threaded=True)
