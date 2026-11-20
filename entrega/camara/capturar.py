"""Toma fotos de las toallas y las entrega al inspector.

Funciona en una Raspberry Pi (cámara CSI con picamera2) o en cualquier PC con
una cámara USB (OpenCV). Cada foto se entrega de una de dos formas:

  --carpeta RUTA     la guarda en una carpeta (la carpeta "entrada" del inspector,
                     o una carpeta compartida de la red que apunte a ella)
  --servidor URL     la envía por la red al inspector (http://IP-DE-LA-PC:8000);
                     requiere permitir_red = si en configuracion.ini

Cuándo se toma la foto:
  --modo tecla       al presionar Enter (pruebas)            [por defecto]
  --modo intervalo   cada N segundos (--segundos N)
  --modo sensor      cuando un sensor conectado a un pin GPIO se activa (--pin 17), solo Raspberry Pi

Ejemplos:
  python3 capturar.py --carpeta /mnt/inspector/entrada --modo sensor --pin 17
  python3 capturar.py --servidor http://192.168.1.50:8000 --modo intervalo --segundos 4
  python capturar.py --fuente usb --carpeta "C:\\Inspector\\entrada"

Las fotos se escriben primero como .part y se renombran al terminar, para que el
inspector nunca lea una foto a medias.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path


# ------------------------------------------------------------------ cameras
class CamaraPi:
    def __init__(self, ancho, alto):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        conf = self.cam.create_still_configuration(main={"size": (ancho, alto)} if ancho else {})
        self.cam.configure(conf)
        self.cam.start()
        time.sleep(2)  # auto-exposición

    def foto(self, destino: Path):
        self.cam.capture_file(str(destino), format="jpeg")

    def cerrar(self):
        self.cam.stop()


class CamaraUSB:
    def __init__(self, indice, ancho, alto):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(indice)
        if ancho:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara USB {indice}")
        for _ in range(5):  # descartar los primeros cuadros (exposición)
            self.cap.read()

    def foto(self, destino: Path):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("La cámara no devolvió imagen")
        self.cv2.imwrite(str(destino), frame, [self.cv2.IMWRITE_JPEG_QUALITY, 95])

    def cerrar(self):
        self.cap.release()


# ----------------------------------------------------------------- delivery
def entregar_carpeta(tmp: Path, carpeta: Path, nombre: str):
    carpeta.mkdir(parents=True, exist_ok=True)
    parcial = carpeta / (nombre + ".part")
    with open(tmp, "rb") as src, open(parcial, "wb") as dst:
        dst.write(src.read())
    os.replace(parcial, carpeta / nombre)
    return f"guardada en {carpeta / nombre}"


def entregar_servidor(tmp: Path, url: str, nombre: str):
    limite = uuid.uuid4().hex
    cuerpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{nombre}\"\r\n"
              f"Content-Type: image/jpeg\r\n\r\n").encode() + tmp.read_bytes() + f"\r\n--{limite}--\r\n".encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/inspect", data=cuerpo, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        res = json.loads(r.read())
    return f"{res.get('decision')} (score {res.get('score', 0):.2f})"


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", choices=["pi", "usb"], default=None, help="pi = cámara de Raspberry Pi; usb = webcam/cámara USB")
    ap.add_argument("--usb", type=int, default=0, help="número de la cámara USB (0, 1, …)")
    ap.add_argument("--ancho", type=int, default=0, help="resolución (ej. 3840); 0 = la de la cámara")
    ap.add_argument("--alto", type=int, default=0)
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--carpeta", help="carpeta donde guardar las fotos (la 'entrada' del inspector)")
    dest.add_argument("--servidor", help="URL del inspector, ej. http://192.168.1.50:8000")
    ap.add_argument("--modo", choices=["tecla", "intervalo", "sensor"], default="tecla")
    ap.add_argument("--segundos", type=float, default=5.0, help="modo intervalo: segundos entre fotos")
    ap.add_argument("--pin", type=int, default=17, help="modo sensor: pin GPIO (numeración BCM)")
    ap.add_argument("--retraso", type=float, default=0.0, help="modo sensor: segundos de espera entre la señal y la foto")
    ap.add_argument("--prefijo", default="toalla")
    args = ap.parse_args()

    fuente = args.fuente
    if fuente is None:
        try:
            import picamera2  # noqa: F401
            fuente = "pi"
        except ImportError:
            fuente = "usb"
    cam = CamaraPi(args.ancho, args.alto) if fuente == "pi" else CamaraUSB(args.usb, args.ancho, args.alto)
    tmpdir = Path.home() / ".capturar_tmp"
    tmpdir.mkdir(exist_ok=True)
    print(f"Cámara: {fuente} · modo: {args.modo} · destino: {args.carpeta or args.servidor}")
    print("Ctrl+C para terminar.\n")

    def tomar():
        nombre = f"{args.prefijo}_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3] + ".jpg"
        tmp = tmpdir / nombre
        try:
            cam.foto(tmp)
            msg = (entregar_carpeta(tmp, Path(args.carpeta), nombre) if args.carpeta
                   else entregar_servidor(tmp, args.servidor, nombre))
            print(f"{datetime.now():%H:%M:%S} {nombre}: {msg}", flush=True)
        except Exception as e:  # no detener la captura por una foto
            print(f"{datetime.now():%H:%M:%S} ERROR {nombre}: {e}", flush=True)
        finally:
            tmp.unlink(missing_ok=True)

    try:
        if args.modo == "tecla":
            while True:
                input("Enter = tomar foto ")
                tomar()
        elif args.modo == "intervalo":
            while True:
                t0 = time.time()
                tomar()
                time.sleep(max(0.0, args.segundos - (time.time() - t0)))
        else:
            from gpiozero import Button

            sensor = Button(args.pin, pull_up=True, bounce_time=0.2)
            print(f"Esperando señal del sensor en GPIO {args.pin}…")
            while True:
                sensor.wait_for_press()
                if args.retraso:
                    time.sleep(args.retraso)
                tomar()
                sensor.wait_for_release()
    except KeyboardInterrupt:
        print("\nCaptura terminada.")
    finally:
        cam.cerrar()


if __name__ == "__main__":
    sys.exit(main())
