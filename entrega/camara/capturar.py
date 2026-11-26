"""Toma fotos de las toallas y las entrega al inspector (PC con la tarjeta gráfica).

Pensado para la Raspberry Pi con la cámara Raspberry Pi High Quality (IMX477,
4056 x 3040, a color), pero también funciona con una cámara USB, y tiene una
fuente "carpeta" que reenvía fotos existentes para probar todo el flujo
(cámara → inspector → PLC) sin cámara.

Entrega de cada foto (elija una):
  --servidor URL   la envía por la red al inspector, ej. http://192.168.1.50:8000
                   (en la PC: configuracion.ini → [pantalla] permitir_red = si)
  --carpeta RUTA   la guarda en una carpeta (la "entrada" del inspector o una
                   carpeta compartida de la red que apunte a ella)

Cuándo se toma la foto (--modo):
  tecla        al presionar Enter (pruebas)                     [por defecto]
  intervalo    cada N segundos (--segundos N)
  sensor       cuando se activa un pin GPIO de la Pi (--pin 17): sensor de presencia
               o salida del PLC a través de un optoacoplador. --retraso espera a que la toalla quede en posición;
               --pin-listo da un pulso cuando la foto ya se tomó (para reanudar la banda).

Imagen estable (IMPORTANTE para el inspector): por defecto la cámara de la Pi
mide la exposición y el balance de blancos automáticos UNA vez al arrancar
(--medir segundos) y los deja FIJOS. Así la luz no "cambia" entre toallas.
También se pueden dar a mano: --exposicion (µs), --ganancia, --balance R,B.

Nombre de archivo: toalla_000123.jpg — el número es un contador que se guarda
entre reinicios (--inicio N para cambiarlo). El inspector toma ese número como
ID de la toalla y se lo pasa al PLC (registros 4-5).

Ejemplos:
  python3 capturar.py --servidor http://192.168.1.50:8000 --modo sensor --pin 17 --retraso 0.2
  python3 capturar.py --carpeta /mnt/inspector --modo tecla
  python3 capturar.py --fuente carpeta --origen fotos_prueba --servidor http://localhost:8000 --modo intervalo --segundos 2
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

ESTADO = Path.home() / ".capturar_estado.json"   # contador de toallas


# ------------------------------------------------------------------ cameras
class CamaraPi:
    """Raspberry Pi HQ Camera (o cualquier cámara de la Pi) con picamera2."""

    def __init__(self, ancho, alto, calidad, medir, exposicion, ganancia, balance):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        size = (ancho, alto) if ancho else self.cam.sensor_resolution  # HQ: 4056 x 3040
        conf = self.cam.create_still_configuration(main={"size": tuple(size)})
        self.cam.configure(conf)
        self.cam.options["quality"] = calidad
        self.cam.start()
        if exposicion or ganancia or balance:
            ctrl = {"AeEnable": False, "AwbEnable": False}
            md = self._medir(medir) if not (exposicion and ganancia and balance) else {}
            ctrl["ExposureTime"] = int(exposicion or md["ExposureTime"])
            ctrl["AnalogueGain"] = float(ganancia or md["AnalogueGain"])
            ctrl["ColourGains"] = tuple(balance or md["ColourGains"])
        else:
            md = self._medir(medir)
            ctrl = {"AeEnable": False, "AwbEnable": False, "ExposureTime": int(md["ExposureTime"]),
                    "AnalogueGain": float(md["AnalogueGain"]), "ColourGains": tuple(md["ColourGains"])}
        self.cam.set_controls(ctrl)
        time.sleep(0.5)
        self.ajustes = ctrl
        print(f"Cámara Pi: {size[0]} x {size[1]} · exposición {ctrl['ExposureTime']} µs · ganancia "
              f"{ctrl['AnalogueGain']:.2f} · balance R,B {ctrl['ColourGains'][0]:.2f},{ctrl['ColourGains'][1]:.2f} (FIJOS)")
        print("  Para repetir exactamente estos valores:  "
              f"--exposicion {ctrl['ExposureTime']} --ganancia {ctrl['AnalogueGain']:.2f} "
              f"--balance {ctrl['ColourGains'][0]:.2f},{ctrl['ColourGains'][1]:.2f}")

    def _medir(self, segundos):
        print(f"Midiendo exposición y balance de blancos automáticos durante {segundos:.0f} s "
              "(deje una toalla BUENA bajo la cámara, con la luz definitiva)…")
        time.sleep(segundos)
        return self.cam.capture_metadata()

    def foto(self, destino: Path):
        self.cam.capture_file(str(destino), format="jpeg")

    def cerrar(self):
        self.cam.stop()


class CamaraUSB:
    def __init__(self, indice, ancho, alto, calidad):
        import cv2

        self.cv2, self.calidad = cv2, calidad
        self.cap = cv2.VideoCapture(indice)
        if ancho:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara USB {indice}")
        for _ in range(10):  # que se estabilice la exposición automática
            self.cap.read()
        # fijar exposición y balance si la cámara lo permite (no todas las webcams lo respetan)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    def foto(self, destino: Path):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("La cámara no devolvió imagen")
        self.cv2.imwrite(str(destino), frame, [self.cv2.IMWRITE_JPEG_QUALITY, self.calidad])

    def cerrar(self):
        self.cap.release()


class CamaraCarpeta:
    """Simula una cámara reenviando las fotos de una carpeta (en orden, y vuelve a empezar)."""

    def __init__(self, origen):
        ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.fotos = sorted(p for p in Path(origen).iterdir() if p.suffix.lower() in ext)
        if not self.fotos:
            raise RuntimeError(f"No hay fotos en {origen}")
        self.i = 0
        print(f"Fuente de prueba: {len(self.fotos)} fotos de {origen}")

    def foto(self, destino: Path):
        shutil.copyfile(self.fotos[self.i % len(self.fotos)], destino)
        self.i += 1

    def cerrar(self):
        pass


# ----------------------------------------------------------------- delivery
def entregar_carpeta(tmp: Path, carpeta: Path, nombre: str):
    carpeta.mkdir(parents=True, exist_ok=True)
    parcial = carpeta / (nombre + ".part")  # el inspector ignora .part: nunca lee una foto a medias
    shutil.copyfile(tmp, parcial)
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
    return f"{res.get('decision')} (score {res.get('score') or 0:.2f})"


def siguiente_numero(inicio=None) -> int:
    try:
        n = json.loads(ESTADO.read_text())["ultimo"] if inicio is None else inicio - 1
    except (OSError, ValueError, KeyError):
        n = 0
    n += 1
    ESTADO.write_text(json.dumps({"ultimo": n}))
    return n


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", choices=["pi", "usb", "carpeta"], default=None,
                    help="pi = cámara de Raspberry Pi (HQ); usb = cámara USB; carpeta = reenviar fotos existentes (pruebas)")
    ap.add_argument("--origen", help="fuente carpeta: carpeta con las fotos a reenviar")
    ap.add_argument("--usb", type=int, default=0, help="número de la cámara USB (0, 1, …)")
    ap.add_argument("--ancho", type=int, default=0, help="resolución; 0 = la máxima del sensor (HQ: 4056 x 3040)")
    ap.add_argument("--alto", type=int, default=0)
    ap.add_argument("--calidad", type=int, default=95, help="calidad JPG (90-98)")
    ap.add_argument("--medir", type=float, default=3.0, help="segundos para medir exposición/balance antes de fijarlos")
    ap.add_argument("--exposicion", type=int, default=0, help="exposición fija en microsegundos (0 = medir)")
    ap.add_argument("--ganancia", type=float, default=0.0, help="ganancia analógica fija (0 = medir)")
    ap.add_argument("--balance", default="", help="balance de blancos fijo 'R,B' (ej. 1.9,1.6; vacío = medir)")
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--carpeta", help="carpeta donde guardar las fotos (la 'entrada' del inspector)")
    dest.add_argument("--servidor", help="URL del inspector, ej. http://192.168.1.50:8000")
    ap.add_argument("--modo", choices=["tecla", "intervalo", "sensor"], default="tecla")
    ap.add_argument("--segundos", type=float, default=5.0, help="modo intervalo: segundos entre fotos")
    ap.add_argument("--pin", type=int, default=17, help="modo sensor: pin GPIO (numeración BCM)")
    ap.add_argument("--retraso", type=float, default=0.0, help="segundos de espera entre la señal y la foto")
    ap.add_argument("--pin-listo", type=int, default=0,
                    help="pin GPIO (BCM) que da un pulso de 0.1 s cuando la foto ya se tomó (ej. para que el PLC reanude la banda)")
    ap.add_argument("--prefijo", default="toalla")
    ap.add_argument("--inicio", type=int, default=None, help="número de la primera toalla (por defecto continúa el contador)")
    ap.add_argument("--max", type=int, default=0, help="detenerse después de N fotos (0 = sin límite)")
    args = ap.parse_args()

    fuente = args.fuente
    if fuente is None:
        try:
            import picamera2  # noqa: F401
            fuente = "pi"
        except ImportError:
            fuente = "usb"
    if fuente == "pi":
        bal = tuple(float(x) for x in args.balance.split(",")) if args.balance else None
        cam = CamaraPi(args.ancho, args.alto, args.calidad, args.medir, args.exposicion, args.ganancia, bal)
    elif fuente == "usb":
        cam = CamaraUSB(args.usb, args.ancho, args.alto, args.calidad)
    else:
        if not args.origen:
            sys.exit("--fuente carpeta necesita --origen CARPETA")
        cam = CamaraCarpeta(args.origen)
    tmpdir = Path.home() / ".capturar_tmp"
    tmpdir.mkdir(exist_ok=True)
    print(f"Fuente: {fuente} · modo: {args.modo} · destino: {args.carpeta or args.servidor}")
    print("Ctrl+C para terminar.\n")
    listo = None
    if args.pin_listo:
        from gpiozero import DigitalOutputDevice

        listo = DigitalOutputDevice(args.pin_listo)
    inicio = [args.inicio]
    tomadas = [0]

    def tomar():
        n = siguiente_numero(inicio[0])
        inicio[0] = None
        nombre = f"{args.prefijo}_{n:06d}.jpg"
        tmp = tmpdir / nombre
        try:
            t0 = time.time()
            cam.foto(tmp)
            t1 = time.time()
            if listo:  # la foto ya está tomada: la toalla se puede mover
                listo.blink(on_time=0.1, n=1, background=True)
            msg = (entregar_carpeta(tmp, Path(args.carpeta), nombre) if args.carpeta
                   else entregar_servidor(tmp, args.servidor, nombre))
            print(f"{datetime.now():%H:%M:%S} {nombre}: {msg}  (foto {1000 * (t1 - t0):.0f} ms)", flush=True)
        except Exception as e:  # no detener la captura por una foto
            print(f"{datetime.now():%H:%M:%S} ERROR {nombre}: {e}", flush=True)
        finally:
            tmp.unlink(missing_ok=True)
        tomadas[0] += 1
        if args.max and tomadas[0] >= args.max:
            raise KeyboardInterrupt

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

            # pull-up interno: la señal "activa" es el pin llevado a GND (salida del optoacoplador o NPN)
            sensor = Button(args.pin, pull_up=True, bounce_time=0.05)
            print(f"Esperando señal en GPIO {args.pin}…")
            while True:
                sensor.wait_for_press()
                if args.retraso:
                    time.sleep(args.retraso)
                tomar()
                sensor.wait_for_release()
    except KeyboardInterrupt:
        print(f"\nCaptura terminada ({tomadas[0]} fotos).")
    finally:
        cam.cerrar()


if __name__ == "__main__":
    sys.exit(main())
