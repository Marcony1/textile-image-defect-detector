"""Inicia el inspector de toallas: vigila la carpeta de entrada y abre el tablero.

Lee ../configuracion.ini. Se ejecuta con INICIAR.bat (Windows) o iniciar.sh (Linux).
"""
import configparser
import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # carpeta del paquete
sys.path.insert(0, str(Path(__file__).resolve().parent))


def ruta(valor: str) -> Path:
    p = Path(valor.strip().strip('"'))
    return p if p.is_absolute() or str(p).startswith("\\\\") else ROOT / p


def leer_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))  # "#" no: los colores llevan #
    cfg.read(ROOT / "configuracion.ini", encoding="utf-8")
    return cfg


def ip_local() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "IP-de-esta-PC"


def main():
    cfg = leer_config()
    g = lambda sec, key, default="": cfg.get(sec, key, fallback=default).strip()  # noqa: E731
    si = lambda v: v.lower() in ("si", "sí", "yes", "true", "1")  # noqa: E731

    entrada, procesadas = ruta(g("carpetas", "entrada", "entrada")), ruta(g("carpetas", "procesadas", "procesadas"))
    ajuste = g("inspeccion", "ajuste", "recall_95")
    if ajuste not in ("recall_90", "recall_95", "recall_98"):
        print(f"Ajuste '{ajuste}' no válido; se usa recall_95")
        ajuste = "recall_95"
    esc = g("inspeccion", "escala", "auto").lower()
    escala = None if esc in ("", "auto") else float(esc.replace(",", "."))
    puerto = int(g("pantalla", "puerto", "8000"))
    red = si(g("pantalla", "permitir_red", "no"))
    logo = g("marca", "logo")
    logo = ruta(logo) if logo else None

    print("=" * 64)
    try:
        from galia.service.version import VERSION
    except Exception:
        VERSION = "?"
    print(f" Inspector de toallas · versión {VERSION}")
    print("=" * 64)
    print(" Cargando programa (la primera vez tarda hasta 1 minuto)…")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        import uvicorn
        from galia.service.app import ServiceConfig, create_app
        from galia.service.store import Store
        from galia.service.worker import InspectionWorker
    except ImportError as e:
        print(f"\n No se encontraron las librerías ({e}).\n Ejecute primero INSTALAR.")
        return

    plc_on = si(g("plc", "habilitar", "no"))
    plc_port = int(g("plc", "puerto", "502"))
    plc_ips = tuple(x.strip() for x in g("plc", "ip_permitidas").replace(";", ",").split(",") if x.strip())
    datos = ROOT / "datos"
    store = Store(datos / "inspecciones.db")
    worker = InspectionWorker(store, entrada, procesadas, ROOT / "modelos", ajuste, escala)
    demo = ROOT / "demo"
    sc = ServiceConfig(line_name=g("inspeccion", "nombre_linea", "Línea de prueba"),
                       streak_alert=int(g("inspeccion", "alerta_seguidas", "5")),
                       rate_alert=float(g("inspeccion", "alerta_fraccion", "0.5")),
                       demo_dir=demo if (demo / "etiquetas.csv").exists() else None,
                       brand_name=g("marca", "nombre"), brand_logo=logo,
                       brand_color=g("marca", "color_principal"), brand_color_2=g("marca", "color_secundario"),
                       settings_file=datos / "ajuste.json", labels_dir=ROOT / "etiquetas",
                       plc_enabled=plc_on, plc_port=plc_port, plc_allowed=plc_ips,
                       plc_host=g("plc", "escuchar_en", "0.0.0.0") or "0.0.0.0")
    app = create_app(store, worker, sc)
    url = f"http://localhost:{puerto}"
    print(f"\n Tablero:            {url}")
    if red:
        print(f" Desde otra PC:      http://{ip_local()}:{puerto}")
    print(f" Carpeta de entrada: {entrada}")
    print(f" Fotos procesadas:   {procesadas}")
    print(f" Ajuste:             {ajuste}")
    if plc_on:
        print(f" PLC (Modbus TCP):   puerto {plc_port} en {ip_local()}"
              + (f" · solo desde {', '.join(plc_ips)}" if plc_ips else ""))
    print("\n Deje esta ventana abierta mientras use el sistema. Para detenerlo, ciérrela.\n")
    if si(g("pantalla", "abrir_navegador", "si")):
        threading.Thread(target=lambda: (time.sleep(2.5), webbrowser.open(url)), daemon=True).start()
    try:
        uvicorn.run(app, host="0.0.0.0" if red else "127.0.0.1", port=puerto, log_level="warning")
    except OSError as e:
        print(f"\n No se pudo abrir el puerto {puerto}: {e}\n ¿Ya hay otro INICIAR abierto? Ciérrelo o cambie el puerto en configuracion.ini.")


if __name__ == "__main__":
    main()
