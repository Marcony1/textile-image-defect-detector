"""Servicio de producción: vigila la carpeta donde la cámara guarda las fotos,
inspecciona cada toalla y publica el tablero en vivo.

    python scripts/serve.py --inbox D:/camara/salida --archive D:/galia/archivo
    # abrir http://localhost:8000 en la pantalla de la línea

Para verlo desde otras computadoras de la planta: --host 0.0.0.0 (solo en la red
interna; el servicio no tiene usuarios ni contraseñas).
"""
import argparse
import logging
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.service.app import ServiceConfig, create_app  # noqa: E402
from galia.service.store import Store  # noqa: E402
from galia.service.worker import InspectionWorker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = C.PROJECT_ROOT / "produccion"
    ap.add_argument("--inbox", default=str(base / "entrada"), help="carpeta donde la cámara deja las fotos")
    ap.add_argument("--archive", default=str(base / "archivo"), help="fotos procesadas, por día y decisión")
    ap.add_argument("--db", default=str(base / "inspecciones.db"))
    ap.add_argument("--models", default=str(C.MODELS_DIR))
    ap.add_argument("--operating-point", default="recall_98", choices=["recall_90", "recall_95", "recall_98"])
    ap.add_argument("--scale", type=float, default=1.0,
                    help="1.0 para la cámara de junio 2025 (3580x2150); 0.62 para la de abril (5120x5120)")
    ap.add_argument("--line-name", default="Línea 1")
    ap.add_argument("--streak-alert", type=int, default=5, help="NO OK seguidas que disparan alerta")
    ap.add_argument("--rate-alert", type=float, default=0.5, help="fracción NO OK (últimas 50) que dispara alerta")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = Store(args.db)
    worker = InspectionWorker(store, Path(args.inbox), Path(args.archive), Path(args.models),
                              args.operating_point, args.scale)
    cfg = ServiceConfig(line_name=args.line_name, streak_alert=args.streak_alert, rate_alert=args.rate_alert)
    app = create_app(store, worker, cfg)
    print(f"Tablero: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    print(f"Carpeta de entrada: {args.inbox}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
