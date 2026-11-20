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
    ap.add_argument("--scale", type=float, default=None,
                    help="escala de la foto; por defecto la de models/calibracion.json o 1.0 (cámara de junio 2025)")
    ap.add_argument("--demo-dir", default=None, help="carpeta con fotos/ y etiquetas.csv para la demostración")
    ap.add_argument("--brand-name", default="")
    ap.add_argument("--brand-logo", default=None)
    ap.add_argument("--brand-color", default="", help="color principal #rrggbb")
    ap.add_argument("--brand-color-2", default="", help="color secundario #rrggbb")
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
    cfg = ServiceConfig(line_name=args.line_name, streak_alert=args.streak_alert, rate_alert=args.rate_alert,
                        demo_dir=Path(args.demo_dir) if args.demo_dir else None, brand_name=args.brand_name,
                        brand_logo=Path(args.brand_logo) if args.brand_logo else None,
                        brand_color=args.brand_color, brand_color_2=args.brand_color_2,
                        settings_file=Path(args.db).parent / "ajuste.json",
                        labels_dir=Path(args.db).parent / "etiquetas")
    app = create_app(store, worker, cfg)
    print(f"Tablero: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    print(f"Carpeta de entrada: {args.inbox}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
