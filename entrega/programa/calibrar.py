"""Calibra el inspector para SU cámara e iluminación.

Uso: ponga entre 30 y 60 fotos de toallas BUENAS (sin defecto), tomadas con la
cámara y la luz que se usarán en la prueba, en la carpeta "calibracion" y ejecute
CALIBRAR.bat (o calibrar.sh). También puede indicar otra carpeta:

    python programa/calibrar.py D:\\fotos\\toallas_buenas
    python programa/calibrar.py --restaurar        # volver a la calibración de fábrica
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", nargs="?", default=str(ROOT / "calibracion"), help="fotos de toallas buenas")
    ap.add_argument("--escala", type=float, default=None, help="escala manual (si no se encuentra la etiqueta)")
    ap.add_argument("--nombre", default="planta")
    ap.add_argument("--restaurar", action="store_true", help="borrar la calibración y usar la de fábrica")
    args = ap.parse_args()
    modelos = ROOT / "modelos"
    cal = modelos / "calibracion.json"

    print("=" * 64)
    print(" Calibración del inspector")
    print("=" * 64)
    if args.restaurar:
        if cal.exists():
            cal.rename(cal.with_suffix(".json.anterior"))
        print(" Listo: se usará la calibración de fábrica (cámara de junio 2025).")
        return
    carpeta = Path(args.carpeta.strip('"'))
    carpeta.mkdir(parents=True, exist_ok=True)
    from galia.calibration import calibrate, list_photos

    n = len(list_photos(carpeta))
    if n < 15:
        print(f"\n En {carpeta} hay {n} fotos. Se necesitan al menos 15 (ideal 30–60)")
        print(" fotos de toallas BUENAS tomadas con su cámara e iluminación.")
        print(" Cópielas a esa carpeta y vuelva a ejecutar CALIBRAR.")
        return
    print(f"\n Fotos de toallas buenas: {n} en {carpeta}")
    print(" Importante: deben ser toallas SIN defecto, completas y con la luz de la prueba.\n")
    if cal.exists():
        cal.replace(cal.with_suffix(".json.anterior"))
    calibrate(carpeta, modelos, name=args.nombre, scale=args.escala, log=lambda m: print(" " + m))
    print("\n Listo. Cierre INICIAR si está abierto y vuelva a abrirlo para usar la nueva calibración.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # mensaje claro para el usuario
        print(f"\n Error: {e}")
