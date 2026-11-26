"""Registra una versión nueva del programa en src/galia/service/static/cambios.json.

    python scripts/nueva_version.py 1.4.0 "Título corto" "Cambio 1" "Cambio 2" ...
    python scripts/nueva_version.py --ver            # muestra la versión actual y la siguiente sugerida

cambios.json es la única fuente de la versión: de ahí salen el nombre del zip, el número del
tablero, el botón «Novedades» y la sección 14 de LEEME.pdf. Después de registrar la versión:

    make zip          (o: python scripts/actualizar.py zip)
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMBIOS = ROOT / "src" / "galia" / "service" / "static" / "cambios.json"


def parse(v: str) -> tuple:
    if not re.fullmatch(r"\d+\.\d+\.\d+", v):
        raise SystemExit(f"Versión inválida «{v}»: use MAYOR.MENOR.PARCHE, por ejemplo 1.4.0")
    return tuple(int(x) for x in v.split("."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?")
    ap.add_argument("titulo", nargs="?")
    ap.add_argument("cambios", nargs="*")
    ap.add_argument("--ver", action="store_true", help="solo mostrar la versión actual")
    a = ap.parse_args()
    data = json.loads(CAMBIOS.read_text(encoding="utf-8"))
    cur = data[0]["version"]
    M, m, p = parse(cur)
    if a.ver or not a.version:
        print(f"Versión actual: {cur} ({data[0]['fecha']}, {data[0]['titulo']})")
        print(f"Siguiente sugerida: {M}.{m}.{p + 1} (arreglo) o {M}.{m + 1}.0 (función nueva)")
        return
    if parse(a.version) <= parse(cur):
        raise SystemExit(f"La versión nueva ({a.version}) debe ser mayor que la actual ({cur}).")
    if not a.titulo or not a.cambios:
        raise SystemExit("Indique un título y al menos un cambio (cada cambio entre comillas).")
    data.insert(0, dict(version=a.version, fecha=date.today().isoformat(), titulo=a.titulo, cambios=a.cambios))
    CAMBIOS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Registrada la versión {a.version}. Ahora genere el paquete: make zip")


if __name__ == "__main__":
    main()
