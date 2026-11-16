"""Simula la cámara de la línea: copia fotos de toallas (por defecto, las de test
de la cámara de junio) a la carpeta de entrada del servicio, una cada N segundos.
Sirve para probar el tablero sin la máquina.

    python scripts/serve.py                       # en una terminal
    python scripts/simulate_line.py --every 4     # en otra
"""
import argparse
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.data import load_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default=str(C.PROJECT_ROOT / "produccion" / "entrada"))
    ap.add_argument("--every", type=float, default=4.0, help="segundos entre fotos")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="test")
    ap.add_argument("--capture", default="2025-06", help="solo fotos de esta cámara (escala de producción)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_manifest()
    df = df[(df.split == args.split) & (df.capture == args.capture)]
    rows = df.sample(min(args.n, len(df)), random_state=args.seed)
    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows.itertuples(), 1):
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{i:04d}{Path(r.path).suffix}"
        tmp = inbox / (name + ".part")  # write then rename, like a camera should
        shutil.copy2(r.path, tmp)
        tmp.rename(inbox / name)
        print(f"[{i}/{len(rows)}] {name}  (real: {r.label})", flush=True)
        time.sleep(args.every * random.uniform(0.7, 1.3))


if __name__ == "__main__":
    main()
