"""Inspect towel photos with the trained ensemble (production entry point).

    python scripts/predict.py foto.jpg
    python scripts/predict.py carpeta_de_fotos/ --out reports/predicciones --save-images
    python scripts/predict.py foto_5120.png --scale 0.62     # photos from the April-2025 camera
    python scripts/predict.py carpeta/ --operating-point recall_98

Writes <out>/predicciones.csv (one row per photo: decision, score, per-model
scores, defects found) and, with --save-images, an annotated JPG per photo.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.inference import TowelInspector  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--out", default=str(C.REPORTS_DIR / "predicciones"))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize factor to the production pixel scale (1.0 = June-2025 camera)")
    ap.add_argument("--operating-point", default="recall_98", choices=["recall_90", "recall_95", "recall_98"])
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inspector = TowelInspector.from_dir(C.MODELS_DIR, args.operating_point)
    print(f"{len(files)} images | threshold {inspector.threshold:.4g} ({args.operating_point})")

    rows = []
    for f in files:
        t0 = time.time()
        r = inspector.inspect(f, scale=args.scale)
        dt = time.time() - t0
        rows.append(dict(file=str(f), decision=r["decision"], score=r["score"], seconds=round(dt, 2),
                         **{f"score_{k}": v for k, v in r["model_scores"].items()},
                         defects=json.dumps([d for d in r["defects"] if d["cls"] != "Etiqueta"], ensure_ascii=False)))
        print(f"{r['decision']:6s} score={r['score']:.3f}  {dt:.1f}s  {f.name}")
        if args.save_images:
            from PIL import Image

            from galia.viz import draw_boxes

            im = Image.open(f).convert("RGB")
            boxes = [d["box"] + [d["conf"], C.CLASSES.index(d["cls"])] for d in r["defects"]]
            vis = draw_boxes(__import__("numpy").asarray(im), boxes)
            vis.thumbnail((2000, 2000))
            vis.save(out / f"{f.stem}_{r['decision'].replace(' ', '_')}.jpg", quality=90)
    pd.DataFrame(rows).to_csv(out / "predicciones.csv", index=False)
    print(f"-> {out / 'predicciones.csv'}")
