"""Build the unified manifest and the tiled datasets for every model.

Usage (from the project root):
    python scripts/04_build_dataset.py            # manifest + all tile sets
    python scripts/04_build_dataset.py --only manifest

Outputs
    data/manifest.csv          one row per towel photo with its split
    data/labels_unified/       YOLO labels with the unified class taxonomy
    data/yolo_tiles/           tiled YOLO dataset (train/val) + data.yaml
    data/cls_tiles/            defect / clean tiles for the tile classifier
    data/patchcore_tiles/      clean tiles from OK towels for PatchCore
"""
import argparse
import random
import shutil
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.data import build_manifest, load_boxes, load_image, load_manifest  # noqa: E402
from galia.roi import tile_coverage, towel_mask  # noqa: E402
from galia.tiling import boxes_in_tile, crop, tile_grid, tile_has_partial_defect  # noqa: E402

# Fraction of tiles with no annotation that are kept (background / clean fabric).
YOLO_KEEP_EMPTY = {"NG": 0.25, "OK": 0.15}
CLS_KEEP_CLEAN = {"NG": 0.30, "OK": 0.12}
PATCHCORE_TILES_PER_TOWEL = 3


def _yolo_tiles_for(row):
    rng = random.Random(zlib.crc32(row["image_id"].encode()))
    img, scale = load_image(row["path"], row["source"])
    boxes = load_boxes(row["label_path"], row["width"], row["height"])
    boxes[:, 1:] *= scale
    H, W = img.shape[:2]
    out_dir = C.YOLO_TILES_DIR / row["split"]
    n = 0
    for k, win in enumerate(tile_grid(W, H, C.YOLO_TILE, C.YOLO_OVERLAP)):
        tb = boxes_in_tile(boxes, win)
        if len(tb) == 0:
            if tile_has_partial_defect(boxes, win) or rng.random() > YOLO_KEEP_EMPTY[row["label"]]:
                continue
        name = f"{row['image_id']}_t{k:03d}"
        Image.fromarray(crop(img, win, C.YOLO_TILE)).save(out_dir / "images" / f"{name}.jpg", quality=95)
        T = C.YOLO_TILE
        lines = [f"{int(c)} {(x1 + x2) / 2 / T:.6f} {(y1 + y2) / 2 / T:.6f} {(x2 - x1) / T:.6f} {(y2 - y1) / T:.6f}"
                 for c, x1, y1, x2, y2 in tb]
        (out_dir / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n


def _cls_tiles_for(row):
    rng = random.Random(zlib.crc32(row["image_id"].encode()))
    img, scale = load_image(row["path"], row["source"])
    boxes = load_boxes(row["label_path"], row["width"], row["height"])
    boxes[:, 1:] *= scale
    H, W = img.shape[:2]
    n = 0
    for k, win in enumerate(tile_grid(W, H, C.CLS_TILE, C.CLS_OVERLAP)):
        tb = boxes_in_tile(boxes, win)
        has_defect = bool(len(tb)) and np.isin(tb[:, 0], C.DEFECT_CLASS_IDS).any()
        if has_defect:
            cls = "defect"
        else:
            if tile_has_partial_defect(boxes, win):
                continue  # ambiguous: part of a defect is visible
            if rng.random() > CLS_KEEP_CLEAN[row["label"]]:
                continue
            cls = "clean"
        d = C.CLS_TILES_DIR / row["split"] / cls
        Image.fromarray(crop(img, win, C.CLS_TILE)).save(d / f"{row['image_id']}_t{k:03d}.jpg", quality=95)
        n += 1
    return n


def _patchcore_tiles_for(row):
    rng = random.Random(zlib.crc32(row["image_id"].encode()))
    img, _ = load_image(row["path"], row["source"])
    H, W = img.shape[:2]
    mask = towel_mask(img)
    # Only tiles that show towel (incl. its borders) - background is ignored at inference.
    wins = [w for w in tile_grid(W, H, C.PATCHCORE_TILE, 0.0) if tile_coverage(mask, w) >= 0.3]
    for k, win in enumerate(rng.sample(wins, min(PATCHCORE_TILES_PER_TOWEL, len(wins)))):
        t = Image.fromarray(crop(img, win, C.PATCHCORE_TILE)).resize((C.PATCHCORE_INPUT,) * 2, Image.BILINEAR)
        t.save(C.PATCHCORE_TILES_DIR / "train" / "good" / f"{row['image_id']}_{k}.png")
    return len(wins)


def _run(fn, rows, workers):
    with ProcessPoolExecutor(workers) as ex:
        return sum(ex.map(fn, rows, chunksize=4))


def make_yolo(df, workers):
    shutil.rmtree(C.YOLO_TILES_DIR, ignore_errors=True)
    for s in ("train", "val"):
        (C.YOLO_TILES_DIR / s / "images").mkdir(parents=True)
        (C.YOLO_TILES_DIR / s / "labels").mkdir(parents=True)
    # YOLO uses towels with box annotations only (A, B, C). Test towels are
    # never tiled: they are evaluated on the full photo, as in production.
    rows = df[df.has_boxes & df.split.isin(["train", "val"])].to_dict("records")
    n = _run(_yolo_tiles_for, rows, workers)
    data = {"path": str(C.YOLO_TILES_DIR), "train": "train/images", "val": "val/images",
            "nc": len(C.CLASSES), "names": C.CLASSES}
    (C.YOLO_TILES_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"YOLO tiles: {n} from {len(rows)} towels")


def make_cls(df, workers):
    shutil.rmtree(C.CLS_TILES_DIR, ignore_errors=True)
    for s in ("train", "val"):
        for c in ("clean", "defect"):
            (C.CLS_TILES_DIR / s / c).mkdir(parents=True)
    # Defect tiles need boxes (A, B); clean tiles come from every towel,
    # including the April OK towels (D) that have no box annotations.
    rows = df[df.split.isin(["train", "val"])].to_dict("records")
    n = _run(_cls_tiles_for, rows, workers)
    print(f"Classifier tiles: {n} from {len(rows)} towels")


def make_patchcore(df, workers):
    shutil.rmtree(C.PATCHCORE_TILES_DIR, ignore_errors=True)
    (C.PATCHCORE_TILES_DIR / "train" / "good").mkdir(parents=True)
    rows = df[(df.label == "OK") & (df.split == "train")].to_dict("records")
    _run(_patchcore_tiles_for, rows, workers)
    print(f"PatchCore normal tiles: {len(list((C.PATCHCORE_TILES_DIR / 'train' / 'good').glob('*')))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["manifest", "yolo", "cls", "patchcore"], default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.only in (None, "manifest"):
        df = build_manifest()
        df.to_csv(C.MANIFEST_CSV, index=False)
        print(df.groupby(["split", "source", "label"]).size().unstack(["source", "label"]).fillna(0).astype(int))
    df = load_manifest()
    if args.only in (None, "yolo"):
        make_yolo(df, args.workers)
    if args.only in (None, "cls"):
        make_cls(df, args.workers)
    if args.only in (None, "patchcore"):
        make_patchcore(df, args.workers)
