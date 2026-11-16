"""Hard-negative mining for the tile classifier.

The towel score is the max over ~50 tiles, so a single over-confident clean tile
turns an OK towel into a false alarm. The first training run saw only ~12 % of the
clean tiles of OK towels. This script scores *every* tile of the OK towels in the
TRAIN split and copies the ones the classifier wrongly flags into
data/cls_tiles/train/clean/ (prefix ``hn_``) so a fine-tuning run can learn them.

Val/test towels are never touched, so the thresholds and the reported metrics
stay honest.

    python scripts/06b_mine_hard_negatives.py --min-prob 0.2 --per-towel 8
    python scripts/06_train_tile_classifier.py --init models/tile_cls_efficientnet_b0.pt --epochs 6 --lr 1e-4 --hn-weight 3
"""
import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.data import load_image, load_manifest  # noqa: E402
from galia.models import ClassifierTilePredictor  # noqa: E402
from galia.tiling import crop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(C.MODELS_DIR / "tile_cls_efficientnet_b0.pt"))
    ap.add_argument("--min-prob", type=float, default=0.2)
    ap.add_argument("--per-towel", type=int, default=8)
    args = ap.parse_args()

    out = C.CLS_TILES_DIR / "train" / "clean"
    for f in out.glob("hn_*.png"):  # idempotent
        f.unlink()
    pred = ClassifierTilePredictor(args.weights)
    df = load_manifest()
    ok = df[(df.split == "train") & (df.label == "OK")]
    n_saved, n_flagged = 0, 0
    for r in tqdm(ok.itertuples(), total=len(ok), desc="mining"):
        img, _ = load_image(r.path, r.source)
        res = pred(img)
        hard = sorted([t for t in res.tile_scores if t[1] >= args.min_prob], key=lambda t: -t[1])
        n_flagged += bool(hard)
        for win, p in hard[: args.per_towel]:
            tile = crop(img, win, C.CLS_TILE)
            cv2.imwrite(str(out / f"hn_{r.image_id}_{win[0]}_{win[1]}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            n_saved += 1
    print(f"{len(ok)} OK train towels, {n_flagged} with a tile >= {args.min_prob}; saved {n_saved} hard negatives -> {out}")


if __name__ == "__main__":
    main()
