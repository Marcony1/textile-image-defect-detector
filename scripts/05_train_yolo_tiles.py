"""Train YOLO11 on 1024 px tiles (run scripts/04_build_dataset.py first).

    python scripts/05_train_yolo_tiles.py --model yolo11s.pt --epochs 80
"""
import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=C.YOLO_TILE)
    ap.add_argument("--name", default="yolo11s_tiles")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(C.YOLO_TILES_DIR / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(C.RUNS_DIR),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        seed=C.SEED,
        patience=25,
        cos_lr=True,
        close_mosaic=10,
        # Towels can appear in any orientation; colour jitter is kept mild
        # because dye differences are themselves a defect class.
        fliplr=0.5,
        flipud=0.5,
        degrees=5.0,
        scale=0.3,
        hsv_h=0.005,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=1.0,
        plots=True,
    )
