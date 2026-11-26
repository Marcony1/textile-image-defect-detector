"""Train the box-supervised tile segmenter (galia.segnet.SegTileNet) on the same
cls_tiles as the classifier, with the same recipe (balanced sampling, AdamW +
OneCycle, 15 epochs, best epoch by tile AP on the val tiles), so the comparison
with 06_train_tile_classifier.py only changes the supervision: box masks + tile label.

    python scripts/06c_train_tile_segmenter.py --arch efficientnet_b0 --seed 7 --tag basic_s7
    -> models/tile_seg_efficientnet_b0_basic_s7.pt   (loads like a classifier: arch "seg:...")
"""
import argparse
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.segnet import SegTileNet, SegTiles, tile_boxes_index  # noqa: E402


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    return 1 - ((2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--w-tile", type=float, default=0.5, help="weight of the tile-level BCE")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    boxes = tile_boxes_index()
    train_ds, val_ds = SegTiles("train", boxes, train=True), SegTiles("val", boxes, train=False)
    targets = np.array(train_ds.targets)
    w = 1.0 / np.bincount(targets)[targets]
    print(f"train tiles: {len(targets)} ({targets.sum()} defect); boxes for {len(boxes)} defect tiles")
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(targets), replacement=True)
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
                          persistent_workers=True)
    val_dl = DataLoader(val_ds, args.batch * 2, shuffle=False, num_workers=args.workers, pin_memory=True,
                        persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegTileNet(args.arch, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    name = f"tile_seg_{args.arch}" + (f"_{args.tag}" if args.tag else "")
    ckpt_path = C.MODELS_DIR / f"{name}.pt"
    hist, best = [], -1
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tl, n = 0.0, 0
        for x, m, y in train_dl:
            x, m, y = x.to(device, non_blocking=True), m.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                ml = model.forward_mask(x)
            ml = ml.float()
            pix = F.binary_cross_entropy_with_logits(ml, m)
            pos = y > 0
            dice = dice_loss(ml[pos], m[pos]).mean() if pos.any() else ml.sum() * 0
            tile = F.binary_cross_entropy_with_logits(SegTileNet.tile_logit(ml).squeeze(1), y)
            loss = pix + dice + args.w_tile * tile
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            n += len(y)

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for x, _, y in val_dl:
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    probs.append(torch.sigmoid(model(x.to(device)).float()).squeeze(1).cpu())
                ys.append(y)
        p, yv = torch.cat(probs).numpy(), torch.cat(ys).numpy()
        ap_, auc = average_precision_score(yv, p), roc_auc_score(yv, p)
        rec = ((p > 0.5) & (yv == 1)).sum() / max(1, (yv == 1).sum())
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_ap=ap_, val_auc=auc, val_recall_at_0_5=rec))
        print(f"ep {ep + 1:02d} loss {tl / n:.4f} | val AP {ap_:.4f} AUC {auc:.4f} recall@0.5 {rec:.3f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        if ap_ > best:
            best = ap_
            torch.save({"arch": f"seg:{args.arch}", "state_dict": model.state_dict(), "epoch": ep + 1,
                        "val_ap": ap_, "config": vars(args)}, ckpt_path)
    (C.RUNS_DIR / name).mkdir(parents=True, exist_ok=True)
    (C.RUNS_DIR / name / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"best val AP {best:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
