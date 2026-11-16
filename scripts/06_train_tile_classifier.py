"""Fine-tune an ImageNet-pretrained CNN (EfficientNet / ResNet via timm) to
classify 512 px tiles as defect / clean. At inference the towel score is the
max tile probability (see galia.models.ClassifierTilePredictor).

    python scripts/06_train_tile_classifier.py --arch efficientnet_b0
    python scripts/06_train_tile_classifier.py --arch resnet50
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.models import build_classifier  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class RandomRot90:
    def __call__(self, x):
        return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from (e.g. after hard-negative mining)")
    ap.add_argument("--hn-weight", type=float, default=1.0,
                    help="sampling weight of hard-negative tiles (hn_*) relative to other clean tiles")
    args = ap.parse_args()
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        # mild photometric jitter: no hue change (dye differences are defects)
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(), RandomRot90(), transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.ImageFolder(C.CLS_TILES_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(C.CLS_TILES_DIR / "val", transform=val_tf)
    assert train_ds.classes == ["clean", "defect"], train_ds.classes

    # Balanced sampling: every batch sees ~50% defect tiles.
    targets = np.array(train_ds.targets)
    hn = np.array([Path(p).name.startswith("hn_") for p, _ in train_ds.samples])
    rel = np.where(hn, args.hn_weight, 1.0)  # within-class weight
    class_mass = np.array([rel[targets == k].sum() for k in (0, 1)])
    w = rel / class_mass[targets]
    print(f"train tiles: {len(targets)} ({(targets == 1).sum()} defect, {hn.sum()} hard negatives x{args.hn_weight})")
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(targets),
                                    replacement=True)
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=args.workers,
                          pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, args.batch * 2, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(args.arch, pretrained=args.init is None)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["state_dict"])
        print(f"fine-tuning from {args.init}")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.BCEWithLogitsLoss()

    out_dir = C.MODELS_DIR
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"tile_cls_{args.arch}.pt"
    hist, best = [], -1
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tl, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            n += len(y)

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for x, y in val_dl:
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
            torch.save({"arch": args.arch, "state_dict": model.state_dict(), "epoch": ep + 1, "val_ap": ap_},
                       ckpt_path)
    (C.RUNS_DIR / f"tile_cls_{args.arch}").mkdir(parents=True, exist_ok=True)
    (C.RUNS_DIR / f"tile_cls_{args.arch}" / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"best val AP {best:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
