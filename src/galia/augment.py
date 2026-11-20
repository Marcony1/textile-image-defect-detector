"""Augmentations for the tile classifier.

* ``strong_transform``: photometric and geometric changes that a real towel photo
  can show (light level, contrast, focus, sensor noise, orientation) but that do
  NOT erase a defect: hue is changed only a little (a dye defect is a colour
  difference *relative* to the rest of the towel, so a tiny global shift keeps
  it), and crops keep >= 85 % of the tile so a defect does not fall outside.
* ``CopyPasteFolder``: defect copy-paste. Real defect crops from TRAIN towels are
  pasted onto clean tiles (colour matched to the destination, feathered edges)
  and the tile becomes a "defect" example. This multiplies the variety of
  backgrounds each defect is seen on, which matters most for the rare classes
  (Hilo, Tejido, Costura). Val/test towels never contribute crops.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from . import config as C

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BANK_DIR = C.DATA_DIR / "cls_defect_bank"


class RandomRot90:
    def __call__(self, x):
        return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))


class GaussNoise:
    def __init__(self, sigma=0.02, p=0.3):
        self.sigma, self.p = sigma, p

    def __call__(self, x):
        if random.random() < self.p:
            x = (x + torch.randn_like(x) * self.sigma).clamp(0, 1)
        return x


def basic_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(), RandomRot90(), transforms.Normalize(MEAN, STD),
    ])


def strong_transform(size=C.CLS_TILE):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.2))], p=0.2),
        transforms.RandomAdjustSharpness(1.8, p=0.2),
        transforms.ToTensor(), GaussNoise(0.02, 0.3), RandomRot90(), transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# --------------------------------------------------------------- defect bank
def build_defect_bank(margin=6, max_side=320, min_side=6, verbose=True) -> list[Path]:
    """Crop every defect box of the TRAIN towels once (cached in data/cls_defect_bank)."""
    from .data import load_boxes, load_image, load_manifest

    out = BANK_DIR / "train"
    files = sorted(out.glob("*.png"))
    if files:
        return files
    out.mkdir(parents=True, exist_ok=True)
    df = load_manifest()
    df = df[(df.split == "train") & (df.n_defect_boxes > 0)]
    for i, r in enumerate(df.itertuples()):
        img, s = load_image(r.path, r.source)
        H, W = img.shape[:2]
        for j, (c, x1, y1, x2, y2) in enumerate(load_boxes(r.label_path, r.width, r.height)):
            if int(c) not in C.DEFECT_CLASS_IDS:
                continue
            x1, y1, x2, y2 = [int(round(v * s)) for v in (x1, y1, x2, y2)]
            w, h = x2 - x1, y2 - y1
            if max(w, h) > max_side or min(w, h) < min_side:
                continue
            a, b = max(0, x1 - margin), max(0, y1 - margin)
            crop = img[b:min(H, y2 + margin), a:min(W, x2 + margin)]
            Image.fromarray(crop).save(out / f"{C.CLASSES[int(c)]}_{r.image_id}_{j:02d}.png")
        if verbose and (i + 1) % 50 == 0:
            print(f"  defect bank: {i + 1}/{len(df)} towels", flush=True)
    return sorted(out.glob("*.png"))


def _feather(h, w, m):
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    return np.clip(np.minimum(yy, xx) / max(m, 1), 0, 1).astype(np.float32)


def _on_towel(px: np.ndarray, min_frac=0.85) -> bool:
    """True if the pixels look like towel fabric (saturated colour), not machine/background."""
    px = px.reshape(-1, 3).astype(np.float32)
    mx = px.max(1)
    sat = (mx - px.min(1)) / np.maximum(mx, 1)
    return bool(((sat > 0.25) & (mx > 40)).mean() >= min_frac)


def paste_defect(tile: np.ndarray, patch: np.ndarray, margin=6, rng=random) -> np.ndarray | None:
    """Paste ``patch`` at a random place of ``tile`` that lies on the towel. Returns None if no place fits."""
    k = rng.randrange(4)
    patch = np.rot90(patch, k)
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    ph, pw = patch.shape[:2]
    T = tile.shape[0]
    ring = np.ones((ph, pw), bool)
    ring[margin:-margin or None, margin:-margin or None] = False
    if ring.sum() < 4:
        ring[:] = True
    if not _on_towel(patch[ring], 0.7):  # defect crop taken at the towel edge: skip
        return None
    if ph >= T - 8 or pw >= T - 8:
        return None
    for _ in range(6):
        y, x = rng.randrange(4, T - ph - 4), rng.randrange(4, T - pw - 4)
        dst = tile[y:y + ph, x:x + pw].astype(np.float32)
        if not _on_towel(dst):  # background / machine, not towel
            continue
        p = patch.astype(np.float32)
        p = p - p[ring].mean(0) + dst[ring].mean(0)  # colour match: keep the defect's relative contrast
        a = _feather(ph, pw, margin)[..., None]
        out = tile.copy()
        out[y:y + ph, x:x + pw] = np.clip(dst * (1 - a) + p * a, 0, 255).astype(np.uint8)
        return out
    return None


class CopyPasteFolder(datasets.ImageFolder):
    """ImageFolder where a clean tile becomes a defect tile with probability ``p``."""

    def __init__(self, root, transform, bank: list[Path], p=0.3):
        super().__init__(root, transform=transform)
        self.bank, self.p = bank, p

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = self.loader(path)
        if y == 0 and self.bank and random.random() < self.p:
            patch = np.asarray(Image.open(random.choice(self.bank)).convert("RGB"))
            out = None
            for _ in range(3):
                out = paste_defect(np.asarray(img), patch)
                if out is not None:
                    break
                patch = np.asarray(Image.open(random.choice(self.bank)).convert("RGB"))
            if out is not None:
                img, y = Image.fromarray(out), 1
        return self.transform(img), y
