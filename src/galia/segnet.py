"""Box-supervised tile segmenter (alternative to the tile classifier).

Idea taken from the ZJU-Leaper benchmark (Zhang et al., IEEE TAI 2020): with the
same images, training with box / mask annotations beats training with image-level
labels. Our towels only have boxes, so the box interiors are used as a coarse mask.

The network predicts a defect map at 1/4 resolution (128 x 128 for a 512 tile).
``forward`` returns ONE logit per tile (mean of the top 0.5 % pixel logits), so it
is a drop-in replacement for the classifier: ClassifierTilePredictor, the heat-map,
the towel aggregation and the evaluation scripts work unchanged.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import config as C

TOPK_FRAC = 0.005


class SegTileNet(nn.Module):
    def __init__(self, arch: str = "efficientnet_b0", pretrained: bool = True, dim: int = 96):
        super().__init__()
        import timm

        self.enc = timm.create_model(arch, features_only=True, pretrained=pretrained, out_indices=(1, 2, 3, 4))
        chs = self.enc.feature_info.channels()
        self.lat = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in chs)
        self.head = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.SiLU(),
                                  nn.Conv2d(dim, 1, 1))

    def forward_mask(self, x: torch.Tensor) -> torch.Tensor:
        f = self.enc(x)  # strides 4, 8, 16, 32
        y = self.lat[-1](f[-1])
        for i in range(len(f) - 2, -1, -1):  # top-down feature pyramid
            y = F.interpolate(y, size=f[i].shape[-2:], mode="nearest") + self.lat[i](f[i])
        return self.head(y)  # (B, 1, H/4, W/4) logits

    @staticmethod
    def tile_logit(mask_logits: torch.Tensor) -> torch.Tensor:
        flat = mask_logits.flatten(1)
        k = max(1, int(flat.shape[1] * TOPK_FRAC))
        return flat.topk(k, dim=1).values.mean(1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tile_logit(self.forward_mask(x))


# ----------------------------------------------------------------- training data
def tile_boxes_index(cache: Path = C.DATA_DIR / "cls_tiles" / "tile_boxes.json") -> dict:
    """{tile file name: [[x1,y1,x2,y2], ...]} for every DEFECT tile, rebuilt from the
    manifest and the tile name (<image_id>_t<k>.jpg) exactly as 04_build_dataset made it."""
    if cache.exists():
        return json.loads(cache.read_text())
    from .data import load_boxes, load_manifest
    from .tiling import boxes_in_tile, tile_grid

    df = load_manifest().set_index("image_id")
    out = {}
    for split in ("train", "val"):
        for p in (C.CLS_TILES_DIR / split / "defect").glob("*.jpg"):
            iid, k = p.stem.rsplit("_t", 1)
            r = df.loc[iid]
            s = C.SOURCE_SCALE[r["source"]]
            W, H = round(int(r["width"]) * s), round(int(r["height"]) * s)
            boxes = load_boxes(r["label_path"], int(r["width"]), int(r["height"]))
            boxes[:, 1:] *= s
            win = tile_grid(W, H, C.CLS_TILE, C.CLS_OVERLAP)[int(k)]
            tb = boxes_in_tile(boxes, win)
            tb = tb[np.isin(tb[:, 0], C.DEFECT_CLASS_IDS)]
            out[p.name] = [[round(float(v), 1) for v in b[1:]] for b in tb]
    cache.write_text(json.dumps(out))
    return out


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SegTiles(torch.utils.data.Dataset):
    """cls_tiles/<split>/{clean,defect} images + box masks; same flips / rotations / colour
    jitter as the classifier's basic recipe, applied jointly to image and mask."""

    def __init__(self, split: str, boxes: dict, train: bool):
        self.samples = [(p, 0) for p in sorted((C.CLS_TILES_DIR / split / "clean").iterdir())] + \
                       [(p, 1) for p in sorted((C.CLS_TILES_DIR / split / "defect").iterdir())]
        self.targets = [t for _, t in self.samples]
        self.boxes, self.train = boxes, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
        m = torch.zeros(1, *x.shape[1:])
        for x1, y1, x2, y2 in self.boxes.get(p.name, []) if y else []:
            m[:, int(y1):max(int(y1) + 1, int(np.ceil(y2))), int(x1):max(int(x1) + 1, int(np.ceil(x2)))] = 1
        if self.train:
            if random.random() < 0.5:
                x, m = x.flip(2), m.flip(2)
            if random.random() < 0.5:
                x, m = x.flip(1), m.flip(1)
            k = random.randint(0, 3)
            x, m = torch.rot90(x, k, (1, 2)), torch.rot90(m, k, (1, 2))
            b, c = 1 + random.uniform(-.25, .25), 1 + random.uniform(-.25, .25)
            mu = x.mean()
            x = ((x - mu) * c + mu) * b
            g = x.mean(0, keepdim=True)
            x = (g + (x - g) * (1 + random.uniform(-.15, .15))).clamp(0, 1)
        m = F.avg_pool2d(m[None], 4)[0]  # mask at 1/4 resolution, soft at box borders
        return (x - MEAN) / STD, m, torch.tensor(float(y))
