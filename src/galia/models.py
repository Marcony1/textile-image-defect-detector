"""Towel-level predictors. Each one takes ONE full photo of a towel (already
scale-normalised with ``data.load_image``) and returns a defect score; the
higher the score, the more likely the towel is NO OK.

    YoloTilePredictor        max confidence of any defect box (YOLO11 on tiles)
    ClassifierTilePredictor  max P(defect) over tiles (EfficientNet/ResNet)
    PatchcorePredictor       max anomaly score inside the towel (PatchCore)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .roi import tile_coverage, towel_mask
from .tiling import crop, nms_merge, tile_grid

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(tiles: list[np.ndarray], device) -> torch.Tensor:
    x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float().div_(255)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device)


@dataclass
class TowelResult:
    score: float
    detections: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))  # x1,y1,x2,y2,conf,cls
    heatmap: np.ndarray | None = None  # low-res map in [0, inf), same aspect as image
    tile_scores: list | None = None   # [(win, score)]


# --------------------------------------------------------------------------- YOLO
class YoloTilePredictor:
    def __init__(self, weights: str | Path, conf: float = 0.01, batch: int = 8, use_roi: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.batch, self.use_roi = conf, batch, use_roi
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def __call__(self, img: np.ndarray) -> TowelResult:
        H, W = img.shape[:2]
        wins = tile_grid(W, H, C.YOLO_TILE, C.YOLO_OVERLAP)
        if self.use_roi:
            mask = towel_mask(img)
            wins = [w for w in wins if tile_coverage(mask, w) > 0.02] or wins
        dets = []
        for i in range(0, len(wins), self.batch):
            chunk = wins[i:i + self.batch]
            # ultralytics expects BGR numpy arrays
            res = self.model.predict([crop(img, w, C.YOLO_TILE)[..., ::-1] for w in chunk],
                                     imgsz=C.YOLO_TILE, conf=self.conf, device=self.device,
                                     verbose=False, half=torch.cuda.is_available())
            for w, r in zip(chunk, res):
                if len(r.boxes):
                    b = r.boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,cls
                    b[:, [0, 2]] += w[0]
                    b[:, [1, 3]] += w[1]
                    dets.append(b)
        dets = nms_merge(np.concatenate(dets), 0.5) if dets else np.zeros((0, 6))
        is_def = np.isin(dets[:, 5].astype(int), C.DEFECT_CLASS_IDS)
        score = float(dets[is_def, 4].max()) if is_def.any() else 0.0
        return TowelResult(score=score, detections=dets)


# ---------------------------------------------------------------- Tile classifier
def build_classifier(arch: str = "efficientnet_b0", pretrained: bool = True):
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=1)


class ClassifierTilePredictor:
    """Tile classifier applied to the whole towel.

    ``weights`` may be one checkpoint (.pt) or a JSON file describing an ensemble::

        {"members": ["tile_cls_a.pt", "tile_cls_b.pt"], "tta": true, "agg": "max"}

    Tile probabilities are averaged over members and TTA views (identity + flips);
    the towel score aggregates the tiles (``max`` or ``top2``/``top3`` = mean of the k
    most suspicious tiles).
    """

    def __init__(self, weights: str | Path, batch: int = 32, use_roi: bool = True,
                 tta: bool | None = None, agg: str | None = None):
        import json

        weights = Path(weights)
        cfg = {"members": [weights.name], "tta": False, "agg": "max"}
        if weights.suffix == ".json":
            cfg.update(json.loads(weights.read_text(encoding="utf-8")))
        self.tta = cfg["tta"] if tta is None else tta
        self.agg = cfg["agg"] if agg is None else agg
        self.device = _device()
        self.models = []
        for m in cfg["members"]:
            ckpt = torch.load(weights.parent / m, map_location="cpu", weights_only=False)
            net = build_classifier(ckpt["arch"], pretrained=False)
            net.load_state_dict(ckpt["state_dict"])
            self.models.append(net.to(self.device).eval())
        self.batch, self.use_roi = batch, use_roi

    @torch.no_grad()
    def tile_probs(self, x: torch.Tensor) -> torch.Tensor:
        views = [x, x.flip(3), x.flip(2)] if self.tta else [x]
        acc = 0
        for net in self.models:
            for v in views:
                with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    acc = acc + torch.sigmoid(net(v).float()).squeeze(1)
        return (acc / (len(self.models) * len(views))).cpu()

    @staticmethod
    def aggregate(probs: np.ndarray, agg: str = "max") -> float:
        if agg.startswith("top"):
            k = int(agg[3:])
            return float(np.sort(probs)[::-1][:k].mean())
        return float(probs.max())

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> TowelResult:
        H, W = img.shape[:2]
        wins = tile_grid(W, H, C.CLS_TILE, C.CLS_OVERLAP)
        if self.use_roi:
            mask = towel_mask(img)
            wins = [w for w in wins if tile_coverage(mask, w) > 0.02] or wins
        probs = []
        for i in range(0, len(wins), self.batch):
            x = _to_tensor([crop(img, w, C.CLS_TILE) for w in wins[i:i + self.batch]], self.device)
            probs.append(self.tile_probs(x))
        probs = torch.cat(probs).numpy()
        # coarse heat-map (one cell per tile stride) for visualisation
        stride = int(C.CLS_TILE * (1 - C.CLS_OVERLAP))
        heat = np.zeros((H // stride + 1, W // stride + 1), np.float32)
        for w, p in zip(wins, probs):
            ys, xs = slice(w[1] // stride, max(w[1] // stride + 1, w[3] // stride)), slice(w[0] // stride, max(w[0] // stride + 1, w[2] // stride))
            heat[ys, xs] = np.maximum(heat[ys, xs], p)
        return TowelResult(score=self.aggregate(probs, self.agg), heatmap=heat, tile_scores=list(zip(wins, probs.tolist())))


# ----------------------------------------------------------------------- PatchCore
class PatchcorePredictor:
    """Anomalib PatchCore model applied tile by tile inside the towel ROI."""

    def __init__(self, weights: str | Path, batch: int = 16):
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.model = PatchcoreModel(backbone=ckpt["backbone"], layers=ckpt["layers"], pre_trained=True,
                                    num_neighbors=ckpt.get("num_neighbors", 9))
        self.model.memory_bank = ckpt["memory_bank"]
        self.device = _device()
        self.model.to(self.device).eval()
        self.batch = batch

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> TowelResult:
        H, W = img.shape[:2]
        T, S = C.PATCHCORE_TILE, C.PATCHCORE_INPUT
        mask = towel_mask(img)
        wins = [w for w in tile_grid(W, H, T, 0.0) if tile_coverage(mask, w) > 0.02] or tile_grid(W, H, T, 0.0)
        r = S / T
        heat = np.zeros((int(np.ceil(H * r)), int(np.ceil(W * r))), np.float32)
        for i in range(0, len(wins), self.batch):
            chunk = wins[i:i + self.batch]
            x = _to_tensor([crop(img, w, T) for w in chunk], self.device)
            x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
            out = self.model(x)
            amap = out.anomaly_map.squeeze(1).cpu().numpy()
            for w, a in zip(chunk, amap):
                y1, x1 = int(w[1] * r), int(w[0] * r)
                h, ww = min(S, heat.shape[0] - y1), min(S, heat.shape[1] - x1)
                heat[y1:y1 + h, x1:x1 + ww] = np.maximum(heat[y1:y1 + h, x1:x1 + ww], a[:h, :ww])
        # ignore everything outside the towel
        small_mask = torch.from_numpy(mask[None, None].astype(np.float32))
        small_mask = F.interpolate(small_mask, size=heat.shape, mode="nearest")[0, 0].numpy() > 0.5
        heat[~small_mask] = 0
        # plain max: the map is already Gaussian-smoothed by anomalib, and a
        # percentile would wash out the smallest defects (a few pixels here).
        score = float(heat[small_mask].max()) if small_mask.any() else float(heat.max())
        return TowelResult(score=score, heatmap=heat)
