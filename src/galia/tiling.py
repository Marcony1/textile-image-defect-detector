"""Tiling helpers shared by training-data generation and production inference."""
from __future__ import annotations

import numpy as np

from . import config as C


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Cover the image with tile x tile windows (last row/col snapped to the border)."""
    stride = max(1, int(tile * (1 - overlap)))

    def starts(n):
        if n <= tile:
            return [0]
        s = list(range(0, n - tile, stride))
        s.append(n - tile)
        return s

    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height) for x in starts(width)]


def crop(img: np.ndarray, win: tuple[int, int, int, int], tile: int) -> np.ndarray:
    """Crop a window and zero-pad to tile x tile if the image is smaller."""
    x1, y1, x2, y2 = win
    out = img[y1:y2, x1:x2]
    if out.shape[0] != tile or out.shape[1] != tile:
        pad = np.zeros((tile, tile, 3), dtype=img.dtype)
        pad[: out.shape[0], : out.shape[1]] = out
        out = pad
    return out


def boxes_in_tile(boxes: np.ndarray, win, min_vis: float = C.MIN_BOX_VISIBILITY) -> np.ndarray:
    """Clip pixel boxes [cls,x1,y1,x2,y2] to a window; return tile-relative boxes.

    A box is kept if enough of it is visible, or if the visible part is itself
    large (long defects such as dye streaks cross several tiles).
    """
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = win
    cx1 = np.clip(boxes[:, 1], x1, x2)
    cy1 = np.clip(boxes[:, 2], y1, y2)
    cx2 = np.clip(boxes[:, 3], x1, x2)
    cy2 = np.clip(boxes[:, 4], y1, y2)
    vw, vh = cx2 - cx1, cy2 - cy1
    ow, oh = boxes[:, 3] - boxes[:, 1], boxes[:, 4] - boxes[:, 2]
    vis = (vw * vh) / np.maximum(ow * oh, 1e-6)
    big_enough = (vw >= np.minimum(ow, 64)) & (vh >= np.minimum(oh, 64))
    keep = (vw > 2) & (vh > 2) & ((vis >= min_vis) | big_enough)
    out = np.stack([boxes[:, 0], cx1 - x1, cy1 - y1, cx2 - x1, cy2 - y1], 1)[keep]
    return out


def tile_has_partial_defect(boxes: np.ndarray, win) -> bool:
    """True if a defect box touches the tile but was NOT kept (ambiguous tile)."""
    if len(boxes) == 0:
        return False
    d = boxes[np.isin(boxes[:, 0], C.DEFECT_CLASS_IDS)]
    if len(d) == 0:
        return False
    x1, y1, x2, y2 = win
    touches = (d[:, 3] > x1) & (d[:, 1] < x2) & (d[:, 4] > y1) & (d[:, 2] < y2)
    kept = boxes_in_tile(d, win)
    return bool(touches.sum() > len(kept))


def nms_merge(dets: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Class-aware NMS over detections [x1,y1,x2,y2,conf,cls] gathered from tiles."""
    if len(dets) == 0:
        return dets
    import torch
    from torchvision.ops import batched_nms

    t = torch.as_tensor(dets, dtype=torch.float32)
    keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), iou_thr).numpy()
    return dets[keep]
