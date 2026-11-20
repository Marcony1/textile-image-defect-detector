"""Towel region-of-interest (ROI) segmentation.

The towels are dyed (saturated colour) while the conveyor / background /
machine parts are black, grey or white (low saturation). A saturation
threshold + largest connected component gives a robust towel mask. It is
used to ignore anomalies outside the towel (PatchCore) and to skip tiles that
contain only background.

Limitation: a white / grey towel would need a different rule (e.g. a fixed
ROI from the camera calibration).
"""
from __future__ import annotations

import cv2
import numpy as np


def towel_mask(img: np.ndarray, downsample: int = 8, margin_px: int = 96) -> np.ndarray:
    """Return a boolean mask (H, W) of the towel, dilated by ``margin_px``."""
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // downsample, H // downsample), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    m = ((s > 70) & (v > 25)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones((H, W), bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    # fill holes (dark stains / labels inside the towel must stay inside)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = cv2.drawContours(np.zeros_like(m), cnts, -1, 1, thickness=-1)
    r = max(1, margin_px // downsample)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def tile_coverage(mask: np.ndarray, win) -> float:
    x1, y1, x2, y2 = win
    return float(mask[y1:y2, x1:x2].mean())
