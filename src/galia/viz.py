"""Plot helpers shared by the notebooks (consistent palette and styling)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from . import config as C

# Validated categorical palette, assigned in fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LABEL_COLORS = {"OK": PALETTE[0], "NG": PALETTE[1]}
MODEL_COLORS = {"yolo": PALETTE[0], "cls": PALETTE[1], "patchcore": PALETTE[2],
                "max": PALETTE[3], "logistic": PALETTE[4]}
MODEL_NAMES = {"yolo": "YOLO11s (recortes)", "cls": "EfficientNet-B0 (recortes)", "patchcore": "PatchCore",
               "max": "Ensamble (máx.)", "logistic": "Ensamble (logístico)"}
# box colours on photos: bright, readable on blue fabric
BOX_COLORS = {"Etiqueta": "#ffffff", "Mancha": "#ff3b30", "Hilo": "#ffd60a", "Costura": "#ff9f0a",
              "Tejido": "#bf5af2", "Tenido": "#64d2ff", "Empalme": "#30d158"}


def setup():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=PALETTE), "legend.frameon": False,
        "axes.titleweight": "bold", "axes.titlesize": 11, "lines.linewidth": 2,
    })


def draw_boxes(img: np.ndarray, boxes, names=C.CLASSES, width: int | None = None, show_conf=True,
               min_conf: float = 0.0) -> Image.Image:
    """boxes: iterable of [x1,y1,x2,y2,conf,cls] or [cls,x1,y1,x2,y2] (gt, conf=None)."""
    im = Image.fromarray(img).copy()
    d = ImageDraw.Draw(im)
    lw = width or max(3, im.width // 500)
    fs = max(18, im.width // 70)
    for b in boxes:
        if len(b) == 6:
            x1, y1, x2, y2, conf, c = b
            if conf < min_conf:
                continue
            txt = f"{names[int(c)]} {conf:.2f}" if show_conf else names[int(c)]
        else:
            c, x1, y1, x2, y2 = b
            txt = names[int(c)]
        col = BOX_COLORS.get(names[int(c)], "#ff3b30")
        pad = lw * 3
        d.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], outline=col, width=lw)
        d.text((x1 - pad, max(0, y1 - pad - fs - 4)), txt, fill=col, font_size=fs)
    return im


def overlay_heatmap(ax, img: np.ndarray, heat: np.ndarray, vmax=None, alpha=0.5, title=None):
    ax.imshow(img)
    ax.imshow(heat, cmap="inferno", alpha=alpha, extent=(0, img.shape[1], img.shape[0], 0),
              vmin=0, vmax=vmax or max(float(heat.max()), 1e-6), interpolation="bilinear")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
