"""Calibrate the inspector to a new camera / lighting set-up.

The models were trained on photos where the towel label measures ~99 px (as YOLO boxes it)
(sqrt of its box area) and the decision threshold is a robust z-score against
the GOOD towels of the same camera. A new camera or new lighting changes both,
so before a real trial we need ~30 photos of GOOD towels taken with the new
set-up:

1. **Scale**: YOLO finds the towel label in each photo; scale = 99 / label size.
2. **Score baseline**: the classifier scores those photos at that scale; their
   median and IQR replace the factory baseline (June-2025 camera).

The result is written to ``<models>/calibracion.json`` and picked up
automatically by :class:`galia.inference.TowelInspector`. Delete the file to go
back to the factory calibration.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C

REF_LABEL_PX = 99.0          # median YOLO label size (sqrt of box area) on train+val photos of the June-2025 camera (scale 1.0)
CANDIDATE_SCALES = (1.0, 0.6, 1.6, 0.4, 2.5)
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CAL_FILE = "calibracion.json"


def list_photos(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)


def _load(path: Path, scale: float) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
    return np.asarray(im)


def _best_label(yolo, path: Path, s: float):
    det = yolo(_load(path, s)).detections
    labels = [d for d in det if int(d[5]) == 0 and d[4] >= 0.35]
    if not labels:
        return None
    x1, y1, x2, y2, conf, _ = max(labels, key=lambda d: d[4])
    return float(np.sqrt((x2 - x1) * (y2 - y1))) / s, float(conf)  # native pixels


def label_size(yolo, path: Path) -> float | None:
    """Label size (native pixels) of the most confident 'Etiqueta' detection.

    First pass: try several scales until the label is found. Second pass: measure
    again at the scale where the label should look like in training (~99 px),
    because boxes of labels much larger/smaller than in training are less accurate.
    """
    best = None
    for s in CANDIDATE_SCALES:
        r = _best_label(yolo, path, s)
        if r and (best is None or r[1] > best[1]):
            best = r
        if r and r[1] >= 0.6:
            break
    if best is None:
        return None
    s2 = float(np.clip(REF_LABEL_PX / best[0], 0.2, 3.0))
    r2 = _best_label(yolo, path, s2)
    return r2[0] if r2 and r2[1] >= 0.35 else best[0]


def calibrate(folder, models_dir=C.MODELS_DIR, name: str = "planta", scale: float | None = None,
              min_photos: int = 15, log=print) -> dict:
    from .inference import TowelInspector
    from .models import YoloTilePredictor
    from .inference import DEFAULT_FILES

    photos = list_photos(folder)
    if len(photos) < min_photos:
        raise ValueError(f"Se necesitan al menos {min_photos} fotos de toallas BUENAS; hay {len(photos)} en {folder}")
    models_dir = Path(models_dir)
    insp = TowelInspector.from_dir(models_dir, use_calibration=False)

    # 1 · scale from the towel label
    sizes = []
    if scale is None:
        yolo = insp.predictors.get("yolo") or YoloTilePredictor(models_dir / DEFAULT_FILES["yolo"])
        sample = photos[:: max(1, len(photos) // 20)][:20]
        log(f"1/2 Midiendo la escala con la etiqueta de la toalla ({len(sample)} fotos)…")
        for i, p in enumerate(sample, 1):
            s = label_size(yolo, p)
            log(f"    {i}/{len(sample)} {p.name}: " + (f"etiqueta {s:.0f} px" if s else "etiqueta no encontrada"))
            if s:
                sizes.append(s)
        if len(sizes) >= 3:
            scale = float(np.clip(REF_LABEL_PX / np.median(sizes), 0.2, 3.0))
            log(f"    -> escala {scale:.3f} (etiqueta mediana {np.median(sizes):.0f} px; referencia {REF_LABEL_PX:.0f} px)")
        else:
            scale = 1.0
            log("    -> no se encontró la etiqueta en suficientes fotos; se usa escala 1.0. "
                "Si la toalla se ve más grande o más chica que en las fotos de demostración, indique la escala a mano.")
    else:
        log(f"1/2 Escala indicada a mano: {scale:.3f}")

    # 2 · baseline of the scores on GOOD towels
    log(f"2/2 Analizando {len(photos)} fotos de toallas buenas…")
    raw = {m: [] for m in insp.ensemble.models}
    for i, p in enumerate(photos, 1):
        img = _load(p, scale)
        for m in insp.ensemble.models:
            raw[m].append(float(insp.predictors[m](img).score))
        if i % 5 == 0 or i == len(photos):
            log(f"    {i}/{len(photos)}")
    norm = {}
    for m, v in raw.items():
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        norm[m] = [float(med), float(max(q3 - q1, 1e-4))]

    # sanity check: how many of these GOOD towels would be flagged at each operating point
    import pandas as pd
    ens = insp.ensemble
    ens.norm[name] = norm
    df = pd.DataFrame({**raw, "capture": name})
    z = ens.score(df)
    flagged = {op: float(np.mean(z >= thr)) for op, thr in ens.thresholds.items()}

    cal = dict(capture=name, scale=scale, norm=norm, n_photos=len(photos), folder=str(Path(folder).resolve()),
               label_px=float(np.median(sizes)) if sizes else None, date=datetime.now().isoformat(timespec="seconds"),
               ok_flagged=flagged)
    (models_dir / CAL_FILE).write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
    log("")
    log(f"Calibración guardada en {models_dir / CAL_FILE}")
    log("Falsas alarmas esperadas en toallas buenas (se miden sobre estas mismas fotos):")
    for op, f in flagged.items():
        log(f"    {op}: {f:.0%}")
    return cal
