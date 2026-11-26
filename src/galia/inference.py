"""Production inference: one photo per towel -> OK / NO OK decision.

    from galia.inference import TowelInspector
    inspector = TowelInspector.from_dir("models")
    result = inspector.inspect("photo.jpg")          # scale defaults to 1.0 (production camera)
    result["decision"], result["score"], result["defects"]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C
from .ensemble import Ensemble, towel_probability  # noqa: F401  (re-exported)
from .models import ClassifierTilePredictor, PatchcorePredictor, YoloTilePredictor

DEFAULT_FILES = {
    "yolo": "yolo11s_tiles.pt",
    "cls": "tile_cls_ensemble.json",
    "patchcore": "patchcore_wrn50.pt",
    "ensemble": "ensemble.json",
}


class TowelInspector:
    def __init__(self, predictors: dict, ensemble: Ensemble, operating_point: str = "recall_98"):
        self.predictors = predictors
        self.ensemble = ensemble
        self.operating_point = operating_point
        self.threshold = ensemble.thresholds[operating_point]

    calibration: dict | None = None  # contents of calibracion.json, if any
    prob_cal: dict | None = None     # "probabilidad" block of curva_operacion.json, if any
    default_scale: float = 1.0

    @classmethod
    def from_dir(cls, models_dir=C.MODELS_DIR, operating_point: str = "recall_98", use_calibration: bool = True):
        """Load the models. If ``calibracion.json`` exists (see galia.calibration) its
        score baseline and photo scale replace the factory ones (June-2025 camera)."""
        import json

        d = Path(models_dir)
        ens = Ensemble.load(d / DEFAULT_FILES["ensemble"])
        builders = {"yolo": YoloTilePredictor, "cls": ClassifierTilePredictor, "patchcore": PatchcorePredictor}
        preds = {m: builders[m](d / DEFAULT_FILES[m]) for m in ens.models}
        if "yolo" not in preds and (d / DEFAULT_FILES["yolo"]).exists():
            preds["yolo"] = YoloTilePredictor(d / DEFAULT_FILES["yolo"])  # still useful to locate defects
        insp = cls(preds, ens, operating_point)
        cal_path = d / "calibracion.json"
        if use_calibration and cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            ens.norm[cal["capture"]] = cal["norm"]
            ens.default_capture = cal["capture"]
            insp.calibration, insp.default_scale = cal, float(cal.get("scale", 1.0))
        curve = d / "curva_operacion.json"
        if curve.exists():
            insp.prob_cal = json.loads(curve.read_text(encoding="utf-8")).get("probabilidad")
        return insp

    def probability(self, z: float) -> float | None:
        """Orientative P(defect) for score ``z`` with the active camera calibration."""
        ens = self.ensemble
        return towel_probability(z, self.prob_cal, ens.norm[ens.default_capture][ens.models[0]])

    def inspect(self, image, scale: float | None = None) -> dict:
        """``image``: path or RGB array from the production camera.

        ``scale`` resizes the photo to the pixel scale the models were trained
        on (1.0 for the June-2025 / production set-up, 0.62 for the April-2025
        5120x5120 camera).
        """
        scale = self.default_scale if scale is None else scale
        if not isinstance(image, np.ndarray):
            im = Image.open(image).convert("RGB")
            if scale != 1.0:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
            image = np.asarray(im)
        raw = {m: p(image) for m, p in self.predictors.items()}
        row = pd.DataFrame([{**{m: r.score for m, r in raw.items()}, "capture": self.ensemble.default_capture}])
        score = float(self.ensemble.score(row)[0])
        defects = []
        if "yolo" in raw:
            for x1, y1, x2, y2, conf, cls in raw["yolo"].detections:
                if conf >= 0.10:
                    defects.append(dict(cls=C.CLASSES[int(cls)], conf=float(conf),
                                        box=[float(v) / scale for v in (x1, y1, x2, y2)]))
        # probabilities shown to the operator: per tile (sigmoid of the networks) and per towel (guide)
        tiles = []
        if "cls" in raw and raw["cls"].tile_scores:
            tiles = [[round(v / scale) for v in w[:4]] + [round(float(p), 4)] for w, p in raw["cls"].tile_scores]
        probs = dict(towel=self.probability(score), tiles=tiles,
                     size=[round(image.shape[1] / scale), round(image.shape[0] / scale)])
        return dict(decision="NO OK" if score >= self.threshold else "OK", score=score,
                    threshold=self.threshold, model_scores={m: r.score for m, r in raw.items()},
                    defects=defects, probs=probs, raw=raw)
