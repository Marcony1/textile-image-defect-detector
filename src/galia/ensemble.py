"""Combine the towel scores of the three models and pick operating thresholds.

Each model's raw score lives on its own scale, and that scale also shifts with
the camera set-up (lighting, distance, sensor). So every score is standardised
**per camera** against the OK towels of that camera in the validation set
(robust z-score: (s - median_OK) / IQR_OK). A new camera only needs a few dozen
OK photos to be calibrated. Two combinations are fitted on validation:

    "max"       OR-rule: the towel is NO OK if ANY model finds it unusual
    "logistic"  logistic regression on the standardised scores (stacking)

The threshold is chosen on validation for a target recall of NO OK towels
(the business priority: miss as few defective towels as possible).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

MODEL_COLS = ["yolo", "cls", "patchcore"]


@dataclass
class Ensemble:
    models: list
    norm: dict          # capture -> model -> (median_ok, iqr_ok)
    method: str         # "max" | "logistic" | single model name
    coef: list | None = None
    intercept: float | None = None
    thresholds: dict | None = None  # name -> threshold on the ensemble score
    default_capture: str = "2025-06"  # production camera

    def standardise(self, df: pd.DataFrame) -> np.ndarray:
        caps = df["capture"] if "capture" in df else pd.Series([self.default_capture] * len(df), index=df.index)
        z = np.zeros((len(df), len(self.models)))
        for cap in caps.unique():
            idx = np.where(caps.to_numpy() == cap)[0]
            nrm = self.norm.get(cap, self.norm[self.default_capture])
            for j, m in enumerate(self.models):
                z[idx, j] = (df[m].to_numpy()[idx] - nrm[m][0]) / nrm[m][1]
        return z

    def score(self, df: pd.DataFrame) -> np.ndarray:
        z = self.standardise(df)
        if self.method in MODEL_COLS:
            return z[:, 0]
        if self.method == "max":
            return z.max(1)
        logit = z @ np.asarray(self.coef) + self.intercept
        return 1 / (1 + np.exp(-logit))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def fit_norm(val: pd.DataFrame, models) -> dict:
    """capture -> model -> (median, IQR) of the OK towels of that camera."""
    out = {}
    for cap, g in val[val.label == "OK"].groupby("capture"):
        out[cap] = {}
        for m in models:
            q1, med, q3 = np.percentile(g[m], [25, 50, 75])
            out[cap][m] = (float(med), float(max(q3 - q1, 1e-6)))
    return out


def threshold_for_recall(scores, y, target_recall: float) -> float:
    """Largest threshold whose recall on (scores, y) is >= target_recall."""
    pos = np.sort(scores[y == 1])[::-1]
    k = int(np.ceil(target_recall * len(pos))) - 1
    return float(pos[max(k, 0)])


def metrics_at(scores, y, thr) -> dict:
    pred = scores >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    return dict(threshold=float(thr), recall_ng=tp / max(tp + fn, 1), false_alarm_rate_ok=fp / max(fp + tn, 1),
                precision_ng=tp / max(tp + fp, 1), accuracy=(tp + tn) / len(y), tp=tp, fn=fn, fp=fp, tn=tn,
                auc=float(roc_auc_score(y, scores)) if len(set(y)) > 1 else float("nan"))


def fit_ensembles(val: pd.DataFrame, models=MODEL_COLS, targets=(0.90, 0.95, 0.98)) -> dict[str, Ensemble]:
    y = (val.label == "NG").astype(int).to_numpy()
    norm = fit_norm(val, models)
    out = {}
    for m in models:
        out[m] = Ensemble(models=[m], norm={c: {m: v[m]} for c, v in norm.items()}, method=m)
    out["max"] = Ensemble(models=list(models), norm=norm, method="max")
    lr_proto = Ensemble(models=list(models), norm=norm, method="logistic")
    z = lr_proto.standardise(val)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    # thresholds come from out-of-fold predictions so they are not optimistic
    oof = cross_val_predict(lr, z, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    lr.fit(z, y)
    lr_proto.coef, lr_proto.intercept = lr.coef_[0].tolist(), float(lr.intercept_[0])
    out["logistic"] = lr_proto

    for name, ens in out.items():
        s = oof if name == "logistic" else ens.score(val)
        ens.thresholds = {f"recall_{int(t * 100)}": threshold_for_recall(s, y, t) for t in targets}
    return out
