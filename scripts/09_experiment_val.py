"""Compare classifier variants on the VALIDATION towels only.

Every model choice (augmentation, architecture, TTA, ensemble, tile aggregation)
is made here. The test split is never read by this script, so the final test
numbers stay an honest estimate. Rules used to avoid overfitting val:

* compare with threshold-free metrics (towel AUC) plus the false-alarm rate at a
  fixed recall, with a stratified bootstrap 95 % interval;
* accept a change only when it improves the production camera (2025-06) and the
  pooled AUC, and the paired bootstrap interval of the gain is mostly above zero;
* prefer the simpler option when results are within noise.

    python scripts/09_experiment_val.py --name b0_base --weights models/tile_cls_efficientnet_b0.pt
    python scripts/09_experiment_val.py --name ens --weights models/a.pt models/b.pt --tta
    python scripts/09_experiment_val.py --report          # table of all experiments
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.data import load_image, load_manifest  # noqa: E402

OUT = C.REPORTS_DIR / "experiments"
PROD = "2025-06"
AGGS = ["max", "top2", "top3"]


def score_val(weights: list[str], tta: bool) -> pd.DataFrame:
    from galia.models import ClassifierTilePredictor

    if len(weights) == 1 and not tta:
        pred = ClassifierTilePredictor(weights[0])
    else:
        cfg = {"members": [str(Path(w).resolve()) for w in weights], "tta": tta, "agg": "max"}
        tmp = Path(tempfile.mkdtemp()) / "ens.json"
        tmp.write_text(json.dumps(cfg))
        pred = ClassifierTilePredictor(tmp)
    df = load_manifest()
    val = df[df.split == "val"]
    rows = []
    for i, r in enumerate(val.itertuples()):
        img, _ = load_image(r.path, r.source)
        res = pred(img)
        p = np.array([s for _, s in res.tile_scores])
        rows.append(dict(image_id=r.image_id, source=r.source, capture=r.capture, label=r.label,
                         category=r.category, **{a: pred.aggregate(p, a) for a in AGGS}))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(val)}", flush=True)
    return pd.DataFrame(rows)


def calibrate(d: pd.DataFrame, col: str) -> np.ndarray:
    """Per-camera robust z-score against the OK towels of that camera (as in galia.ensemble)."""
    z = np.zeros(len(d))
    for cap, g in d.groupby("capture"):
        ok = g.loc[g.label == "OK", col]
        q1, med, q3 = np.percentile(ok, [25, 50, 75])
        z[d.index.get_indexer(g.index)] = (g[col] - med) / max(q3 - q1, 1e-6)
    return z


def fa_at_recall(s, y, r):
    pos = np.sort(s[y == 1])[::-1]
    thr = pos[int(np.ceil(r * len(pos))) - 1]
    return float((s[y == 0] >= thr).mean())


def metrics(d: pd.DataFrame, col: str, n_boot=1000, seed=0) -> dict:
    d = d.reset_index(drop=True)
    y = (d.label == "NG").astype(int).to_numpy()
    z = calibrate(d, col)
    prod = (d.capture == PROD).to_numpy()
    out = dict(auc=roc_auc_score(y, z), auc_prod=roc_auc_score(y[prod], z[prod]),
               fa95=fa_at_recall(z, y, 0.95), fa95_prod=fa_at_recall(z[prod], y[prod], 0.95),
               fa90_prod=fa_at_recall(z[prod], y[prod], 0.90))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(y[idx], z[idx]))
    out["auc_lo"], out["auc_hi"] = np.percentile(boots, [2.5, 97.5])
    return out


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str, n_boot=1000, seed=0):
    """Bootstrap CI of AUC(a) - AUC(b) on the same towels (pooled and production camera)."""
    m = a[["image_id", "label", "capture"]].copy()
    m["za"] = calibrate(a.reset_index(drop=True), col_a)
    bb = b.set_index("image_id").loc[m.image_id].reset_index()
    m["zb"] = calibrate(bb, col_b)
    y = (m.label == "NG").astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    res = {}
    for name, mask in [("all", np.ones(len(m), bool)), ("prod", (m.capture == PROD).to_numpy())]:
        yy, za, zb = y[mask], m.za.to_numpy()[mask], m.zb.to_numpy()[mask]
        pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
        ds = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            ds.append(roc_auc_score(yy[idx], za[idx]) - roc_auc_score(yy[idx], zb[idx]))
        res[name] = (float(np.mean(ds)), *np.percentile(ds, [2.5, 97.5]).tolist(), float(np.mean(np.array(ds) > 0)))
    return res


def report(baseline: str):
    files = sorted(OUT.glob("*_val.csv"))
    rows = []
    base = pd.read_csv(OUT / f"{baseline}_val.csv", keep_default_na=False) if (OUT / f"{baseline}_val.csv").exists() else None
    for f in files:
        name = f.name[:-8]
        d = pd.read_csv(f, keep_default_na=False)
        for agg in AGGS:
            m = metrics(d, agg)
            row = dict(experimento=name, agg=agg, **{k: round(v, 3) for k, v in m.items()})
            if base is not None and name != baseline:
                dl = paired_delta(d, base, agg, "max")
                row["dAUC_prod"] = f"{dl['prod'][0]:+.3f} [{dl['prod'][1]:+.3f},{dl['prod'][2]:+.3f}] P>0={dl['prod'][3]:.2f}"
                row["dAUC_all"] = f"{dl['all'][0]:+.3f} [{dl['all'][1]:+.3f},{dl['all'][2]:+.3f}] P>0={dl['all'][3]:.2f}"
            rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.to_string(index=False))
    t.to_csv(OUT / "summary.csv", index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--weights", nargs="+")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--baseline", default="b0_hn")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args.baseline)
    else:
        d = score_val(args.weights, args.tta)
        d.to_csv(OUT / f"{args.name}_val.csv", index=False)
        (OUT / f"{args.name}.json").write_text(json.dumps(dict(weights=args.weights, tta=args.tta), indent=2))
        for agg in AGGS:
            print(args.name, agg, {k: round(v, 3) for k, v in metrics(d, agg).items()})
