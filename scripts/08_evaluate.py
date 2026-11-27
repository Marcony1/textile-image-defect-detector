"""Towel-level evaluation of every model + ensemble selection.

1. Runs each trained model on every val and test towel (full photo, tiled
   exactly as in production) and caches the scores in reports/scores_<split>.csv
2. Fits the ensembles and thresholds on VALIDATION only.
3. Reports metrics on the untouched TEST towels -> reports/metrics.json/.md
4. Saves the best ensemble (highest specificity at the target recall on
   validation) to models/ensemble.json, used by galia.inference.

    python scripts/08_evaluate.py                 # score + report
    python scripts/08_evaluate.py --report-only   # reuse cached scores
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.data import load_image, load_manifest  # noqa: E402
from galia.ensemble import MODEL_COLS, fit_ensembles, metrics_at  # noqa: E402
from galia.inference import DEFAULT_FILES  # noqa: E402

PROD_CAPTURE = "2025-06"
TARGET = "recall_98"


def score_split(split: str, models: list[str]) -> pd.DataFrame:
    from galia.models import ClassifierTilePredictor, PatchcorePredictor, YoloTilePredictor

    builders = {"yolo": YoloTilePredictor, "cls": ClassifierTilePredictor, "patchcore": PatchcorePredictor}
    preds = {m: builders[m](C.MODELS_DIR / DEFAULT_FILES[m]) for m in models}
    df = load_manifest()
    df = df[df.split == split].reset_index(drop=True)
    out_csv = C.REPORTS_DIR / f"scores_{split}.csv"
    prev = pd.read_csv(out_csv, keep_default_na=False) if out_csv.exists() else None
    dets = {}
    rows = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        img, scale = load_image(r.path, r.source)
        row = dict(image_id=r.image_id, source=r.source, capture=r.capture, label=r.label,
                   visibility=r.visibility, category=r.category, scale=scale)
        for m, p in preds.items():
            res = p(img)
            row[m] = res.score
            if m == "yolo":
                dets[r.image_id] = res.detections.round(4).tolist()
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {split}: {i + 1}/{len(df)}  ({(time.time() - t0) / (i + 1):.2f}s/towel)", flush=True)
    new = pd.DataFrame(rows)
    if prev is not None:  # keep scores of models not re-run
        keep = [c for c in MODEL_COLS if c in prev.columns and c not in models]
        new = new.merge(prev[["image_id"] + keep], on="image_id", how="left")
    new.to_csv(out_csv, index=False)
    if "yolo" in models:
        (C.REPORTS_DIR / f"yolo_detections_{split}.json").write_text(json.dumps(dets))
    return new


def breakdown(df, s, thr, col):
    out = {}
    for k, g in df[df.label == "NG"].groupby(col):
        if k == "":
            continue
        out[k] = dict(n=len(g), recall=float((s[g.index] >= thr).mean()))
    return out


def report():
    val = pd.read_csv(C.REPORTS_DIR / "scores_val.csv", keep_default_na=False)
    test = pd.read_csv(C.REPORTS_DIR / "scores_test.csv", keep_default_na=False)
    models = [m for m in MODEL_COLS if m in val.columns and m in test.columns]
    ens = fit_ensembles(val, models)

    yv = (val.label == "NG").astype(int).to_numpy()
    yt = (test.label == "NG").astype(int).to_numpy()
    results, lines = {}, []
    lines.append("| Método | Umbral | Recall NO OK (val) | Falsas alarmas OK (val) | Recall NO OK (test) | "
                 "Falsas alarmas OK (test) | Precisión NO OK (test) | AUC (test) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, e in ens.items():
        sv, st = e.score(val), e.score(test)
        results[name] = {}
        for op, thr in e.thresholds.items():
            mv, mt = metrics_at(sv, yv, thr), metrics_at(st, yt, thr)
            mt["by_source"] = breakdown(test, pd.Series(st), thr, "source")
            mt["by_visibility"] = breakdown(test[test.source == "A"], pd.Series(st), thr, "visibility")
            mt["by_category"] = breakdown(test, pd.Series(st), thr, "category")
            ok_src = {}
            for k, g in test[test.label == "OK"].groupby("source"):
                ok_src[k] = dict(n=len(g), false_alarm=float((st[g.index] >= thr).mean()))
            mt["false_alarm_by_source"] = ok_src
            prod = (test.capture == PROD_CAPTURE).to_numpy()
            mp = metrics_at(st[prod], yt[prod], thr)
            pv = (val.capture == PROD_CAPTURE).to_numpy()
            results[name][op] = dict(val=mv, test=mt, test_produccion=mp, val_produccion=metrics_at(sv[pv], yv[pv], thr))
            if op == TARGET:
                lines.append(f"| {name} | {thr:.4g} | {mv['recall_ng']:.1%} | {mv['false_alarm_rate_ok']:.1%} | "
                             f"{mt['recall_ng']:.1%} ({mt['tp']}/{mt['tp'] + mt['fn']}) | "
                             f"{mt['false_alarm_rate_ok']:.1%} ({mt['fp']}/{mt['fp'] + mt['tn']}) | "
                             f"{mt['precision_ng']:.1%} | {mt['auc']:.3f} |")

    # choose on VALIDATION: lowest false-alarm rate at the target recall
    best = min(ens, key=lambda n: results[n][TARGET]["val"]["false_alarm_rate_ok"]
               - 1e-3 * results[n][TARGET]["val"]["auc"])
    ens[best].save(C.MODELS_DIR / "ensemble.json")
    for name, e in ens.items():
        e.save(C.REPORTS_DIR / f"ensemble_{name}.json")
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(dict(selected=best, target=TARGET, results=results),
                                                           indent=2, ensure_ascii=False), encoding="utf-8")
    sel = results[best]
    prod_lines = ["| Punto de operación | Recall NO OK | Falsas alarmas OK | Precisión NO OK |", "|---|---|---|---|"]
    for op, r in sel.items():
        m = r["test_produccion"]
        prod_lines.append(f"| {op} | {m['recall_ng']:.1%} ({m['tp']}/{m['tp'] + m['fn']}) | "
                          f"{m['false_alarm_rate_ok']:.1%} ({m['fp']}/{m['fp'] + m['tn']}) | {m['precision_ng']:.1%} |")
    md = [f"# Resultados a nivel toalla (objetivo: {TARGET.replace('_', ' ')}% en validación)", "",
          f"Val: {len(val)} toallas ({yv.sum()} NO OK). Test: {len(test)} toallas ({yt.sum()} NO OK).", "",
          *lines, "", f"**Seleccionado en validación:** `{best}` -> `models/ensemble.json`", "",
          f"## Cámara de producción ({PROD_CAPTURE}), test, método `{best}`", "",
          f"{int((test.capture == PROD_CAPTURE).sum())} toallas "
          f"({int(((test.capture == PROD_CAPTURE) & (test.label == 'NG')).sum())} NO OK).", "", *prod_lines]
    (C.REPORTS_DIR / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, help="subset of models to (re)score")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    C.REPORTS_DIR.mkdir(exist_ok=True)
    if not args.report_only:
        models = args.models or [m for m in MODEL_COLS if (C.MODELS_DIR / DEFAULT_FILES[m]).exists()]
        print("scoring with", models)
        for split in ("val", "test"):
            score_split(split, models)
    report()
