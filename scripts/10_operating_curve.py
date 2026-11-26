"""Curva de operación para el control de exigencia del tablero.

Para cada umbral posible (en unidades del score calibrado por cámara) estima qué
fracción de toallas con defecto se detecta y qué fracción de toallas buenas se
manda a revisión. Usa los scores por toalla de validación y test (reports/
scores_*.csv) con el modelo ya elegido: no selecciona nada, solo describe el
compromiso para que el usuario decida con información.

Además ajusta una probabilidad orientativa por toalla, P(defecto) = sigmoide(a·x + b), con
x = z / z_max (z relativo al máximo que alcanza esa cámara), SOLO con validación; test se usa
únicamente para reportar qué tan bien calibrada queda. Refleja la proporción de defectos de validación
(~32 %): en planta, donde hay menos defectos, la probabilidad real es menor.

    python scripts/10_operating_curve.py      # -> models/curva_operacion.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.ensemble import Ensemble, towel_probability  # noqa: E402

RECALL_RANGE = (0.80, 0.995)  # extremos del control
N_POINTS = 41


def fit_probability(ens: Ensemble) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    val, test = (pd.read_csv(C.REPORTS_DIR / f"scores_{s}.csv", keep_default_na=False) for s in ("val", "test"))
    m = ens.models[0]
    norm = lambda d: [ens.norm[c][m] for c in d.capture]  # noqa: E731
    rel = lambda d: np.array([z * n[1] / (1 - n[0]) for z, n in zip(ens.score(d), norm(d))])  # noqa: E731
    yv, yt = (val.label == "NG").to_numpy().astype(int), (test.label == "NG").to_numpy().astype(int)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(rel(val)[:, None], yv)
    cal = dict(transform="z_relativo", a=round(float(lr.coef_[0][0]), 5), b=round(float(lr.intercept_[0]), 5))
    pt = np.array([towel_probability(z, cal, n) for z, n in zip(ens.score(test), norm(test))])
    cal.update(fit="validación", n=int(len(yv)), prevalencia=round(float(yv.mean()), 4),
               brier_test=round(float(brier_score_loss(yt, pt)), 4),
               brier_test_sin_modelo=round(float(brier_score_loss(yt, np.full(len(yt), yv.mean()))), 4))
    return cal


def main():
    ens = Ensemble.load(C.MODELS_DIR / "ensemble.json")
    prob = fit_probability(ens)
    prod = ens.norm[ens.default_capture][ens.models[0]]  # la columna prob es para la cámara de producción
    d = pd.concat([pd.read_csv(C.REPORTS_DIR / f"scores_{s}.csv", keep_default_na=False) for s in ("val", "test")])
    z = ens.score(d.reset_index(drop=True))
    y = (d.label == "NG").to_numpy()
    pos, neg = np.sort(z[y]), z[~y]
    # umbrales que dan recall entre los extremos (cuantiles de los positivos)
    lo = np.quantile(pos, 1 - RECALL_RANGE[1])
    hi = np.quantile(pos, 1 - RECALL_RANGE[0])
    ths = np.linspace(hi, lo, N_POINTS)  # de menos a más exigente
    points = [dict(threshold=round(float(t), 4), recall=round(float((pos >= t).mean()), 4),
                   false_alarms=round(float((neg >= t).mean()), 4), prob=round(towel_probability(t, prob, prod), 4))
              for t in ths]
    out = dict(
        points=points,
        presets={k: round(float(v), 4) for k, v in ens.thresholds.items()},
        n_defect=int(y.sum()), n_ok=int((~y).sum()),
        probabilidad=prob,
        note="Estimado con fotos de validación y prueba de las cámaras de 2025 (ya calibradas). "
             "Con otra cámara o iluminación los valores reales pueden variar.",
    )
    path = C.MODELS_DIR / "curva_operacion.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{path}: {len(points)} puntos, recall {points[0]['recall']:.0%}–{points[-1]['recall']:.0%}, "
          f"falsas alarmas {points[0]['false_alarms']:.0%}–{points[-1]['false_alarms']:.0%}")
    for k, t in ens.thresholds.items():
        print(f"  {k}: umbral {t:.3f} -> recall {(pos >= t).mean():.1%}, falsas alarmas {(neg >= t).mean():.1%}, "
              f"P(defecto) {towel_probability(t, prob, prod):.0%}")
    print(f"  probabilidad: a={prob['a']} b={prob['b']} · Brier en test {prob['brier_test']} "
          f"(sin modelo {prob['brier_test_sin_modelo']})")


if __name__ == "__main__":
    main()
