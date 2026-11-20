"""Curva de operación para el control de exigencia del tablero.

Para cada umbral posible (en unidades del score calibrado por cámara) estima qué
fracción de toallas con defecto se detecta y qué fracción de toallas buenas se
manda a revisión. Usa los scores por toalla de validación y test (reports/
scores_*.csv) con el modelo ya elegido: no selecciona nada, solo describe el
compromiso para que el usuario decida con información.

    python scripts/10_operating_curve.py      # -> models/curva_operacion.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402
from galia.ensemble import Ensemble  # noqa: E402

RECALL_RANGE = (0.80, 0.995)  # extremos del control
N_POINTS = 41


def main():
    ens = Ensemble.load(C.MODELS_DIR / "ensemble.json")
    d = pd.concat([pd.read_csv(C.REPORTS_DIR / f"scores_{s}.csv", keep_default_na=False) for s in ("val", "test")])
    z = ens.score(d.reset_index(drop=True))
    y = (d.label == "NG").to_numpy()
    pos, neg = np.sort(z[y]), z[~y]
    # umbrales que dan recall entre los extremos (cuantiles de los positivos)
    lo = np.quantile(pos, 1 - RECALL_RANGE[1])
    hi = np.quantile(pos, 1 - RECALL_RANGE[0])
    ths = np.linspace(hi, lo, N_POINTS)  # de menos a más exigente
    points = [dict(threshold=round(float(t), 4), recall=round(float((pos >= t).mean()), 4),
                   false_alarms=round(float((neg >= t).mean()), 4)) for t in ths]
    out = dict(
        points=points,
        presets={k: round(float(v), 4) for k, v in ens.thresholds.items()},
        n_defect=int(y.sum()), n_ok=int((~y).sum()),
        note="Estimado con fotos de validación y prueba de las cámaras de 2025 (ya calibradas). "
             "Con otra cámara o iluminación los valores reales pueden variar.",
    )
    path = C.MODELS_DIR / "curva_operacion.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{path}: {len(points)} puntos, recall {points[0]['recall']:.0%}–{points[-1]['recall']:.0%}, "
          f"falsas alarmas {points[0]['false_alarms']:.0%}–{points[-1]['false_alarms']:.0%}")
    for k, t in ens.thresholds.items():
        print(f"  {k}: umbral {t:.3f} -> recall {(pos >= t).mean():.1%}, falsas alarmas {(neg >= t).mean():.1%}")


if __name__ == "__main__":
    main()
