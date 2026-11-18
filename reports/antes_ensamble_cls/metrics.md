# Resultados a nivel toalla (objetivo: recall 98% en validación)

Val: 230 toallas (73 NO OK). Test: 231 toallas (74 NO OK).

| Método | Umbral | Recall NO OK (val) | Falsas alarmas OK (val) | Recall NO OK (test) | Falsas alarmas OK (test) | Precisión NO OK (test) | AUC (test) |
|---|---|---|---|---|---|---|---|
| yolo | -0.3093 | 98.6% | 65.6% | 91.9% (68/74) | 74.5% (117/157) | 36.8% | 0.815 |
| cls | 0.5622 | 98.6% | 28.7% | 89.2% (66/74) | 23.6% (37/157) | 64.1% | 0.900 |
| patchcore | -0.4292 | 98.6% | 82.8% | 97.3% (72/74) | 79.0% (124/157) | 36.7% | 0.756 |
| max | 0.9261 | 98.6% | 42.0% | 94.6% (70/74) | 35.7% (56/157) | 55.6% | 0.911 |
| logistic | 0.04661 | 100.0% | 93.6% | 100.0% (74/74) | 94.9% (149/157) | 33.2% | 0.897 |

**Seleccionado en validación:** `cls` -> `models/ensemble.json`

## Cámara de producción (2025-06), test, método `cls`

77 toallas (40 NO OK).

| Punto de operación | Recall NO OK | Falsas alarmas OK | Precisión NO OK |
|---|---|---|---|
| recall_90 | 72.5% (29/40) | 10.8% (4/37) | 87.9% |
| recall_95 | 80.0% (32/40) | 16.2% (6/37) | 84.2% |
| recall_98 | 85.0% (34/40) | 27.0% (10/37) | 77.3% |