# Resultados a nivel toalla (objetivo: recall 98% en validación)

Val: 230 toallas (73 NO OK). Test: 231 toallas (74 NO OK).

| Método | Umbral | Recall NO OK (val) | Falsas alarmas OK (val) | Recall NO OK (test) | Falsas alarmas OK (test) | Precisión NO OK (test) | AUC (test) |
|---|---|---|---|---|---|---|---|
| yolo | -0.3093 | 98.6% | 65.6% | 91.9% (68/74) | 74.5% (117/157) | 36.8% | 0.815 |
| cls | 0.3703 | 98.6% | 31.8% | 91.9% (68/74) | 28.0% (44/157) | 60.7% | 0.936 |
| patchcore | -0.4292 | 98.6% | 82.8% | 97.3% (72/74) | 79.0% (124/157) | 36.7% | 0.756 |
| max | 1.009 | 98.6% | 38.9% | 95.9% (71/74) | 31.8% (50/157) | 58.7% | 0.935 |
| logistic | 0.0454 | 100.0% | 93.6% | 100.0% (74/74) | 94.9% (149/157) | 33.2% | 0.900 |

**Seleccionado en validación:** `cls` -> `models/ensemble.json`

## Cámara de producción (2025-06), test, método `cls`

77 toallas (40 NO OK).

| Punto de operación | Recall NO OK | Falsas alarmas OK | Precisión NO OK |
|---|---|---|---|
| recall_90 | 82.5% (33/40) | 8.1% (3/37) | 91.7% |
| recall_95 | 87.5% (35/40) | 16.2% (6/37) | 85.4% |
| recall_98 | 90.0% (36/40) | 37.8% (14/37) | 72.0% |