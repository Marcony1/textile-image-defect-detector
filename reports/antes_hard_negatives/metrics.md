# Resultados a nivel toalla (objetivo: recall 95% en validación)

Val: 230 toallas (73 NO OK). Test: 231 toallas (74 NO OK).

| Método | Umbral | Recall NO OK (val) | Falsas alarmas OK (val) | Recall NO OK (test) | Falsas alarmas OK (test) | Precisión NO OK (test) | AUC (test) |
|---|---|---|---|---|---|---|---|
| yolo | 0.04724 | 95.9% | 88.5% | 97.3% (72/74) | 91.1% (143/157) | 33.5% | 0.697 |
| cls | 0.6707 | 95.9% | 47.1% | 95.9% (71/74) | 51.0% (80/157) | 47.0% | 0.888 |
| patchcore | 41.05 | 95.9% | 81.5% | 97.3% (72/74) | 74.5% (117/157) | 38.1% | 0.786 |
| max | 0.4497 | 95.9% | 63.1% | 91.9% (68/74) | 58.6% (92/157) | 42.5% | 0.788 |
| logistic | 0.1395 | 97.3% | 53.5% | 94.6% (70/74) | 61.1% (96/157) | 42.2% | 0.872 |

**Seleccionado en validación:** `cls` -> `models/ensemble.json`