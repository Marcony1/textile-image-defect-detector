# textile-image-defect-detector

Detección automática de defectos en toallas a partir de **una foto por toalla**, tomada por la cámara de la línea de inspección. Responde **OK / NO OK** y, además, marca dónde está el defecto.

El objetivo de negocio es **no dejar pasar toallas defectuosas**. Por eso el sistema se calibra para tener **recall alto**: detectar la mayor proporción posible de toallas NO OK, aunque eso implique algunas falsas alarmas que un operador revisa.

---

## Resultados

Evaluación **por toalla** sobre 231 toallas de prueba (74 NO OK, 157 OK) que ningún modelo vio durante el entrenamiento. Los umbrales se fijaron en validación.

| | v1 (CNN en foto reducida) | **v2 (punto `recall_98`)** | v2 (punto `recall_95`) |
|---|---|---|---|
| Recall de toallas NO OK | 22 % | **89.2 % (66/74)** | 86.5 % (64/74) |
| Falsas alarmas (OK enviadas a revisión) | — | 23.6 % (37/157) | 16.6 % (26/157) |
| AUC por toalla | — | 0.90 | 0.90 |

Solo la **cámara de producción** (junio 2025; 77 toallas de prueba, 40 NO OK):

| Punto de operación | Recall NO OK | Falsas alarmas OK |
|---|---|---|
| `recall_90` | 72.5 % (29/40) | 10.8 % (4/37) |
| `recall_95` | 80.0 % (32/40) | 16.2 % (6/37) |
| **`recall_98` (por defecto)** | **85.0 % (34/40)** | 27.0 % (10/37) |

- El modelo elegido en validación es el **clasificador EfficientNet-B0 por recortes**, con minería de negativos difíciles y calibración por cámara. YOLO11 sigue en el pipeline para **ubicar y nombrar** el defecto. En recortes de validación, YOLO subió su mAP50 de 0.25 a 0.47; la etiqueta de control se detecta con mAP50 = 0.96.
- Los defectos que más se escapan son **manchas pequeñas (~40 px), costuras en esquinas e hilos de ~15 px**, y algunas **franjas de teñido muy tenues** que atraviesan la toalla. Son los tipos donde más anotaciones ayudarían.
- Con 40 toallas NO OK por cámara, cada toalla equivale a 2.5 puntos de recall. Las cifras tienen un margen de ±5–10 puntos.
- El detalle completo está en [`reports/metrics.md`](reports/metrics.md) y en el notebook `09_evaluacion_ensamble`.

---

## Cómo funciona (pipeline v2)

```text
foto de la toalla (1 por pieza)
        │  normalización de escala (cámara abril 5120² → escala de producción)
        │  máscara de la toalla (ignora fondo y máquina)
        ├──► YOLO11s en recortes de 1024 px  ──► cajas + clase de defecto
        ├──► EfficientNet-B0 en recortes de 512 px ──► P(defecto) por recorte
        └──► PatchCore (Anomalib) en recortes de 512 px ──► mapa de anomalía
                             │
   cada score se calibra contra las toallas OK de validación de la misma cámara
                             │
                   ensamble → score de la toalla → umbral → OK / NO OK
```

**Por qué recortes:** el modelo original reducía la foto completa a 640 px (8 veces menos resolución). Un hilo suelto de 40 px quedaba en 5 px y desaparecía. Ahora la foto se procesa a resolución nativa, dividida en recortes con 25 % de traslape. Las detecciones se unen con NMS y el resultado es **una decisión por toalla**. Esto es compatible con producción: entra una foto y sale una decisión.

**Por qué tres modelos:**

| Modelo | Aprende de | Fortaleza |
|---|---|---|
| YOLO11s (recortes) | cajas anotadas | localiza y nombra el defecto |
| EfficientNet-B0 (recortes) | recortes con/sin defecto | sensible a defectos pequeños; preentrenado en ImageNet |
| PatchCore (Anomalib) | **solo toallas OK** | detecta defectos de tipos nunca vistos; usa las ~800 fotos OK sin anotar |

**Etiqueta = control:** la etiqueta de la toalla se anota como clase `Etiqueta`. No es un defecto y **no cuenta para la decisión OK/NO OK**. Sirve para verificar que el detector funciona: si detecta bien la etiqueta, la escala y el pipeline de datos son correctos.

---

## Datos

`data/` está en `.gitignore` por privacidad de las imágenes.

| Fuente | Carpeta | Toallas | Etiqueta | Captura | Anotaciones |
|---|---|---|---|---|---|
| A | `data/01_annotated_data` | 226 | NO OK | abril 2025 (5120×5120) | cajas YOLO, 14 clases |
| B | `data/02_New_Annotated/imagenes no ok` | 262 | NO OK | junio 2025 (3580×2150) | cajas YOLO, 19 clases |
| C | `data/02_New_Annotated/imagenes ok` | 249 | OK | junio 2025 | solo `Etiqueta` |
| D | `data/new_by_category/PIEZAS OK` | 801 | OK | abril 2025 | sin cajas |

- Las fotos de abril se reducen ×0.62 para igualar la escala física de junio, que es la de producción.
- `data/raw` (2024, fondo blanco, 2448×2048) **no se usa**: es otra cámara y otra configuración.
- **Taxonomía unificada** (`src/galia/config.py`): las 19 clases originales se agrupan en `Etiqueta` (control), `Mancha`, `Hilo`, `Costura`, `Tejido`, `Tenido` y `Empalme`.
- **Split único por toalla** (70/15/15), estratificado por fuente y etiqueta, y compartido por todos los modelos (`data/manifest.csv`). Ningún recorte de una toalla de prueba se usa para entrenar.
- ⚠️ `data/new_by_category/PIEZAS OK/20250402_114712702_0000000143_T0000_01_CAM.png` tiene un **error de CRC en disco** y no se puede leer completo. Se excluye automáticamente (`readable=False` en el manifest). Conviene revisar el disco o recuperar una copia.

---

## Instalación

```bash
conda env create -f environment.yml
conda activate galia_v2
python -m ipykernel install --user --name galia_v2
```

Requiere GPU NVIDIA con CUDA 12.1. Se probó en una RTX 3060 Laptop de 6 GB. En Windows conviene exportar `PYTHONIOENCODING=utf-8`.

## Reproducir el entrenamiento

```bash
python scripts/04_build_dataset.py                 # manifest + split + recortes YOLO / clasificador / PatchCore
python scripts/05_train_yolo_tiles.py --epochs 80  # ≈3.5 h   -> runs/yolo11s_tiles/weights/best.pt
cp runs/yolo11s_tiles/weights/best.pt models/yolo11s_tiles.pt
python scripts/06_train_tile_classifier.py --arch efficientnet_b0 --epochs 15   # ≈1 h
cp models/tile_cls_efficientnet_b0.pt models/tile_cls_efficientnet_b0_v1.pt
python scripts/06b_mine_hard_negatives.py --weights models/tile_cls_efficientnet_b0_v1.pt   # recortes OK que el modelo confunde
python scripts/06_train_tile_classifier.py --init models/tile_cls_efficientnet_b0_v1.pt --epochs 6 --lr 1e-4 --hn-weight 3
python scripts/07_train_patchcore.py --max-tiles 1500 --coreset 0.01            # ≈7 min -> models/patchcore_wrn50.pt
python scripts/08_evaluate.py                      # scores por toalla, ensamble, umbrales -> reports/, models/ensemble.json
```

`models/` está en `.gitignore`. Los pesos de YOLO quedan versionados en `runs/yolo11s_tiles/weights/best.pt`. Los del clasificador y de PatchCore hay que compartirlos aparte (por ejemplo, en OneDrive).

## Uso en producción

### Servicio de línea con tablero en vivo (recomendado)

```bash
python scripts/serve.py --inbox D:/camara/salida --archive D:/galia/archivo --line-name "Línea 1"
# abrir http://localhost:8000 en la pantalla de la línea (otras PCs de la planta: --host 0.0.0.0)
```

- La cámara guarda cada foto en `--inbox`. El servicio la analiza (~0.8 s por toalla en una RTX 3060) y la mueve a `archivo/AAAA-MM-DD/{OK,NO_OK,errores}/`. Cada resultado queda registrado en `inspecciones.db` (SQLite).
- **Tablero en vivo:** KPIs del turno, última toalla con mapa de calor y cajas, lista de inspecciones, toallas por hora, tipos de defecto y alertas (NO OK seguidas, tasa alta, errores, línea detenida).
- **Revisión del operador:** cada toalla se marca como *defecto real / falsa alarma* (teclas D/F) u *OK correcto / tenía defecto* (O/X). Así se mide la precisión real y se acumulan etiquetas para reentrenar.
- API: `POST /api/inspect` (subir una foto), `GET /api/summary`, `GET /api/inspections`, `GET /api/export.csv?window=hoy`, `GET /api/health`.
- Para probarlo sin la máquina: `python scripts/simulate_line.py --every 4`, que copia fotos de test a la carpeta de entrada.
- ⚠️ El servicio no tiene usuarios ni contraseñas. Úsalo solo en la red interna.

Los reportes técnico y ejecutivo, con la propuesta de mejora, están en [`docs/reportes/`](docs/reportes/).

### Inferencia por lotes

```bash
python scripts/predict.py carpeta_o_foto.png --out reports/predicciones --save-images
```

Escribe `predicciones.csv` con la decisión, el score, el umbral y los defectos encontrados. Con `--save-images` guarda además la foto con las cajas.

Desde Python:

```python
from galia.inference import TowelInspector
insp = TowelInspector.from_dir("models", operating_point="recall_98")
res = insp.inspect("foto.png")       # o un array RGB
res["decision"], res["score"], res["defects"]
```

**Puntos de operación** (`recall_90`, `recall_95`, `recall_98`): el umbral se fija en validación para alcanzar ese recall de toallas NO OK. Un punto más alto detecta más defectos, pero genera más falsas alarmas. El valor por defecto es `recall_98`, porque la prioridad es no dejar pasar defectos.

> **Calibración por cámara:** la escala de los scores depende de la cámara (luz, fondo, sensor). Las toallas OK de junio reciben scores más altos que las de abril, así que con un umbral global casi todas saldrían como NO OK. Por eso cada score se normaliza contra las toallas OK de validación de su cámara, y en producción se usa la calibración de junio 2025. **Si cambia la cámara o la iluminación, hay que recalibrar** con unas decenas de fotos OK (y, de preferencia, algunas NO OK) volviendo a correr `08_evaluate.py`.
>
> Las fotos de producción deben tener la escala de la cámara de junio 2025 (≈3580×2150). Si la cámara cambia de resolución o distancia, use `--scale` para que un milímetro de tela ocupe los mismos píxeles.

---

## Estructura

```text
src/galia/          código reutilizable (config, datos, recortes, máscara, modelos, ensamble, inferencia, gráficas)
scripts/04-08       construcción del dataset, entrenamiento y evaluación
scripts/predict.py  inferencia por lotes
scripts/serve.py    servicio de línea + tablero en vivo (src/galia/service/)
notebooks/05-09     análisis y visualizaciones del pipeline v2
notebooks/01-04     modelos históricos (v1), se conservan como referencia
reports/            métricas (metrics.md / metrics.json) y scores por toalla
runs/               curvas y pesos de entrenamiento de YOLO
```

| Notebook | Contenido |
|---|---|
| `05_dataset_unificado_y_recortes` | fuentes, split, escala, taxonomía, tamaños de defecto, recortes y máscara |
| `06_yolo11_recortes` | curvas, mAP por clase, inferencia en foto completa |
| `07_clasificador_efficientnet` | entrenamiento, ROC/PR por recorte, recortes difíciles, mapas de calor |
| `08_patchcore_anomalib` | memoria de tela normal, distribución de scores, mapas de anomalía |
| `09_evaluacion_ensamble` | **métricas por toalla**, puntos de operación, matriz de confusión, recall por tipo de defecto, galerías de errores, demo de producción |

## Limitaciones conocidas

- **"NG No Visibles"** (48 toallas de la fuente A): el defecto no es visible en la foto. Ningún modelo de visión puede detectarlas de forma confiable, y se reportan por separado en `reports/metrics.md`.
- La máscara de la toalla usa saturación de color. En toallas blancas o grises sobre fondo similar puede incluir algo de fondo; en ese caso solo se pierde eficiencia, no se pierden defectos.
- Hay pocas anotaciones de `Hilo`, `Tejido` y `Costura`. Son las clases con menor recall de YOLO, y más ejemplos anotados de esas clases tendrían el mayor impacto.
- Validación y prueba tienen 231 toallas cada una, y solo ~40 NO OK por cámara. Los porcentajes tienen un margen de error de varios puntos.
- En la fuente A, el modelo también marca el 80 % de las toallas **"No visible"**. Puede ser que el defecto sí deje rastro en la foto, o que el modelo aproveche diferencias de la sesión de captura de abril (las toallas NO OK y OK de abril se fotografiaron por separado). Lo más confiable es evaluar con toallas de producción, donde las OK y NO OK salen de la misma línea.
- PatchCore solo (AUC 0.76) quedó por debajo del clasificador supervisado, porque la textura de rizo y los bordes varían mucho entre toallas OK. Se conserva para el ensamble y para detectar tipos de defecto nuevos.

---

## Historia (v1)

Los notebooks 01–04 contienen el primer enfoque:

- CNN entrenadas desde cero sobre fotos reducidas a 256 px: **recall de 22 %** de toallas NO OK.
- YOLO11 sobre la foto completa a 640 px: **mAP50 = 0.25**.

Los scripts `01_data_preprocessing.py` (aumentación + split OK/NO_OK) y `02_unzipper.py` (descomprimir exportaciones de Label Studio) pertenecen a esa etapa.

### Anotación

Las imágenes se anotan en [Label Studio](https://labelstud.io/) y se exportan en formato YOLO. Las credenciales **no** deben guardarse en el repositorio.

### Splitter YOLO (v1)

```bash
python scripts/03_data_splitter.py --input data/01_annotated_data --ratios 0.7 0.2 0.1 --seed 42 --stratify True
```

Genera `train/`, `valid/`, `test/` y `data.yaml`. El pipeline v2 usa en su lugar el split por toalla de `04_build_dataset.py`.
