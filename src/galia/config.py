"""Central configuration: paths, class taxonomy and tiling parameters.

Everything that more than one script/notebook needs lives here so the
whole pipeline (dataset -> YOLO / classifier / PatchCore -> ensemble)
stays consistent.
"""
from pathlib import Path
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Raw sources ----------------------------------------------------------------
# A: April 2025 capture, 5120x5120 PNG, NG pieces annotated in Label Studio.
SRC_A_DIR = DATA_DIR / "01_annotated_data"
# B/C: June 2025 capture, 3580x2150 JPG (current production set-up).
SRC_B_DIR = DATA_DIR / "02_New_Annotated" / "imagenes no ok"
SRC_C_DIR = DATA_DIR / "02_New_Annotated" / "imagenes ok"
# D: April 2025 capture, OK pieces (no box annotations).
SRC_D_DIR = DATA_DIR / "new_by_category" / "PIEZAS OK"
# NG pieces grouped by category / visibility (same images as source A).
SRC_NG_BY_CATEGORY = DATA_DIR / "new_by_category" / "PIEZAS NO OK"

# Generated artefacts ---------------------------------------------------------
MANIFEST_CSV = DATA_DIR / "manifest.csv"
YOLO_TILES_DIR = DATA_DIR / "yolo_tiles"
CLS_TILES_DIR = DATA_DIR / "cls_tiles"
PATCHCORE_TILES_DIR = DATA_DIR / "patchcore_tiles"

# Scale normalisation ---------------------------------------------------------
# The April camera (5120x5120) sees the towel ~1.55x larger than the June
# camera (labels measure ~170 px vs ~100 px). We rescale April images so
# every image has the pixel scale of the June / production set-up.
SOURCE_SCALE = {"A": 0.62, "B": 1.0, "C": 1.0, "D": 0.62}

# Class taxonomy --------------------------------------------------------------
# The two annotation rounds used different names (and typos). Everything is
# mapped to 6 defect groups + the "Etiqueta" control class. Grouping gives
# each class enough examples to learn; for the business decision (OK / NO OK)
# only "is there any defect" matters.
CLASSES = [
    "Etiqueta",      # 0 - control, NOT a defect
    "Mancha",        # 1 - oil / grease / rust / generic stain
    "Hilo",          # 2 - loose / trapped / agglomerated thread, raffia fibre
    "Costura",       # 3 - Over, RO, Recta, skipped stitch (seam defects)
    "Tejido",        # 4 - weave defects, holes
    "Tenido",        # 5 - dye differences
    "Empalme",       # 6 - splices
]
CONTROL_CLASSES = {"Etiqueta"}
DEFECT_CLASS_IDS = [i for i, c in enumerate(CLASSES) if c not in CONTROL_CLASSES]


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(name.lower().split())


_RAW_TO_GROUP = {
    # control
    "etiqueta": "Etiqueta",
    # stains
    "mancha de aceite": "Mancha", "macha aceite": "Mancha", "mancha de grasa": "Mancha",
    "mancha grasa": "Mancha", "mancha de oxido": "Mancha", "mancha oxido": "Mancha",
    "mancha": "Mancha",
    # threads / fibres
    "hilo": "Hilo", "hilo aglomerado": "Hilo", "hilo atrapado": "Hilo", "hilo suelto": "Hilo",
    "rafia": "Hilo",
    # seams
    "over": "Costura", "falla over": "Costura", "ro": "Costura", "falla ro": "Costura",
    "ro atrapado": "Costura", "recta": "Costura", "falla recta": "Costura",
    "puntada brincada": "Costura",
    # weave
    "tejido": "Tejido", "falla en tejido": "Tejido", "perforadas": "Tejido", "perforada": "Tejido",
    # dye
    "tenido": "Tenido", "diferencia en tenido": "Tenido",
    # splices
    "empalmes": "Empalme", "empalme": "Empalme",
}


def map_raw_class(raw_name: str) -> int:
    """Map an annotation class name (any round, any spelling) to a CLASSES index."""
    key = _norm(raw_name)
    if key not in _RAW_TO_GROUP:
        raise KeyError(f"Unknown annotation class: {raw_name!r}")
    return CLASSES.index(_RAW_TO_GROUP[key])


# Tiling ----------------------------------------------------------------------
# Detection / classification run on tiles cut from the full-resolution photo
# (after scale normalisation). Production is one photo per towel: that photo
# is tiled the same way at inference time, so train and production match.
YOLO_TILE = 1024
YOLO_OVERLAP = 0.25
CLS_TILE = 512
CLS_OVERLAP = 0.25
PATCHCORE_TILE = 512
PATCHCORE_INPUT = 256  # tiles are resized to this before the backbone

# A box is assigned to a tile if at least this fraction of it is visible.
MIN_BOX_VISIBILITY = 0.35

SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test, at TOWEL level
