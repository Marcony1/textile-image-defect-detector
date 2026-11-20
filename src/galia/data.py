"""Dataset manifest: one row per towel photo, with a fixed train/val/test split.

The split is done at TOWEL level (before any tiling or augmentation) so no
towel ever contributes tiles to more than one split. All three models
(YOLO, tile classifier, PatchCore) share this split, which makes the final
ensemble evaluation on the test towels honest.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config as C

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
UNIFIED_LABELS_DIR = C.DATA_DIR / "labels_unified"


def _read_classes(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _convert_labels(src_label: Path, raw_classes: list[str], dst_label: Path) -> list[int]:
    """Rewrite a Label Studio YOLO label file with unified class ids."""
    lines, groups = [], []
    if src_label.exists():
        for line in src_label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = C.map_raw_class(raw_classes[int(parts[0])])
            groups.append(cid)
            lines.append(" ".join([str(cid)] + parts[1:5]))
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""))
    return groups


def _ng_category_lookup() -> dict[str, tuple[str, str]]:
    """filename -> (visibility, category) for the April NG pieces."""
    out = {}
    for p in C.SRC_NG_BY_CATEGORY.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            vis, cat = p.relative_to(C.SRC_NG_BY_CATEGORY).parts[:2]
            out[p.name] = ("No visible" if "No Visible" in vis else "Visible", cat)
    return out


def build_manifest(seed: int = C.SEED) -> pd.DataFrame:
    rows = []
    ng_lookup = _ng_category_lookup()

    # A: April NG, annotated (Label Studio prefixes file names with a uuid)
    raw_a = _read_classes(C.SRC_A_DIR / "classes.txt")
    for img in sorted((C.SRC_A_DIR / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        iid = f"A_{img.stem}"
        groups = _convert_labels(C.SRC_A_DIR / "labels" / f"{img.stem}.txt", raw_a,
                                 UNIFIED_LABELS_DIR / f"{iid}.txt")
        vis, cat = ng_lookup.get(img.name.split("-", 1)[1], ("Visible", "?"))
        rows.append(dict(image_id=iid, source="A", capture="2025-04", path=img, label="NG",
                         visibility=vis, category=cat, groups=groups))

    # B: June NG, annotated
    raw_b = _read_classes(C.SRC_B_DIR / "datos" / "classes.txt")
    for img in sorted((C.SRC_B_DIR / "imagenes" / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        iid = f"B_{img.stem}"
        groups = _convert_labels(C.SRC_B_DIR / "datos" / "labels" / f"{img.stem}.txt", raw_b,
                                 UNIFIED_LABELS_DIR / f"{iid}.txt")
        defect_groups = sorted({C.CLASSES[g] for g in groups if g in C.DEFECT_CLASS_IDS})
        rows.append(dict(image_id=iid, source="B", capture="2025-06", path=img, label="NG",
                         visibility="Visible", category="+".join(defect_groups) or "?",
                         groups=groups))

    # C: June OK, only the control class (Etiqueta) is annotated
    raw_c = _read_classes(C.SRC_C_DIR / "datos" / "classes.txt")
    for img in sorted((C.SRC_C_DIR / "imagenes" / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        iid = f"C_{img.stem}"
        groups = _convert_labels(C.SRC_C_DIR / "datos" / "labels" / f"{img.stem}.txt", raw_c,
                                 UNIFIED_LABELS_DIR / f"{iid}.txt")
        rows.append(dict(image_id=iid, source="C", capture="2025-06", path=img, label="OK",
                         visibility="", category="OK", groups=groups))

    # D: April OK, no boxes (Etiqueta not annotated -> not used to train YOLO)
    for img in sorted(C.SRC_D_DIR.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        rows.append(dict(image_id=f"D_{img.stem}", source="D", capture="2025-04", path=img,
                         label="OK", visibility="", category="OK", groups=None))

    df = pd.DataFrame(rows)
    df["has_boxes"] = df["groups"].notna()
    df["n_defect_boxes"] = df["groups"].apply(
        lambda g: sum(1 for x in g if x in C.DEFECT_CLASS_IDS) if isinstance(g, list) else 0)
    df["label_path"] = df.apply(
        lambda r: str(UNIFIED_LABELS_DIR / f"{r.image_id}.txt") if r.has_boxes else "", axis=1)
    df["path"] = df["path"].astype(str)

    sizes = [Image.open(p).size for p in df["path"]]
    df["width"] = [s[0] for s in sizes]
    df["height"] = [s[1] for s in sizes]
    # Flag files that cannot be fully read (e.g. disk CRC errors). They stay in
    # the manifest so the split of every other towel is unchanged, but
    # load_manifest() drops them.
    df["readable"] = [_is_readable(p) for p in df["path"]]

    # Towel-level stratified split by source x label
    strat = df["source"] + "_" + df["label"]
    tr, va, te = C.SPLIT_RATIOS
    idx_train, idx_tmp = train_test_split(df.index, test_size=va + te, stratify=strat,
                                          random_state=seed)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=te / (va + te),
                                         stratify=strat[idx_tmp], random_state=seed)
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"
    return df.drop(columns=["groups"])


def _is_readable(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            while f.read(1 << 24):
                pass
        return True
    except OSError:
        return False


def load_manifest(include_unreadable: bool = False) -> pd.DataFrame:
    df = pd.read_csv(C.MANIFEST_CSV, keep_default_na=False)
    for col in ("has_boxes", "readable"):
        df[col] = df[col].astype(str).str.lower() == "true"
    if not include_unreadable:
        df = df[df.readable].reset_index(drop=True)
    return df


def load_boxes(label_path: str, width: int, height: int) -> np.ndarray:
    """Return (N, 5) array [cls, x1, y1, x2, y2] in pixels of the ORIGINAL image."""
    if not label_path or not Path(label_path).exists():
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    c, x, y, w, h = arr[:, 0], arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    return np.stack([c, x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def load_image(path: str, source: str) -> tuple[np.ndarray, float]:
    """Load an RGB image already rescaled to the production pixel scale."""
    scale = C.SOURCE_SCALE[source]
    for attempt in range(5):  # OneDrive can briefly lock files while syncing
        try:
            im = Image.open(path).convert("RGB")
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1.0 + attempt)
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
    return np.asarray(im), scale
