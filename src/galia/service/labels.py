"""Labels drawn on the dashboard, saved in the format the models train with.

    etiquetas/
      images/<id>_<foto>.jpg      copy of the original photo (full resolution)
      labels/<id>_<foto>.txt      YOLO: "<clase> <cx> <cy> <w> <h>" normalised to 0-1
      clases.txt                  class names in id order (same taxonomy as the models)
      registro.csv                one row per labelled photo: who/when/decision/review/boxes per class

A photo saved with no boxes is a valid "clean towel" example. Saving again
overwrites the previous labels of that photo.
"""
from __future__ import annotations

import csv
import shutil
import threading
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image

from .. import config as C

FIELDS = ["archivo", "inspeccion", "fecha", "decision_sistema", "revision", "n_cajas", "clases", "nota"]


class LabelStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.images, self.labels = self.root / "images", self.root / "labels"
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)
        (self.root / "clases.txt").write_text("\n".join(C.CLASSES) + "\n", encoding="utf-8")
        self._lock = threading.Lock()

    @staticmethod
    def name_for(row: dict) -> str:
        return f"{row['id']:07d}_{Path(row['file']).stem}"

    # ------------------------------------------------------------------ read
    def get(self, row: dict) -> dict:
        txt = self.labels / f"{self.name_for(row)}.txt"
        if not txt.exists():
            return dict(saved=False, boxes=[])
        img = self._image_file(row)
        W, H = Image.open(img).size
        boxes = []
        for line in txt.read_text(encoding="utf-8").split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            boxes.append(dict(cls=C.CLASSES[c], x1=(cx - w / 2) * W, y1=(cy - h / 2) * H,
                              x2=(cx + w / 2) * W, y2=(cy + h / 2) * H))
        return dict(saved=True, boxes=boxes)

    def _image_file(self, row: dict) -> Path:
        hit = sorted(self.images.glob(f"{self.name_for(row)}.*"))
        return hit[0] if hit else Path(row["image_path"])

    def stats(self) -> dict:
        per_class, n_img, n_empty = Counter(), 0, 0
        for txt in self.labels.glob("*.txt"):
            n_img += 1
            lines = [ln.split()[0] for ln in txt.read_text(encoding="utf-8").split("\n") if ln.strip()]
            if not lines:
                n_empty += 1
            for c in lines:
                per_class[C.CLASSES[int(c)]] += 1
        return dict(images=n_img, empty=n_empty, boxes=sum(per_class.values()),
                    per_class={c: per_class.get(c, 0) for c in C.CLASSES}, folder=str(self.root.resolve()))

    # ----------------------------------------------------------------- write
    def save(self, row: dict, boxes: list[dict], note: str = "") -> dict:
        src = Path(row["image_path"])
        if not src.exists():
            raise FileNotFoundError(f"No se encuentra la foto original: {src}")
        W, H = Image.open(src).size
        lines, classes = [], Counter()
        for b in boxes:
            if b["cls"] not in C.CLASSES:
                raise ValueError(f"Clase desconocida: {b['cls']}")
            x1, x2 = sorted((max(0.0, min(W, float(b["x1"]))), max(0.0, min(W, float(b["x2"])))))
            y1, y2 = sorted((max(0.0, min(H, float(b["y1"]))), max(0.0, min(H, float(b["y2"])))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{C.CLASSES.index(b['cls'])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            classes[b["cls"]] += 1
        name = self.name_for(row)
        with self._lock:
            dst = self.images / f"{name}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)
            (self.labels / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            self._log(dict(archivo=dst.name, inspeccion=row["id"], fecha=datetime.now().isoformat(timespec="seconds"),
                           decision_sistema=row["decision"], revision=row.get("review") or "",
                           n_cajas=len(lines), clases=" ".join(f"{k}:{v}" for k, v in classes.items()), nota=note))
        return dict(saved=True, n_boxes=len(lines), file=dst.name)

    def delete(self, row: dict) -> bool:
        name = self.name_for(row)
        with self._lock:
            hit = False
            for f in [self.labels / f"{name}.txt", *self.images.glob(f"{name}.*")]:
                if f.exists():
                    f.unlink()
                    hit = True
        return hit

    def delete_by_photo(self, files: set[str]) -> int:
        """Delete the labels of these original photo names (any inspection id)."""
        stems, n = {Path(f).stem for f in files}, 0
        with self._lock:
            for f in list(self.labels.glob("*.txt")) + list(self.images.glob("*.*")):
                if f.stem.split("_", 1)[-1] in stems:
                    f.unlink()
                    n += f.suffix == ".txt"
        return n

    def _log(self, rec: dict):
        path = self.root / "registro.csv"
        new = not path.exists()
        with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(rec)

    def export_zip(self, dest: Path) -> Path:
        """Write every label + photo to ``dest`` (zip on disk: the photos can add up to hundreds of MB)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:  # photos are already compressed
            for f in sorted(self.root.rglob("*")):
                if f.is_file():
                    z.write(f, Path("etiquetas") / f.relative_to(self.root))
        return dest
