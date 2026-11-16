"""Inspection worker: takes towel photos from an inbox folder (where the line
camera saves them) or from HTTP uploads, runs the TowelInspector on the GPU,
archives the photo, renders an annotated preview and logs the result.

Only this thread touches the models, so GPU memory and ordering stay simple.
"""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import config as C
from ..inference import TowelInspector
from .store import Store

log = logging.getLogger("galia.worker")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PREVIEW_WIDTH = 1280
CLS_STRIDE = int(C.CLS_TILE * (1 - C.CLS_OVERLAP))


@dataclass
class Job:
    path: Path
    future: Future | None = None
    move: bool = True  # move from inbox to archive (False: copy, the caller keeps the file)


@dataclass
class WorkerStatus:
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    last_activity: str | None = None
    processed: int = 0
    errors: int = 0
    queue: int = 0
    model_loaded: bool = False


class InspectionWorker(threading.Thread):
    def __init__(self, store: Store, inbox: Path, archive: Path, models_dir: Path = C.MODELS_DIR,
                 operating_point: str = "recall_98", scale: float = 1.0, poll_s: float = 0.5,
                 on_result=None):
        super().__init__(daemon=True, name="inspection-worker")
        self.store, self.inbox, self.archive = store, Path(inbox), Path(archive)
        self.models_dir, self.operating_point, self.scale = Path(models_dir), operating_point, scale
        self.poll_s, self.on_result = poll_s, on_result
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.status = WorkerStatus()
        self.inspector: TowelInspector | None = None
        self._seen: dict[Path, tuple[int, float]] = {}
        self._halt = threading.Event()
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- control
    def submit(self, path: Path, move: bool = True) -> Future:
        fut: Future = Future()
        self.jobs.put(Job(Path(path), fut, move))
        return fut

    def stop(self):
        self._halt.set()

    @property
    def threshold(self) -> float | None:
        return self.inspector.threshold if self.inspector else None

    # -------------------------------------------------------------------- loop
    def run(self):
        log.info("loading models from %s (operating point %s)", self.models_dir, self.operating_point)
        self.inspector = TowelInspector.from_dir(self.models_dir, self.operating_point)
        self.status.model_loaded = True
        log.info("models loaded; watching %s", self.inbox)
        while not self._halt.is_set():
            try:
                self._scan_inbox()
            except Exception:
                log.exception("inbox scan failed")
            try:
                job = self.jobs.get(timeout=self.poll_s)
            except queue.Empty:
                continue
            self.status.queue = self.jobs.qsize()
            res = self._process(job)
            if job.future is not None:
                job.future.set_result(res)

    def _scan_inbox(self):
        """Queue files whose size has been stable for one poll (camera finished writing)."""
        queued = {j.path for j in list(self.jobs.queue)}
        files = []
        for p in self.inbox.iterdir():
            if p.suffix.lower() not in IMAGE_EXT or p in queued:
                continue
            try:  # the camera may rename/delete files while we scan
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, p, st.st_size))
        for _, p, size in sorted(files):
            prev = self._seen.get(p)
            if prev and prev[0] == size and size > 0:
                self._seen.pop(p)
                self.jobs.put(Job(p))
            else:
                self._seen[p] = (size, time.time())
        self.status.queue = self.jobs.qsize()

    # ----------------------------------------------------------------- process
    def _process(self, job: Job) -> dict:
        t0 = time.perf_counter()
        day = datetime.now().strftime("%Y-%m-%d")
        try:
            out = self.inspector.inspect(str(job.path), scale=self.scale)
            elapsed = (time.perf_counter() - t0) * 1000
            folder = "NO_OK" if out["decision"] == "NO OK" else "OK"
            dst = self._archive(job, self.archive / day / folder)
            defects = [d for d in out["defects"] if d["cls"] not in C.CONTROL_CLASSES]
            top = max(defects, key=lambda d: d["conf"])["cls"] if defects and out["decision"] == "NO OK" else None
            iid = self.store.add(file=job.path.name, decision=out["decision"], score=out["score"],
                                 threshold=out["threshold"], operating_point=self.operating_point,
                                 model_scores=out["model_scores"], defects=defects, top_defect=top,
                                 elapsed_ms=elapsed, image_path=str(dst))
            prev = self.archive / day / "previews" / f"{iid:07d}.jpg"
            self._render_preview(dst, out, defects, prev)
            self.store.set_preview(iid, str(prev))
            self.status.processed += 1
        except Exception as e:  # never stop the line because of one bad file
            log.exception("failed on %s", job.path)
            elapsed = (time.perf_counter() - t0) * 1000
            dst = self._archive(job, self.archive / day / "errores") if job.path.exists() else job.path
            iid = self.store.add(file=job.path.name, decision="ERROR", elapsed_ms=elapsed, image_path=str(dst),
                                 error=f"{type(e).__name__}: {e}")
            self.status.errors += 1
        self.status.last_activity = datetime.now().isoformat(timespec="seconds")
        row = self.store.get(iid)
        if self.on_result:
            self.on_result(row)
        return row

    def _archive(self, job: Job, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / job.path.name
        if dst.exists():
            dst = folder / f"{job.path.stem}_{int(time.time() * 1000)}{job.path.suffix}"
        (shutil.move if job.move else shutil.copy2)(str(job.path), dst)
        return dst

    def _render_preview(self, src: Path, out: dict, defects: list, dst: Path):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        k = PREVIEW_WIDTH / W
        img = cv2.resize(img, (PREVIEW_WIDTH, round(H * k)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        raw = out.get("raw", {})
        if out["decision"] == "NO OK" and "cls" in raw and raw["cls"].heatmap is not None:
            heat = raw["cls"].heatmap
            cell = CLS_STRIDE / self.scale * k  # grid cell size in preview pixels
            big = cv2.resize(heat, (max(1, round(heat.shape[1] * cell)), max(1, round(heat.shape[0] * cell))),
                             interpolation=cv2.INTER_LINEAR)
            big = np.pad(big, ((0, max(0, h - big.shape[0])), (0, max(0, w - big.shape[1]))))[:h, :w]
            big = cv2.GaussianBlur(big, (0, 0), cell / 3)
            color = cv2.applyColorMap((np.clip(big, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            alpha = (np.clip((big - 0.15) / 0.6, 0, 1) * 0.55)[..., None]
            img = (img * (1 - alpha) + color * alpha).astype(np.uint8)
        for d in defects:
            if d["conf"] < 0.25:
                continue
            x1, y1, x2, y2 = [int(v * k) for v in d["box"]]
            pad = 8
            cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (59, 59, 208), 2)
            cv2.putText(img, f"{d['cls']} {d['conf']:.2f}", (x1 - pad, max(14, y1 - pad - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 59, 208), 1, cv2.LINE_AA)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
