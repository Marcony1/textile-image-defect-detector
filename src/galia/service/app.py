"""Web service for the inspection line: REST API + live dashboard.

    python scripts/serve.py --inbox D:/camara/salida --archive D:/galia/archivo

Endpoints (all JSON unless noted):
    GET  /                          live dashboard (HTML)
    GET  /api/health                worker / model status
    GET  /api/summary?window=turno  KPIs, hourly counts, defect types, alerts (turno | hoy | 24h | 7d)
    GET  /api/inspections?limit=50  latest inspections
    GET  /api/inspections/{id}      one inspection
    GET  /api/inspections/{id}/preview   annotated JPEG
    POST /api/inspections/{id}/review    {"review": "defecto_real|falsa_alarma|ok_confirmado|defecto_no_detectado", "note": ""}
    POST /api/inspect               multipart upload of one photo -> decision (synchronous)
    GET  /api/events                Server-Sent Events stream (new inspection / review)
    GET  /api/export.csv?window=hoy CSV for the shift report
    GET  /api/setup                 folders, device, calibration, demo availability, branding
    POST /api/demo                  copy the demo photos into the inbox, one every ~1.5 s
    GET  /api/demo/result           demo decisions vs. the known answers (etiquetas.csv)
    POST /api/batch {"folder": ...} analyse every photo of a folder (the originals are not moved)
    GET  /brand/logo                logo file from the branding config
    GET  /api/threshold             current / recommended threshold and the operating curve
    POST /api/threshold             {"threshold": z} to change strictness, {"reset": true} to go back
    POST /api/history/demo          delete the demo inspections (and their archived copies)
    POST /api/history/new           back up the history database and start an empty one
    GET  /api/inspections/{id}/image   original photo (full resolution) for the labeller
    GET/POST/DELETE /api/labels/{id}   boxes drawn on that photo ({"boxes": [{cls,x1,y1,x2,y2}], "note"})
    GET  /api/plc                   Modbus TCP server state for the line PLC (registers, last read, PLC heartbeat)
    GET  /api/labels                counts per class;  GET /api/labels/export.zip  everything for retraining
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import shutil
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .. import config as C
from .labels import LabelStore
from .plc import PLCServer
from .store import REVIEW_LABELS, Store, window_start
from .worker import IMAGE_EXT, InspectionWorker

STATIC = Path(__file__).parent / "static"


@dataclass
class ServiceConfig:
    line_name: str = "Línea de inspección"
    streak_alert: int = 5          # consecutive NO OK towels -> alert
    rate_window: int = 50          # last N towels used for the NO OK rate alert
    rate_alert: float = 0.50       # NO OK share above this -> alert
    idle_minutes: int = 10         # no towel for this long -> notice
    expected_nook_rate: float | None = None  # optional reference shown on the dashboard
    demo_dir: Path | None = None   # folder with demo photos + etiquetas.csv (archivo,real,tipo)
    brand_name: str = ""           # company name shown next to the logo
    brand_logo: Path | None = None  # PNG/SVG/JPG logo
    brand_color: str = ""          # main brand colour (#rrggbb): header bar and buttons
    brand_color_2: str = ""        # secondary brand colour (#rrggbb)
    note: str = ""                 # free text shown in the setup panel
    settings_file: Path | None = None  # JSON where the dashboard strictness setting survives restarts
    labels_dir: Path | None = None     # where the dashboard labeller saves photos + YOLO labels
    plc_enabled: bool = False          # Modbus TCP server for the line PLC (see galia.service.plc)
    plc_port: int = 502
    plc_host: str = "0.0.0.0"          # 0.0.0.0 = reachable from the plant network (the PLC); 127.0.0.1 = this PC only
    plc_allowed: tuple = ()            # IPs allowed to connect; empty = any on the network


def _demo_labels(demo_dir: Path | None) -> dict:
    if not demo_dir or not (demo_dir / "etiquetas.csv").exists():
        return {}
    with open(demo_dir / "etiquetas.csv", encoding="utf-8-sig", newline="") as f:
        return {r["archivo"]: r for r in csv.DictReader(f)}


def create_app(store: Store, worker: InspectionWorker, cfg: ServiceConfig = ServiceConfig()) -> FastAPI:
    app = FastAPI(title="Galia · inspección de toallas", version="2.0")
    subscribers: set[asyncio.Queue] = set()
    loop_holder: dict = {}

    def broadcast(kind: str, payload: dict):
        loop = loop_holder.get("loop")
        if not loop:
            return
        msg = json.dumps({"type": kind, "data": payload}, default=str)
        for q in list(subscribers):
            loop.call_soon_threadsafe(q.put_nowait, msg)

    plc = PLCServer(cfg.plc_port, cfg.plc_host, allowed=set(cfg.plc_allowed)) if cfg.plc_enabled else None
    if plc:
        plc.status_fn = lambda: (2 if worker.status.load_error else 0 if not worker.status.model_loaded
                                 else 1 if worker.is_alive() else 3)

    def on_result(row):
        if plc:
            plc.publish(row, worker.jobs.qsize())
        broadcast("inspection", _public(row))

    worker.on_result = on_result

    @app.on_event("startup")
    async def _startup():
        loop_holder["loop"] = asyncio.get_running_loop()
        if not worker.is_alive():
            worker.start()
        if plc:
            plc.start()

    @app.on_event("shutdown")
    async def _shutdown():
        worker.stop()
        if plc:
            plc.stop()

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (STATIC / "dashboard.html").read_text(encoding="utf-8")

    # ------------------------------------------------------------- strictness
    curve_path = Path(worker.models_dir) / "curva_operacion.json"
    curve = json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.exists() else None
    if cfg.settings_file and Path(cfg.settings_file).exists():
        try:
            saved = json.loads(Path(cfg.settings_file).read_text(encoding="utf-8")).get("threshold")
            if saved is not None:
                worker.set_threshold(float(saved))
        except (ValueError, OSError):
            pass

    def threshold_state() -> dict:
        return dict(threshold=worker.threshold, default=worker.default_threshold,
                    custom=worker.custom_threshold is not None, operating_point=worker.operating_point,
                    model_loaded=worker.status.model_loaded, curve=curve)

    @app.get("/api/threshold")
    def get_threshold():
        return threshold_state()

    @app.post("/api/threshold")
    def post_threshold(body: dict = Body(...)):
        if not worker.status.model_loaded:
            raise HTTPException(409, "El modelo todavía se está cargando")
        if body.get("reset"):
            value = None
        else:
            try:
                value = float(body["threshold"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(422, "threshold debe ser un número")
            if curve:
                ts = [pt["threshold"] for pt in curve["points"]] + list(curve["presets"].values())
                lo, hi = min(ts) - 0.5, max(ts) + 0.5
                if not lo <= value <= hi:
                    raise HTTPException(422, f"Fuera del rango permitido ({lo:.2f} a {hi:.2f})")
        worker.set_threshold(value)
        if cfg.settings_file:
            Path(cfg.settings_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.settings_file).write_text(json.dumps(dict(
                threshold=value, changed=datetime.now().isoformat(timespec="seconds"))), encoding="utf-8")
        state = threshold_state()
        broadcast("settings", {k: v for k, v in state.items() if k != "curve"})
        return state

    # ------------------------------------------------------------------ history
    def _remove_files(rows):
        for r in rows:
            for k in ("image_path", "preview_path"):
                try:
                    if r.get(k) and Path(r[k]).exists():
                        Path(r[k]).unlink()
                except OSError:
                    pass

    @app.post("/api/history/demo")
    def clear_demo():
        if demo_state["running"]:
            raise HTTPException(409, "Espere a que termine la demostración")
        rows = store.delete_files(set(_demo_labels(cfg.demo_dir)))
        _remove_files(rows)  # archived copies of the demo photos (the originals stay in demo/)
        labels.delete_by_photo(set(_demo_labels(cfg.demo_dir)))  # demo photos are TEST photos: never training labels
        broadcast("reset", {"what": "demo", "deleted": len(rows)})
        return dict(deleted=len(rows))

    @app.post("/api/history/new")
    def new_history():
        backup = store.path.parent / "respaldos" / f"inspecciones_{datetime.now():%Y-%m-%d_%H%M%S}.db"
        n = store.backup_and_clear(backup)
        broadcast("reset", {"what": "all", "deleted": n})
        return dict(deleted=n, backup=str(backup.resolve()))

    # ------------------------------------------------------------------ labels
    labels = LabelStore(cfg.labels_dir or (store.path.parent / "etiquetas"))

    def _row_or_404(iid: int) -> dict:
        r = store.get(iid)
        if not r or not r.get("image_path") or r["decision"] == "ERROR":
            raise HTTPException(404, "Inspección sin foto")
        return r

    @app.get("/api/inspections/{iid}/image")
    def original_image(iid: int):
        r = _row_or_404(iid)
        f = labels._image_file(r)
        if not f.exists():
            raise HTTPException(404, "La foto original ya no está en la carpeta de procesadas")
        return FileResponse(f, headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/labels")
    def label_stats():
        return labels.stats()

    @app.get("/api/labels/export.zip")
    def label_export():
        dest = Path(labels.root).parent / f"etiquetas_{datetime.now():%Y%m%d_%H%M}.zip"
        labels.export_zip(dest)
        return FileResponse(dest, media_type="application/zip", filename=dest.name)

    @app.get("/api/labels/{iid}")
    def label_get(iid: int):
        r = _row_or_404(iid)
        out = labels.get(r)
        out["suggestions"] = [dict(cls=d["cls"], x1=d["box"][0], y1=d["box"][1], x2=d["box"][2], y2=d["box"][3],
                                   conf=d["conf"]) for d in (r.get("defects") or []) if d["conf"] >= 0.25]
        out["classes"] = list(C.CLASSES)
        return out

    @app.post("/api/labels/{iid}")
    def label_save(iid: int, body: dict = Body(...)):
        r = _row_or_404(iid)
        if r["file"] in _demo_labels(cfg.demo_dir):
            raise HTTPException(422, "Las fotos de la demostración no se guardan: son fotos de prueba del modelo. "
                                     "Úselas solo para practicar; etiquete fotos de su línea.")
        try:
            res = labels.save(r, body.get("boxes") or [], str(body.get("note", ""))[:500])
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise HTTPException(422, str(e))
        broadcast("labels", {"id": iid, **res})
        return res

    @app.delete("/api/labels/{iid}")
    def label_delete(iid: int):
        r = _row_or_404(iid)
        return dict(deleted=labels.delete(r))

    # ------------------------------------------------------- setup / demo / batch
    demo_state = {"running": False, "sent": 0, "total": 0}

    @app.get("/api/setup")
    def setup():
        labels = _demo_labels(cfg.demo_dir)
        return dict(inbox=str(worker.inbox.resolve()), archive=str(worker.archive.resolve()),
                    device=worker.status.device, calibration=worker.status.calibration,
                    model_loaded=worker.status.model_loaded, load_error=worker.status.load_error,
                    operating_point=worker.operating_point, custom_threshold=worker.custom_threshold is not None,
                    demo_count=len(labels), demo=demo_state,
                    total_inspections=store.count() if hasattr(store, "count") else None,
                    brand=dict(name=cfg.brand_name, color=cfg.brand_color, color2=cfg.brand_color_2,
                               logo=(int(Path(cfg.brand_logo).stat().st_mtime) if cfg.brand_logo and Path(cfg.brand_logo).exists() else False)),
                    note=cfg.note, line_name=cfg.line_name,
                    plc=plc.state() if plc else dict(enabled=False))

    @app.get("/api/plc")
    def plc_state():
        return plc.state() if plc else dict(enabled=False)

    @app.get("/brand/logo")
    def brand_logo():
        if not cfg.brand_logo or not Path(cfg.brand_logo).exists():
            raise HTTPException(404)
        return FileResponse(cfg.brand_logo, headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/demo")
    def demo():
        labels = _demo_labels(cfg.demo_dir)
        if not labels:
            raise HTTPException(404, "No hay fotos de demostración")
        if demo_state["running"]:
            return demo_state
        files = [cfg.demo_dir / "fotos" / a for a in labels if (cfg.demo_dir / "fotos" / a).exists()]
        demo_state.update(running=True, sent=0, total=len(files), started=datetime.now().isoformat(timespec="seconds"))

        def feed():  # behaves like a camera: writes .part and renames when complete
            try:
                for f in files:
                    tmp = worker.inbox / (f.name + ".part")
                    shutil.copyfile(f, tmp)
                    tmp.replace(worker.inbox / f.name)
                    demo_state["sent"] += 1
                    time.sleep(1.5)
            finally:
                demo_state["running"] = False

        threading.Thread(target=feed, daemon=True, name="demo-feed").start()
        return demo_state

    @app.get("/api/demo/result")
    def demo_result():
        labels = _demo_labels(cfg.demo_dir)
        since = demo_state.get("started")
        latest = {}
        for r in reversed(store.recent(5000, since)):
            if r["file"] in labels and r["decision"] != "ERROR":
                latest[r["file"]] = r
        out = dict(tp=0, fn=0, fp=0, tn=0, n=len(latest), total=len(labels), items=[])
        for f, r in latest.items():
            real = labels[f]["real"].strip().upper().replace("_", " ")
            pred = r["decision"]
            key = ("tp" if pred == "NO OK" else "fn") if real == "NO OK" else ("fp" if pred == "NO OK" else "tn")
            out[key] += 1
            out["items"].append(dict(id=r["id"], file=f, real=real, pred=pred, tipo=labels[f].get("tipo", ""),
                                     correct=key in ("tp", "tn")))
        return out

    @app.post("/api/batch")
    def batch(body: dict = Body(...)):
        folder = Path(str(body.get("folder", "")).strip().strip('"'))
        if not str(folder) or not folder.is_dir():
            raise HTTPException(404, f"No existe la carpeta: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise HTTPException(422, "La carpeta no tiene fotos (.jpg, .png, .bmp, .tif)")
        for p in files:
            worker.submit(p, move=False)  # originals stay where they are; a copy goes to the archive
        return dict(queued=len(files), folder=str(folder))

    # -------------------------------------------------------------------- api
    @app.get("/api/health")
    def health():
        return dict(status="ok" if worker.is_alive() else "detenido", worker=asdict(worker.status),
                    operating_point=worker.operating_point, threshold=worker.threshold,
                    inbox=str(worker.inbox), archive=str(worker.archive), line_name=cfg.line_name)

    @app.get("/api/inspections")
    def inspections(limit: int = Query(50, le=500), window: str | None = None, decision: str | None = None):
        rows = store.recent(limit, window_start(window) if window else None, decision)
        return [_public(r) for r in rows]

    @app.get("/api/inspections/{iid}")
    def inspection(iid: int):
        r = store.get(iid)
        if not r:
            raise HTTPException(404)
        return _public(r)

    @app.get("/api/inspections/{iid}/preview")
    def preview(iid: int):
        r = store.get(iid)
        if not r or not r.get("preview_path") or not Path(r["preview_path"]).exists():
            raise HTTPException(404)
        return FileResponse(r["preview_path"], media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})

    @app.post("/api/inspections/{iid}/review")
    def review(iid: int, body: dict = Body(...)):
        label = body.get("review")
        if label not in REVIEW_LABELS:
            raise HTTPException(422, f"review debe ser uno de {sorted(REVIEW_LABELS)}")
        if not store.review(iid, label, body.get("note", "")):
            raise HTTPException(404)
        row = _public(store.get(iid))
        broadcast("review", row)
        return row

    @app.post("/api/inspect")
    async def inspect(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in IMAGE_EXT:
            raise HTTPException(415, f"formato no soportado: {ext}")
        tmp = worker.archive / "_uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        path = tmp / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{Path(file.filename).name}"
        path.write_bytes(await file.read())
        fut = worker.submit(path, move=True)
        row = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=120)
        return _public(row)

    @app.get("/api/summary")
    def summary(window: str = "turno"):
        return _summary(store, worker, cfg, window)

    @app.get("/api/events")
    async def events():
        q: asyncio.Queue = asyncio.Queue()
        subscribers.add(q)

        async def gen():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {msg}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                subscribers.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/export.csv")
    def export(window: str = "hoy"):
        rows = store.recent(100000, window_start(window))
        buf = io.StringIO()
        cols = ["id", "ts", "file", "decision", "score", "threshold", "top_defect", "elapsed_ms", "review",
                "review_note", "image_path", "error"]
        w = csv.DictWriter(buf, cols, extrasaction="ignore")
        w.writeheader()
        for r in reversed(rows):
            w.writerow(r)
        name = f"inspecciones_{window}_{datetime.now():%Y%m%d_%H%M}.csv"
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    return app


def _public(r: dict | None) -> dict | None:
    if r is None:
        return None
    d = {k: v for k, v in r.items() if k not in ("image_path", "preview_path")}
    d["has_preview"] = bool(r.get("preview_path"))
    return d


def _summary(store: Store, worker: InspectionWorker, cfg: ServiceConfig, window: str) -> dict:
    start = window_start(window)
    rows = store.rows_since(start)
    done = [r for r in rows if r["decision"] != "ERROR"]
    nook = [r for r in done if r["decision"] == "NO OK"]
    now = datetime.now()

    # buckets: hourly (daily for 7d)
    daily = window == "7d"
    t0 = datetime.fromisoformat(start)
    t0 = t0.replace(hour=0) if daily else t0
    t0 = t0.replace(minute=0, second=0)
    step = timedelta(days=1) if daily else timedelta(hours=1)
    buckets, t = [], t0
    while t <= now:
        buckets.append(dict(start=t.isoformat(timespec="minutes"),
                            label=t.strftime("%d/%m") if daily else t.strftime("%H:00"), ok=0, nook=0, error=0))
        t += step
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        i = int((ts - t0) / step)
        if 0 <= i < len(buckets):
            buckets[i]["nook" if r["decision"] == "NO OK" else "ok" if r["decision"] == "OK" else "error"] += 1

    types = Counter(r["top_defect"] or "Sin tipo (solo clasificador)" for r in nook)
    reviewed = Counter(r["review"] for r in rows if r["review"])
    last15 = [r for r in done if datetime.fromisoformat(r["ts"]) >= now - timedelta(minutes=15)]
    recent_ms = [r["elapsed_ms"] for r in done[-50:] if r["elapsed_ms"]]

    alerts = []
    tail = store.recent(max(cfg.rate_window, cfg.streak_alert))
    tail_done = [r for r in tail if r["decision"] != "ERROR"]
    streak = 0
    for r in tail_done:
        if r["decision"] != "NO OK":
            break
        streak += 1
    if streak >= cfg.streak_alert:
        alerts.append(dict(level="critical", title=f"{streak} toallas NO OK seguidas",
                           detail="Revisar la máquina o el lote. Si las toallas se ven bien, revisar cámara e iluminación."))
    if len(tail_done) >= 20:
        rate = sum(r["decision"] == "NO OK" for r in tail_done[:cfg.rate_window]) / len(tail_done[:cfg.rate_window])
        if rate > cfg.rate_alert:
            alerts.append(dict(level="warning", title=f"{rate:.0%} NO OK en las últimas {len(tail_done[:cfg.rate_window])} toallas",
                               detail="Más alto de lo normal: posible problema de producción o cambio en la cámara (recalibrar)."))
    errs = sum(r["decision"] == "ERROR" for r in tail)
    if errs:
        alerts.append(dict(level="warning", title=f"{errs} fotos con error de lectura",
                           detail="Revisar la carpeta 'errores' del archivo: archivo incompleto o dañado."))
    last = tail[0]["ts"] if tail else None
    if last and now - datetime.fromisoformat(last) > timedelta(minutes=cfg.idle_minutes):
        alerts.append(dict(level="info", title="Sin toallas nuevas",
                           detail=f"Última inspección {datetime.fromisoformat(last):%H:%M}. ¿Línea detenida o cámara sin guardar fotos?"))
    if worker.status.gpu_problem:
        alerts.append(dict(level="warning", title="La tarjeta gráfica no se pudo usar: se analiza con el procesador (más lento)",
                           detail="La versión de PyTorch instalada no es compatible con esta tarjeta. Cierre el programa y "
                                  "ejecute INSTALAR de nuevo; elegirá la versión correcta. Detalle: " + worker.status.gpu_problem))
    if not worker.is_alive():
        alerts.append(dict(level="critical", title="El servicio de inspección está detenido",
                           detail=(f"No se pudieron cargar los modelos: {worker.status.load_error}" if worker.status.load_error
                                   else "Cierre esta ventana y vuelva a iniciar el programa.")))

    confirmed_real = reviewed.get("defecto_real", 0)
    false_alarm = reviewed.get("falsa_alarma", 0)
    return dict(
        window=window, start=start, now=now.isoformat(timespec="seconds"), line_name=cfg.line_name,
        operating_point=worker.operating_point, threshold=worker.threshold,
        custom_threshold=worker.custom_threshold is not None,
        total=len(done), ok=len(done) - len(nook), nook=len(nook), errors=len(rows) - len(done),
        nook_rate=len(nook) / len(done) if done else None,
        per_hour_last15=len(last15) * 4,
        avg_ms=sum(recent_ms) / len(recent_ms) if recent_ms else None,
        pending_review=sum(1 for r in nook if not r["review"]),
        reviews=dict(reviewed),
        observed_precision=confirmed_real / (confirmed_real + false_alarm) if confirmed_real + false_alarm else None,
        missed_reported=reviewed.get("defecto_no_detectado", 0),
        buckets=buckets, defect_types=types.most_common(), alerts=alerts,
        worker=asdict(worker.status), expected_nook_rate=cfg.expected_nook_rate,
    )
