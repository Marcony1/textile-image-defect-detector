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
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

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

    worker.on_result = lambda row: broadcast("inspection", _public(row))

    @app.on_event("startup")
    async def _startup():
        loop_holder["loop"] = asyncio.get_running_loop()
        if not worker.is_alive():
            worker.start()

    @app.on_event("shutdown")
    async def _shutdown():
        worker.stop()

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (STATIC / "dashboard.html").read_text(encoding="utf-8")

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
    if not worker.is_alive():
        alerts.append(dict(level="critical", title="El servicio de inspección está detenido",
                           detail="Reiniciar scripts/serve.py."))

    confirmed_real = reviewed.get("defecto_real", 0)
    false_alarm = reviewed.get("falsa_alarma", 0)
    return dict(
        window=window, start=start, now=now.isoformat(timespec="seconds"), line_name=cfg.line_name,
        operating_point=worker.operating_point, threshold=worker.threshold,
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
