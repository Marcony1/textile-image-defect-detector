"""Tablas y figuras del reporte de experimentos (07) a partir de los resultados guardados.

    python scripts/build_experimentos.py          # luego: python scripts/build_reportes.py experimentos
    make experimentos                             # ambos pasos

Lee (no corre ningún modelo ni GPU):
    reports/experiments/<nombre>_val.csv   scores por toalla de validación (scripts/09_experiment_val.py)
    runs/<carpeta>/history.json            curvas de entrenamiento
    docs/reportes/fuente/experimentos.json nombre, ronda y tipo de cada experimento; cuál es producción
Escribe:
    docs/reportes/fuente/generado/experimentos_*.html   fragmentos que el reporte incluye con {{INC:...}}
    docs/reportes/fuente/img/exp_*.png                  figuras
    reports/experiments/comparacion_produccion.csv      la misma tabla en CSV

Para agregar un experimento: evaluarlo con 09_experiment_val.py, describirlo en experimentos.json
y correr `make experimentos`. El veredicto de la tabla sale de una regla fija (ver VEREDICTO).
"""
import datetime as dt
import html
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
FUENTE = ROOT / "docs" / "reportes" / "fuente"
GEN, IMG = FUENTE / "generado", FUENTE / "img"
AGG = "top2"  # agregación de producción: promedio de los 2 recortes más sospechosos
N_BOOT = 1000
VEREDICTO = """Regla del veredicto (contra producción, mismas toallas de validación, bootstrap pareado):
«mejor» si el ΔAUC global tiene P(>0) ≥ 0.95, no empeora la cámara de producción y no sube las falsas alarmas;
«peor» si P(>0) ≤ 0.05 en el global o en la cámara de producción; si no, «sin diferencia clara»."""

# colores: paleta categórica validada (skill dataviz), en orden fijo
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, LINE = "#1b1b1a", "#52514e", "#e2e1dc"


def cargar_09():
    spec = importlib.util.spec_from_file_location("exp09", ROOT / "scripts" / "09_experiment_val.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tabla(meta: dict) -> pd.DataFrame:
    e9 = cargar_09()
    prod = meta["produccion"]
    base = pd.read_csv(EXP / f"{prod}_val.csv", keep_default_na=False)
    rows = []
    for f in sorted(EXP.glob("*_val.csv")):
        name = f.name[:-8]
        info = meta["experimentos"].get(name, {})
        d = pd.read_csv(f, keep_default_na=False)
        m = e9.metrics(d, AGG, n_boot=N_BOOT)
        r = dict(experimento=name, nombre=info.get("nombre", f"{name} (sin descripción en experimentos.json)"),
                 ronda=info.get("ronda", 99), tipo=info.get("tipo", "?"), produccion=name == prod,
                 **{k: float(v) for k, v in m.items()})
        if name != prod:
            dl = e9.paired_delta(d, base, AGG, AGG, n_boot=N_BOOT)
            for k in ("all", "prod"):
                r[f"d_{k}"], r[f"d_{k}_lo"], r[f"d_{k}_hi"], r[f"p_{k}"] = dl[k]
        rows.append(r)
        print(f"  {name}", flush=True)
    t = pd.DataFrame(rows)
    p = t[t.produccion].iloc[0]

    def veredicto(r):
        if r.produccion:
            return "producción"
        if r.p_all <= 0.05 or r.p_prod <= 0.05:
            return "peor"
        if r.p_all >= 0.95 and r.d_prod >= 0 and r.fa95_prod <= p.fa95_prod:
            return "mejor"
        return "sin diferencia clara"

    t["veredicto"] = t.apply(veredicto, axis=1)
    return t.sort_values(["ronda", "produccion", "auc"], ascending=[True, False, False]).reset_index(drop=True)


def pct(x):
    return f"{100 * x:.0f} %"


def delta(r, k):
    if r.produccion:
        return "—"
    return f"{r[f'd_{k}']:+.3f} <span class=\"ci\">[{r[f'd_{k}_lo']:+.3f}, {r[f'd_{k}_hi']:+.3f}]</span>"


def html_tabla(t: pd.DataFrame, meta: dict) -> str:
    out = []
    for ronda, g in t.groupby("ronda", sort=True):
        titulo = meta["rondas"].get(str(ronda), "Sin ronda asignada")
        out.append(f"<h3>{html.escape(titulo)}</h3>\n<table class=\"exp\">\n<tr><th>Experimento</th><th class=\"n\">AUC "
                   "todas<br><span class=\"ci\">[IC 95 %]</span></th><th class=\"n\">AUC cámara<br>producción</th>"
                   "<th class=\"n\">F.A. al 95 %<br>todas</th><th class=\"n\">F.A. al 95 %<br>producción</th>"
                   "<th class=\"n\">ΔAUC vs producción<br>todas</th><th class=\"n\">ΔAUC vs producción<br>cámara prod.</th>"
                   "<th>Veredicto</th></tr>")
        for r in g.itertuples():
            r = t.loc[r.Index]
            cls = ' class="hl"' if r.produccion else ""
            tipo = "ensamble" if r.tipo == "ensamble" else "modelo"
            out.append(f"<tr{cls}><td>{html.escape(r.nombre)} <span class=\"tag\">{tipo}</span></td>"
                       f"<td class=\"n\">{r.auc:.3f} <span class=\"ci\">[{r.auc_lo:.3f}, {r.auc_hi:.3f}]</span></td>"
                       f"<td class=\"n\">{r.auc_prod:.3f}</td><td class=\"n\">{pct(r.fa95)}</td>"
                       f"<td class=\"n\">{pct(r.fa95_prod)}</td><td class=\"n\">{delta(r, 'all')}</td>"
                       f"<td class=\"n\">{delta(r, 'prod')}</td><td><span class=\"v v-{r.veredicto.split()[0]}\">"
                       f"{r.veredicto}</span></td></tr>")
        out.append("</table>")
    out.append(f"<p class=\"nota\">{html.escape(VEREDICTO)} Agregación por toalla: <code>{AGG}</code>. "
               f"F.A. = falsas alarmas (toallas OK marcadas NO OK) con el umbral que detecta el 95 % de las NO OK. "
               f"Intervalos: bootstrap estratificado, {N_BOOT} remuestreos.</p>")
    return "\n".join(out)


def html_resumen(t: pd.DataFrame) -> str:
    p = t[t.produccion].iloc[0]
    alt = t[~t.produccion]
    mejor = alt.loc[alt.auc.idxmax()]
    n_mejor = int((alt.veredicto == "mejor").sum())
    return (f"<div class=\"kpis\">"
            f"<div class=\"kpi u\"><div class=\"v\">{len(alt)}</div><div class=\"l\">alternativas comparadas "
            f"contra el modelo de producción</div></div>"
            f"<div class=\"kpi\"><div class=\"v\">{p.auc:.3f}</div><div class=\"l\">AUC de producción en validación "
            f"({p.auc_prod:.3f} en la cámara de producción; {pct(p.fa95_prod)} de falsas alarmas al 95 %)</div></div>"
            f"<div class=\"kpi {'g' if n_mejor else 'b'}\"><div class=\"v\">{n_mejor}</div><div class=\"l\">"
            f"alternativas con mejora clara. Mayor AUC: {html.escape(mejor.nombre)} ({mejor.auc:.3f}, "
            f"{mejor.veredicto})</div></div></div>")


def estilo(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="x", color=LINE, lw=0.6)
    ax.set_axisbelow(True)


def fig_delta(t: pd.DataFrame, path: Path):
    a = t[~t.produccion].iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.28 * len(a) + 1.2), sharey=True, sharex=True)
    for ax, k, titulo in zip(axes, ("all", "prod"), ("Todas las cámaras", "Cámara de producción (2025-06)")):
        estilo(ax)
        y = np.arange(len(a))
        col = [SERIES[0] if r == 1 else SERIES[1] for r in a.ronda]
        ax.hlines(y, a[f"d_{k}_lo"], a[f"d_{k}_hi"], color=col, lw=2)
        ax.scatter(a[f"d_{k}"], y, s=36, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(titulo, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("ΔAUC contra producción (IC 95 %)", fontsize=8.5, color=INK2)
    axes[0].set_yticks(np.arange(len(a)), a.nombre, fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=SERIES[0], marker="o", lw=2), Line2D([], [], color=SERIES[1], marker="o", lw=2)],
               ["Ronda 1", "Ronda 2"], loc="upper right", frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_curvas(meta: dict, path: Path) -> str:
    runs = [(info["nombre"], ROOT / "runs" / info["curva"] / "history.json")
            for info in meta["experimentos"].values() if info.get("curva")]
    runs = [(n, p) for n, p in runs if p.exists()]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    estilo(ax)
    ax.grid(axis="y", color=LINE, lw=0.6)
    filas = []
    for i, (n, p) in enumerate(runs):
        h = json.loads(p.read_text())
        ep, ap = [r["epoch"] for r in h], [r["val_ap"] for r in h]
        c = SERIES[i % len(SERIES)]
        ax.plot(ep, ap, color=c, lw=2, label=n)
        b = int(np.argmax(ap))
        ax.scatter([ep[b]], [ap[b]], s=40, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        filas.append(f"<tr><td>{html.escape(n)}</td><td class=\"n\">{ap[b]:.3f}</td><td class=\"n\">{ep[b]}</td>"
                     f"<td class=\"n\">{len(ep)}</td></tr>")
    ax.set_xlabel("Época", fontsize=8.5, color=INK2)
    ax.set_ylabel("AP por recorte (validación)", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return ("<table><tr><th>Entrenamiento (misma receta: 15 épocas, muestreo balanceado)</th><th class=\"n\">Mejor AP "
            "por recorte</th><th class=\"n\">Época</th><th class=\"n\">Épocas</th></tr>" + "".join(filas) + "</table>")


def main():
    meta = json.loads((FUENTE / "experimentos.json").read_text(encoding="utf-8"))
    GEN.mkdir(exist_ok=True)
    print("Calculando métricas y bootstrap pareado contra producción...")
    t = tabla(meta)
    t.drop(columns=["tipo"]).round(4).to_csv(EXP / "comparacion_produccion.csv", index=False)
    (GEN / "experimentos_tabla.html").write_text(html_tabla(t, meta), encoding="utf-8")
    (GEN / "experimentos_resumen.html").write_text(html_resumen(t), encoding="utf-8")
    (GEN / "experimentos_curvas.html").write_text(fig_curvas(meta, IMG / "exp_curvas_entrenamiento.png"), encoding="utf-8")
    fig_delta(t, IMG / "exp_delta_auc.png")
    (GEN / "experimentos_fecha.html").write_text(dt.date.today().isoformat(), encoding="utf-8")
    print(t[["experimento", "auc", "auc_prod", "fa95_prod", "veredicto"]].to_string(index=False))
    print(f"-> {GEN.relative_to(ROOT)}/experimentos_*.html, {IMG.relative_to(ROOT)}/exp_*.png")


if __name__ == "__main__":
    main()
