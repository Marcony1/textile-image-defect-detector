"""Arma la carpeta de entrega para la empresa: programa + modelos + demo + instrucciones.

    python scripts/build_delivery.py                # -> dist/Inspector_Toallas.zip (lo único que se entrega)
    python scripts/build_delivery.py --carpeta      # además deja la carpeta descomprimida en dist/Inspector_Toallas

Se arma en una carpeta temporal, así nunca choca con una copia abierta o instalada en dist/.

La demo usa toallas del split de TEST de la cámara de producción (junio 2025),
que ningún modelo vio al entrenar; etiquetas.csv guarda la respuesta correcta.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from galia import config as C  # noqa: E402
from galia.data import load_manifest  # noqa: E402

ENTREGA = ROOT / "entrega"
GALIA_MODULES = ["__init__.py", "config.py", "ensemble.py", "inference.py", "models.py", "roi.py", "tiling.py",
                 "calibration.py"]
MODEL_FILES = ["ensemble.json", "tile_cls_ensemble.json", "tile_cls_efficientnet_b0.pt",
               "tile_cls_efficientnet_b0_basic_s7.pt", "yolo11s_tiles.pt", "curva_operacion.json"]
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = ENTREGA / "estilo.css"


def crlf(src: Path, dst: Path):
    dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


def lf(src: Path, dst: Path):
    dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))


def pick_demo(n: int, seed: int) -> list[tuple]:
    df = load_manifest()
    t = df[(df.split == "test") & (df.capture == "2025-06")]
    ng = t[t.label == "NG"].sample(n // 2, random_state=seed)
    ok = t[t.label == "OK"].sample(n - n // 2, random_state=seed)
    both = pd.concat([ng, ok])
    both = both.sample(frac=1, random_state=seed)  # mezclar OK y NO OK
    return list(both.itertuples())


PY_VERSION = "3.11.9"
PY_PACKAGE = f"https://www.nuget.org/api/v2/package/python/{PY_VERSION}"  # official portable CPython (python.org on NuGet)


def add_portable_python(dest: Path):
    """Ship Python 3.11 inside the package so INSTALAR never downloads or unpacks an interpreter
    (that download-and-extract step is what antivirus heuristics flagged)."""
    import urllib.request
    import zipfile

    cache = ROOT / "data" / "cache" / f"python-{PY_VERSION}.nupkg"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"  descargando Python {PY_VERSION} portátil (una sola vez)…")
        tmp = cache.with_suffix(".part")
        urllib.request.urlretrieve(PY_PACKAGE, tmp)
        tmp.replace(cache)
    with zipfile.ZipFile(cache) as z:
        for m in z.namelist():
            if m.startswith("tools/") and not m.endswith("/"):
                target = dest / m
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(m))
    exe = dest / "tools" / "python.exe"
    out = subprocess.run([str(exe), "-c", "import sys, venv, ensurepip; print(sys.version.split()[0])"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode or out.stdout.strip() != PY_VERSION:
        raise RuntimeError(f"El Python portátil no funciona: {out.stdout} {out.stderr}")
    print(f"  python311: Python {out.stdout.strip()} portátil incluido")


def build_pdf(html: Path, pdf: Path):
    if not Path(EDGE).exists():
        print("  (Edge no encontrado: solo se genera el HTML)")
        return
    prof = Path(tempfile.gettempdir()) / "galia_edgeprof"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()], timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demo", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--carpeta", action="store_true", help="dejar también la carpeta descomprimida en dist/")
    args = ap.parse_args()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="galia_entrega_"))
    out = staging / "Inspector_Toallas"
    out.mkdir()

    # programa
    print("programa…")
    prog = out / "programa"
    (prog / "galia" / "service" / "static").mkdir(parents=True)
    for m in GALIA_MODULES:
        shutil.copy2(C.PROJECT_ROOT / "src" / "galia" / m, prog / "galia" / m)
    for f in ["__init__.py", "app.py", "store.py", "worker.py", "labels.py", "plc.py"]:
        shutil.copy2(ROOT / "src" / "galia" / "service" / f, prog / "galia" / "service" / f)
    shutil.copy2(ROOT / "src" / "galia" / "service" / "static" / "dashboard.html", prog / "galia" / "service" / "static")
    for f in ["iniciar.py", "calibrar.py", "probar_plc.py", "requirements.txt"]:
        shutil.copy2(ENTREGA / "programa" / f, prog / f)

    # scripts de arranque, configuración, cámara, marca
    for f in ["INSTALAR.bat", "INICIAR.bat", "CALIBRAR.bat", "PROBAR_PLC.bat", "configuracion.ini"]:
        crlf(ENTREGA / f, out / f)
    for f in ["instalar.sh", "iniciar.sh", "calibrar.sh"]:
        lf(ENTREGA / f, out / f)
    (out / "camara").mkdir()
    lf(ENTREGA / "camara" / "capturar.py", out / "camara" / "capturar.py")
    shutil.copytree(ENTREGA / "marca", out / "marca")
    add_portable_python(out / "python311")
    (out / "calibracion").mkdir()
    (out / "calibracion" / "LEEME.txt").write_text(
        "Copie aquí entre 30 y 60 fotos de toallas BUENAS (sin defecto), tomadas con su cámara\r\n"
        "y su iluminación, y ejecute CALIBRAR.bat. Vea la sección 6 de LEEME.pdf.\r\n", encoding="utf-8")

    # modelos (sin calibración: el paquete sale con la de fábrica)
    print("modelos…")
    (out / "modelos").mkdir()
    for f in MODEL_FILES:
        shutil.copy2(C.MODELS_DIR / f, out / "modelos" / f)

    # demo
    print("demo…")
    fotos = out / "demo" / "fotos"
    fotos.mkdir(parents=True)
    rows = []
    for i, r in enumerate(pick_demo(args.n_demo, args.seed), 1):
        name = f"demo_{i:02d}{Path(r.path).suffix.lower()}"
        shutil.copy2(r.path, fotos / name)
        tipo = "" if r.label == "OK" else re.sub(r"\s+", " ", str(r.category or "Defecto")).replace("Tenido", "Teñido")
        rows.append(dict(archivo=name, real="OK" if r.label == "OK" else "NO OK", tipo=tipo))
    with open(out / "demo" / "etiquetas.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["archivo", "real", "tipo"])
        w.writeheader()
        w.writerows(rows)
    n_ng = sum(r["real"] == "NO OK" for r in rows)
    print(f"  {len(rows)} fotos ({n_ng} NO OK, {len(rows) - n_ng} OK)")

    # instrucciones
    print("instrucciones…")
    html = (ENTREGA / "LEEME.html").read_text(encoding="utf-8")
    html = html.replace("{{CSS}}", CSS.read_text(encoding="utf-8")).replace("{{N_DEMO}}", str(len(rows)))
    import base64

    html = re.sub(r"\{\{IMG:([^}]+)\}\}", lambda m: "data:image/png;base64," + base64.b64encode(
        (ENTREGA / "guia" / m.group(1)).read_bytes()).decode(), html)
    meses = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
    html = html.replace("{{FECHA}}", f"{meses[date.today().month - 1]} {date.today().year}")
    (out / "LEEME.html").write_text(html, encoding="utf-8")
    build_pdf(out / "LEEME.html", out / "LEEME.pdf")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    z = Path(shutil.make_archive(str(staging / "Inspector_Toallas"), "zip", staging, out.name))
    final = dist / "Inspector_Toallas.zip"
    shutil.move(str(z), final)
    print(f"zip: {final} ({final.stat().st_size / 1e6:.0f} MB; {size:.0f} MB descomprimido)")
    if args.carpeta:
        dst = dist / "Inspector_Toallas"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out, dst)
        print(f"carpeta: {dst}")
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
