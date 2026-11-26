"""Genera los reportes (HTML autocontenido + PDF) a partir de sus fuentes editables.

    python scripts/build_reportes.py                 # técnico y ejecutivo
    python scripts/build_reportes.py tecnico         # solo uno

Fuentes:  docs/reportes/fuente/reporte_<nombre>.html   (texto; edítelo aquí)
          docs/reportes/fuente/img/*.png               (figuras)
          docs/diagramas/*.svg                         (python docs/diagramas/generar_diagramas.py)
Salida:   docs/reportes/reporte_<nombre>.html y .pdf

Marcadores en la fuente:  {{IMG:archivo.png}} -> imagen incrustada · {{SVG:archivo.svg}} -> diagrama incrustado.
El PDF se imprime con Microsoft Edge en modo headless; no hace falta ningún modelo ni GPU.
"""
import base64
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "reportes" / "fuente"
OUT = ROOT / "docs" / "reportes"
DIAGRAMAS = ROOT / "docs" / "diagramas"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def inline_svg(name: str) -> str:
    s = (DIAGRAMAS / name).read_text(encoding="utf-8")
    s = re.sub(r'<svg xmlns="http://www.w3.org/2000/svg" viewBox="([^"]+)" width="[\d.]+" height="[\d.]+"',
               r'<svg xmlns="http://www.w3.org/2000/svg" viewBox="\1" style="width:100%;height:auto;display:block"', s, count=1)
    return s.replace("background:#ffffff", "background:#ffffff;width:100%;height:auto")


def render(name: str) -> Path:
    html = (SRC / f"reporte_{name}.html").read_text(encoding="utf-8")
    html = re.sub(r"\{\{IMG:([^}]+)\}\}", lambda m: "data:image/png;base64," + base64.b64encode(
        (SRC / "img" / m.group(1)).read_bytes()).decode(), html)
    html = re.sub(r"\{\{SVG:([^}]+)\}\}", lambda m: inline_svg(m.group(1)), html)
    left = re.findall(r"\{\{[A-Z]+:[^}]*\}\}", html)
    if left:
        raise SystemExit(f"Marcadores sin resolver en {name}: {left}")
    out = OUT / f"reporte_{name}.html"
    out.write_text(html, encoding="utf-8")
    return out


def to_pdf(html: Path) -> Path:
    pdf = html.with_suffix(".pdf")
    tmp = pdf.with_name(pdf.stem + ".nuevo.pdf")  # el PDF anterior solo se reemplaza si Edge tuvo éxito
    tmp.unlink(missing_ok=True)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={tempfile.mkdtemp(prefix='edge_')}",
                    "--no-pdf-header-footer", f"--print-to-pdf={tmp}", html.resolve().as_uri()], timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):  # Edge a veces se relanza (actualización pendiente): esperar el archivo
        if tmp.exists() and tmp.stat().st_size > 10_000:
            time.sleep(1)
            break
        time.sleep(1)
    if not tmp.exists() or tmp.stat().st_size < 10_000:
        raise SystemExit(f"Edge no generó {pdf.name}; se conserva el anterior.")
    tmp.replace(pdf)
    return pdf


def main():
    names = sys.argv[1:] or ["tecnico", "ejecutivo"]
    for n in names:
        html = render(n)
        msg = f"{html.relative_to(ROOT)} ({html.stat().st_size // 1024} KB)"
        if Path(EDGE).exists():
            pdf = to_pdf(html)
            msg += f" -> {pdf.relative_to(ROOT)} ({pdf.stat().st_size // 1024} KB)"
        else:
            msg += " (Edge no encontrado: solo HTML)"
        print(msg)


if __name__ == "__main__":
    main()
