"""Genera docs/06_propuesta_estacion_captura_v<versión>.pdf (número: scripts/documentos.py) a partir de su HTML (Microsoft Edge en modo headless).

    python scripts/build_propuesta.py
"""
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from documentos import nombre_archivo  # noqa: E402

SRC = ROOT / "docs" / "propuesta_estacion" / "propuesta_estacion_captura.html"
DOC_VERSION = "1.1"  # súbala cuando cambie el contenido (y anote el cambio en el recuadro de la portada)
PDF = ROOT / "docs" / f"{nombre_archivo('propuesta')}_v{DOC_VERSION}.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def main():
    css = (ROOT / "entrega" / "estilo.css").read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp()) / "propuesta.html"
    tmp.write_text(SRC.read_text(encoding="utf-8").replace("{{CSS}}", css), encoding="utf-8")
    if PDF.exists():
        PDF.unlink()
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={Path(tempfile.mkdtemp(prefix='edge_'))}",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF}", tmp.as_uri()], timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Edge a veces se relanza (actualización pendiente) y el primer proceso termina enseguida: esperar el archivo
    import time
    for _ in range(90):
        if PDF.exists() and PDF.stat().st_size > 10_000:
            time.sleep(1)
            break
        time.sleep(1)
    if not PDF.exists() or PDF.stat().st_size < 10_000:
        raise SystemExit("Edge no generó el PDF; no se borró ninguna versión anterior.")
    for old in (ROOT / "docs").glob("*propuesta_estacion_captura*.pdf"):  # solo la versión vigente
        if old != PDF:
            old.unlink()
    print(f"{PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
