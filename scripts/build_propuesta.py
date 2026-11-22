"""Genera docs/propuesta_estacion_captura.pdf a partir de su HTML (Microsoft Edge en modo headless).

    python scripts/build_propuesta.py
"""
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "propuesta_estacion" / "propuesta_estacion_captura.html"
PDF = ROOT / "docs" / "propuesta_estacion_captura.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def main():
    css = (ROOT / "entrega" / "estilo.css").read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp()) / "propuesta.html"
    tmp.write_text(SRC.read_text(encoding="utf-8").replace("{{CSS}}", css), encoding="utf-8")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={Path(tempfile.gettempdir()) / 'galia_edgeprof'}",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF}", tmp.as_uri()], timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
