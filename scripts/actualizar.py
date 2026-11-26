"""Regenera los archivos derivados del repositorio (sin entrenar ni usar ningún asistente de IA).

    python scripts/actualizar.py                 # todo: diagramas, curva, reportes, propuesta y zip
    python scripts/actualizar.py reportes zip    # solo esos pasos, en ese orden
    python scripts/actualizar.py --lista         # qué hace cada paso

Equivale a `make todo` / `make reportes zip` (ver Makefile). Se detiene en el primer error.
Pasos que NO incluye porque cambian el modelo: entrenamiento (scripts 05-09) y evaluación (08).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

PASOS = {  # nombre: (descripción, comando)
    "diagramas": ("Diagramas SVG de la red neuronal -> docs/diagramas/*.svg",
                  [PY, "docs/diagramas/generar_diagramas.py"]),
    "curva": ("Curva de Exigencia y probabilidad orientativa -> models/curva_operacion.json "
              "(usa reports/scores_*.csv; no corre el modelo)",
              [PY, "scripts/10_operating_curve.py"]),
    "reportes": ("Reportes técnico y ejecutivo (HTML + PDF) desde docs/reportes/fuente/",
                 [PY, "scripts/build_reportes.py"]),
    "propuesta": ("Propuesta de estación de captura -> docs/propuesta_estacion_captura_v<versión>.pdf",
                  [PY, "scripts/build_propuesta.py"]),
    "zip": ("Paquete para la empresa -> dist/Inspector_Toallas_v<versión>.zip (incluye LEEME.pdf)",
            [PY, "scripts/build_delivery.py"]),
}


def main():
    args = sys.argv[1:]
    if "--lista" in args or "-h" in args or "--help" in args:
        print(__doc__)
        for k, (d, _) in PASOS.items():
            print(f"  {k:<10} {d}")
        return
    pasos = args or list(PASOS)
    malos = [p for p in pasos if p not in PASOS]
    if malos:
        raise SystemExit(f"Paso desconocido: {', '.join(malos)}. Opciones: {', '.join(PASOS)}")
    for p in pasos:
        desc, cmd = PASOS[p]
        print(f"\n=== {p}: {desc}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            raise SystemExit(f"Falló el paso «{p}» (código {r.returncode}). Los pasos siguientes no se ejecutaron.")
        print(f"--- {p} listo en {time.time() - t0:.0f} s")
    print("\nTodo listo. Revise `git status` y haga commit de lo que cambió (dist/ no se sube a GitHub).")


if __name__ == "__main__":
    main()
