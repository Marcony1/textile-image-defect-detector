"""Numeración de los documentos del proyecto (01, 02, 03...). Un solo lugar para cambiarla.

Los documentos 01-03 son los reportes de los notebooks de la versión 1 (docs/0X_*.pdf, ya no se regeneran).
Para agregar un documento nuevo: darle el siguiente número aquí y usar `nombre_archivo()` en su script.
"""
DOCUMENTOS = {  # clave: (número, nombre de archivo sin número ni extensión, título)
    "baseline": ("01", "textile_defect_training_baseline", "Notebook v1: entrenamiento base"),
    "raw_split": ("02", "textile_defect_training_baseline_raw_split", "Notebook v1: partición original"),
    "balanced": ("03", "textile_defect_training_balanced", "Notebook v1: clases balanceadas"),
    "tecnico": ("04", "reporte_tecnico", "Reporte técnico (pipeline v2)"),
    "ejecutivo": ("05", "reporte_ejecutivo", "Reporte ejecutivo"),
    "propuesta": ("06", "propuesta_estacion_captura", "Propuesta de estación de captura"),
    "experimentos": ("07", "reporte_experimentos", "Reporte de experimentos de modelos"),
}


def nombre_archivo(clave: str) -> str:
    """'tecnico' -> '04_reporte_tecnico'"""
    num, nombre, _ = DOCUMENTOS[clave]
    return f"{num}_{nombre}"
