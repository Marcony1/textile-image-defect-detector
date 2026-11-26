"""Genera los diagramas SVG de la red neuronal del clasificador (EfficientNet-B0, recortes de 512 px).

    python docs/diagramas/generar_diagramas.py      # -> docs/diagramas/*.svg

Las dimensiones salen de timm.create_model("efficientnet_b0") con una entrada de 512 x 512.
"""
import math
from pathlib import Path

OUT = Path(__file__).parent
FONT = "font-family:'Segoe UI',system-ui,-apple-system,Roboto,sans-serif"
INK, INK2, INK3 = "#1b1b1a", "#52514e", "#8a8984"
BLUE, BLUE_L = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_L = "#d0702b", "#fde2c4"
GREEN, GREEN_L = "#0ca30c", "#e7f6e7"
GREY_L = "#f4f4f1"


def svg(w, h, body, title, mid="ar"):
    body = body.replace("url(#ar)", f"url(#{mid})")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{title}" style="{FONT};background:#ffffff">\n<title>{title}</title>\n'
            f'<defs><marker id="{mid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0 L8 4 L0 8 Z" fill="{INK2}"/></marker></defs>\n{body}\n</svg>\n')


def text(x, y, s, size=12, color=INK, anchor="middle", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def arrow(x1, y1, x2, y2, color=INK2):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1.5" marker-end="url(#ar)"/>'


def box(x, y, w, h, fill, stroke, label, sub=None, bold=True):
    b = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
    cy = y + h / 2 + (-2 if sub else 4)
    b += text(x + w / 2, cy, label, 12.5, INK, weight=600 if bold else 400)
    if sub:
        for i, line in enumerate(sub.split("|")):
            b += text(x + w / 2, cy + 15 + 13 * i, line, 10.5, INK2)
    return b


# --------------------------------------------------------------- 1. pipeline
def pipeline():
    steps = [("Foto de la toalla", "3580 × 2150 px", GREY_L, INK3),
             ("Máscara de la toalla", "ignora fondo y máquina", GREY_L, INK3),
             ("Recortes de 512 px", "25 % de traslape|≈ 40–60 por toalla", GREY_L, INK3),
             ("2 redes EfficientNet-B0", "misma arquitectura,|distinto entrenamiento", BLUE_L, BLUE),
             ("P(defecto) por recorte", "promedio de las 2 redes", BLUE_L, BLUE),
             ("Score de la toalla", "promedio de los 2|recortes más sospechosos", ORANGE_L, ORANGE),
             ("Calibración por cámara", "z = (score − mediana OK)|÷ rango intercuartil OK", ORANGE_L, ORANGE),
             ("Decisión", "z ≥ umbral → NO OK|+ mapa de calor", GREEN_L, GREEN)]
    w, h, gap = 150, 74, 26
    per_row = 4
    body = ""
    pos = []
    for i, (a, b, f, s) in enumerate(steps):
        r, c = divmod(i, per_row)
        x = 20 + c * (w + gap)
        y = 30 + r * (h + 60)
        pos.append((x, y))
        body += box(x, y, w, h, f, s, a, b)
    for i in range(len(steps) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:
            body += arrow(x1 + w, y1 + h / 2, x2 - 2, y2 + h / 2)
        else:  # salto de renglón
            body += (f'<path d="M{x1 + w / 2} {y1 + h} L{x1 + w / 2} {y1 + h + 30} L{x2 + w / 2} {y1 + h + 30} '
                     f'L{x2 + w / 2} {y2 - 2}" fill="none" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ar)"/>')
    body += text(20, 290, "Azul: la red neuronal · Naranja: cómo se combinan y calibran sus salidas · Verde: decisión que se envía al tablero y al PLC",
                 10.5, INK3, "start")
    return svg(20 + per_row * (w + gap) - gap + 20, 300, body, "De la foto a la decisión: recorrido completo", "ar_pipe")


# -------------------------------------------------------------------- 2. CNN
LAYERS = [  # nombre, detalle, lado (px), canales
    ("Entrada", "recorte RGB", 512, 3),
    ("Stem", "conv 3×3 /2", 256, 32),
    ("Etapa 1", "MBConv1 3×3 ×1", 256, 16),
    ("Etapa 2", "MBConv6 3×3 ×2", 128, 24),
    ("Etapa 3", "MBConv6 5×5 ×2", 64, 40),
    ("Etapa 4", "MBConv6 3×3 ×3", 32, 80),
    ("Etapa 5", "MBConv6 5×5 ×3", 32, 112),
    ("Etapa 6", "MBConv6 5×5 ×4", 16, 192),
    ("Etapa 7", "MBConv6 3×3 ×1", 16, 320),
    ("Cabeza", "conv 1×1", 16, 1280),
]


def cnn():
    body = ""
    x, cy = 24, 175
    dx, dy = 10, -9
    tops = []
    for i, (name, det, side, ch) in enumerate(LAYERS):
        hgt = 34 + 26 * math.log2(side / 16)        # alto ∝ tamaño espacial
        wid = 6 + 5.0 * math.log2(ch / 3 + 1)      # ancho ∝ canales (escala log)
        y = cy - hgt / 2
        fill, stroke = (GREY_L, INK3) if i == 0 else (BLUE_L, BLUE) if i < len(LAYERS) - 1 else (ORANGE_L, ORANGE)
        # bloque 3D: cara frontal + cara superior + cara lateral
        body += (f'<polygon points="{x},{y} {x + dx},{y + dy} {x + wid + dx},{y + dy} {x + wid},{y}" fill="{fill}" stroke="{stroke}"/>'
                 f'<polygon points="{x + wid},{y} {x + wid + dx},{y + dy} {x + wid + dx},{y + hgt + dy} {x + wid},{y + hgt}" fill="{fill}" stroke="{stroke}" opacity=".85"/>'
                 f'<rect x="{x}" y="{y}" width="{wid}" height="{hgt}" fill="{fill}" stroke="{stroke}"/>')
        lab_y = cy + 100 if i % 2 == 0 else cy - 120
        body += text(x + wid / 2 + 4, lab_y, name, 13, INK, weight=600)
        body += text(x + wid / 2 + 4, lab_y + 15, det, 11.5, INK2)
        body += text(x + wid / 2 + 4, lab_y + 30, f"{side}×{side}×{ch}", 11.5, BLUE if i else INK2)
        body += f'<line x1="{x + wid / 2 + 4}" y1="{(y + hgt + 4) if i % 2 == 0 else (y + dy - 4)}" x2="{x + wid / 2 + 4}" y2="{(lab_y - 13) if i % 2 == 0 else (lab_y + 35)}" stroke="#d6d5cf"/>'
        tops.append((x, wid))
        x += wid + dx + 21
    # pooling y capa densa
    px = x + 4
    body += arrow(x - 16, cy, px, cy)
    body += box(px + 2, cy - 30, 96, 60, ORANGE_L, ORANGE, "Pool global", "promedio de|cada canal")
    nx = px + 124
    body += arrow(px + 100, cy, nx - 8, cy)
    # 1280 neuronas -> 1 neurona
    ys = [cy - 84 + 21 * k for k in range(9)]
    out_x = nx + 100
    for k, yy in enumerate(ys):
        if k == 4:
            body += text(nx + 8, yy + 4, "⋮", 16, INK2)
            continue
        body += f'<line x1="{nx + 8}" y1="{yy}" x2="{out_x - 12}" y2="{cy}" stroke="#c9c8c2" stroke-width=".8"/>'
        body += f'<circle cx="{nx + 8}" cy="{yy}" r="7" fill="{ORANGE_L}" stroke="{ORANGE}"/>'
    body += text(nx + 8, cy + 114, "1 280 neuronas", 12, INK, weight=600)
    body += text(nx + 8, cy + 129, "(una por canal)", 11, INK2)
    body += f'<circle cx="{out_x}" cy="{cy}" r="12" fill="{GREEN_L}" stroke="{GREEN}" stroke-width="1.5"/>'
    body += text(out_x, cy + 36, "1 neurona", 12, INK, weight=600)
    body += text(out_x, cy + 51, "+ sigmoide", 11, INK2)
    body += arrow(out_x + 14, cy, out_x + 52, cy)
    body += text(out_x + 58, cy - 4, "P(defecto)", 13.5, GREEN, "start", 600)
    body += text(out_x + 58, cy + 13, "entre 0 y 1", 11, INK2, "start")
    W = out_x + 135
    body += text(24, 400, "Alto del bloque ∝ tamaño de la imagen (512 → 16 px) · Ancho ∝ canales (3 → 1 280), escala logarítmica · 4.0 M parámetros",
                 11.5, INK3, "start")
    body += text(24, 418, "Cada canal es un detector aprendido (bordes, hilos, manchas, texturas…) que se aplica en cada posición de la imagen.",
                 11.5, INK3, "start")
    return svg(W, 430, body, "Red neuronal convolucional EfficientNet-B0 aplicada a un recorte de 512 × 512 píxeles", "ar_cnn")


# ---------------------------------------------------------------- 3. MBConv
def mbconv():
    items = [("Entrada", "H×W×C", GREY_L, INK3),
             ("Conv 1×1", "expande ×6 (×1 en etapa 1)|H×W×6C", BLUE_L, BLUE),
             ("Conv k×k", "por canal (depthwise)|k = 3 o 5", BLUE_L, BLUE),
             ("Atención SE", "pesa cada canal|(squeeze-excitation)", ORANGE_L, ORANGE),
             ("Conv 1×1", "proyecta a C'|H'×W'×C'", BLUE_L, BLUE),
             ("Suma", "con la entrada|(si C' = C)", GREEN_L, GREEN)]
    w, h, gap = 128, 70, 22
    body = ""
    for i, (a, b, f, s) in enumerate(items):
        x = 16 + i * (w + gap)
        body += box(x, 30, w, h, f, s, a, b)
        if i:
            body += arrow(x - gap, 65, x - 2, 65)
    # conexión residual
    x0, x5 = 16 + w / 2, 16 + 5 * (w + gap) + w / 2
    body += (f'<path d="M{x0} 100 L{x0} 128 L{x5} 128 L{x5} 102" fill="none" stroke="{GREEN}" stroke-width="1.5" '
             f'stroke-dasharray="5 4" marker-end="url(#ar)"/>')
    body += text((x0 + x5) / 2, 145, "conexión residual: la red aprende solo la corrección; facilita entrenar redes profundas", 10.5, GREEN)
    body += text(16, 172, "Bloque MBConv (convolución invertida con cuello de botella). Las capas convolucionales usan normalización por lotes y activación SiLU.",
                 10.5, INK3, "start")
    return svg(16 + 6 * (w + gap) - gap + 16, 182, body, "Bloque MBConv de EfficientNet", "ar_mb")


if __name__ == "__main__":
    for name, fn in [("pipeline_decision.svg", pipeline), ("cnn_efficientnet_b0.svg", cnn), ("bloque_mbconv.svg", mbconv)]:
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(OUT / name)
