"""
Diagramas de bloques de la arquitectura de cada modelo del proyecto.
Genera un PNG por modelo en figuras/diagramas/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OUTPUT_DIR = f"{DATA_DIR}/figuras/diagramas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IN, MID, OUTC = "#cfe8ff", "#eeeeee", "#ffe0b3"   # colores entrada / intermedio / salida
LOSSC = "#fff2c2"                               # etiqueta de función de pérdida
WB, HB = 2.9, 1.9                                 # tamaño de caja (unidades de datos)
STEP = 3.5                                        # separación entre cajas de una cadena
SCALE = 0.5                                       # pulgadas por unidad de datos
FS_HEAD, FS_BODY, FS_TITLE = 10, 7.5, 13          # tamaños de letra


def _box(ax, cx, cy, head, body, fc, w=WB, h=HB, fs_head=FS_HEAD, fs_body=FS_BODY):
    # Dibuja un bloque del diagrama
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 linewidth=1.3, edgecolor="black", facecolor=fc))
    # Si body no está vacío
    if body:
        # Título arriba, detalle debajo en letra menor
        ax.text(cx, cy + 0.28 * h, head, ha="center", va="center",
                fontsize=fs_head, weight="bold")
        ax.text(cx, cy - 0.13 * h, body, ha="center", va="center",
                fontsize=fs_body, color="#333333", linespacing=1.25)
    else:
        ax.text(cx, cy, head, ha="center", va="center", fontsize=fs_head, weight="bold")


def _arrow(ax, x0, y0, x1, y1):
    # Dibuja flecha entre dos puntos conectando los bloques
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))


def _lossbox(ax, cx, cy, loss):
    # Etiqueta de la función de pérdida de entrenamiento
    w, h = 3.0, 0.6
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 linewidth=1.0, edgecolor="#b58900", facecolor=LOSSC))
    ax.text(cx, cy, f"Pérdida: {loss}", ha="center", va="center",
            fontsize=FS_BODY, weight="bold", color="#5a4a00")


def _finish(fig, ax, title, outname, xlim, ylim):
    # Fija los límites de la figura
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # El título
    ax.set_title(title, fontsize=FS_TITLE, weight="bold")
    # Quita los ejes
    ax.axis("off")
    fig.tight_layout()
    # Guarda
    fig.savefig(f"{OUTPUT_DIR}/{outname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", outname)


def chain(blocks, title, outname, note=None, loss=None, hb=HB, fs_head=FS_HEAD, fs_body=FS_BODY):
    n = len(blocks)
    # Extensión horizontal
    xlim = (-WB / 2 - 0.3, (n - 1) * STEP + WB / 2 + 0.3)
    # Margen vertical
    edge = max(1.4, hb / 2 + 0.35)
    # Suma si hay loss y resta si hay note
    bottom = -edge - (0.9 if loss else 0.0) - (0.7 if note else 0.0)
    # Altura del diagrama
    ylim = (bottom, edge)
    # Subplot
    fig, ax = plt.subplots(figsize=((xlim[1] - xlim[0]) * SCALE, (ylim[1] - ylim[0]) * SCALE))
    # Dibuja los bloques y las flechas
    for i, (head, body, fc) in enumerate(blocks):
        cx = i * STEP
        _box(ax, cx, 0, head, body, fc, h=hb, fs_head=fs_head, fs_body=fs_body)
        if i > 0:
            _arrow(ax, (i - 1) * STEP + WB / 2, 0, cx - WB / 2, 0)
    # Dibuja el cuadro de pérdida si está activo
    cxmid = (xlim[0] + xlim[1]) / 2
    y = -edge - 0.5
    if loss:
        _lossbox(ax, cxmid, y, loss)
        y -= 0.8
    # Si hay nota, la añade
    if note:
        ax.text(cxmid, y, note, ha="center", va="center",
                fontsize=fs_body, style="italic", color="#333333", linespacing=1.25)
    # Cierra el diagrama
    _finish(fig, ax, title, outname, xlim, ylim)


def branched(boxes, arrows, title, outname, xlim, ylim, note=None, loss=None,
             hb=HB, fs_head=FS_HEAD, fs_body=FS_BODY):
    # Cuando hay varias filas en el diagrama
    # Si hay loss, se actualiza altura
    if loss:
        ylim = (ylim[0] - 0.9, ylim[1])
    # Subplots
    fig, ax = plt.subplots(figsize=((xlim[1] - xlim[0]) * SCALE, (ylim[1] - ylim[0]) * SCALE))
    # Dibuja los bloques
    for cx, cy, head, body, fc in boxes:
        _box(ax, cx, cy, head, body, fc, h=hb, fs_head=fs_head, fs_body=fs_body)
    # Dibuja las flechas
    for a in arrows:
        _arrow(ax, *a)
    cxmid = (xlim[0] + xlim[1]) / 2
    # Si loss está activo, dibuja la caja de loss
    if loss:
        _lossbox(ax, cxmid, ylim[0] + 0.45, loss)
    # Si hay note, la añade
    if note:
        ax.text(cxmid, ylim[0] + (1.15 if loss else 0.35), note, ha="center", va="bottom",
                fontsize=fs_body, style="italic", color="#333333", linespacing=1.25)
    # Cierra el diagrama
    _finish(fig, ax, title, outname, xlim, ylim)


def main():
    # Bloques reutilizados
    ENC = ("Encoder LSTM", "128 unidades\ndropout 0.3\nrec. dropout 0.2", MID)
    DEC = ("Decoder LSTM", "128 unidades\nRepeatVector +\nestado del encoder", MID)
    DEN = ("Densa", "64, ReLU\nTimeDistributed", MID)
    NOI = ("Ruido\ngaussiano", "σ = 0.05\ncapa, solo en train", MID)
    AUG = ("Aumento\nde datos", "reflejo lateral\n(train ×2)", MID)

    OUT_3D = ("Salida", "Dense(2)\n8 × 2\ndesplazamiento\nfuturo (1.6 s), m", OUTC)

    # LSTM 3D: N_obs observados -> N_pred predicciones
    chain([("Entrada", "N_obs × 9\nobservación\n(metros)", IN), ENC, DEC, DEN,
           ("Salida", "Dense(2)\nN_pred × 2\ndesplazamiento\nfuturo (metros)", OUTC)],
          "Modelo LSTM 3D (etiquetas / detecciones)", "arq_lstm_3d", loss="MAE")

    # Adición de entorno LiDAR (05)
    chain([("Entrada", "8 × 16\n9 de detecciones +\n7 de entorno", IN), ENC, DEC, DEN, OUT_3D],
          "Modelo LSTM 3D + entorno", "arq_lstm_3d_entorno", loss="MAE")

    # LSTM Social (05c)
    chain([("Entrada escena", "peatones × 8 × 9\n+ máscara y\nposición 3D (m)", IN),
           ("Encoder LSTM", "128 unidades\ncompartido entre\npeatones", MID),
           ("Social pooling", "estados de vecinos\na < 2 m (distancia\nfísica real)", MID),
           ("Densa\ncombinación", "estado propio +\ncontexto social", MID),
           DEC, DEN,
           ("Salida", "Dense(2)\npeatones × 8 × 2\ndesplazamiento\nfuturo (1.6 s), m", OUTC)],
          "Social LSTM 3D (metros)", "arq_social_3d", loss="MAE", hb=2.4, fs_head=11, fs_body=9)

    # Adición de información de pose (05f)
    chain([("Entrada", "8 × 17\n9 de detecciones +\n8 de pose corporal", IN), ENC, DEC, DEN, OUT_3D],
          "Modelo LSTM 3D + pose", "arq_pose_3d", loss="MAE")

    # Generalización 3D: ruido gaussiano (06)
    chain([("Entrada", "8 × 9\n(metros)", IN), NOI, ENC, DEC, DEN, OUT_3D],
          "Modelo LSTM 3D con generalización — ruido gaussiano", "arq_generalizacion_3d_ruido",
          loss="MAE", hb=2.4, fs_head=11, fs_body=9)

    # Generalización 3D: reflejo lateral (06b)
    chain([("Entrada", "8 × 9\n(metros)", IN), AUG, ENC, DEC, DEN, OUT_3D],
          "Modelo LSTM 3D con generalización — reflejo lateral", "arq_generalizacion_3d_reflejo",
          loss="MAE", hb=2.4, fs_head=11, fs_body=9)

    # Fusión temprana 07
    chain([("Entrada", "8 × 20\n11 de imagen +\n9 de detección 3D", IN), ENC, DEC, DEN, OUT_3D],
          "Combinación 2D + 3D — fusión temprana", "arq_combinado_temprana", loss="MAE")

    # Multitrayectoria 09
    chain([("Entrada", "8 × 20\n11 de imagen +\n9 de detección 3D", IN), ENC, DEC, DEN,
           ("Proyección final", "Densa K × 2 = 40 +\nReshape + Permute\na K hipótesis", MID),
           ("Salida", "20 × 8 × 2\nK = 20 trayectorias\n(1.6 s), m", OUTC)],
          "Modelo multitrayectoria sobre fusión temprana", "arq_multitrayectoria_3d",
          loss="MAE (best-of-K)", hb=2.4, fs_head=11, fs_body=9)

    # Fusión tardía (08): dos modelos independientes -> promedio
    branched(
        boxes=[(0, 1.6, "Entrada 2D", "8 × 11\n(píxeles)", IN),
               (3.7, 1.6, "Modelo LSTM", "2D → metros\nentrenado", MID),
               (0, -1.6, "Entrada 3D", "8 × 9\n(metros)", IN),
               (3.7, -1.6, "Modelo LSTM", "3D → metros\nentrenado", MID),
               (7.4, 0, "Promedio", "media de las dos\npredicciones", MID),
               (11.1, 0, "Salida", "8 × 2\ndesplazamiento\nfuturo (1.6 s), m", OUTC)],
        arrows=[(0 + WB / 2, 1.6, 3.7 - WB / 2, 1.6),
                (0 + WB / 2, -1.6, 3.7 - WB / 2, -1.6),
                (3.7 + WB / 2, 1.6, 7.4 - WB / 2, 0.4),
                (3.7 + WB / 2, -1.6, 7.4 - WB / 2, -0.4),
                (7.4 + WB / 2, 0, 11.1 - WB / 2, 0)],
        title="Combinación 2D + 3D — fusión tardía",
        outname="arq_fusion_tardia",
        xlim=(-1.7, 12.9), ylim=(-3.0, 3.0), loss="MAE", fs_head=12, fs_body=9)

    print(f"\nTodos los diagramas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
