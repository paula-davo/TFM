"""
Gráfica de barras comparativa de la evaluación.

Leyenda:
  - VERDE = menor error (mejor modelo).
  - AZUL  = resto de modelos.
Se excluye el GT. Datos copiados de la salida de 01_evaluate_metrics_oficial.py. 
Guarda figuras/barras_3d.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "/mnt/c/Users/paula/Desktop/TFM/evaluate_predictions/figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICAS = ["MAE", "RMSE", "MSE", "MASE", "ADE", "FDE"]

LABELS = ["Det", "Det+\nEntorno", "Det+\nSocial", "Det+\nPose",
          "Det+\nRuido", "Det+\nReflejo", "Fus.\ntemp.", "Fus.temp.\nreg.", "Fus.\ntard.", "Multitr.\n(franja)"]

# Métricas [MAE, RMSE, MSE, MASE, ADE, FDE], mismo orden que LABELS. Última corrida de 01 (incluye el paso 10).
DATA = [
    [0.13080, 0.25587, 6.55e-2, 0.803, 0.20668, 0.31021],   # Detecciones 3D
    [0.13547, 0.26087, 6.81e-2, 0.832, 0.21369, 0.31781],   # Det + entorno
    [0.13358, 0.25859, 6.69e-2, 0.825, 0.21073, 0.31582],   # Det + social
    [0.13334, 0.25960, 6.74e-2, 0.819, 0.21048, 0.31122],   # Det + pose
    [0.13224, 0.26088, 6.81e-2, 0.812, 0.20903, 0.31440],   # Det + generalización (ruido)
    [0.13545, 0.25670, 6.59e-2, 0.832, 0.21340, 0.31185],   # Det + generalización (reflejo)
    [0.12785, 0.24775, 6.14e-2, 0.785, 0.20208, 0.30287],   # Fusión temprana (07)
    [0.12794, 0.24789, 6.14e-2, 0.786, 0.20162, 0.30323],   # Fusión temprana regularizada (10)
    [0.13804, 0.26895, 7.23e-2, 0.848, 0.21899, 0.32274],   # Fusión tardía
    [0.13084, 0.25461, 6.48e-2, 0.803, 0.20662, 0.31093],   # Multitrayectoria
]

BASE_C, BEST_C = "tab:blue", "tab:green"


def barras_grupo(labels, data):
    # (10 modelos, 6 métricas)
    M = np.array(data)
    # 6 subplots (uno por métrica)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    for j, (ax, met) in enumerate(zip(axes.ravel(), METRICAS)):
        # Para cada métrica, obtiene los valores de esa métrica para cada modelo
        vals = M[:, j]
        # Asigna el color azul a todos menos al mejor (verde)
        colors = [BASE_C] * len(vals)
        colors[int(np.argmin(vals))] = BEST_C
        # Dibujo
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(met, fontsize=16, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3)
        if met == "MSE":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.4f}" if v < 1 else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, vals.max() * 1.18)
    fig.suptitle("Espacio físico 3D (metros)", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = f"{OUTPUT_DIR}/barras_3d.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {path}")


if __name__ == "__main__":
    barras_grupo(LABELS, DATA)
    print("Leyenda: verde = mejor modelo (menor error); azul = resto.")
