"""
Ilustración del registro ICP. Se toman dos barridos LiDAR consecutivos y se muestra
antes de ICP (nubes desplazadas por el movimiento) y después de ICP (nubes alineadas).

Uso:
    python evitacion/01a_ilustracion_icp.py [secuencia] [frame] [salto]
Por defecto: huang-2-2019-01-25_0, frame 250, salto 25.
"""
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evitacion_common import load_cloud_ego

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
PCD = TRAIN_DIR / "pointclouds" / "lower_velodyne"
OUTPUT_DIR = Path(DATA_DIR) / "evitacion" / "figuras"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOXEL = 0.25       # submuestreo de la nube
CROP_R = 25.0      # recorte
ICP_DIST = 1.2     # máxima distancia de correspondencia


def draw_cloud(ax, P, color):
    # Dibuja nube de puntos
    ax.scatter(P[:, 0], P[:, 1], s=5, c=color, alpha=0.55, linewidths=0)


def draw_robot(ax, pos, head, color):
    # Dibujo del robot: triángulo orientado según la dirección de avance
    h = head / (np.linalg.norm(head) + 1e-9)
    ang_deg = np.degrees(np.arctan2(h[1], h[0])) - 90.0
    ax.scatter(*pos, marker=(3, 0, ang_deg), s=420, c=color, edgecolors="white",
               linewidths=1.6, zorder=7)
    ax.annotate("", xy=pos + 2.0 * h, xytext=pos,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=3.2), zorder=6)


def h_cloud(color, label):
    # Leyenda para la nube de puntos
    return Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
                  markeredgecolor="none", markersize=14, label=label)


def h_robot(color, label):
    # Leyenda para el robot
    return Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=color,
                  markeredgecolor="white", markersize=17, label=label)


def main():
    SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0"
    F = int(sys.argv[2]) if len(sys.argv) > 2 else 250     # frame de la nube objetivo
    GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 25    # salto hasta la nube fuente

    # 1. Objetivo y fuente
    target = load_cloud_ego(PCD, SEQ, F, VOXEL, CROP_R)
    source = load_cloud_ego(PCD, SEQ, F + GAP, VOXEL, CROP_R)
    if target is None or source is None:
        raise SystemExit(f"No existe la nube de {SEQ} (frames {F} o {F + GAP})")

    # 2. ICP punto a plano: alinea la fuente sobre el objetivo
    reg = o3d.pipelines.registration.registration_icp(
        source, target, ICP_DIST, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    T = reg.transformation
    print(f"ICP: fitness={reg.fitness:.3f}  rmse={reg.inlier_rmse:.3f}")

    # 3. Nubes en 2D (vista cenital, plano X-Y)
    tgt = np.asarray(target.points)[:, :2]
    src = np.asarray(source.points)[:, :2]
    src_aligned = (T @ np.c_[np.asarray(source.points), np.ones(len(source.points))].T).T[:, :2]

    # Pose del robot en el marco del objetivo
    rob0 = np.array([0.0, 0.0])
    rob1 = T[:2, 3]
    head0 = np.array([1.0, 0.0])                  # orientación en F
    head1 = T[:2, 0]                              # orientación en F+GAP
    # Desplazamiento estimado por el ICP: traslación y giro
    disp = float(np.linalg.norm(rob1))
    ang = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    print(f"Desplazamiento estimado: Δ = {disp:.2f} m, {ang:+.1f}°")

    # 4. Figura de dos plots
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: antes de ICP
    draw_cloud(axA, tgt, "steelblue")
    draw_cloud(axA, src, "darkorange")
    axA.set_title("Antes de ICP", fontsize=22)

    # Plot 2: después de ICP
    draw_cloud(axB, tgt, "steelblue")
    draw_cloud(axB, src_aligned, "darkorange")
    axB.plot([rob0[0], rob1[0]], [rob0[1], rob1[1]], "k--", lw=2.0, zorder=5)  # desplazamiento estimado
    draw_robot(axB, rob0, head0, "black")
    draw_robot(axB, rob1, head1, "red")
    axB.text(0.03, 0.03, f"Desplazamiento estimado (ICP):\n$\\Delta$ = {disp:.2f} m,  {ang:+.1f}°",
             transform=axB.transAxes, fontsize=17, va="bottom", ha="left", zorder=8,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.9))
    axB.set_title("Después de ICP", fontsize=22)

    # Zoom: encuadra el 90% de los puntos más cercanos al robot
    allpts = np.vstack([tgt, src, src_aligned])
    span = float(np.percentile(np.linalg.norm(allpts, axis=1), 90))
    for ax in (axA, axB):
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        ax.set_xlabel("x (m)", fontsize=19); ax.set_ylabel("y (m)", fontsize=19)
        ax.tick_params(labelsize=16)

    # Leyenda común
    leg_handles = [h_cloud("steelblue", "Nube objetivo (barrido $k$)"),
                   h_cloud("darkorange", "Nube fuente (barrido $k{+}1$)"),
                   h_robot("black", "Robot en $k$"),
                   h_robot("red", "Robot en $k{+}1$")]
    fig.suptitle("Registro ICP de dos barridos LiDAR y pose del robot",
                 fontsize=24, fontweight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.legend(handles=leg_handles, loc="lower center", ncol=4, fontsize=18,
               frameon=True, bbox_to_anchor=(0.5, 0.02))
    out = OUTPUT_DIR / f"fase2_icp_ilustracion_{SEQ}_{F}-{F + GAP}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
