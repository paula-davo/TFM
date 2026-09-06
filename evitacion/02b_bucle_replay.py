"""
El robot continúa su camino grabado y los peatones su trayectoria, mostrando
las predicciones realizadas cada K pasos.

Uso:
    python evitacion/02b_bucle_replay.py [secuencia] [frame_ini] [frame_fin] [K]
Por defecto: huang-2-2019-01-25_0, 120, 360, K=3.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LABELS_DIR = TRAIN_DIR / "labels" / "labels_3d"
DET_DIR = TRAIN_DIR / "detections" / "detections_3d"
RECON_DIR = Path(DATA_DIR) / "evitacion" / "reconstruccion"
OUTPUT_DIR = Path(DATA_DIR) / "evitacion" / "figuras"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from evitacion_common import (load_gt, load_det, load_poses, load_img2d,
                              nearest_det, fill, to_world, predict_scene)

OBS, PRED, STRIDE = 8, 8, 3
TRAIL = 8        # Frames observación
VIEW_R = 3.0     # Margen de recorte de vista en torno al robot


def ped_trail(c, t, poses, det, gt):
    # Calcula la estela de los frames de observación -> detecciones 3D
    # Frames necesarios (con stride y trail) con pose del robot y etiqueta del peatón
    fr = [c - STRIDE * (TRAIL - 1 - i) for i in range(TRAIL)]
    fr = [f for f in fr if f in poses and t in gt.get(f, {})]
    if not fr:
        return None
    # Detección más cercana a la etiqueta en cada frame (rellena huecos) -> marco mundo
    od = np.array(fill([nearest_det(gt[f][t], det.get(f, np.zeros((0, 2), np.float32))) for f in fr]),
                  np.float32)
    return np.array([to_world(poses, f, od[i]) for i, f in enumerate(fr)])


def draw(ax, c, fans, poses, gt, det, color, CS, view):
    # Vista fija entorno al robot
    VX0, VX1, VY0, VY1 = view
    ax.clear()
    ax.set_xlim(VX0, VX1); ax.set_ylim(VY0, VY1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)", fontsize=19); ax.set_ylabel("y (m)", fontsize=19); ax.tick_params(labelsize=16)
    # Dibuja el recorrido del robot, marcando la posición actual
    up_to = [poses[cc][:2, 3] for cc in CS if cc <= c]
    rp = np.array(up_to)
    ax.plot(rp[:, 0], rp[:, 1], color="0.6", lw=1.5, ls=":")
    ax.scatter(*poses[c][:2, 3], c="k", marker="^", s=150, zorder=6, edgecolors="white")
    # Recorre los peatones en el frame actual
    for t in gt.get(c, {}):
        # Solo peatones dentro de la ventana fija
        pw = to_world(poses, c, gt[c][t])
        if not (VX0 <= pw[0] <= VX1 and VY0 <= pw[1] <= VY1):
            continue
        # Dibuja su observación
        tr = ped_trail(c, t, poses, det, gt)
        if tr is None or len(tr) < 2:
            continue
        col = color[t]
        ax.plot(tr[:, 0], tr[:, 1], "-", color=col, lw=2.0, zorder=4)
        ax.scatter(tr[-1, 0], tr[-1, 1], color=col, s=22, zorder=5)
        # Si el peatón tiene predicción -> la dibuja
        if t in fans:
            seg = np.vstack([tr[-1], fans[t]])
            ax.plot(seg[:, 0], seg[:, 1], "--", color=col, lw=1.8, alpha=0.9, zorder=2)


def main():
    SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0"  # Secuencia
    F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 120                 # Frame inicial
    F1 = int(sys.argv[3]) if len(sys.argv) > 3 else 360                 # Frame final
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 3          # re-planificar cada K pasos

    # Frames actuales
    CS = list(range(F0, F1 + 1, STRIDE))

    # 1. Cargar poses de robot -> marco mundo
    # Frames para los que se necesita pose
    need_lo, need_hi = min(CS) - STRIDE * (OBS - 1), max(CS)
    poses = load_poses(RECON_DIR, SEQ, range(need_lo, need_hi + 1))
    if poses is None:
        raise SystemExit(f"No hay .npz que cubra {need_lo}..{need_hi}. Ejecuta antes 01b con ese rango.")

    # 2. Carga de datos: etiquetas, detecciones 3D y trayectorias 2D de imagen
    gt, det = load_gt(LABELS_DIR, SEQ), load_det(DET_DIR, SEQ)
    traj2d = load_img2d(f"{DATA_DIR}/jrdb_clean_trajectories.csv", SEQ)

    # 3. Preparación de los datos
    # Carga imagen 2D (11) + detecciones 3D (9), con partición oficial
    Ximg = np.load(f"{DATA_DIR}/X_ds.npy")            # (N,8,11)
    X3d = np.load(f"{DATA_DIR}/X3d_det_motion.npy")   # (N,8,9)
    X20 = np.concatenate([Ximg, X3d], axis=2)         # (N,8,20)
    ids = np.load(f"{DATA_DIR}/subtrack_ids_ds.npy", allow_pickle=True)
    tm, _ = official_masks(ids)
    # Un scaler de 20 características (fusión) y otro de 9 (detecciones 3D)
    scaler20 = StandardScaler().fit(X20[tm].reshape(-1, 20))
    scaler9 = StandardScaler().fit(X3d[tm].reshape(-1, 9))
    # Modelos: fusión temprana (2D+3D) y detecciones 3D
    model_fus = load_model(f"{DATA_DIR}/prediccion_3d/best/combined3d_reg_best.keras", compile=False)
    model_base = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_det_mae_best.keras", compile=False)

    # IDs -> peatones -> obtenidos de los frames presentes en las etiquetas de la secuencia
    all_tids = sorted({t for c in CS for t in gt.get(c, {})})
    # Asigna un color a cada peatón
    color = {t: plt.get_cmap("tab20")(i % 20) for i, t in enumerate(all_tids)}

    # 4. Predicciones y Figuras
    # Ventana fija: caja que engloba la posición inicial y final del robot,
    # con un margen de VIEW_R metros.
    r_ini, r_fin = poses[CS[0]][:2, 3], poses[CS[-1]][:2, 3]
    VX0, VX1 = min(r_ini[0], r_fin[0]) - VIEW_R, max(r_ini[0], r_fin[0]) + VIEW_R
    VY0, VY1 = min(r_ini[1], r_fin[1]) - VIEW_R, max(r_ini[1], r_fin[1]) + VIEW_R
    view = (VX0, VX1, VY0, VY1)

    # Figura 1. GIF
    state = {"fans": {}}
    fig, ax = plt.subplots(figsize=(9, 9))

    def update(i):
        # Dibuja la escena
        c = CS[i]
        if i % K == 0:
            state["fans"] = predict_scene(c, gt, det, traj2d, scaler20, scaler9,
                                          model_fus, model_base, poses)[0]
        draw(ax, c, state["fans"], poses, gt, det, color, CS, view)
        ax.set_title(f"Predicción en {SEQ}  (frame {c})", fontsize=11)

    # Genera y guarda la animación
    print(f"Generando animación ({len(CS)} pasos)...")
    anim = FuncAnimation(fig, update, frames=len(CS), interval=200)
    gif = OUTPUT_DIR / f"fase2_replay_{SEQ}_{F0}-{F1}.gif"
    anim.save(gif, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"Guardado: {gif}")

    # Figura 2. Capturas en distintos frames de una escena.
    idxs = np.linspace(0, len(CS) - 1, 3).astype(int)
    # Proporción ancho/alto de la ventana
    A = (VX1 - VX0) / (VY1 - VY0)
    if A >= 1.0:
        # Ancho/alto de cada plot
        pw = 7.0; ph = pw / A
        fig = plt.figure(figsize=(2 * pw + 1.4, 2 * ph + 2.4))
        gs = fig.add_gridspec(2, 4)
        axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]
    else:
        pw = 4.6; ph = min(pw / A, 9.5)
        fig, axs = plt.subplots(1, 3, figsize=(3 * pw + 1.6, ph + 2.2))
        axes = list(axs)
    # Dibuja 3 instantes
    for ax, i in zip(axes, idxs):
        fans = predict_scene(CS[i], gt, det, traj2d, scaler20, scaler9, model_fus, model_base, poses)[0]
        draw(ax, CS[i], fans, poses, gt, det, color, CS, view)
        ax.set_title(f"frame {CS[i]}", fontsize=23)
    h1, = axes[0].plot([], [], "-", color="0.3", lw=2, label="Observación (detecciones)")
    h2, = axes[0].plot([], [], "--", color="0.3", lw=1.8, label="Predicción")
    h3, = axes[0].plot([], [], marker="^", color="k", linestyle="none", markersize=12,
                       markeredgecolor="white", label="Robot")
    h4, = axes[0].plot([], [], color="0.6", lw=1.5, ls=":", label="Recorrido del robot")
    # Leyenda común fuera de los subgráficos
    fig.suptitle(f"Predicción en {SEQ}", fontsize=27, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.legend(handles=[h1, h2, h3, h4], loc="lower center", ncol=4, fontsize=21, frameon=True,
               bbox_to_anchor=(0.5, 0.012))
    # Guarda la figura
    png = OUTPUT_DIR / f"fase2_replay_{SEQ}_{F0}-{F1}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {png}")


if __name__ == "__main__":
    main()
