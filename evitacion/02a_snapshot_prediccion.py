"""
Predicción de intención de trayectorias de peatones en una escena reconstruida.
- Se reconstruye la observación a partir de detecciones 3D.
- Se predice:
    - Fusión temprana si se tienen datos 2D y 3D para los frames.
    - Modelo base de detecciones 3D si no se tienen datos 2D.
- Se dibuja.

Uso:
    python evitacion/02a_snapshot_prediccion.py [secuencia] [frame_actual]
Por defecto: huang-2-2019-01-25_0, frame 300.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from evitacion_common import (fill, nearest_det, load_gt, load_det, load_img2d, to_world, predict_scene)

OBS, PRED, STRIDE = 8, 8, 3


def main():
    SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0"
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 300  # Frame actual

    # 1. Poses
    # 8 frames observados
    frames8 = [C - STRIDE * (OBS - 1 - i) for i in range(OBS)]
    need = set(frames8)
    poses = None
    # Para cada npz de la secuencia seleccionada, se carga el fichero y los frames que lo cubren
    for npz in sorted(RECON_DIR.glob(f"{SEQ}_*.npz")):
        d = np.load(npz)
        fr = set(int(x) for x in d["frames"])
        # Si todos los frames están en este conjunto, se construye el diccionario:
        # frame -> matriz de pose 4x4
        if need <= fr:
            poses = {int(f): P for f, P in zip(d["frames"], d["poses"])}
            print(f"Poses cargadas de {npz.name}")
            break
    if poses is None:
        raise SystemExit(f"No hay .npz que cubra los frames {min(frames8)}..{C}. "
                         f"Ejecuta antes: python evitacion/01b_reconstruir_escena.py {SEQ} "
                         f"{max(0, min(frames8))} {C}")

    # 2. Carga de datos: etiquetas, detecciones 3D, y trayectorias 2D de imagen.
    gt = load_gt(LABELS_DIR, SEQ)
    det = load_det(DET_DIR, SEQ)
    traj2d = load_img2d(f"{DATA_DIR}/jrdb_clean_trajectories.csv", SEQ)

    # 3. Preparación de los datos
    # Peatones existentes en los 8 frames observados
    tids = [t for t in gt[C] if all(t in gt.get(f, {}) for f in frames8)]
    print(f"Peatones con observación completa en frame {C}: {len(tids)}")

    # Carga imagen 2D (11) + detecciones 3D (9), con partición oficial
    Ximg = np.load(f"{DATA_DIR}/X_ds.npy")            # (N,8,11)
    X3d = np.load(f"{DATA_DIR}/X3d_det_motion.npy")   # (N,8,9)
    X20 = np.concatenate([Ximg, X3d], axis=2)         # (N,8,20)
    ids = np.load(f"{DATA_DIR}/subtrack_ids_ds.npy", allow_pickle=True)
    tm, _ = official_masks(ids)
    # Un scaler de 20 características (fusión) y otro de 9 (detecciones 3D)
    scaler20 = StandardScaler().fit(X20[tm].reshape(-1, 20))
    scaler9 = StandardScaler().fit(X3d[tm].reshape(-1, 9))
    # Modelos del híbrido: fusión temprana (imagen+3D) y base (detecciones 3D)
    model_fus = load_model(f"{DATA_DIR}/prediccion_3d/best/combined3d_reg_best.keras", compile=False)
    model_base = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_det_mae_best.keras", compile=False)

    # 4. Predicción
    # Observación en marco mundo
    obs_world = {}
    for t in tids:
        # Trayectoria de detecciones en frames observados (rellena los huecos) -> marco mundo
        od = np.array(fill([nearest_det(gt[f][t], det.get(f, np.zeros((0, 2), np.float32)))
                            for f in frames8]), np.float32)
        obs_world[t] = np.array([to_world(poses, frames8[i], od[i]) for i in range(OBS)])

    # Predice: fusión (2D+3D) vs base (sin imagen)
    pred_world, of, ob = predict_scene(C, gt, det, traj2d, scaler20, scaler9, model_fus, model_base, poses)
    print(f"Peatones -> fusión (con 2D): {len(of)} | base solo-LiDAR (sin 2D): {len(ob)}")

    # 5. Dibujo
    fig, ax = plt.subplots(figsize=(11, 11))
    cmap = plt.get_cmap("tab20")
    # Trayectoria del robot
    robxy = np.array([poses[f][:2, 3] for f in sorted(poses)])
    ax.plot(robxy[:, 0], robxy[:, 1], color="0.6", lw=1.5, ls=":", label="Camino del robot")
    # Posición actual del robot
    ax.scatter(*poses[C][:2, 3], c="k", marker="^", s=170, zorder=6, edgecolors="white",
               label="Robot (ahora)")

    # Por cada peatón
    for i, t in enumerate(tids):
        col = cmap(i % 20)
        # Trayectoria observada
        o = obs_world[t]
        ax.plot(o[:, 0], o[:, 1], "-", color=col, lw=2.2, zorder=4)
        ax.scatter(o[-1, 0], o[-1, 1], color=col, s=25, zorder=5)
        # Si se ha predicho
        if t in pred_world:
            # Se dibuja la trayectoria predicha (une última observación con la predicción)
            seg = np.vstack([o[-1], pred_world[t]])
            ax.plot(seg[:, 0], seg[:, 1], "--", color=col, lw=1.8, alpha=0.9, zorder=2)

    ax.plot([], [], "-", color="0.3", lw=2.2, label="Observación (detecciones)")
    ax.plot([], [], "--", color="0.3", lw=1.8, label="Predicción (fusión / base LiDAR)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Snapshot de predicción - {SEQ}, frame {C}\n({len(tids)} peatones)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    # Se guarda la figura
    out = OUTPUT_DIR / f"fase2_snapshot_{SEQ}_{C}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
