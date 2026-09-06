"""
Reconstrucción de una escena en marco mundo.
- Estima movimiento del robot por ICP.
- Transforma las trayectorias de los peatones al marco mundo.
- Dibuja trayectoria del robot y de los peatones.
- Guarda las poses.

Uso:
    python evitacion/01b_reconstruir_escena.py [secuencia] [frame_ini] [frame_fin]
Por defecto: huang-2-2019-01-25_0, secuencia completa.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evitacion_common import load_cloud_ego

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
PCD_DIR = TRAIN_DIR / "pointclouds" / "lower_velodyne"
LABELS_DIR = TRAIN_DIR / "labels" / "labels_3d"
OUTPUT_FIG_DIR = Path(DATA_DIR) / "evitacion" / "figuras"
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REC_DIR = Path(DATA_DIR) / "evitacion" / "reconstruccion"
OUTPUT_REC_DIR.mkdir(parents=True, exist_ok=True)

VOXEL = 0.25       # submuestreo de la nube
CROP_R = 25.0      # recorte
ICP_DIST = 0.6     # máxima distancia de correspondencia


def main():
    SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0" # Secuencia a reconstruir
    F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0                  # Frame inicial
    F1 = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9              # Frame final

    # Ordena frames de la secuencia con filtro al rango [F0, F1]
    frames = sorted(int(p.stem) for p in (PCD_DIR / SEQ).glob("*.pcd") if F0 <= int(p.stem) <= F1)
    if not frames:
        raise SystemExit(f"No hay nubes para {SEQ} en {F0}..{F1}")
    print(f"Secuencia {SEQ}: {len(frames)} frames en {frames[0]}..{frames[-1]}")

    # ICP
    # Poses del robot
    poses = {}
    prev = None
    # Pose acumulada del robot
    T = np.eye(4)
    # Para cada frame
    for i, fr in enumerate(frames):
        # Carga su nube de puntos lidar procesada
        cur = load_cloud_ego(PCD_DIR, SEQ, fr, VOXEL, CROP_R)
        if cur is None:
            continue
        # Si no es el primer frame
        if prev is not None:
            # Alinea la nube actual con la anterior
            reg = o3d.pipelines.registration.registration_icp(cur, prev, ICP_DIST, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            # pose(fr) = pose(prev) @ T_{prev<-curr}: T = T0 @ T1 @ T2...
            T = T @ reg.transformation
        # Guarda la pose acumulada
        poses[fr] = T.copy()
        prev = cur
        if i % 100 == 0:
            print(f"  ICP {i}/{len(frames)}")

    fkeys = [f for f in frames if f in poses]
    # Desplazamiento del robot
    disp = np.linalg.norm(poses[fkeys[-1]][:2, 3] - poses[fkeys[0]][:2, 3])
    print(f"Desplazamiento neto del robot: {disp:.2f} m  (~0 => robot parado)")

    # Transforma las trayectorias de los peatones al marco mundo
    # Carga el JSON de etiquetas de la secuencia
    data = json.load(open(LABELS_DIR / f"{SEQ}.json"))["labels"]
    tr = defaultdict(list)
    # Para cada frame etiquetado
    for k, anns in data.items():
        fr = int(k.replace(".pcd", ""))
        if fr not in poses:
            continue
        # Pose del robot en ese frame
        Tw = poses[fr]
        # Para cada peatón:
        for a in anns:
            # Obtiene la etiqueta 3D, y descarta la Z
            b = a["box"]
            # Aplica la pose del robot -> coordenadas del peatón en el marco mundo
            w = Tw @ np.array([b["cx"], b["cy"], 0.0, 1.0])
            tr[a["label_id"]].append((fr, w[0], w[1]))

    # Figura
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")
    # Trayectoria del robot
    rob = np.array([poses[f][:2, 3] for f in fkeys])
    ax.plot(rob[:, 0], rob[:, 1], "k-", lw=3.0, label="Robot", zorder=4)
    # Punto de inicio
    ax.scatter(rob[0, 0], rob[0, 1], c="k", marker="^", s=230, zorder=5,
               edgecolors="white", linewidths=1.5, label="Inicio robot")
    # Recorre los tracks ID, obteniendo los puntos del peatón y los dibuja
    n = 0
    ped_pts = []
    for i, (tid, pts) in enumerate(sorted(tr.items())):
        pts = np.array(sorted(pts))
        # Descarta trayectorias cortas
        if len(pts) < 5:
            continue
        ax.plot(pts[:, 1], pts[:, 2], "-", color=cmap(i % 20), lw=1.6, alpha=0.85)
        ped_pts.append(pts[:, 1:3])
        n += 1
    # Zoom
    if ped_pts:
        P = np.vstack(ped_pts)
        lo = np.minimum(np.percentile(P, 2, axis=0), rob.min(axis=0))
        hi = np.maximum(np.percentile(P, 98, axis=0), rob.max(axis=0))
    else:
        lo, hi = rob.min(axis=0), rob.max(axis=0)
    ctr = (lo + hi) / 2
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 * 1.08 + 1.0
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)", fontsize=19); ax.set_ylabel("y (m)", fontsize=19); ax.tick_params(labelsize=16)
    ax.set_title(f"Reconstrucción de {SEQ} en el mundo\n"
                 f"(frames {fkeys[0]}..{fkeys[-1]}, robot {disp:.1f} m, {n} peatones)", fontsize=20)
    ax.legend(loc="upper right", fontsize=18)
    fig.tight_layout()
    # Guarda la figura
    out = OUTPUT_FIG_DIR / f"fase2_mundo_{SEQ}_{fkeys[0]}-{fkeys[-1]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")

    # Guarda las poses para reutilizarlas
    npz = OUTPUT_REC_DIR / f"{SEQ}_{fkeys[0]}-{fkeys[-1]}.npz"
    np.savez(npz, frames=np.array(fkeys), poses=np.array([poses[f] for f in fkeys]))
    print(f"Guardado: {npz}")


if __name__ == "__main__":
    main()
