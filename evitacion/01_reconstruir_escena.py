"""
Fase 2 - Paso 1: reconstrucción de una escena en MARCO MUNDO.

Estima el movimiento del robot por ICP sobre las nubes lidar aplicando la calibración.
Transforma las trayectorias de los peatones al marco mundo.
Dibuja la trayectoria del robot y de los peatones en marco mundo.
Guarda las poses para reutilizarlas en las siguientes fases.

Uso:
    python evitacion/01_reconstruir_escena.py [secuencia] [frame_ini] [frame_fin]
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

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
PCD = TRAIN_DIR / "pointclouds" / "lower_velodyne"
LAB = TRAIN_DIR / "labels" / "labels_3d"
OUTF = Path(DATA_DIR) / "evitacion" / "figuras"
OUTF.mkdir(parents=True, exist_ok=True)
OUTD = Path(DATA_DIR) / "evitacion" / "reconstruccion"; OUTD.mkdir(parents=True, exist_ok=True)

# Matrices de transformación obtenidas de la calibración (lidars.yaml)
# Permiten pasar de lidar lower a marco del robot
upper2ego = np.array([
    [0.9963896745022904, -0.08489768280241602, 0.0, 0.0],
    [0.08489768280241602, 0.9963896745022904, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.33529],
    [0.0, 0.0, 0.0, 1.0]])
lower2upper = np.array([
    [0.9967586199688887, 0.08033737338163417, 0.0, 0.0],
    [-0.0803373733816342, 0.9967586199688887, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.4720000013709068],
    [0.0, 0.0, 0.0, 1.0]])
LOWER2EGO = upper2ego @ lower2upper

SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0" # Secuencia a reconstruir
F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0                  # Frame inicial
F1 = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9              # Frame final

VOXEL = 0.25       # m, submuestreo de la nube para ICP -> de 25 cm se queda con un punto
CROP_R = 25.0      # m, recorte radial (centra el ICP en estructura cercana) -> descarta puntos más lejanos
ICP_DIST = 0.6     # m, máx distancia de correspondencia -> empareja ptos entre frames a menos de esta dist


def load_cloud_ego(fr):
    # Carga nube de puntos lidar y la prepara para ICP
    p = PCD / SEQ / f"{fr:06d}.pcd"
    if not p.exists():
        return None
    # Carga los puntos de la nube -> (x, y, z)
    pts = np.asarray(o3d.io.read_point_cloud(str(p)).points)
    if pts.size == 0:
        return None
    # Transforma al marco del robot (ego)
    pe = (LOWER2EGO @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    pe = pe[np.linalg.norm(pe[:, :2], axis=1) < CROP_R]           # recorte radial
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(pe)
    # Submuestrea por voxel
    c = c.voxel_down_sample(VOXEL)
    # Calcula las normales de cada punto
    c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 3, max_nn=30))
    return c

# Ordena todos los frames de la secuencia, filtrados al rango [F0, F1]
frames = sorted(int(p.stem) for p in (PCD / SEQ).glob("*.pcd") if F0 <= int(p.stem) <= F1)
if not frames:
    raise SystemExit(f"No hay nubes para {SEQ} en {F0}..{F1}")
print(f"Secuencia {SEQ}: {len(frames)} frames en {frames[0]}..{frames[-1]}")

# ICP
poses = {}      # poses del robot
prev = None
T = np.eye(4)   # Pose acumulada del frame actual -> mundo
# Para cada frame
for i, fr in enumerate(frames):
    # Carga su nube de puntos lidar procesada
    cur = load_cloud_ego(fr)
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

fkeys = [f for f in frames if f in poses]   # frames con pose
disp = np.linalg.norm(poses[fkeys[-1]][:2, 3] - poses[fkeys[0]][:2, 3]) # desplazamiento del robot
print(f"Desplazamiento neto del robot: {disp:.2f} m  (~0 => robot parado)")

# Transforma las trayectorias de los peatones (marco robot) al marco mundo
# Carga el JSON de etiquetas de la secuencia
data = json.load(open(LAB / f"{SEQ}.json"))["labels"]
tr = defaultdict(list)
# Para cada frame etiquetado
for k, anns in data.items():
    fr = int(k.replace(".pcd", ""))
    if fr not in poses:
        continue
    Tw = poses[fr]  # Pose del robot en ese frame
    # Para cada peatón:
    for a in anns:
        # Obtiene la etiqueta 3D, y descarta la Z (proyección X e Y)
        b = a["box"]
        # Aplica la pose del robot -> coordenadas del peatón en el marco mundo
        w = Tw @ np.array([b["cx"], b["cy"], 0.0, 1.0])
        tr[a["label_id"]].append((fr, w[0], w[1]))

# Figura
fig, ax = plt.subplots(figsize=(10, 10))
cmap = plt.get_cmap("tab20")
# Trayectoria del robot
rob = np.array([poses[f][:2, 3] for f in fkeys])
ax.plot(rob[:, 0], rob[:, 1], "k-", lw=2.5, label="Robot", zorder=4)
# Punto de inicio
ax.scatter(rob[0, 0], rob[0, 1], c="k", marker="^", s=150, zorder=5,
           edgecolors="white", label="Inicio robot")
# Recorre los tracks ID, obteniendo los puntos del peatón (ordenados por frames) y los dibuja
n = 0
for i, (tid, pts) in enumerate(sorted(tr.items())):
    pts = np.array(sorted(pts))
    # Descarta trayectorias cortas
    if len(pts) < 5:
        continue
    ax.plot(pts[:, 1], pts[:, 2], "-", color=cmap(i % 20), lw=1.2, alpha=0.85)
    n += 1
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Reconstrucción de {SEQ} en el mundo\n"
             f"(frames {fkeys[0]}..{fkeys[-1]}, robot {disp:.1f} m, {n} peatones)")
ax.legend(loc="upper right")
fig.tight_layout()
# Guarda la figura
out = OUTF / f"fase2_mundo_{SEQ}_{fkeys[0]}-{fkeys[-1]}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")

# Guarda las poses para reutilizarlas
npz = OUTD / f"{SEQ}_{fkeys[0]}-{fkeys[-1]}.npz"
np.savez(npz, frames=np.array(fkeys), poses=np.array([poses[f] for f in fkeys]))
print(f"Guardado: {npz}")
