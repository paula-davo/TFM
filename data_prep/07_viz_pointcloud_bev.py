"""
Figura de nube de puntos del lidar en vista BEV (desde arriba),
con la caja 3D de un peatón y el radio de recorte de 3,5 m (recorte de entorno)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LABELS_3D = TRAIN_DIR / "labels" / "labels_3d"
PCD_DIR = TRAIN_DIR / "pointclouds" / "lower_velodyne"
OUT = f"{DATA_DIR}/data_prep/figuras"; os.makedirs(OUT, exist_ok=True)

SEQ = "bytes-cafe-2019-02-07_0"
R = 3.5   # radio de recorte de entorno

# Carga etiquetas 3D de la secuencia
labels = json.load(open(LABELS_3D / f"{SEQ}.json"))["labels"]

# Elige frame con peatón (> 100) y nube de puntos
chosen = None
for pcd_key, anns in labels.items():
    fr = int(pcd_key.replace(".pcd", ""))
    if not (PCD_DIR / SEQ / f"{fr:06d}.pcd").exists():
        continue
    best = max(anns, key=lambda a: a.get("attributes", {}).get("num_points", 0), default=None)
    if best and best.get("attributes", {}).get("num_points", 0) > 100:
        chosen = (fr, best)
        break

# Extrae datos de bbox
fr, ped = chosen
box = ped["box"]
cx, cy, l, w, rot = box["cx"], box["cy"], box["l"], box["w"], box["rot_z"]
# Extrae nube de puntos y filtra
pts = np.asarray(o3d.io.read_point_cloud(str(PCD_DIR / SEQ / f"{fr:06d}.pcd")).points)
pxy = pts[(pts[:, 2] > -0.5) & (pts[:, 2] < 2.0)][:, :2]   # mismo filtrado de suelo

# Dibuja el cuadro delimitador
c, s = np.cos(rot), np.sin(rot)
corners = np.array([[ l/2,  w/2], [ l/2, -w/2], [-l/2, -w/2], [-l/2,  w/2], [ l/2, w/2]])
corners = corners @ np.array([[c, s], [-s, c]]) + [cx, cy]

# Dibuja la nube de puntos y la caja 3D
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(pxy[:, 0], pxy[:, 1], s=1, c="0.6", label="Nube LiDAR")
circ = plt.Circle((cx, cy), R, fill=False, ls="--", color="tab:blue", lw=1.5, label=f"Radio entorno ({R} m)")
ax.add_patch(circ)
ax.plot(corners[:, 0], corners[:, 1], "-", color="tab:red", lw=2, label="Caja 3D peatón")
ax.plot(cx, cy, "x", color="tab:red", ms=10)
ax.scatter(0, 0, c="tab:green", marker="^", s=120, label="Robot (origen)", zorder=5)
ax.set_aspect("equal")
ax.set_xlim(cx - R - 2, cx + R + 2); ax.set_ylim(cy - R - 2, cy + R + 2)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Nube LiDAR — {SEQ}, frame {fr}", fontsize=10)
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/pointcloud_bev.png", dpi=140, bbox_inches="tight")
print(f"Guardado: {OUT}/pointcloud_bev.png  (frame {fr}, num_points={ped['attributes'].get('num_points')})")
