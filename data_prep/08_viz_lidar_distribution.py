"""
Distribución de los peatones en el espacio físico.
1) Mapa de densidad de posiciones (cx, cy) en metros, vista cenital (BEV) ->
360 grados alrededor del robot.
2) Histograma de la distancia de cada peatón al robot.
Datos obtenidos de labels_3d -> etiquetas
"""
import os
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
LABELS_3D = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity" / "labels" / "labels_3d"
OUT = f"{DATA_DIR}/data_prep/figuras"; os.makedirs(OUT, exist_ok=True)

# Lee todas las secuencias de labels_3d, extrayendo cx, cy y distancia al robot
cx_all, cy_all, dist_all = [], [], []
for jf in LABELS_3D.glob("*.json"):
    with open(jf) as f:
        labels = json.load(f)["labels"]
    for anns in labels.values():
        for a in anns:
            b = a["box"]
            cx_all.append(b["cx"])
            cy_all.append(b["cy"])
            d = a.get("attributes", {}).get("distance")
            dist_all.append(d if d is not None else np.hypot(b["cx"], b["cy"]))

cx = np.array(cx_all)
cy = np.array(cy_all)
dist = np.array(dist_all)
print(f"Peatones (detecciones 3D): {len(cx)}")
print(f"cx: media {cx.mean():+.2f} m  | cy: media {cy.mean():+.2f} m | distancia: media {dist.mean():.2f} m, max {dist.max():.1f} m")

# 1) Densidad espacial (BEV)
fig, ax = plt.subplots(figsize=(7, 6))
lim = max(np.percentile(np.abs(cx), 99.5), np.percentile(np.abs(cy), 99.5)) * 1.05
hb = ax.hexbin(cx, cy, gridsize=70, bins="log", cmap="viridis",
               extent=(-lim, lim, -lim, lim))
ax.scatter(0, 0, c="red", marker="^", s=140, label="Robot (origen)", zorder=5, edgecolors="white")
ax.set_aspect("equal")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("x — profundidad (m)", fontsize=16); ax.set_ylabel("y — lateral (m)", fontsize=16)
ax.tick_params(labelsize=14)
ax.set_title(textwrap.fill("Distribución de peatones en el espacio 3D", 48), fontsize=18)
ax.legend(loc="upper right", fontsize=14)
cb = fig.colorbar(hb, ax=ax, label="Número de detecciones (log)")
cb.set_label("Número de detecciones (log)", fontsize=15); cb.ax.tick_params(labelsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/dataset_lidar_posiciones.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {OUT}/dataset_lidar_posiciones.png")

# 2) Histograma de distancia al robot
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(dist, bins=60, range=(0, np.percentile(dist, 99.5)), color="tab:blue", alpha=0.85)
ax.axvline(dist.mean(), color="red", ls="--", lw=1.5, label=f"Media = {dist.mean():.2f} m")
ax.set_xlabel("Distancia al robot (m)", fontsize=16); ax.set_ylabel("Número de detecciones", fontsize=16)
ax.tick_params(labelsize=14)
ax.set_title(textwrap.fill("Distribución de la distancia peatón-robot", 48), fontsize=18)
ax.legend(fontsize=14); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/dataset_lidar_distancia.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {OUT}/dataset_lidar_distancia.png")
