"""
Figura que comprueba la relación detección-etiqueta 3D, midiendo el error de localización
y tasa de emparejamiento -> pero en este caso sobre el connjunto filtrado de subtrayectorias, 
submuestreadas (STRIDE=3), y solo sobre frames observados.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LABELS_3D = TRAIN_DIR / "labels" / "labels_3d"
DET_3D = TRAIN_DIR / "detections" / "detections_3d"
OUT = f"{DATA_DIR}/data_prep/figuras"; os.makedirs(OUT, exist_ok=True)

OBS_LEN, PRED_LEN, STRIDE = 8, 12, 3
WIN = OBS_LEN + PRED_LEN
MATCH_THRESH = 1.0 # Umbral máximo para emparejar detección-etiqueta (m)

def load_gt():
    # Lee las etiquetas 3D de todas las secuencias y devuelve
    # un diccionario con centro de peaton por secuencia, frame y track_id.
    gt = defaultdict(lambda: defaultdict(dict))
    for jf in LABELS_3D.glob("*.json"):
        seq = jf.stem
        for k, anns in json.load(open(jf))["labels"].items():
            fr = int(k.replace(".pcd", ""))
            for a in anns:
                tid = int(a["label_id"].split(":")[1])
                gt[seq][fr][tid] = (a["box"]["cx"], a["box"]["cy"])
    return gt


def load_det():
    # Lee las detecciones 3D de todas las secuencias y devuelve
    # un diccionario con centros de detecciones por secuencia y frame.
    det = defaultdict(dict)
    for jf in DET_3D.glob("*.json"):
        seq = jf.stem
        for k, ds in json.load(open(jf))["detections"].items():
            fr = int(k.replace(".pcd", ""))
            det[seq][fr] = np.array([(d["box"]["cx"], d["box"]["cy"]) for d in ds], np.float32) if ds else np.zeros((0, 2), np.float32)
    return det


# Carga etiquetas, detecciones y CSV de subtrayectorias filtradas
gt = load_gt()
det = load_det()
df = pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv")

errors = []
total, matched = 0, 0
for sid, traj in df.groupby("subtrack_id"):
    # Coge 1 de cada 3 frames (STRIDE=3)
    # Descarta las que tienen lon de menos de WIN frames (OBS+PRED)
    traj = traj.sort_values("frame")
    traj_ds = traj.iloc[::STRIDE]
    if len(traj_ds) < WIN:
        continue
    # Extrae frame, secuencia y track_id de la subtrayectoria
    frames = traj_ds["frame"].values.astype(int)
    seq = traj_ds["sequence"].iloc[0]
    tid = int(traj_ds["track_id"].iloc[0])
    # Para cada ventana de WIN frames
    # Se recorren solo los frames observados (OBS_LEN)
    for start in range(len(traj_ds) - WIN + 1):
        for f in frames[start:start + OBS_LEN]:
            # Extrae la posición de la etiqueta
            gpos = gt.get(seq, {}).get(int(f), {}).get(tid)
            if gpos is None:
                continue
            total += 1
            # Busca la detección más cercana y calcula el error de localización
            darr = det.get(seq, {}).get(int(f), np.zeros((0, 2), np.float32))
            if darr.shape[0] == 0:
                continue
            d = np.linalg.norm(darr - np.array(gpos), axis=1)
            j = int(np.argmin(d))
            if d[j] <= MATCH_THRESH:
                matched += 1
                errors.append(d[j])

errors = np.array(errors)
rate = 100 * matched / total
print(f"Observaciones (conjunto entrenamiento): {total}   emparejadas: {matched} ({rate:.1f}%)")
print(f"Error de localización — media {errors.mean():.3f} m | mediana {np.median(errors):.3f} m | p95 {np.percentile(errors,95):.3f} m")

# Dibuja histograma
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(errors, bins=50, range=(0, MATCH_THRESH), color="tab:green", alpha=0.85)
ax.axvline(errors.mean(), color="red", ls="--", lw=1.5, label=f"Media = {errors.mean():.3f} m")
ax.axvline(np.median(errors), color="black", ls=":", lw=1.5, label=f"Mediana = {np.median(errors):.3f} m")
ax.set_xlabel("Error de localización detección-etiqueta (m)")
ax.set_ylabel("Número de detecciones")
ax.set_title(f"Ruido del detector LiDAR — conjunto de entrenamiento (emparejamiento {rate:.1f}%)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/dataset_deteccion_error_filtrado.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {OUT}/dataset_deteccion_error_filtrado.png")
