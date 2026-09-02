"""
Figura que comprueba la relación detección-etiqueta 3D, midiendo el error
de localización de las detecciones y la tasa de emparejamiento.
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
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LABELS_3D = TRAIN_DIR / "labels" / "labels_3d"
DET_3D = TRAIN_DIR / "detections" / "detections_3d"
OUT = f"{DATA_DIR}/data_prep/figuras"; os.makedirs(OUT, exist_ok=True)
MATCH_THRESH = 1.0 # Umbral máximo para emparejar detección-etiqueta (m)


def load_gt(seq):
    # Lee las etiquetas 3D de la secuencia y devuelve
    # [(cx, cy), ...] por frame con peatones
    gt = {}
    for k, anns in json.load(open(LABELS_3D / f"{seq}.json"))["labels"].items():
        fr = int(k.replace(".pcd", ""))
        gt[fr] = [(a["box"]["cx"], a["box"]["cy"]) for a in anns]
    return gt


def load_det(seq):
    # Lee las detecciones 3D de la secuencia y devuelve
    # [(cx, cy), ...] por frame con peatones
    p = DET_3D / f"{seq}.json"
    if not p.exists():
        return {}
    det = {}
    for k, ds in json.load(open(p))["detections"].items():
        fr = int(k.replace(".pcd", ""))
        det[fr] = np.array([(d["box"]["cx"], d["box"]["cy"]) for d in ds], dtype=np.float32) if ds else np.zeros((0, 2), np.float32)
    return det

# Calcula el error de localización detección-etiqueta para
# todas las secuencias y frames
errors = []
total, matched = 0, 0
for jf in LABELS_3D.glob("*.json"):
    seq = jf.stem
    gt = load_gt(seq); det = load_det(seq)
    for fr, peds in gt.items():
        darr = det.get(fr, np.zeros((0, 2), np.float32))
        for (cx, cy) in peds:
            total += 1
            if darr.shape[0] == 0:
                continue
            d = np.linalg.norm(darr - np.array([cx, cy]), axis=1)
            j = int(np.argmin(d))
            if d[j] <= MATCH_THRESH:
                matched += 1
                errors.append(d[j])

errors = np.array(errors)
print(f"Peatones (GT): {total}   emparejados (<{MATCH_THRESH} m): {matched} ({100*matched/total:.1f}%)")
print(f"Error de localización — media {errors.mean():.3f} m | mediana {np.median(errors):.3f} m | p95 {np.percentile(errors,95):.3f} m")

# Dibuja histograma
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(errors, bins=50, range=(0, MATCH_THRESH), color="tab:orange", alpha=0.85)
ax.axvline(errors.mean(), color="red", ls="--", lw=1.5, label=f"Media = {errors.mean():.3f} m")
ax.axvline(np.median(errors), color="black", ls=":", lw=1.5, label=f"Mediana = {np.median(errors):.3f} m")
ax.set_xlabel("Error de localización detección-etiqueta (m)", fontsize=16)
ax.set_ylabel("Número de detecciones", fontsize=16)
ax.tick_params(labelsize=14)
ax.set_title(textwrap.fill(f"Ruido del detector LiDAR (emparejamiento {100*matched/total:.1f}%)", 48), fontsize=18)
ax.legend(fontsize=14); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/dataset_deteccion_error.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {OUT}/dataset_deteccion_error.png")
