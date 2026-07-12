"""
Figura de una escena completa con todos los peatones anotados y sus cuadros
delimitadores 2D dibujados.
"""
import os
import sys
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
IMG_DIR = f"{DATA_DIR}/train_dataset_with_activity/train_dataset_with_activity/images"
JSON = f"{DATA_DIR}/jrdb_mmpose_train/jrdb_mmpose_train/train_individual.json"
OUT = f"{DATA_DIR}/data_prep/figuras"; os.makedirs(OUT, exist_ok=True)

data = json.load(open(JSON))
img_info = {im["id"]: im for im in data["images"]}

# Anotaciones por imagen
by_img = defaultdict(list)
for ann in data["annotations"]:
    by_img[ann["image_id"]].append(ann)

# Frame: indicado por CLI o, por defecto, el de más peatones
if len(sys.argv) > 1:
    target = sys.argv[1]
    img_id = next((iid for iid, im in img_info.items() if im["file_name"] == target), None)
    if img_id is None:
        raise SystemExit(f"No se encontró la imagen {target}")
else:
    img_id = max(by_img, key=lambda iid: len(by_img[iid]))

# Carga imagen y anotaciones
img_sel = img_info[img_id]
anns = by_img[img_id]
img_dir = os.path.join(IMG_DIR, img_sel["file_name"])
img = plt.imread(img_dir)

# Se dibuja imagen y bboxes
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(img)
for ann in anns:
    x, y, bw, bh = ann["bbox"]
    ax.add_patch(patches.Rectangle((x, y), bw, bh, fill=False,
                                   edgecolor="tab:red", lw=1.8))
    ax.text(x, max(y - 4, 0), f"{ann['track_id']}", color="white", fontsize=7,
            bbox=dict(facecolor="tab:red", edgecolor="none", pad=0.5, alpha=0.8))

ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/escena_cajas.png", dpi=150, bbox_inches="tight")
print(f"Guardado: {OUT}/escena_cajas.png  ({len(anns)} peatones, {img_sel['file_name']})")
