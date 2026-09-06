"""
Genera una figura con una anotación 2D (bbox + 17 kp de pose) dibujada sobre la
imagen real.
Guarda figuras/anotacion_2d.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
IMG_ROOT = f"{DATA_DIR}/train_dataset_with_activity/train_dataset_with_activity/images"
JSON = f"{DATA_DIR}/jrdb_mmpose_train/jrdb_mmpose_train/train_individual.json"
OUTPUT_DIR = f"{DATA_DIR}/evaluate_predictions/figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Anotación objetivo; si no se encuentra, la de mejor calidad
TARGET = {"file_name": "image_4/bytes-cafe-2019-02-07_0/000861.jpg", "track_id": 6}

if __name__ == "__main__":
    # Carga el JSON
    with open(JSON) as f:
        data = json.load(f)
    # Pares de kp: [hombro, codo], [codo, muñeca], ...
    skeleton = data["categories"][0]["skeleton"]
    # Diccionario de búsqueda
    img_info = {im["id"]: im for im in data["images"]}

    # Busca la anotación objetivo
    ann_sel, img_sel = None, None
    for ann in data["annotations"]:
        im = img_info[ann["image_id"]]
        if im["file_name"] == TARGET["file_name"] and ann["track_id"] == TARGET["track_id"]:
            ann_sel, img_sel = ann, im
            break
    # Si no la encuentra, busca la anotación con más kp visibles (o bbox más grande)
    if ann_sel is None:
        best = -1
        for ann in data["annotations"]:
            kp = ann["keypoints"]
            vis = sum(1 for i in range(17) if kp[i*3+2] > 0)
            area = ann["bbox"][2] * ann["bbox"][3]
            score = vis * 1000 + area / 1000
            if vis >= 15 and score > best:
                best, ann_sel, img_sel = score, ann, img_info[ann["image_id"]]

    # Carga la imagen
    img_path = os.path.join(IMG_ROOT, img_sel["file_name"])
    img = plt.imread(img_path)

    # Carga los kp y el cuadro delimitador de la anotación
    kp = np.array(ann_sel["keypoints"], dtype=float).reshape(17, 3)
    x, y, bw, bh = ann_sel["bbox"]

    # Crea subplots
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img)
    # Dibuja el cuadro delimitador (bbox)
    ax.add_patch(patches.Rectangle((x, y), bw, bh, fill=False, edgecolor="tab:red", lw=2))
    # Muestra el esqueleto
    for a, b in skeleton:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]], "-", color="tab:cyan", lw=1.5)
    # Dibuja los kp visibles
    vis = kp[:, 2] > 0
    ax.scatter(kp[vis, 0], kp[vis, 1], c="yellow", s=25, edgecolors="black", zorder=3)
    ax.set_title(f"Anotación 2D: bbox + 17 keypoints\n{img_sel['file_name']}  (track {ann_sel['track_id']})", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    # Guarda la figura
    fig.savefig(f"{OUTPUT_DIR}/anotacion_2d.png", dpi=140, bbox_inches="tight")
    print(f"Guardado: {OUTPUT_DIR}/anotacion_2d.png")
