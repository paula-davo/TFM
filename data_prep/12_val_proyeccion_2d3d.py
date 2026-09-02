"""
Validación de la correspondencia 2D–3D que entra al modelo de fusión temprana.

Para un (secuencia, frame): proyecta la caja 3D de cada peatón (labels_3d, marco del
LiDAR superior) sobre la imagen de su cámara usando la calibración de JRDB
(lidars.yaml: upper2cam, K distorsionada y coeficientes de distorsión D), y la
superpone con la caja 2D anotada (del CSV). Si ambas cajas coinciden, se confirma
que los datos 2D y 3D corresponden al MISMO peatón en el MISMO frame — que es lo que
asume la fusión.

La validación cuantitativa es con el CENTRO: comprueba que el centro de la caja 3D
proyectada cae DENTRO de la caja 2D. La caja 3D proyectada (roja) sobreestima el
tamaño 2D porque es el volumen 3D completo (profundidad incluida), así que el solape
de cajas (IoU) es engañoso; el centro es la señal robusta de correspondencia.

Uso:
    python data_prep/12_val_proyeccion_2d3d.py [secuencia] [frame]
Por defecto: bytes-cafe-2019-02-07_0, frame 0.

Nota: los labels_3d de JRDB están en el marco EGO (base del robot), por eso el marco
correcto es "ego" (inversa de cam2ego) — verificado: 100% de los centros 3D caen
dentro de la caja 2D. Se deja "upper" como alternativa por si se necesita comparar.
"""
import sys
import csv
import json
from pathlib import Path
import numpy as np
try:
    import yaml
except ImportError:
    sys.exit("Falta el paquete pyyaml. Instálalo con: pip install pyyaml")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
CALIB = TRAIN / "calibration"
LABELS_3D = TRAIN / "labels" / "labels_3d"
IMAGES = TRAIN / "images"
OUT = Path(DATA_DIR) / "data_prep" / "figuras"; OUT.mkdir(parents=True, exist_ok=True)

SEQ = sys.argv[1] if len(sys.argv) > 1 else "bytes-cafe-2019-02-07_0"
FR = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# Marco de los labels 3D: "ego" (correcto, -> inv(cam2ego)) o "upper" (-> upper2cam).
# Verificado: con "ego" el 100% de los centros 3D caen dentro de la caja 2D (offset 0,15).
FRAME_3D = sys.argv[3] if len(sys.argv) > 3 else "ego"


def load_calib():
    # Calibración por cámara: image_N -> sensor_N (N = 0,2,4,6,8)
    cal = yaml.safe_load(open(CALIB / "lidars.yaml"))
    cams = {}
    for n in (0, 2, 4, 6, 8):
        s = cal[f"sensor_{n}"]
        M = np.array(s["upper2cam"], dtype=float) if FRAME_3D == "upper" \
            else np.linalg.inv(np.array(s["cam2ego"], dtype=float))
        cams[f"image_{n}"] = dict(
            M=M,
            K=np.array(s["distorted_img_K"], dtype=float),
            D=np.array(s["D"], dtype=float))
    return cams


def load_boxes3d():
    # Cajas 3D del frame: id -> (cx, cy, cz, l, w, h, rot_z)
    data = json.load(open(LABELS_3D / f"{SEQ}.json"))["labels"]
    key = f"{FR:06d}.pcd"
    boxes = {}
    for a in data.get(key, []):
        tid = int(a["label_id"].split(":")[1])
        b = a["box"]
        boxes[tid] = (b["cx"], b["cy"], b["cz"], b["l"], b["w"], b["h"], b["rot_z"])
    return boxes


def load_boxes2d():
    # Cajas 2D del CSV: camara -> {track_id: (x0, y0, w, h)} (esquina sup-izq y tamaño)
    boxes = defaultdict(dict)
    with open(f"{DATA_DIR}/jrdb_clean_trajectories.csv") as f:
        for r in csv.DictReader(f):
            if r["sequence"] == SEQ and int(r["frame"]) == FR:
                x, y = float(r["x"]), float(r["y"])            # centro del bbox
                bw, bh = float(r["bbox_w"]), float(r["bbox_h"])
                boxes[r["camera"]][int(r["track_id"])] = (x - bw / 2, y - bh / 2, bw, bh)
    return boxes


def box_corners(cx, cy, cz, l, w, h, rot_z):
    # 8 esquinas de la caja 3D orientada por rot_z (marco del LiDAR)
    c, s = np.cos(rot_z), np.sin(rot_z)
    pts = []
    for X in (-l / 2, l / 2):
        for Y in (-w / 2, w / 2):
            for Z in (-h / 2, h / 2):
                pts.append([cx + c * X - s * Y, cy + s * X + c * Y, cz + Z])
    return np.array(pts)


def project(pts, M, K, D):
    # Proyecta puntos 3D (marco LiDAR) a píxeles de la cámara (modelo pinhole + distorsión)
    P = (M @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]     # marco cámara (z hacia delante)
    z = P[:, 2]
    valid = z > 0.05                                        # solo lo que está delante de la cámara
    x, y = P[:, 0] / z, P[:, 1] / z
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = D
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return np.c_[u, v], valid


def main():
    print(f"FRAME_3D = {FRAME_3D}  ·  {SEQ}  frame {FR}")
    cams = load_calib()
    b3d = load_boxes3d()
    b2d = load_boxes2d()
    if not b2d:
        raise SystemExit(f"No hay cajas 2D en el CSV para {SEQ} frame {FR}.")

    centers_in, offsets = [], []
    for cam, peds in b2d.items():
        if cam not in cams:
            continue
        img_path = IMAGES / cam / SEQ / f"{FR:06d}.jpg"
        if not img_path.exists():
            print(f"  (falta imagen) {img_path}")
            continue
        img = plt.imread(img_path)
        H, W = img.shape[:2]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        for tid, box2d in peds.items():
            # Caja 2D anotada (verde)
            x0, y0, bw, bh = box2d
            ax.add_patch(Rectangle((x0, y0), bw, bh, fill=False, edgecolor="lime", lw=2.5))
            ax.text(x0, y0 - 4, f"2D id {tid}", color="lime", fontsize=11, fontweight="bold")

            # Proyección de la caja 3D del peatón (si la tiene)
            if tid not in b3d:
                continue
            corners = box_corners(*b3d[tid])
            uv, valid = project(corners, cams[cam]["M"], cams[cam]["K"], cams[cam]["D"])
            if valid.sum() < 4:
                continue                                    # la caja cae fuera/atrás de esta cámara
            uv = uv[valid]
            px0, py0 = uv[:, 0].min(), uv[:, 1].min()
            px1, py1 = uv[:, 0].max(), uv[:, 1].max()
            # Caja 3D proyectada (roja, fina) -> solo contexto; sobreestima por incluir la profundidad
            ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0,
                                   fill=False, edgecolor="red", lw=1.3, linestyle="--", alpha=0.6))
            # CENTRO 3D proyectado -> señal robusta: debe caer DENTRO de la caja 2D
            cx3, cy3, cz3 = b3d[tid][0], b3d[tid][1], b3d[tid][2]
            uvc, _ = project(np.array([[cx3, cy3, cz3]]), cams[cam]["M"], cams[cam]["K"], cams[cam]["D"])
            uc, vc = float(uvc[0, 0]), float(uvc[0, 1])
            ax.plot(uc, vc, "rx", markersize=13, markeredgewidth=3, zorder=5)
            inside = (x0 <= uc <= x0 + bw) and (y0 <= vc <= y0 + bh)
            off = float(np.hypot(uc - (x0 + bw / 2), vc - (y0 + bh / 2)) / (0.5 * (bw + bh)))
            centers_in.append(inside); offsets.append(off)
            ax.text(x0, y0 + bh + 14, f"centro {'DENTRO' if inside else 'fuera'} (off={off:.2f})",
                    color="red", fontsize=10, fontweight="bold")

        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(f"Validación 2D–3D  ·  {SEQ}  frame {FR}  ·  {cam}\n"
                     "verde = caja 2D anotada, rojo = caja 3D proyectada", fontsize=13)
        ax.axis("off")
        out = OUT / f"val_proyeccion_2d3d_{SEQ}_{FR}_{cam}.png"
        fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"Guardado: {out.name}")

    if offsets:
        frac = 100 * np.mean(centers_in)
        print(f"\nCentro 3D proyectado DENTRO de la caja 2D: {frac:.0f}% "
              f"({int(np.sum(centers_in))}/{len(centers_in)} peatones)")
        print(f"Offset medio centro3D–centro2D (normalizado por tamaño de caja): {np.mean(offsets):.2f}")
        print("Alto % dentro y offset bajo (<0,5) confirma que 2D y 3D son el mismo peatón en el mismo frame.")
        print("NOTA: la caja 3D proyectada (roja) sale MÁS GRANDE que la 2D porque es el volumen 3D "
              "completo (profundidad incluida); por eso se valida con el CENTRO, no con el IoU de cajas.")
    else:
        print("\nNo se pudo proyectar ninguna caja 3D sobre las 2D (revisa FRAME_3D).")


if __name__ == "__main__":
    main()
