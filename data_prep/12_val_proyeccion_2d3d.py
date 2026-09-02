"""
Validación de la correspondencia 2D–3D.
Uso:
    python data_prep/12_val_proyeccion_2d3d.py [secuencia] [frame]
Por defecto: bytes-cafe-2019-02-07_0, frame 0.
"""
import sys
import csv
import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

# Valores por defecto (se pueden pasar por línea de comandos)
SEQ_DEFAULT = "bytes-cafe-2019-02-07_0"
FR_DEFAULT = 0
FRAME_3D_DEFAULT = "ego"   # Marco de los labels 3D: ego


class JRDBProjection2D3DValidator:

    """
    JRDBProjection2D3DValidator valida la correspondencia 2D–3D proyectando las cajas 3D sobre la imagen.

    1) Carga calibración de cámaras, cajas 3D (GT) y cajas 2D anotadas
    2) Proyecta cada caja 3D a píxeles
    3) Comprueba que el centro 3D proyectado cae dentro de la caja 2D del mismo peatón
    4) Guarda una figura por cámara y reporta % dentro y offset medio
    """

    def __init__(self, seq=SEQ_DEFAULT, fr=FR_DEFAULT, frame_3d=FRAME_3D_DEFAULT, data_dir=DATA_DIR):
        # Directorio y subcarpetas del dataset
        self.data_dir = data_dir
        train_dir = Path(data_dir) / "train_dataset_with_activity" / "train_dataset_with_activity"
        self.calib_dir = train_dir / "calibration"
        self.labels_3d_dir = train_dir / "labels" / "LABELS_3D_DIR"
        self.images_dir = train_dir / "IMAGES_DIR"
        self.out_dir = Path(data_dir) / "data_prep" / "figuras"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Secuencia, frame y marco 3D a validar
        self.seq = seq
        self.fr = fr
        self.frame_3d = frame_3d

        # Datos y resultados
        self.cams = None
        self.b3d = None
        self.b2d = None
        self.centers_in = []
        self.offsets = []

    def load_calib(self) -> dict:
        # Se carga la calibración de las cámaras
        cal = yaml.safe_load(open(self.calib_dir / "lidars.yaml"))
        cams = {}
        # 5 cámaras
        for n in (0, 2, 4, 6, 8):
            s = cal[f"sensor_{n}"]
            M = np.array(s["upper2cam"], dtype=float) if self.frame_3d == "upper" \
                else np.linalg.inv(np.array(s["cam2ego"], dtype=float))
            # M = extrínsecos (pos/orient de la cámara respecto 3D),
            # K: matriz extrínseca (3D a cámara); D = coef. distorsión lente.
            cams[f"image_{n}"] = dict(
                M=M,
                K=np.array(s["distorted_img_K"], dtype=float),
                D=np.array(s["D"], dtype=float))
        self.cams = cams
        return cams

    def load_boxes3d(self) -> dict:
        # Carga las cajas 3D para la secuencia y frame
        # Cajas 3D: id -> (cx, cy, cz, l, w, h, rot_z)
        data = json.load(open(self.labels_3d_dir / f"{self.seq}.json"))["labels"]
        key = f"{self.fr:06d}.pcd"
        boxes = {}
        for a in data.get(key, []):
            tid = int(a["label_id"].split(":")[1])
            b = a["box"]
            boxes[tid] = (b["cx"], b["cy"], b["cz"], b["l"], b["w"], b["h"], b["rot_z"])
        self.b3d = boxes
        return boxes

    def load_boxes2d(self) -> dict:
        # Carga las cajas 2D para la secuencia y frame
        # Cajas 2D: camara -> {track_id: (x0, y0, w, h)}
        boxes = defaultdict(dict)
        with open(f"{self.data_dir}/jrdb_clean_trajectories.csv") as f:
            for r in csv.DictReader(f):
                if r["sequence"] == self.seq and int(r["frame"]) == self.fr:
                    x, y = float(r["x"]), float(r["y"])            # centro del bbox
                    bw, bh = float(r["bbox_w"]), float(r["bbox_h"])
                    boxes[r["camera"]][int(r["track_id"])] = (x - bw / 2, y - bh / 2, bw, bh)
        self.b2d = boxes
        return boxes

    @staticmethod
    def box_corners(cx, cy, cz, l, w, h, rot_z):
        # Calcula las 8 esquinas de la caja 3D
        c, s = np.cos(rot_z), np.sin(rot_z)
        pts = []
        for X in (-l / 2, l / 2):
            for Y in (-w / 2, w / 2):
                for Z in (-h / 2, h / 2):
                    pts.append([cx + c * X - s * Y, cy + s * X + c * Y, cz + Z])
        return np.array(pts)

    @staticmethod
    def project(pts, M, K, D):
        # Proyecta puntos 3D a píxeles de la cámara
        # Extrínsecos: 3D -> cámara
        P = (M @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        # Filtra distancias positivas (delante de la cámara)
        z = P[:, 2]
        valid = z > 0.05
        # Normaliza y aplica la distorsión de lente
        x, y = P[:, 0] / z, P[:, 1] / z
        r2 = x * x + y * y
        k1, k2, p1, p2, k3 = D
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        # Intrínsecos: cámara -> píxeles
        u = K[0, 0] * xd + K[0, 2]
        v = K[1, 1] * yd + K[1, 2]
        return np.c_[u, v], valid

    def validate(self):
        if not self.b2d:
            raise SystemExit(f"No hay cajas 2D en el CSV para {self.seq} frame {self.fr}.")

        # Para cada cámara y peatón
        self.centers_in, self.offsets = [], []
        for cam, peds in self.b2d.items():
            if cam not in self.cams:
                continue
            img_path = self.images_dir / cam / self.seq / f"{self.fr:06d}.jpg"
            if not img_path.exists():
                print(f"  (falta imagen) {img_path}")
                continue
            img = plt.imread(img_path)
            H, W = img.shape[:2]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            for tid, box2d in peds.items():
                # Dibuja la caja 2D
                x0, y0, bw, bh = box2d
                ax.add_patch(Rectangle((x0, y0), bw, bh, fill=False, edgecolor="lime", lw=2.5))
                ax.text(x0, y0 - 4, f"2D id {tid}", color="lime", fontsize=11, fontweight="bold")

                # Proyecta la caja 3D
                if tid not in self.b3d:
                    continue
                corners = self.box_corners(*self.b3d[tid])
                uv, valid = self.project(corners, self.cams[cam]["M"], self.cams[cam]["K"], self.cams[cam]["D"])
                if valid.sum() < 4:
                    continue                                    # la caja cae fuera/atrás de esta cámara
                uv = uv[valid]
                px0, py0 = uv[:, 0].min(), uv[:, 1].min()
                px1, py1 = uv[:, 0].max(), uv[:, 1].max()
                ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0,
                                       fill=False, edgecolor="red", lw=1.3, linestyle="--", alpha=0.6))
                # Proyecta el centro de la caja 3D y comprueba si cae
                # dentro de la caja 2D y la distancia entre los centros
                cx3, cy3, cz3 = self.b3d[tid][0], self.b3d[tid][1], self.b3d[tid][2]
                uvc, _ = self.project(np.array([[cx3, cy3, cz3]]), self.cams[cam]["M"], self.cams[cam]["K"], self.cams[cam]["D"])
                uc, vc = float(uvc[0, 0]), float(uvc[0, 1])
                ax.plot(uc, vc, "rx", markersize=13, markeredgewidth=3, zorder=5)
                inside = (x0 <= uc <= x0 + bw) and (y0 <= vc <= y0 + bh)
                off = float(np.hypot(uc - (x0 + bw / 2), vc - (y0 + bh / 2)) / (0.5 * (bw + bh)))
                self.centers_in.append(inside); self.offsets.append(off)
                ax.text(x0, y0 + bh + 14, f"centro {'DENTRO' if inside else 'fuera'} (off={off:.2f})",
                        color="red", fontsize=10, fontweight="bold")

            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_title(f"Validación 2D–3D  ·  {self.seq}  frame {self.fr}  ·  {cam}\n"
                         "verde = caja 2D anotada, rojo = caja 3D proyectada", fontsize=13)
            ax.axis("off")
            out_path = self.out_dir / f"val_proyeccion_2d3d_{self.seq}_{self.fr}_{cam}.png"
            fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
            print(f"Guardado: {out_path.name}")

    def report(self):
        if self.offsets:
            frac = 100 * np.mean(self.centers_in)
            print(f"\nCentro 3D proyectado dentro de la caja 2D: {frac:.0f}% "
                  f"({int(np.sum(self.centers_in))}/{len(self.centers_in)} peatones)")
            print(f"Offset medio centro3D–centro2D: {np.mean(self.offsets):.2f}")
        else:
            print("\nNo se pudo proyectar ninguna caja 3D sobre las 2D.")

    def run(self):
        # Carga datos de calibración, cajas 3D y cajas 2D
        print(f"FRAME_3D = {self.frame_3d}  ·  {self.seq}  frame {self.fr}")
        # 1. Carga calibración de las cámaras
        self.load_calib()
        # 2. Carga cajas 3D (GT)
        self.load_boxes3d()
        # 3. Carga cajas 2D anotadas
        self.load_boxes2d()
        # 4. Proyecta y valida (guarda una figura por cámara)
        self.validate()
        # 5. Reporta % dentro y offset medio
        self.report()


if __name__ == "__main__":

    seq = sys.argv[1] if len(sys.argv) > 1 else SEQ_DEFAULT
    fr = int(sys.argv[2]) if len(sys.argv) > 2 else FR_DEFAULT
    frame_3d = sys.argv[3] if len(sys.argv) > 3 else FRAME_3D_DEFAULT

    validator = JRDBProjection2D3DValidator(seq=seq, fr=fr, frame_3d=frame_3d)
    validator.run()
