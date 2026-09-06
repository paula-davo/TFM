"""
Obtención de datos de entorno lidar y de las detecciones 3D -> 8 frames observados, 12 predichos.

Características de entrada: (N, 8, 7): (nº de puntos sobre el peatón, obstáculo más cercano,
densidad total en el recorte, obstáculo más cercano en las 4 direcciones).
"""

import os
import json
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

OBS_LEN, PRED_LEN, STRIDE = 8, 12, 3
WIN = OBS_LEN + PRED_LEN
R, SELF_RADIUS, Z_MIN, Z_MAX = 3.5, 0.5, -0.5, 2.0
MATCH_THRESH = 1.0   # umbral para emparejar detección y etiqueta (m)

# Calibración lower_velodyne -> ego de lidars.yaml. 
# Las detecciones/etiquetas 3D están en el marco EGO
# La nube (que viene en el marco del LiDAR inferior) se lleva a
# ego antes de recortar.
_UPPER2EGO = np.array([[0.9963896745022904, -0.08489768280241602, 0.0, 0.0],
                       [0.08489768280241602, 0.9963896745022904, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.33529],
                       [0.0, 0.0, 0.0, 1.0]])
_LOWER2UPPER = np.array([[0.9967586199688887, 0.08033737338163417, 0.0, 0.0],
                         [-0.0803373733816342, 0.9967586199688887, 0.0, 0.0],
                         [0.0, 0.0, 1.0, -0.4720000013709068],
                         [0.0, 0.0, 0.0, 1.0]])
LOWER2EGO = _UPPER2EGO @ _LOWER2UPPER


class Lidar3DEnvBuilder:

    """
    Lidar3DEnvBuilder genera las características de entorno LiDAR (N, 8, 7).

    1) Carga etiquetas 3D y detecciones 3D.
    2) Construye las ventanas submuestreadas (STRIDE=3)
    3) Por cada nube recorta el entorno de la detección del peatón y calcula las 7 características
    4) Guardado en .npy
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN, stride=STRIDE,
                 r=R, self_radius=SELF_RADIUS, z_min=Z_MIN, z_max=Z_MAX, match_thresh=MATCH_THRESH):
        # Directorio y subcarpetas del dataset
        self.data_dir = data_dir
        train_dir = Path(data_dir) / "train_dataset_with_activity" / "train_dataset_with_activity"
        self.labels_3d_dir = train_dir / "labels" / "labels_3d"
        self.det_3d_dir = train_dir / "detections" / "detections_3d"
        self.pcd_dir = train_dir / "pointclouds" / "lower_velodyne"

        # Longitudes de ventana y submuestreo temporal
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.win = obs_len + pred_len
        # Parámetros del recorte de entorno y del emparejamiento
        self.r = r
        self.self_radius = self_radius
        self.z_min = z_min
        self.z_max = z_max
        self.match_thresh = match_thresh

        # Datos y resultados
        self.gt = None
        self.det = None
        self.df = None
        self.win_meta = None
        self.X = None
        self.flags = None

    def load_gt(self) -> dict:
        # Etiquetas 3D -> centro (cx, cy) del peatón por secuencia, frame y track_id.
        # Se usan para emparejar cada detección con el peatón seguido.
        gt = defaultdict(lambda: defaultdict(dict))
        for jf in self.labels_3d_dir.glob("*.json"):
            seq = jf.stem
            with open(jf) as f:
                data = json.load(f)["labels"]
            for k, anns in data.items():
                fr = int(k.replace(".pcd", ""))
                for a in anns:
                    tid = int(a["label_id"].split(":")[1])
                    gt[seq][fr][tid] = (a["box"]["cx"], a["box"]["cy"])
        self.gt = gt
        return gt

    def load_dets(self) -> dict:
        # Detecciones 3D -> por secuencia y frame: posiciones (K,2) y cajas (K,7):
        # (cx, cy, cz, l, w, h, rot_z).
        det = defaultdict(dict)
        for jf in self.det_3d_dir.glob("*.json"):
            seq = jf.stem
            with open(jf) as f:
                data = json.load(f)["detections"]
            for k, dets in data.items():
                fr = int(k.replace(".pcd", ""))
                if dets:
                    pos = np.array([(d["box"]["cx"], d["box"]["cy"]) for d in dets], np.float32)
                    bx = np.array([(d["box"]["cx"], d["box"]["cy"], d["box"]["cz"],
                                    d["box"]["l"], d["box"]["w"], d["box"]["h"], d["box"]["rot_z"])
                                   for d in dets], np.float32)
                else:
                    pos = np.zeros((0, 2), np.float32)
                    bx = np.zeros((0, 7), np.float32)
                det[seq][fr] = (pos, bx)
        self.det = det
        return det

    @staticmethod
    def nearest_box(gpos, pos, boxes, thresh=MATCH_THRESH):
        # Caja de la detección más cercana a la etiqueta, si está dentro del umbral.
        if pos.shape[0] == 0:
            return None
        d = np.linalg.norm(pos - np.array(gpos), axis=1)
        j = int(np.argmin(d))
        return boxes[j] if d[j] <= thresh else None

    @staticmethod
    def count_in_box(pts, cx, cy, cz, l, w, h, rot_z):
        # Cuenta los puntos LiDAR dentro de la caja detectada.
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        dz = pts[:, 2] - cz
        c, s = np.cos(-rot_z), np.sin(-rot_z)
        xb = c * dx - s * dy
        yb = s * dx + c * dy
        return int(np.count_nonzero((np.abs(xb) <= l / 2) & (np.abs(yb) <= w / 2) & (np.abs(dz) <= h / 2)))

    def feats(self, pts, pxy, box):
        # 7 características del entorno alrededor de la posición detectada del peatón.
        cx, cy, cz, l, w, h, rot_z = box
        # 1) Número de puntos sobre el peatón (dentro de la caja)
        npnt = self.count_in_box(pts, cx, cy, cz, l, w, h, rot_z)

        # Recorte del entorno, centrado en la detección
        dx = pxy[:, 0] - cx
        dy = pxy[:, 1] - cy
        d = np.sqrt(dx * dx + dy * dy)
        # Recorte de los puntos más lejanos del radio definido
        inw = d < self.r
        d_w, dx_w, dy_w = d[inw], dx[inw], dy[inw]
        # Excluye al propio peatón
        ns = d_w > self.self_radius
        d_ns, dx_ns, dy_ns = d_w[ns], dx_w[ns], dy_w[ns]
        # Guarda el obstáculo más cercano
        near = d_ns.min() if d_ns.size else self.r

        if d_ns.size:
            # Cálculo del ángulo relativo al rumbo del peatón
            an = np.arctan2(dy_ns, dx_ns) - rot_z
            an = (an + np.pi) % (2 * np.pi) - np.pi
            # Definición del cálculo de distancia al obstáculo más cercano
            g = lambda m: (d_ns[m].min() if m.any() else self.r)
            # Define los 4 sectores: delante, izq, detrás, derecha
            ff = g(np.abs(an) < np.pi / 4)
            lf = g((an >= np.pi / 4) & (an < 3 * np.pi / 4))
            bf = g(np.abs(an) >= 3 * np.pi / 4)
            rf = g((an <= -np.pi / 4) & (an > -3 * np.pi / 4))
        else:
            ff = lf = bf = rf = self.r
        # Devuelve un vector de 7 características: nº de puntos sobre el peatón, obstáculo más cercano,
        # densidad total de puntos en el recorte, y obstáculo más cercano en las 4 direcciones.
        return np.array([np.log1p(npnt), min(near, self.r), np.log1p(d_w.size),
                         min(ff, self.r), min(lf, self.r), min(bf, self.r), min(rf, self.r)], dtype=np.float32)

    def load_dataset(self) -> pd.DataFrame:
        # Carga CSV de datos 2D
        self.df = pd.read_csv(f"{self.data_dir}/jrdb_clean_trajectories.csv")
        return self.df

    def build_windows(self):
        win_meta = []   # (seq, tid, [8 frames])
        # Para cada trayectoria
        for sid, traj in self.df.groupby("subtrack_id"):
            # Ordena por frame, submuestrea (1 de cada 3 frames) y filtra las trayectorias
            # con menos de 20 frames
            traj = traj.sort_values("frame")
            traj_ds = traj.iloc[::self.stride]
            if len(traj_ds) < self.win:
                continue
            # Número de frames, secuencia, track ID
            fr = traj_ds["frame"].values.astype(int)
            seq = traj_ds["sequence"].iloc[0]
            tid = int(traj_ds["track_id"].iloc[0])
            # Ventana deslizante y guarda los 8 primeros frames: (seq, tid, [8 frames])
            for start in range(len(traj_ds) - self.win + 1):
                win_meta.append((seq, tid, fr[start:start + self.obs_len].tolist()))

        # Cuenta el número de ventanas y comprueba que sea igual al número 2D
        n = len(win_meta)
        Y = np.load(f"{self.data_dir}/data/Y_ds.npy")
        print(f"Ventanas: {n}  vs Y_ds: {Y.shape[0]}")
        if n != Y.shape[0]:
            raise SystemExit("ERROR: no alineado con Y_ds.")
        self.win_meta = win_meta
        return win_meta

    def build_features(self):
        n = len(self.win_meta)
        self.X = np.zeros((n, self.obs_len, 7), np.float32)
        self.flags = np.zeros((n, self.obs_len), bool)

        need = defaultdict(list)
        # Agrupa por nube -> recorre las ventanas
        for i, (seq, tid, frames) in enumerate(self.win_meta):
            # Recorre los 8 frames observados de la ventana
            for slot, fr in enumerate(frames):
                # Guarda para cada secuencia y frame -> (fila, slot 0-7, tid)
                need[(seq, int(fr))].append((i, slot, tid))

        print(f"Nubes únicas: {len(need)}")
        # Para cada nube -> secuencia, frame, fila
        for k, ((seq, fr), members) in enumerate(need.items()):
            if k % 1000 == 0:
                print(f"\r  {k}/{len(need)}", end="", flush=True)
            # Carga el pcd correspondiente
            pp = self.pcd_dir / seq / f"{fr:06d}.pcd"
            if not pp.exists():
                continue
            # Lee la nube de puntos (3D completa: se usa para contar puntos dentro de la caja)
            pts = np.asarray(o3d.io.read_point_cloud(str(pp)).points)
            if pts.size == 0:
                continue
            # Lleva la nube al marco EGO (igual que las detecciones/etiquetas 3D)
            pts = (LOWER2EGO @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            # Filtra por altura (-0.5 a 2) y se queda con plano XY para las features de entorno
            pxy = pts[(pts[:, 2] > self.z_min) & (pts[:, 2] < self.z_max)][:, :2]
            # Etiquetas (para emparejar) y detecciones (cajas) de ese frame
            gb = self.gt.get(seq, {}).get(fr, {})
            pos, boxes = self.det.get(seq, {}).get(fr, (np.zeros((0, 2), np.float32), np.zeros((0, 7), np.float32)))
            # Para cada frame (0-7) en cada ventana en cada track ID
            for i, slot, tid in members:
                # Posición etiquetada del peatón seguido (para buscar su detección)
                gpos = gb.get(tid)
                if gpos is None:
                    continue
                # Detección más cercana: si no hay, se deja la fila a cero (sin entorno)
                box = self.nearest_box(gpos, pos, boxes, self.match_thresh)
                if box is None:
                    continue
                # Guarda en la fila ventana i, y posición slot (frame 0-7) -> las 7 características
                self.X[i, slot] = self.feats(pts, pxy, box)
                # Marca que el frame tiene entorno
                self.flags[i, slot] = True

        print(f"Cobertura por-frame: {100*self.flags.mean():.1f}%")
        return self.X, self.flags

    def save_numpy(self):
        # Guarda las entradas y flags
        os.makedirs(f"{self.data_dir}/data", exist_ok=True)
        np.save(f"{self.data_dir}/data/X_lidar_seq.npy", self.X)
        np.save(f"{self.data_dir}/data/lidar_seq_flags.npy", self.flags)
        print("Guardado: X_lidar_seq.npy, lidar_seq_flags.npy")

    def run(self):
        # 1. Carga etiquetas 3D (solo para emparejar) y detecciones 3D (fuente de las cajas)
        self.load_gt()
        self.load_dets()
        # 2. Carga CSV de datos 2D y construye las ventanas
        self.load_dataset()
        self.build_windows()
        # 3. Calcula las 7 características de entorno por frame observado
        self.build_features()
        # 4. Guarda en .npy
        self.save_numpy()


if __name__ == "__main__":

    builder = Lidar3DEnvBuilder()
    builder.run()
