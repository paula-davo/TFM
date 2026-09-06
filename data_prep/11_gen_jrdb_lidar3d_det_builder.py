"""
Datos 3D a partir de las detecciones del lidar.
Salidas: X3d_det_motion.npy (N,8,9), Y3d_det.npy (N,12,2)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

OBS_LEN, PRED_LEN, STRIDE = 8, 12, 3
MATCH_THRESH = 1.0   # Umbral máximo para emparejar detección-etiqueta (m)


class JRDBLidar3DDetBuilder:

    """
    JRDBLidar3DDetBuilder genera las muestras 3D (X, Y) a partir de las detecciones 3D.

    1) Carga etiquetas 3D (GT) y detecciones 3D
    2) Ventana deslizante submuestreada (STRIDE=3): 8 frames de entrada, 12 de salida
    3) Entrada = detecciones emparejadas a la GT; salida = desplazamiento relativo GT
    4) GUARDADO en .npy
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 stride=STRIDE, match_thresh=MATCH_THRESH):
        # Directorio y subcarpetas del dataset
        self.data_dir = data_dir
        train_dir = Path(data_dir) / "train_dataset_with_activity" / "train_dataset_with_activity"
        self.labels_3d_dir = train_dir / "labels" / "labels_3d"
        self.det_3d_dir = train_dir / "detections" / "detections_3d"

        # Longitudes de ventana y parámetros del emparejamiento
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride # Submuestreo temporal (pasos) 
        self.win = obs_len + pred_len
        self.match_thresh = match_thresh # Umbral máximo para relacionar detección - etiqueta (m)

        # Datos y resultados
        self.gt = None
        self.det = None
        self.df = None
        self.X3d = None
        self.Y3d = None
        self.matched = 0
        self.total = 0

    def load_gt_xy(self) -> dict:
        # Etiquetas 3D -> centro del peatón por secuencia, frame y track_id
        xy = defaultdict(lambda: defaultdict(dict))
        for jf in self.labels_3d_dir.glob("*.json"):
            seq = jf.stem
            with open(jf) as f:
                labels = json.load(f)["labels"]
            for k, anns in labels.items():
                fr = int(k.replace(".pcd", ""))
                for a in anns:
                    tid = int(a["label_id"].split(":")[1])
                    xy[seq][fr][tid] = (a["box"]["cx"], a["box"]["cy"])
        self.gt = xy
        return xy

    def load_det_xy(self) -> dict:
        # Detecciones 3D -> centros de detecciones por secuencia y frame
        det = defaultdict(dict)
        for jf in self.det_3d_dir.glob("*.json"):
            seq = jf.stem
            with open(jf) as f:
                data = json.load(f)["detections"]
            for k, dets in data.items():
                fr = int(k.replace(".pcd", ""))
                pts = [(d["box"]["cx"], d["box"]["cy"]) for d in dets]
                det[seq][fr] = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)
        self.det = det
        return det

    @staticmethod
    def nearest_det(gt_xy, det_arr, match_thresh=MATCH_THRESH):
        # Detección más cercana a la etiqueta si está dentro del umbral
        if det_arr.shape[0] == 0:
            return None
        d = np.linalg.norm(det_arr - np.array(gt_xy), axis=1)
        j = np.argmin(d)
        return tuple(det_arr[j]) if d[j] <= match_thresh else None

    @staticmethod
    def fill(seq_pos):
        # Rellena huecos (None) de una lista de posiciones:
        # 1) con la última posición conocida, 2) con la siguiente, 3) con (0, 0)
        # Casos donde una detección ha fallado y no se empareja, y esto deja un hueco en la secuencia de 
        # observación o predicción.
        n = len(seq_pos); last = None
        for i in range(n):
            if seq_pos[i] is None: seq_pos[i] = last
            else: last = seq_pos[i]
        nxt = None
        for i in range(n - 1, -1, -1):
            if seq_pos[i] is None: seq_pos[i] = nxt
            else: nxt = seq_pos[i]
        return [p if p is not None else (0.0, 0.0) for p in seq_pos]

    def load_dataset(self) -> pd.DataFrame:
        # Carga el CSV de subtrayectorias filtradas
        self.df = pd.read_csv(f"{self.data_dir}/jrdb_clean_trajectories.csv")
        return self.df

    def build(self):
        X3d, Y3d = [], []
        self.matched, self.total = 0, 0

        for sid, traj in self.df.groupby("subtrack_id"):
            # Para cada trayectoria, submuestrea (STRIDE) y descarta las que tienen menos de WIN frames
            traj = traj.sort_values("frame")
            traj_ds = traj.iloc[::self.stride]
            if len(traj_ds) < self.win:
                continue
            # Guarda frame, secuencia y track_id de la subtrayectoria
            frames = traj_ds["frame"].values.astype(int)
            seq = traj_ds["sequence"].iloc[0]
            tid = int(traj_ds["track_id"].iloc[0])

            # Para cada ventana de WIN frames
            for start in range(len(traj_ds) - self.win + 1):
                wf = frames[start:start + self.win]
                # OBS (emparejadas a la GT)
                obs_det = []
                for f in wf[:self.obs_len]:
                    # Extra posición de la etiqueta y busca la detección más cercana (si hay)
                    gpos = self.gt.get(seq, {}).get(int(f), {}).get(tid)
                    self.total += 1
                    if gpos is None:
                        obs_det.append(None); continue
                    dpos = self.nearest_det(gpos, self.det.get(seq, {}).get(int(f), np.zeros((0, 2), np.float32)), self.match_thresh)
                    if dpos is not None:
                        self.matched += 1
                    obs_det.append(dpos)
                obs_det = np.array(self.fill(obs_det), dtype=np.float32)

                # PRED: posiciones reales GT
                pred_gt = []
                for f in wf[self.obs_len:]:
                    pred_gt.append(self.gt.get(seq, {}).get(int(f), {}).get(tid))
                pred_gt = np.array(self.fill(pred_gt), dtype=np.float32)

                # Se calcula las características de movimiento a partir de las detecciones observadas (OBS_LEN)
                # Centro (cx, cy), velocidad (vx, vy), aceleración (ax, ay), velocidad lineal y ángulo (sin, cos)
                cx, cy = obs_det[:, 0], obs_det[:, 1]
                vx = np.diff(cx, prepend=cx[0]) 
                vy = np.diff(cy, prepend=cy[0])
                ax = np.diff(vx, prepend=vx[0]) 
                ay = np.diff(vy, prepend=vy[0])
                speed = np.sqrt(vx**2 + vy**2)
                ang = np.arctan2(vy, vx)
                feats = np.stack([cx, cy, vx, vy, ax, ay, speed, np.sin(ang), np.cos(ang)], axis=1)

                # Desplazamiento futuro (GT) desde la última posición observada DETECTADA
                last_det = obs_det[self.obs_len - 1] if self.obs_len <= len(obs_det) else obs_det[-1]
                y = pred_gt - last_det

                # Guarda la muestra (X, Y)
                X3d.append(feats); 
                Y3d.append(y)

        self.X3d = np.array(X3d, dtype=np.float32)
        self.Y3d = np.array(Y3d, dtype=np.float32)
        return self.X3d, self.Y3d

    def save_numpy(self):
        # Comprueba alineación con Y_ds y guarda las muestras
        Yref = np.load(f"{self.data_dir}/data/Y_ds.npy")
        print(f"\nX3d_det: {self.X3d.shape}  Y3d_det: {self.Y3d.shape}  (Y_ds: {Yref.shape[0]})")
        if self.X3d.shape[0] != Yref.shape[0]:
            raise SystemExit("ERROR: no alineado con Y_ds.")
        print(f"Tasa de emparejamiento detección↔peatón: {100*self.matched/self.total:.1f}%")

        os.makedirs(f"{self.data_dir}/data", exist_ok=True)
        np.save(f"{self.data_dir}/data/X3d_det_motion.npy", self.X3d)
        np.save(f"{self.data_dir}/data/Y3d_det.npy", self.Y3d)
        print("Guardado: X3d_det_motion.npy, Y3d_det.npy")

    def run(self):
        print("Cargando GT 3D y detecciones 3D...")
        # 1. Carga etiquetas 3D
        self.load_gt_xy()
        # 2. Carga detecciones 3D
        self.load_det_xy()
        # 3. Carga subtrayectorias filtradas
        self.load_dataset()
        # 4. Construye muestras (X, Y)
        self.build()
        # 5. Guarda en .npy
        self.save_numpy()


if __name__ == "__main__":

    builder = JRDBLidar3DDetBuilder()
    builder.run()
