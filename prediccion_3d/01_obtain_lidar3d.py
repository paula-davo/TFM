"""
Obtención de datos de etiquetas 3D del lidar -> 8 frames de observación, 12 de predicción.

Características de entrada: cx, cy, vx, vy, ax, ay, speed, sin_dir, cos_dir (metros)
Características de salida: desplazamiento relativo (X e Y) en metros.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

OBS_LEN, PRED_LEN, STRIDE = 8, 12, 3
WIN = OBS_LEN + PRED_LEN


class Lidar3DLabelsBuilder:

    """
    Lidar3DLabelsBuilder genera las muestras 3D (X, Y) a partir de las etiquetas 3D (GT).

    1) Carga etiquetas 3D
    2) Ventana deslizante submuestreada (STRIDE=3): 8 frames de entrada, 12 de salida
    3) Entrada = características de movimiento; salida = desplazamiento relativo GT
    4) Guardado en .npy
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN, stride=STRIDE):
        # Directorio y subcarpeta de etiquetas 3D del dataset
        self.data_dir = data_dir
        self.labels_3d_dir = Path(data_dir) / "train_dataset_with_activity" / "train_dataset_with_activity" / "labels" / "labels_3d"

        # Longitudes de ventana y submuestreo temporal
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.win = obs_len + pred_len

        # Datos y resultados
        self.xy = None
        self.df = None
        self.X3d = None
        self.Y3d = None

    def load_xy(self) -> dict:
        xy = defaultdict(lambda: defaultdict(dict))
        # Recorre los JSON con etiquetas 3D
        for jf in self.labels_3d_dir.glob("*.json"):
            # Guarda la secuencia
            seq = jf.stem
            # Carga el JSON
            with open(jf) as f:
                data = json.load(f)
            # Para cada frame
            for pcd_key, anns in data["labels"].items():
                fr = int(pcd_key.replace(".pcd", ""))
                # Y para cada anotación en el frame
                for ann in anns:
                    # Guarda el track ID, y la detección
                    tid = int(ann["label_id"].split(":")[1])
                    xy[seq][fr][tid] = (ann["box"]["cx"], ann["box"]["cy"])
        # xy[seq][frame][tid] = (cx, cy) (en metros)
        self.xy = xy
        return xy

    @staticmethod
    def fill_missing(seq_pos):
        # Longitud de la ventana
        n = len(seq_pos)
        # 1. Rellena hacia delante -> recorre de izquierda a derecha. Si el frame es None, lo
        # rellena con la última posición válida vista
        last = None
        for i in range(n):
            if seq_pos[i] is None:
                seq_pos[i] = last
            else:
                last = seq_pos[i]
        # 2. Rellena hacia detrás -> recorre de derecha a izquierda. Si el frame es None, lo rellena
        # con la siguiente posición válida vista
        nxt = None
        for i in range(n - 1, -1, -1):
            if seq_pos[i] is None:
                seq_pos[i] = nxt
            else:
                nxt = seq_pos[i]
        # Si continúa habiendo algún None -> lo sustituye por 0.0.
        return [p if p is not None else (0.0, 0.0) for p in seq_pos]

    def load_dataset(self) -> pd.DataFrame:
        # Carga datos 2D
        self.df = pd.read_csv(f"{self.data_dir}/jrdb_clean_trajectories.csv")
        return self.df

    def build(self):
        X3d, Y3d = [], []
        # Para cada trayectoria (por subtrack ID)
        for sid, traj in self.df.groupby("subtrack_id"):
            # Ordena los frames, submuestrea (1 de cada 3 frames) y descarta las trayectorias con menos de
            # 20 frames
            traj = traj.sort_values("frame")
            traj_ds = traj.iloc[::self.stride]
            if len(traj_ds) < self.win:
                continue
            # Trayectoria con: frame, secuencia, track ID, y etiquetas 3D
            frames = traj_ds["frame"].values.astype(int)
            seq = traj_ds["sequence"].iloc[0]
            tid = int(traj_ds["track_id"].iloc[0])
            fb = self.xy.get(seq, {})

            # Ventana deslizante
            for start in range(len(traj_ds) - self.win + 1):
                # 20 frames de ventana
                wf = frames[start:start + self.win]
                # Posiciones 3D del peatón en cada frame
                pos = [fb.get(int(f), {}).get(tid) for f in wf]
                pos = np.array(self.fill_missing(pos), dtype=np.float32)   # (20,2) metros

                # 8 frames de observación: cálculo de posición (X e Y) en metros, velocidad (X e Y),
                # aceleración (X e Y), velocidad lineal, ángulo de avance (seno y coseno de la dirección).
                # Todas en metros.
                obs = pos[:self.obs_len]
                cx, cy = obs[:, 0], obs[:, 1]
                vx = np.diff(cx, prepend=cx[0])
                vy = np.diff(cy, prepend=cy[0])
                ax = np.diff(vx, prepend=vx[0])
                ay = np.diff(vy, prepend=vy[0])
                speed = np.sqrt(vx**2 + vy**2)
                ang = np.arctan2(vy, vx)
                feats = np.stack([cx, cy, vx, vy, ax, ay, speed, np.sin(ang), np.cos(ang)], axis=1)

                # 12 frames de salida: desplazamiento relativo en metros
                last = pos[self.obs_len - 1]
                y = pos[self.obs_len:] - last

                X3d.append(feats)
                Y3d.append(y)

        # Arrays -> entrada (N, 8, 9) y salida (N, 12, 2)
        self.X3d = np.array(X3d, dtype=np.float32)
        self.Y3d = np.array(Y3d, dtype=np.float32)
        return self.X3d, self.Y3d

    def save_numpy(self):
        # Carga la salida del dataset 2D para comprobar que el número de muestras es el mismo.
        Y_img = np.load(f"{self.data_dir}/data/Y_ds.npy")
        print(f"X3d_motion: {self.X3d.shape}   Y3d: {self.Y3d.shape}   (Y_ds: {Y_img.shape[0]})")
        if self.X3d.shape[0] != Y_img.shape[0]:
            raise SystemExit("ERROR: no alineado con Y_ds.")
        print("Alineación correcta")
        # Imprime desplazamiento medio
        print(f"\nY3d (metros) — desplazamiento medio: {np.linalg.norm(self.Y3d, axis=2).mean():.3f} m")
        # Guarda los datos
        os.makedirs(f"{self.data_dir}/data", exist_ok=True)
        np.save(f"{self.data_dir}/data/X3d_motion.npy", self.X3d)
        np.save(f"{self.data_dir}/data/Y3d.npy", self.Y3d)
        print("Guardado: X3d_motion.npy, Y3d.npy")

    def run(self):
        # 1. Carga posiciones 3D (etiquetas)
        self.load_xy()
        # 2. Carga datos 2D
        self.load_dataset()
        # 3. Construye muestras (X, Y)
        self.build()
        # 4. Guarda en .npy
        self.save_numpy()


if __name__ == "__main__":

    builder = Lidar3DLabelsBuilder()
    builder.run()
