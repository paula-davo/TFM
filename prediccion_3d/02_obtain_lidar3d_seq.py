"""
Obtención de datos de entorno lidar por cada frame para los 8 frames observados de cada ventana
submuestreada.

Características de entrada: (N, 8, 8): (densidad, dist robot, obstáculo más cercano,
clutter, obstáculo más cercano en las 4 direcciones).
"""

import json
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LABELS_3D_DIR = TRAIN_DIR / "labels" / "labels_3d"
PCD_DIR = TRAIN_DIR / "pointclouds" / "lower_velodyne"

OBS_LEN, PRED_LEN, STRIDE = 8, 12, 3
WIN = OBS_LEN + PRED_LEN
R, SELF_RADIUS, Z_MIN, Z_MAX = 3.5, 0.5, -0.5, 2.0


def load_boxes():
    boxes = defaultdict(lambda: defaultdict(dict))
    # Recorre los JSON con etiquetas 3D
    for jf in LABELS_3D_DIR.glob("*.json"):
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
                # Guarda el track ID, la detección (centro X e Y), la rotación del peatón, el
                # número de puntos lidar sobre este, y la distancia al robot
                tid = int(ann["label_id"].split(":")[1]); b = ann["box"]; at = ann.get("attributes", {})
                boxes[seq][fr][tid] = (b["cx"], b["cy"], b["rot_z"], at.get("num_points", 0), at.get("distance", 0.0))
    # boxes[seq][frame][tid] = (cx, cy, rot_z, num_points, distance)
    return boxes


def feats(pts_xy, cx, cy, rot_z, num_points, distance):
    # Distancia de cada punto lidar al peatón
    dx = pts_xy[:, 0]-cx
    dy = pts_xy[:, 1]-cy
    d = np.sqrt(dx*dx+dy*dy)
    # Recorte de los puntos más lejanos del radio definido
    inw = d < R
    d_w, dx_w, dy_w = d[inw], dx[inw], dy[inw]
    # Excluye al peatón
    ns = d_w > SELF_RADIUS
    d_ns, dx_ns, dy_ns = d_w[ns], dx_w[ns], dy_w[ns]
    # Guarda el obstáculo más cercano
    near = d_ns.min() if d_ns.size else R
    
    if d_ns.size:
        # Cálculo del ángulo relativo al rumbo del peatón
        an = np.arctan2(dy_ns, dx_ns) - rot_z
        an = (an+np.pi) % (2*np.pi) - np.pi
        # Definición del cálculo de distancia al obstáculo más cercano
        g = lambda m: (d_ns[m].min() if m.any() else R)
        # Define los 4 sectores: delante, izq, detrás, derecha
        ff = g(np.abs(an) < np.pi/4)
        lf = g((an >= np.pi/4) & (an < 3*np.pi/4))
        bf = g(np.abs(an) >= 3*np.pi/4)
        rf = g((an <= -np.pi/4) & (an > -3*np.pi/4))
    else:
        ff = lf = bf = rf = R
    # Devuelve un vector de 8 características: densidad de puntos alrededor del peatón,
    # distancia al robot, obstáculo más cercano, aglomeración en torno al peatón, obstáculo 
    # más cercano en las 4 direcciones.
    return np.array([np.log1p(num_points), distance, min(near, R), np.log1p(d_w.size),
                     min(ff, R), min(lf, R), min(bf, R), min(rf, R)], dtype=np.float32)


if __name__ == "__main__":
    # Carga información 3D del peatón
    boxes = load_boxes()
    # Carga CSV de datos 2D
    df = pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv")

    win_meta = []   # (seq, tid, [8 frames])
    # Para cada trayectoria
    for sid, traj in df.groupby("subtrack_id"):
        # Ordena por frame, submuestrea (1 de cada 3 frames) y filtra las trayectorias
        # con menos de 20 frames
        traj = traj.sort_values("frame")
        traj_ds = traj.iloc[::STRIDE]
        if len(traj_ds) < WIN:
            continue
        # Número de frames, secuencia, track ID
        fr = traj_ds["frame"].values.astype(int)
        seq = traj_ds["sequence"].iloc[0]
        tid = int(traj_ds["track_id"].iloc[0])
        # Ventana deslizante y guarda los 8 primeros frames: (seq, tid, [8 frames])
        for start in range(len(traj_ds) - WIN + 1):
            win_meta.append((seq, tid, fr[start:start+OBS_LEN].tolist()))

    # Cuenta el número de ventanas y comprueba que sea igual al número 2D
    n = len(win_meta)
    Y = np.load(f"{DATA_DIR}/Y_ds.npy")
    print(f"Ventanas: {n}  vs Y_ds: {Y.shape[0]}")
    if n != Y.shape[0]:
        raise SystemExit("ERROR: no alineado con Y_ds.")

    X = np.zeros((n, OBS_LEN, 8), np.float32)
    flags = np.zeros((n, OBS_LEN), bool)

    need = defaultdict(list)
    # Agrupa por nube -> recorre las ventanas
    for i, (seq, tid, frames) in enumerate(win_meta):
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
        pp = PCD_DIR / seq / f"{fr:06d}.pcd"
        if not pp.exists():
            continue
        # Lee la nube de puntos
        pts = np.asarray(o3d.io.read_point_cloud(str(pp)).points)
        if pts.size == 0:
            continue
        # Filtra por altura (-0.5 a 2) y se queda con plano XY
        pxy = pts[(pts[:, 2] > Z_MIN) & (pts[:, 2] < Z_MAX)][:, :2]
        # Obtiene las etiquetas 3D de ese frame
        fb = boxes.get(seq, {}).get(fr, {})
        # Para cada frame (0-7) en cada ventana en cada track ID
        for i, slot, tid in members:
            # Caja del peatón
            b = fb.get(tid)
            if b is None:
                continue
            # Guarda en la fila ventana i, y posición slot (frame 0-7) -> las 8 características
            X[i, slot] = feats(pxy, *b)
            # Marca que el frame tiene entorno
            flags[i, slot] = True

    print(f"Cobertura por-frame: {100*flags.mean():.1f}%")

    # Guarda las entradas y flags
    np.save(f"{DATA_DIR}/X_lidar_seq.npy", X)
    np.save(f"{DATA_DIR}/lidar_seq_flags.npy", flags)
    print("Guardado: X_lidar_seq.npy, lidar_seq_flags.npy")
