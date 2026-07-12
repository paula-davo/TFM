"""
Obtiene el dataset con datos lidar con aumento del horizonte.
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
N_FEATURES = 8
R, SELF_RADIUS, Z_MIN, Z_MAX = 3.5, 0.5, -0.5, 2.0


def load_all_boxes():
    boxes = defaultdict(lambda: defaultdict(dict))
    # Recorre todos los JSON de etiquetas 3D
    for jf in LABELS_3D_DIR.glob("*.json"):
        # Identifica la secuencia
        seq = jf.stem
        # Carga el JSON
        with open(jf) as f:
            data = json.load(f)
        # Por cada frame
        for pcd_key, anns in data["labels"].items():
            # Por cada peatón anotado
            fr = int(pcd_key.replace(".pcd", ""))
            for ann in anns:
                # Obtiene el track ID y el cuadro delimitador
                tid = int(ann["label_id"].split(":")[1])
                b = ann["box"]; at = ann.get("attributes", {})
                # Guarda el centro del cuadro delimitador (cx, cy) en metros, la rotación de este,
                # el número de puntos lidar en el cuadro, y la distancia peatón-robot. Identificados
                # por track ID, frame y secuencia.
                boxes[seq][fr][tid] = (b["cx"], b["cy"], b["rot_z"],
                                       at.get("num_points", 0), at.get("distance", 0.0))
    return boxes


def compute_features(pts_xy, cx, cy, rot_z, num_points, distance):
    # 1. Recorte del peatón
    # Distancia punto-peatón
    dx = pts_xy[:, 0] - cx
    dy = pts_xy[:, 1] - cy
    d = np.sqrt(dx*dx + dy*dy)
    # Recorta una ventana de radio 3.5 m
    inw = d < R
    d_w, dx_w, dy_w = d[inw], dx[inw], dy[inw]
    # Excluye los puntos que pertenecen al propio peatón
    ns = d_w > SELF_RADIUS
    d_ns, dx_ns, dy_ns = d_w[ns], dx_w[ns], dy_w[ns]
    # 2. Características escalares
    # Número de puntos dentro de la caja del peatón
    f_np = np.log1p(num_points)
    # Distancia peatón-robot
    f_dist = distance
    # Distancia al obstáculo más cercano
    f_near = d_ns.min() if d_ns.size else R
    # Mide la aglomeración: densidad de obstáculos
    f_clut = np.log1p(d_w.size)
    # 3. Características direccionales
    if d_ns.size:
        # Ángulo de cada obstáculo relativo a la rotación del peatón
        ang = np.arctan2(dy_ns, dx_ns) - rot_z
        ang = (ang + np.pi) % (2*np.pi) - np.pi
        # Ángulos en las 4 direcciones (delante, izq, detrás, derecha)
        fr_ = np.abs(ang) < np.pi/4
        lf = (ang >= np.pi/4) & (ang < 3*np.pi/4)
        bk = np.abs(ang) >= 3*np.pi/4
        rt = (ang <= -np.pi/4) & (ang > -3*np.pi/4)
        # Obstáculo más cercano en cada dirección
        f_f = d_ns[fr_].min() if fr_.any() else R
        f_l = d_ns[lf].min() if lf.any() else R
        f_b = d_ns[bk].min() if bk.any() else R
        f_r = d_ns[rt].min() if rt.any() else R
    else:
        f_f = f_l = f_b = f_r = R
    # Devuelve 8 características: densidad, dist_robot, obstáculo más cercano, clutter, delante, izq, detrás, derecha.
    return np.array([f_np, f_dist, min(f_near, R), f_clut,
                     min(f_f, R), min(f_l, R), min(f_b, R), min(f_r, R)], dtype=np.float32)


def build_sample_meta_ds(df):
    meta = []
    # Para cada trayectoria
    for _, traj in df.groupby("subtrack_id"):
        # Ordena por frame, submuestrea (1 de cada 3 frames) y descarta los que tienen menos de 20 frames
        traj = traj.sort_values("frame")
        traj_ds = traj.iloc[::STRIDE]
        if len(traj_ds) < WIN:
            continue
        # Obtiene frames, secuencias, track ID
        frames = traj_ds["frame"].values
        seq = traj_ds["sequence"].iloc[0]
        tid = int(traj_ds["track_id"].iloc[0])
        # Ventana deslizante 
        for start in range(len(traj_ds) - WIN + 1):
            meta.append((seq, int(frames[start + OBS_LEN - 1]), tid))
    # Devuelve lista con: secuencia, último frame observado, track ID
    return meta


if __name__ == "__main__":
    # 1. Carga los datos Lidar 3D empleados
    boxes = load_all_boxes()
    # 2. Carga el CSV de datos 2D
    df = pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv")
    # 3. Construye lista con ventana deslizante
    meta = build_sample_meta_ds(df)
    n = len(meta)
    # 4. Comprueba tamaño entrada-salida
    Y = np.load(f"{DATA_DIR}/Y_ds.npy")
    print(f"Muestras: {n}  vs  Y_ds: {Y.shape[0]}")
    if n != Y.shape[0]:
        raise SystemExit("ERROR: no alineado con Y_ds (¿mismo STRIDE?).")
    print("Alineación correcta")
    # 5. Obtención de entradas
    # Características por muestra (N, 8) y vector que indica si se ha obtenido características
    # lidar para esa muestra
    feats = np.zeros((n, N_FEATURES), dtype=np.float32)
    flags = np.zeros(n, dtype=bool)

    # Agrupa las muestras por nube: (secuencia, frame) -> [(muestra, track ID)]
    groups = defaultdict(list)
    for i, (seq, fr, tid) in enumerate(meta):
        groups[(seq, fr)].append((i, tid))

    # Por cada nube de puntos lidar
    print(f"Nubes únicas: {len(groups)}")
    hits = 0
    for k, ((seq, fr), members) in enumerate(groups.items()):
        if k % 500 == 0:
            print(f"\r  {k}/{len(groups)}", end="", flush=True)
        # Lee la nube de puntos
        pcd_path = PCD_DIR / seq / f"{fr:06d}.pcd"
        if not pcd_path.exists():
            continue
        pts = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points)
        if pts.size == 0:
            continue
        # Se queda con plano (X, Y)
        pts_xy = pts[(pts[:, 2] > Z_MIN) & (pts[:, 2] < Z_MAX)][:, :2]
        # Obtiene los datos 3D de ese frame
        fb = boxes.get(seq, {}).get(fr, {})

        # Para cada peatón de esa nube
        for i, tid in members:
            # Obtiene la caja 3D del peatón concreto
            b = fb.get(tid)
            if b is None:
                continue
            # Calcula las características y lo marca como guardado
            feats[i] = compute_features(pts_xy, *b)
            flags[i] = True
            hits += 1
            
    print(f"Cobertura: {hits}/{n} ({100*hits/n:.1f}%)")
    
    # 6. Guarda los datos
    np.save(f"{DATA_DIR}/lidar_feats_ds.npy", feats)
    np.save(f"{DATA_DIR}/lidar_flags_ds.npy", flags)
    print("Guardado: lidar_feats_ds.npy, lidar_flags_ds.npy")
