"""
Genera los datos con información de pose: 11 características 2D +
8 características de pose. del modelo LSTM con información de pose.
"""

import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
JSON_FILES = [f"{DATA_DIR}/jrdb_mmpose_train/jrdb_mmpose_train/train_individual.json",
              f"{DATA_DIR}/jrdb_mmpose_train/jrdb_mmpose_train/val_individual.json"]

OBS_LEN = 8
PRED_LEN = 12
STRIDE = 3
WIN = OBS_LEN + PRED_LEN

POSE_FEATURES = ["sin_sh", "cos_sh", "sh_width", "sin_hip", "cos_hip",
                 "hip_width", "stride", "pose_valid"]


def pose_vector(kp, bbox_w, bbox_h):
    # Agrupa keypoints -> (x, y, confianza)
    p = [(kp[i*3], kp[i*3+1], kp[i*3+2]) for i in range(17)]
    # Visibilidad -> confianza > 0
    vis = lambda i: p[i][2] > 0
    # bbox normalizados y sin 0.0
    bw = bbox_w if bbox_w > 1e-6 else 1.0
    bh = bbox_h if bbox_h > 1e-6 else 1.0
    sin_sh = cos_sh = sh_w = sin_hip = cos_hip = hip_w = stride = 0.0
    # Hombros -> si kp 3 y 5 son visibles -> calcula el vector que los une, su ángulo (seno y coseno
    # de la dirección), y la longitud
    if vis(3) and vis(5):
        dx, dy = p[3][0]-p[5][0], p[3][1]-p[5][1]
        a = np.arctan2(dy, dx)
        sin_sh, cos_sh = np.sin(a), np.cos(a)
        sh_w = np.hypot(dx, dy)/bw
    # Caderas -> si kp 10 y 11 son visibles -> calcula el vector que los une, su ángulo (seno y coseno
    # de la dirección), y la longitud
    if vis(10) and vis(11):
        dx, dy = p[10][0]-p[11][0], p[10][1]-p[11][1]
        a = np.arctan2(dy, dx)
        sin_hip, cos_hip = np.sin(a), np.cos(a)
        hip_w = np.hypot(dx, dy)/bw
    # Zancada -> si kp 15 y 16 son visibles -> mide la distancia entre los pies
    if vis(15) and vis(16):
        stride = np.hypot(p[15][0]-p[16][0], p[15][1]-p[16][1])/bh
    # Calidad de la pose -> cuantos de estos kp clave son visibles
    pose_valid = sum(1 for j in [3, 5, 10, 11, 15, 16] if vis(j)) / 6.0
    return [sin_sh, cos_sh, sh_w, sin_hip, cos_hip, hip_w, stride, pose_valid]


def build_pose_lookup():
    lookup = {}
    # Recorre los JSON
    for json_file in JSON_FILES:
        # Carga JSON
        with open(json_file) as f:
            data = json.load(f)
        # Índice de imágenes
        image_info = {img["id"]: img["file_name"] for img in data["images"]}
        # Para cada anotación
        for ann in data["annotations"]:
            # Cámara, secuencia, nombre fichero
            camera, sequence, fn = image_info[ann["image_id"]].split("/")
            # Obtiene el frame del nombre del fichero
            frame = int(fn.replace(".jpg", ""))
            # Cuadro delimitador del peatón
            bbox = ann["bbox"]
            # Pose
            lookup[(sequence, camera, frame, int(ann["track_id"]))] = pose_vector(ann["keypoints"], bbox[2], bbox[3])
    return lookup


def create_sequences(df):
    X, subtrack_ids = [], []
    # Para cada trayectoria
    for subtrack_id, traj in df.groupby("subtrack_id"):
        # Las ordena por frames. Filtra por stride (1 de cada 3 pasos). Descarta la que no cumplen 20 frames.
        traj = traj.sort_values("frame")
        traj_ds = traj.iloc[::STRIDE]
        if len(traj_ds) < WIN:
            continue
        
        # Obtiene posición (X e Y) normalizada, ancho y alto del bbox (normalizado), y pose
        x = traj_ds["x_norm"].values.astype(np.float32)
        y = traj_ds["y_norm"].values.astype(np.float32)
        bw = traj_ds["bbox_w_norm"].values.astype(np.float32)
        bh = traj_ds["bbox_h_norm"].values.astype(np.float32)
        pose = traj_ds[POSE_FEATURES].values.astype(np.float32)

        # Dinámicas: velocidad, aceleración, velocidad lineal, ángulo
        vx = np.diff(x, prepend=x[0]); vy = np.diff(y, prepend=y[0])
        ax = np.diff(vx, prepend=vx[0]); ay = np.diff(vy, prepend=vy[0])
        speed = np.sqrt(vx**2 + vy**2)
        ang = np.arctan2(vy, vx)
        # base -> 11 características base
        base = np.stack([x, y, vx, vy, ax, ay, speed, np.sin(ang), np.cos(ang), bw, bh], axis=1)
        # feats -> concatena las base con la pose -> (L, 11) + (L, 8) = (L, 19)
        feats = np.concatenate([base, pose], axis=1)

        # Ventana deslizante
        L = len(traj_ds)
        for start in range(L - WIN + 1):
            obs_end = start + OBS_LEN
            X.append(feats[start:obs_end])
            subtrack_ids.append(subtrack_id)

    # Devuelve X
    return np.array(X, dtype=np.float32), np.array(subtrack_ids)


if __name__ == "__main__":
    # 1. Genera datos de pose
    lookup = build_pose_lookup()
    # 2. Carga CSV de datos 2D
    df = pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv")
    # 3. Construye matriz con el vector de pose de cada fila del CSV
    pose_cols = np.array([
        lookup.get((r.sequence, r.camera, int(r.frame), int(r.track_id)), [0.0]*8)
        for r in df.itertuples(index=False)
    ], dtype=np.float32)
    # 4. Añade la pose al df
    for j, col in enumerate(POSE_FEATURES):
        df[col] = pose_cols[:, j]
    # 5. Genera la nueva X (entrada) -> datos base + datos pose
    print(f"STRIDE = {STRIDE}")
    X, subtrack_ids = create_sequences(df)
    # 6. Carga la Y (salida)
    Y = np.load(f"{DATA_DIR}/data/Y_ds.npy")
    # 7. Comprueba que tengan el mismo tamaño, que estén alineados
    print(f"\nX_ds_pose: {X.shape}  vs  Y_ds: {Y.shape[0]} muestras")
    if X.shape[0] != Y.shape[0]:
        raise SystemExit("ERROR: no alineado con Y_ds (¿mismo STRIDE que obtain_downsampled_data.py?).")
    print("Alineación correcta")
    # 8. Guarda la nueva entrada con datos de pose
    os.makedirs(f"{DATA_DIR}/data", exist_ok=True)
    np.save(f"{DATA_DIR}/data/X_ds_pose.npy", X)
    print(f"Guardado: X_ds_pose.npy")
