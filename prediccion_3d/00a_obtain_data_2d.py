"""
Crea el dataset 2D.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

OBS_LEN = 8
PRED_LEN = 12
STRIDE = 3          # 1 de cada 3 frames
WIN = OBS_LEN + PRED_LEN


def create_sequences(df):
    X, Y, subtrack_ids = [], [], []
    total, valid = 0, 0

    # Para cada trayectoria
    for subtrack_id, traj in df.groupby("subtrack_id"):
        total += 1
        # Ordena por frames
        traj = traj.sort_values("frame")

        # Submuestreo temporal -> stride=3 -> de cada 3 frames, seleccinoa 1
        traj_ds = traj.iloc[::STRIDE]
        # Descarta las que tienen menos frames que el mínimo
        if len(traj_ds) < WIN:
            continue
        valid += 1

        # Posición (X e Y) normalizada
        x = traj_ds["x_norm"].values.astype(np.float32)
        y = traj_ds["y_norm"].values.astype(np.float32)
        # Weight y Height del bbox normalizado
        bw = traj_ds["bbox_w_norm"].values.astype(np.float32)
        bh = traj_ds["bbox_h_norm"].values.astype(np.float32)

        # Velocidad y aceleración (X e Y)
        vx = np.diff(x, prepend=x[0])
        vy = np.diff(y, prepend=y[0])
        ax = np.diff(vx, prepend=vx[0])
        ay = np.diff(vy, prepend=vy[0])
        # Velocidad lineal.
        speed = np.sqrt(vx**2 + vy**2)
        # Dirección (seno y coseno)
        ang = np.arctan2(vy, vx)
        sin_dir, cos_dir = np.sin(ang), np.cos(ang)

        # Guarda las características
        feats = np.stack([x, y, vx, vy, ax, ay, speed, sin_dir, cos_dir, bw, bh], axis=1)
        positions = np.stack([x, y], axis=1)

        # Desliza la ventana por cada uno de los 20 frames (WIN)
        L = len(traj_ds)
        for start in range(L - WIN + 1):
            # 8 de entrada y 12 de salida
            obs_end = start + OBS_LEN
            pred_end = obs_end + PRED_LEN
            # Posición del último frame observado
            last = positions[obs_end - 1]
            # Guarda entrada y salida
            X.append(feats[start:obs_end])
            Y.append(positions[obs_end:pred_end] - last)
            subtrack_ids.append(subtrack_id)

    # Devuelve el conjunto de entrada, de salida, los IDs de subtrack, el 
    # total de trayectorias y el total de trayectorias válidas con este nuevo stride.
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            np.array(subtrack_ids), total, valid)


if __name__ == "__main__":

    print(f"STRIDE = {STRIDE}  (1 de cada {STRIDE} frames)")
    # Carga el CSV de trayectorias
    df = pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv")

    # Crea el conjunto de entrada y salida con el nuevo stride (3)
    X, Y, subtrack_ids, total, valid = create_sequences(df)

    print(f"\n========== DATASET SUBMUESTREADO ==========")
    print(f"Subtrayectorias: {valid}/{total} con >= {WIN*STRIDE} frames originales")
    print(f"X: {X.shape}   Y: {Y.shape}")

    if len(X) == 0:
        raise SystemExit("Sin muestras: stride demasiado grande para las trayectorias.")

    print(f"\nY (desplazamientos) — std: {Y.std():.4f}")

    # Guarda ambos conjuntos asi como los subtrack_ids.
    os.makedirs(f"{DATA_DIR}/data", exist_ok=True)
    np.save(f"{DATA_DIR}/data/X_ds.npy", X)
    np.save(f"{DATA_DIR}/data/Y_ds.npy", Y)
    np.save(f"{DATA_DIR}/data/subtrack_ids_ds.npy", subtrack_ids)
    print(f"\nGuardado: X_ds.npy, Y_ds.npy, subtrack_ids_ds.npy")
