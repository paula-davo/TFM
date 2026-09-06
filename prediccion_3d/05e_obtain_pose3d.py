"""
Obtención del dataset de pose: 9 características de detecciones 3D +
8 características de pose corporal (de mmpose).

Las 8 características de pose provienen de la cámara (keypoints). 
Como X3d_det_motion y X_ds_pose están alineados por el mismo subtrack_ids_ds, basta con concatenarlos:
"""

import os
import numpy as np

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

if __name__ == "__main__":
    # 9 features de movimiento 3D (detecciones del sensor, metros)
    X3d = np.load(f"{DATA_DIR}/data/X3d_det_motion.npy")      # (N, 8, 9)
    # Dataset 2D con: 11 base 2D + 8 de pose -> las 8 de pose son las columnas 11:19
    Xpose2d = np.load(f"{DATA_DIR}/data/X_ds_pose.npy")       # (N, 8, 19)
    pose = Xpose2d[:, :, 11:19]                          # (N, 8, 8) características de pose

    # Comprobación de alineación (ambos indexados por subtrack_ids_ds)
    if X3d.shape[0] != pose.shape[0]:
        raise SystemExit(f"ERROR: no alineado ({X3d.shape[0]} vs {pose.shape[0]}).")

    # Concatena movimiento 3D + pose -> 17 características
    X = np.concatenate([X3d, pose], axis=2).astype(np.float32)   # (N, 8, 17)
    print(f"X_3d_pose: {X.shape}  (9 movimiento 3D + 8 pose)")

    os.makedirs(f"{DATA_DIR}/data", exist_ok=True)
    np.save(f"{DATA_DIR}/data/X_3d_pose_ds.npy", X)
    print("Guardado: X_3d_pose_ds.npy")
