"""
Obtención de datos de etiquetas 3D del lidar -> 8 frames de observación, 8 de predicción.

Se reprocesa a partir de X3d_motion.npy / Y3d.npy (mismas ventanas):
la observación es igual, y la salida se recorta a los 8 primeros pasos
de predicción.
"""

import numpy as np

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OBS_LEN, PRED_LEN = 8, 8

if __name__ == "__main__":
    X = np.load(f"{DATA_DIR}/X3d_motion.npy")   # (N, 8, 9) observación (etiquetas),
    Y = np.load(f"{DATA_DIR}/Y3d.npy")          # (N, 12, 2) desplazamiento futuro

    Xb = X                                      # misma observación
    Yb = Y[:, :PRED_LEN, :]                     # 8 primeros pasos de predicción

    print(f"X: {Xb.shape}  Y: {Yb.shape}")
    np.save(f"{DATA_DIR}/X3d_motion_8obs_8pred.npy", Xb)
    np.save(f"{DATA_DIR}/Y3d_8obs_8pred.npy", Yb)
    print("Guardado: X3d_motion_8obs_8pred.npy, Y3d_8obs_8pred.npy")
