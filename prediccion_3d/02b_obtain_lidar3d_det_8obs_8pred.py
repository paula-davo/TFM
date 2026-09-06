"""
Obtención de datos de entorno lidar y de las detecciones 3D -> 8 frames observados, 8 predichos.

Se reprocesa a partir de X3d_det_motion.npy / Y3d_det.npy (mismas ventanas):
la observación es idéntica a la del 8/12, y la salida se recorta
a los 8 primeros pasos de predicción.
"""

import numpy as np

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OBS_LEN, PRED_LEN = 8, 8

if __name__ == "__main__":
    X = np.load(f"{DATA_DIR}/X3d_det_motion.npy")   # (N, 8, 9) observación (detecciones)
    Y = np.load(f"{DATA_DIR}/Y3d_det.npy")          # (N, 12, 2) desplazamiento GT

    Xb = X                                          # misma observación
    Yb = Y[:, :PRED_LEN, :]                         # 8 primeros pasos de predicción

    print(f"X: {Xb.shape}  Y: {Yb.shape}")
    np.save(f"{DATA_DIR}/X3d_det_8obs_8pred.npy", Xb)
    np.save(f"{DATA_DIR}/Y3d_det_8obs_8pred.npy", Yb)
    print("Guardado: X3d_det_8obs_8pred.npy, Y3d_det_8obs_8pred.npy")
