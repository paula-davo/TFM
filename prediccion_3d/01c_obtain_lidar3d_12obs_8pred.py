"""
Obtención de datos de etiquetas 3D del lidar -> 12 frames de observación, 8 de predicción.

Se reprocesa a partir de X3d_motion.npy / Y3d.npy (mismas ventanas):
1) Reconstruye las 20 posiciones GT de cada ventana
2) Recalcula las 9 características sobre 12 frames de observación
3) La salida es el desplazamiento de los 8 frames de predicción a continuación de la observación.
"""

import numpy as np

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OBS_LEN, PRED_LEN = 12, 8

if __name__ == "__main__":
    X = np.load(f"{DATA_DIR}/X3d_motion.npy")   # (N, 8, 9)
    Y = np.load(f"{DATA_DIR}/Y3d.npy")          # (N, 12, 2)
    N = X.shape[0]

    # Reconstruye las 20 posiciones GT
    pos = np.zeros((N, 20, 2), dtype=np.float32)
    pos[:, :8, :] = X[:, :, :2]
    pos[:, 8:, :] = X[:, 7:8, :2] + Y

    # 12 frames de observación -> 9 características
    obs = pos[:, :OBS_LEN, :]
    cx, cy = obs[:, :, 0], obs[:, :, 1]
    vx = np.diff(cx, axis=1, prepend=cx[:, :1])
    vy = np.diff(cy, axis=1, prepend=cy[:, :1])
    ax = np.diff(vx, axis=1, prepend=vx[:, :1])
    ay = np.diff(vy, axis=1, prepend=vy[:, :1])
    speed = np.sqrt(vx**2 + vy**2)
    ang = np.arctan2(vy, vx)
    Xc = np.stack([cx, cy, vx, vy, ax, ay, speed, np.sin(ang), np.cos(ang)], axis=2).astype(np.float32)

    # 8 frames de predicción -> desplazamiento relativo al último frame observado
    Yc = (pos[:, OBS_LEN:, :] - pos[:, OBS_LEN - 1:OBS_LEN, :]).astype(np.float32)

    print(f"X: {Xc.shape}  Y: {Yc.shape}")
    print(f"Y (metros) — desplazamiento medio: {np.linalg.norm(Yc, axis=2).mean():.3f} m")
    np.save(f"{DATA_DIR}/X3d_motion_12obs_8pred.npy", Xc)
    np.save(f"{DATA_DIR}/Y3d_12obs_8pred.npy", Yc)
    print("Guardado: X3d_motion_12obs_8pred.npy, Y3d_12obs_8pred.npy")
