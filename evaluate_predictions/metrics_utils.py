"""
Código común para la evaluación de las predicciones.
"""
import numpy as np


def classic_metrics(disp_true, disp_pred, last_pos):
    # Cálculo de las métricas de evaluación
    # Error vectorial entre trayectoria predicha y real
    err = disp_pred - disp_true
    flat = err.reshape(-1)
    # MAE, MSE, RMSE
    mae = np.abs(flat).mean()
    mse = (flat ** 2).mean()
    rmse = np.sqrt(mse)

    # MASE
    mae_naive = np.abs(disp_true.reshape(-1)).mean() + 1e-12
    mase = mae / mae_naive

    # MAPE sobre posiciones absolutas
    abs_true = disp_true + last_pos[:, None, :]
    abs_pred = disp_pred + last_pos[:, None, :]
    mape = np.mean(np.abs((abs_true - abs_pred) / (np.abs(abs_true) + 1e-6))) * 100

    # ADE y FDE
    ade = np.mean(np.linalg.norm(err, axis=2))
    fde = np.mean(np.linalg.norm(err[:, -1, :], axis=1))
    return dict(MAE=mae, MSE=mse, RMSE=rmse, MASE=mase, MAPE=mape, ADE=ade, FDE=fde)


def dense_band_pred(preds, radius):
    # Elegir la trayectoria representativa entre las K predicciones
    N, K_, T, _ = preds.shape
    # Punto final de cada predicción (N, K, 2)
    end = preds[:, :, -1, :]
    # Distancia entre todos los puntos finales (N, K, K)
    d = np.linalg.norm(end[:, :, None, :] - end[:, None, :, :], axis=3)
    # Predicciones dentro del radio de cada predicción (N, K, K)
    within = d <= radius
    # Predicción con más vecinas (moda) -> (N,)
    mode = within.sum(axis=2).argmax(axis=1)
    # Promedia las predicciones de esa franja densa
    rep = np.empty((N, T, 2), dtype=np.float32)
    for i in range(N):
        rep[i] = preds[i, within[i, mode[i]]].mean(axis=0)
    return rep


def angle(a, b, degrees=True):
    # Ángulo entre dos vectores, o entre dos lotes de vectores fila a fila (arccos del coseno).
    da = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    db = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    ang = np.arccos(np.clip(np.sum(da * db, axis=-1), -1, 1))
    return np.degrees(ang) if degrees else ang


def build_meta(df, obs_len, stride, win):
    seqs, frs, tids = [], [], []
    # Agrupa los datos por subtrack_id
    for _, traj in df.groupby("subtrack_id"):
        # Para cada trayectoria, las ordena por frame y les aplica el submuestreo
        traj = traj.sort_values("frame")
        traj_ds = traj.iloc[::stride]
        # Descarta las que no son válidas
        if len(traj_ds) < win:
            continue
        # Extrae el array de frames, secuencia y track ID
        frames = traj_ds["frame"].values.astype(int)
        seq = traj_ds["sequence"].iloc[0]
        tid = int(traj_ds["track_id"].iloc[0])
        # Para cada ventana deslizante, guarda la secuencia, el frame actual y el track ID
        for start in range(len(traj_ds) - win + 1):
            seqs.append(seq); frs.append(int(frames[start + obs_len - 1])); tids.append(tid)
    return np.array(seqs), np.array(frs, dtype=int), np.array(tids, dtype=int)
