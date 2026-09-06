"""
Visualización de ejemplos generados con el modelo LSTM con detecciones 3D.
Observación (azul), real (verde), predicción (rojo).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OUTPUT_DIR = f"{DATA_DIR}/evaluate_predictions/figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OBS, PRED, NF = 8, 8, 9
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from metrics_utils import angle

if __name__ == "__main__":
    # Carga los datos de detecciones 3D
    X = np.load(f"{DATA_DIR}/data/X3d_det_motion.npy")         # (N,8,9): cx,cy,vx,vy,ax,ay,speed,sin,cos
    Y = np.load(f"{DATA_DIR}/data/Y3d_det.npy")[:, :PRED, :]   # (N,8,2) metros (8/8)
    # Partición oficial
    ids = np.load(f"{DATA_DIR}/data/subtrack_ids_ds.npy", allow_pickle=True)
    sck = np.array([s.rsplit("_", 2)[0] for s in ids])
    tm, vm = official_masks(sck)

    # Ajusta scaler y transforma validación
    sc = StandardScaler().fit(X[tm].reshape(-1, NF))
    Xva, Yva = X[vm], Y[vm]
    Xs = sc.transform(Xva.reshape(-1, NF)).reshape(Xva.shape)
    # Carga el modelo LSTM con detecciones 3D (04e)
    model = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_det_mae_best.keras", compile=False)
    # Genera predicciones
    pred = model.predict(Xs, batch_size=512, verbose=0)

    # Trayectoria observada, última posición observada, futuro real y futuro predicho
    obs = Xva[:, :, 0:2]
    last = obs[:, -1]
    gt = Yva + last[:, None, :]
    pr = pred + last[:, None, :]

    # Busca los mejores ejemplos
    # Vector de dirección entre primera y última posición observada y futura
    v_obs = obs[:, -1] - obs[:, 0]
    v_fut = gt[:, -1] - last
    # Giro del peatón
    turn = angle(v_obs, v_fut, degrees=False)
    # Movimiento futuro del peatón
    move = np.linalg.norm(v_fut, axis=1)
    # Filtra los que se mueven y coge los 9 que más giran
    cand = np.where(move > np.percentile(move, 60))[0]
    cand = cand[np.argsort(-turn[cand])][:9]

    # Dibuja figura: 9 subgráficos - 9 ejemplos.
    # Muestra trayectoria observada, real y predicha.
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, i in zip(axes.ravel(), cand):
        ax.plot(obs[i, :, 0], obs[i, :, 1], "o-", color="tab:blue", label="Observado")
        ax.plot(gt[i, :, 0], gt[i, :, 1], "o-", color="tab:green", label="Real")
        ax.plot(pr[i, :, 0], pr[i, :, 1], "x--", color="tab:red", label="Predicho")
        ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.suptitle("Modelo LiDAR 3D (detecciones) — vista cenital en METROS", fontsize=13)
    fig.tight_layout()
    # Guarda la figura.
    fig.savefig(f"{OUTPUT_DIR}/viz_lidar3d.png", dpi=130)
    print(f"Guardado: {OUTPUT_DIR}/viz_lidar3d.png")
