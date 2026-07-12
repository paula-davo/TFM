"""
Visualización del modelo de múltiples predicciones (ds_multimodal_best.keras): las K=20 hipótesis
de futuro (gris) cubriendo distintos modos, la mejor (rojo) y el futuro real (verde).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OUT = f"{DATA_DIR}/prediccion_2d/figuras"; os.makedirs(OUT, exist_ok=True)
NF = 11 # Número de características

sys.path.insert(0, DATA_DIR)
from lstm_common import load_val_and_scaler

# Carga de datos con el split oficial y aplicación de scaler
Xva, Yva, Xs = load_val_and_scaler(DATA_DIR, NF)
# Carga el modelo de múltiples trayectorias 2D
model = load_model(f"{DATA_DIR}/prediccion_2d/best/ds_multimodal_best.keras", compile=False)
# Predicción con el modelo de múltiples trayectorias
predK = model.predict(Xs, batch_size=256, verbose=0)   # (N,K,12,2)
K = predK.shape[1]

# 8 frames observados y último frame observado
obs = Xva[:, :, 0:2]
last = obs[:, -1]
# Posiciones reales y predichas
gt = Yva + last[:, None, :]
prK = predK + last[:, None, None, :]                   # (N,K,12,2)

# Mide la dispersión del punto final en las hipótesis, cuánto divergen
spread = prK[:, :, -1, :].std(axis=1).sum(axis=1)
# Movimiento del peatón
move = np.linalg.norm(gt[:, -1] - last, axis=1)
# Filtra los que se mueven poco
cand = np.where(move > np.percentile(move, 50))[0]
# Se queda con los 9 con mayor dispersión
cand = cand[np.argsort(-spread[cand])][:9]

# Muestra resultados
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for ax, i in zip(axes.ravel(), cand):
    # Muestra todas las hipótesis
    for k in range(K):
        ax.plot(prK[i, k, :, 0], prK[i, k, :, 1], "-", color="0.75", lw=0.8)
    # Calcula su ADE
    ade_k = np.linalg.norm(prK[i] - gt[i][None], axis=2).mean(axis=1)
    kb = ade_k.argmin()
    ax.plot(obs[i, :, 0], obs[i, :, 1], "o-", color="tab:blue", label="Observado")
    ax.plot(gt[i, :, 0], gt[i, :, 1], "o-", color="tab:green", label="Real")
    ax.plot(prK[i, kb, :, 0], prK[i, kb, :, 1], "x--", color="tab:red", label="Mejor hipótesis")
    ax.invert_yaxis(); ax.set_aspect("equal", "datalim")
    ax.set_xticks([]); ax.set_yticks([])
axes.ravel()[0].legend(fontsize=8, loc="best")
fig.suptitle(f"Modelo multimodal — {K} hipótesis (gris), mejor (rojo), real (verde)", fontsize=13)
fig.tight_layout()
# Guarda la figura
fig.savefig(f"{OUT}/viz_multimodal.png", dpi=130)
print(f"Guardado: {OUT}/viz_multimodal.png")
