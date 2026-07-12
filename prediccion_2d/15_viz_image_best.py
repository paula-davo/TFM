"""
Visualización del mejor modelo 2D determinista (LSTM + aug+reg+ruido, ds_aug2_best.keras).
Rejilla de ejemplos de validación: observación (azul), futuro real (verde), predicción (rojo).
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
# Carga el modelo determinista LSTM con aumento de horizonte de predicción y técnicas de generalización
model = load_model(f"{DATA_DIR}/prediccion_2d/best/ds_aug2_best.keras", compile=False)
# Predicción con el modelo determinista
pred = model.predict(Xs, batch_size=512, verbose=0)

# 8 frames observados y último frame observado
obs = Xva[:, :, 0:2]                       # (N,8,2)
last = obs[:, -1]                          # (N,2)
# Posiciones reales y predichas
gt = Yva + last[:, None, :]                # (N,12,2)
pr = pred + last[:, None, :]

# Vector de observación y de futuro
v_obs = obs[:, -1] - obs[:, 0]
v_fut = gt[:, -1] - last

def ang(a, b):
    na = np.linalg.norm(a, axis=1) + 1e-9; nb = np.linalg.norm(b, axis=1) + 1e-9
    cos = np.sum(a*b, axis=1)/(na*nb)
    return np.arccos(np.clip(cos, -1, 1))

# Ángulo entre las dos direcciones
turn = ang(v_obs, v_fut)
# Movimiento en el futuro
move = np.linalg.norm(v_fut, axis=1)
# Descarta los que se mueven poco
cand = np.where(move > np.percentile(move, 60))[0]
# Se queda con los 9 que más giran
cand = cand[np.argsort(-turn[cand])][:9]

# Muestra
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for ax, i in zip(axes.ravel(), cand):
    ax.plot(obs[i, :, 0], obs[i, :, 1], "o-", color="tab:blue", label="Observado")
    ax.plot([obs[i, -1, 0], gt[i, 0, 0]], [obs[i, -1, 1], gt[i, 0, 1]], "--", color="0.6")
    ax.plot(gt[i, :, 0], gt[i, :, 1], "o-", color="tab:green", label="Real")
    ax.plot(pr[i, :, 0], pr[i, :, 1], "x--", color="tab:red", label="Predicho")
    ax.invert_yaxis(); ax.set_aspect("equal", "datalim")
    ax.set_xticks([]); ax.set_yticks([])
axes.ravel()[0].legend(fontsize=8, loc="best")
fig.suptitle("Mejor modelo de imagen (LSTM+aug+reg+ruido) — obs/real/predicho (norm.)", fontsize=13)
fig.tight_layout()
# Guarda figura
fig.savefig(f"{OUT}/viz_imagen_best.png", dpi=130)
print(f"Guardado: {OUT}/viz_imagen_best.png")
