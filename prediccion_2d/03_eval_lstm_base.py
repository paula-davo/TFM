import sys
import numpy as np
import joblib
from tensorflow.keras.models import load_model

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import ade_fde, cv_baseline

OBS_LEN = 8
PRED_LEN = 12
N_FEATURES = 11

# Script para evaluación del modelo LSTM base.

# Carga el modelo
model = load_model(f"{DATA_DIR}/prediccion_2d/best/lstm_best.keras")
# Carga el scaler
scaler = joblib.load(f"{DATA_DIR}/feature_scaler.pkl")

# Carga las entradas y salidas
X = np.load(f"{DATA_DIR}/X_train.npy")
Y = np.load(f"{DATA_DIR}/Y_train.npy")
subtrack_ids = np.load(f"{DATA_DIR}/subtrack_ids.npy", allow_pickle=True)

_, val_mask = official_masks(subtrack_ids)

# Carga las entradas y salidas de validación
X_val = X[val_mask]
Y_val = Y[val_mask]

# Estandariza con el scaler guardado durante entrenamiento
X_val_scaled = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(X_val.shape[0], OBS_LEN, N_FEATURES)

# Predicción de desplazamientos relativos
pred_delta = model.predict(X_val_scaled, verbose=1)

# Métricas: LSTM vs CV
ade, fde = ade_fde(pred_delta, X_val, Y_val)
ade_cv, fde_cv = cv_baseline(X_val, Y_val, PRED_LEN)

print("\n===== RESULTADOS =====")
print(f"{'Métrica':<8}  {'LSTM':>10}  {'CV baseline':>12}  {'mejora':>8}")
print(f"{'-'*44}")
print(f"{'ADE':<8}  {ade:>10.6f}  {ade_cv:>12.6f}  {(1 - ade/ade_cv)*100:>7.1f}%")
print(f"{'FDE':<8}  {fde:>10.6f}  {fde_cv:>12.6f}  {(1 - fde/fde_cv)*100:>7.1f}%")
