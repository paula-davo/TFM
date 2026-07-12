"""
LSTM sobre el dataset con el aumento del horizonte. Comparación con CV.
"""

import os
import sys
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val, ade_fde, cv_baseline

OBS_LEN = 8
PRED_LEN = 12
N_FEATURES = 11


class LSTMHorizonTrainer:

    """
    LSTMHorizonTrainer entrena el LSTM sobre el dataset con aumento de horizonte.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=0.2,
                 batch_size=256, epochs=100):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos, scaler, modelo e historial
        self.X_train = self.Y_train = None
        self.X_val = self.Y_val = None
        self.X_train_s = self.X_val_s = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga el dataset con aumento del horizonte y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_ds.npy")
        Y = np.load(f"{self.data_dir}/Y_ds.npy")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"X: {X.shape}  Y: {Y.shape}")

        # Se obtiene entrada y salida de train y valid
        train_mask, val_mask = official_masks(subtrack_ids)
        self.X_train, self.X_val = X[train_mask], X[val_mask]
        self.Y_train, self.Y_val = Y[train_mask], Y[val_mask]
        print(f"Train: {self.X_train.shape[0]}  Val: {self.X_val.shape[0]}")

    def scale_features(self):
        # Scaler -> guarda el scaler de aumento de horizonte.
        self.X_train_s, self.X_val_s, self.scaler = scale_train_val(
            self.X_train, self.X_val, self.n_features)
        joblib.dump(self.scaler, f"{self.data_dir}/downsampled_scaler.pkl")

    def build_model(self):
        # Arquitectura igual que el modelo base pero cambio de los datos
        self.model = build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                                   units=self.lstm_units, dropout=self.dropout,
                                   optimizer="adam", metrics=["mae"], named=True)
        # Summary
        self.model.summary()
        # Devuelve el modelo
        return self.model

    def train(self):
        # Entrena con EarlyStopping, ReduceLROnPlateau (patience=5) y ModelCheckpoint
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)
        # Callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_2d/best/downsampled_lstm_best.keras", monitor="val_loss", save_best_only=True),
        ]
        # Entrena el modelo
        t0 = time.time()
        self.history = self.model.fit(self.X_train_s, self.Y_train, validation_data=(self.X_val_s, self.Y_val), epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])

        # Cálculo del tiempo de inferencia
        _ = self.model.predict(self.X_val_s[:256], verbose=0)
        t0 = time.time()
        _ = self.model.predict(self.X_val_s, batch_size=512, verbose=0)
        t_inf_ms = (time.time() - t0) * 1000 / len(self.X_val_s)
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")
        print(f"Inferencia: {t_inf_ms:.3f} ms/muestra")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_ds_lstm.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
        plt.title("Curva de entrenamiento — LSTM (horizonte largo)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_ds_lstm.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_ds_lstm.png, history/hist_ds_lstm.npy")

    def evaluate(self):
        # Evaluación de: aumento horizonte vs CV baselina
        # Predice desplazamiento relativo
        pred_delta = self.model.predict(self.X_val_s, batch_size=512, verbose=0)
        # Cálculo de métricas aumento horizonte vs CV baseline
        ade, fde = ade_fde(pred_delta, self.X_val, self.Y_val)
        ade_cv, fde_cv = cv_baseline(self.X_val, self.Y_val, self.pred_len)

        print("\n===== HORIZONTE LARGO (submuestreado) =====")
        print(f"{'Modelo':<20}  {'ADE':>10}  {'FDE':>10}  {'Mejora ADE':>12}")
        print(f"{'-'*56}")
        print(f"{'CV baseline':<20}  {ade_cv:>10.6f}  {fde_cv:>10.6f}  {'(referencia)':>12}")
        print(f"{'LSTM':<20}  {ade:>10.6f}  {fde:>10.6f}  {(1-ade/ade_cv)*100:>11.1f}%")
        print("\nRecordatorio: estos números NO son comparables con el horizonte corto")
        print("(ADE 0.0129). Aquí el CV se degrada y el LSTM debería ganarle por mucho más.")

    def run(self):
        # Flujo completo: carga -> escalado -> modelo -> entrenamiento -> guardado -> evaluación
        # 1. Carga de datos
        self.load_data()
        # 2. Scaler
        self.scale_features()
        # 3. Modelo
        self.build_model()
        # 4. Entrenamiento
        self.train()
        # 5. Guarda salidas
        self.save_outputs()
        # 6. Evalúa
        self.evaluate()


if __name__ == "__main__":

    trainer = LSTMHorizonTrainer()
    trainer.run()
