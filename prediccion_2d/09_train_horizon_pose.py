"""
LSTM + POSE en horizonte largo (submuestreado). Mismo modelo y split por seq_cam.
Compara contra el LSTM-solo a horizonte largo (ADE 0.025215 / FDE 0.043597) y el CV.
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

OBS_LEN, PRED_LEN, N_FEATURES = 8, 12, 19

# Resultados de LSTM base (con aumento de horizonte)
LSTM_LARGO_ADE, LSTM_LARGO_FDE = 0.028178, 0.051457


class LSTMPoseHorizonTrainer:

    """
    LSTMPoseHorizonTrainer entrena el LSTM con features de pose (19 características) con el
    aumento del horizonte. Misma arquitectura y split que el LSTM base.
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
        # Carga el dataset con pose (19 features) + Y submuestreada y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_ds_pose.npy")
        Y = np.load(f"{self.data_dir}/Y_ds.npy")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"X: {X.shape}  Y: {Y.shape}")

        train_mask, val_mask = official_masks(subtrack_ids)
        self.X_train, self.X_val = X[train_mask], X[val_mask]
        self.Y_train, self.Y_val = Y[train_mask], Y[val_mask]
        print(f"Train: {self.X_train.shape[0]}  Val: {self.X_val.shape[0]}")

    def scale_features(self):
        # Estandariza las 19 features y guarda el scaler de pose
        self.X_train_s, self.X_val_s, self.scaler = scale_train_val(
            self.X_train, self.X_val, self.n_features)
        joblib.dump(self.scaler, f"{self.data_dir}/downsampled_pose_scaler.pkl")

    def build_model(self):
        # Misma arquitectura que el LSTM base, pero con 19 features de entrada
        self.model = build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                                   units=self.lstm_units, dropout=self.dropout,
                                   optimizer="adam", metrics=["mae"], named=True)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)
        # Define los callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_2d/best/downsampled_pose_lstm_best.keras",
                            monitor="val_loss", save_best_only=True),
        ]
        # Entrenamiento, midiendo el tiempo.
        t0 = time.time()
        self.history = self.model.fit(self.X_train_s, self.Y_train,
                                      validation_data=(self.X_val_s, self.Y_val),
                                      epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_ds_pose.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
        plt.title("Curva de entrenamiento — LSTM + pose")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_ds_pose.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_ds_pose.png, history/hist_ds_pose.npy")

    def evaluate(self):
        # Predice con el modelo LSTM con pose
        pred = self.model.predict(self.X_val_s, batch_size=512, verbose=0)
        # Cálculo de métricas -> LSTM con pose vs CV baseline
        ade, fde = ade_fde(pred, self.X_val, self.Y_val)
        ade_cv, _ = cv_baseline(self.X_val, self.Y_val, self.pred_len)

        # Comparación: LSTM+pose vs LSTM-solo vs CV baseline
        print("\n===== HORIZONTE LARGO: LSTM+POSE vs LSTM vs CV =====")
        print(f"{'Modelo':<22}  {'ADE':>10}  {'FDE':>10}  {'Mejora ADE vs CV':>16}")
        print(f"{'-'*64}")
        print(f"{'CV baseline':<22}  {ade_cv:>10.6f}  {'':>10}  {'(referencia)':>16}")
        print(f"{'LSTM (solo mov)':<22}  {LSTM_LARGO_ADE:>10.6f}  {LSTM_LARGO_FDE:>10.6f}  {(1-LSTM_LARGO_ADE/ade_cv)*100:>15.1f}%")
        print(f"{'LSTM + pose':<22}  {ade:>10.6f}  {fde:>10.6f}  {(1-ade/ade_cv)*100:>15.1f}%")
        print(f"\nMejora pose sobre LSTM-solo largo: {(1-ade/LSTM_LARGO_ADE)*100:+.1f}% ADE")

    def run(self):
        # 1. Carga de datos
        self.load_data()
        # 2. Scaler
        self.scale_features()
        # 3. Define el modelo
        self.build_model()
        # 4. Entrenamiento
        self.train()
        # 5. Guarda las salidas
        self.save_outputs()
        # 6. Evaluación
        self.evaluate()


if __name__ == "__main__":

    trainer = LSTMPoseHorizonTrainer()
    trainer.run()
