"""
LSTM de movimiento a horizonte largo + data augmentation (reflejo horizontal) +
regularización más fuerte (dropout + recurrent_dropout). Ataca el overfitting
(train_loss baja / val_loss plano) que limita todos los modelos.

Reflejo horizontal -> espejo en X -> Y se mantiene sin cambios.
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
from lstm_common import build_seq2seq, scale_train_val, ade_fde, cv_baseline, flip_x

OBS_LEN, PRED_LEN, N_FEATURES = 8, 12, 11
LSTM_LARGO_ADE, LSTM_LARGO_FDE = 0.028178, 0.051457  # Resultados del modelo LSTM base (con aumento de horizonte)

DROPOUT = 0.3
REC_DROPOUT = 0.2


class LSTMAugHorizonTrainer:

    """
    LSTMAugHorizonTrainer entrena el LSTM (aumento del horizonte) con técnicas de generalización:
    aumento de los datos (reflejo horizontal en train), aumento del dropout y dropout recurrente.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=DROPOUT,
                 rec_dropout=REC_DROPOUT, batch_size=256, epochs=120):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos de train/val (X_tr_aug/Y_tr_aug = train con reflejo), scaler, modelo e historial
        self.X_tr = self.Y_tr = None
        self.X_va = self.Y_va = None
        self.X_tr_aug = self.Y_tr_aug = None
        self.X_tr_s = self.X_va_s = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga el dataset submuestreado y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_ds.npy")
        Y = np.load(f"{self.data_dir}/Y_ds.npy")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"X: {X.shape}")

        train_mask, val_mask = official_masks(subtrack_ids)
        self.X_tr, self.X_va = X[train_mask], X[val_mask]
        self.Y_tr, self.Y_va = Y[train_mask], Y[val_mask]

    def augment(self):
        # Data augmentation: añade las copias reflejadas (solo al train)
        Xf, Yf = flip_x(self.X_tr, self.Y_tr)
        self.X_tr_aug = np.concatenate([self.X_tr, Xf], axis=0)
        self.Y_tr_aug = np.concatenate([self.Y_tr, Yf], axis=0)
        print(f"Train: {self.X_tr.shape[0]} -> {self.X_tr_aug.shape[0]} (con reflejo)   Val: {self.X_va.shape[0]}")

    def scale_features(self):
        # Scaler (train aumentado)
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(
            self.X_tr_aug, self.X_va, self.n_features)
        # Guarda el scaler
        joblib.dump(self.scaler, f"{self.data_dir}/ds_aug_scaler.pkl")

    def build_model(self):
        # Define el modelo -> igual que LSTM base (aumento de horizonte) pero con recurrent_dropout activado
        # y aumento del dropout
        self.model = build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                                   units=self.lstm_units, dropout=self.dropout,
                                   rec_dropout=self.rec_dropout, optimizer="adam",
                                   metrics=["mae"], named=True)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)
        # Define los callbacks
        cbs = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_2d/best/ds_aug_best.keras",
                            monitor="val_loss", save_best_only=True),
        ]
        # Entrena el modelo, midiendo el tiempo de entrenamiento
        t0 = time.time()
        self.history = self.model.fit(self.X_tr_s, self.Y_tr_aug,
                                      validation_data=(self.X_va_s, self.Y_va),
                                      epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_ds_lstm_aug.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
        plt.title("Curva de entrenamiento — LSTM + aug + reg")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_ds_lstm_aug.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_ds_lstm_aug.png, history/hist_ds_lstm_aug.npy")

    def evaluate(self):
        # Predice con el modelo LSTM con aumento de horizonte y técnicas de generalización
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Obtiene métricas: LSTM aug vs CV baseline
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, _ = cv_baseline(self.X_va, self.Y_va, self.pred_len)

        # Comparación: LSTM aug vs LSTM base vs CV base
        print("\n===== HORIZONTE LARGO: LSTM+aug+reg vs LSTM vs CV =====")
        print(f"{'Modelo':<24}  {'ADE':>10}  {'FDE':>10}  {'Mejora vs CV':>14}")
        print(f"{'-'*64}")
        print(f"{'CV baseline':<24}  {ade_cv:>10.6f}  {'':>10}  {'(ref)':>14}")
        print(f"{'LSTM (solo mov)':<24}  {LSTM_LARGO_ADE:>10.6f}  {LSTM_LARGO_FDE:>10.6f}  {(1-LSTM_LARGO_ADE/ade_cv)*100:>13.1f}%")
        print(f"{'LSTM + aug + reg':<24}  {ade:>10.6f}  {fde:>10.6f}  {(1-ade/ade_cv)*100:>13.1f}%")
        print(f"\nMejora aug+reg sobre LSTM-solo largo: {(1-ade/LSTM_LARGO_ADE)*100:+.1f}% ADE")

    def run(self):
        # 1. Carga de datos
        self.load_data()
        # 2. Aumento de los datos
        self.augment()
        # 3. Scaler
        self.scale_features()
        # 4. Definición del modelo
        self.build_model()
        # 5. Entrenamiento
        self.train()
        # 6. Guardado de las salidas
        self.save_outputs()
        # 7. Evaluación
        self.evaluate()


if __name__ == "__main__":

    trainer = LSTMAugHorizonTrainer()
    trainer.run()
