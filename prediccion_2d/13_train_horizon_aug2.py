"""
LSTM + reflejo + regularización + ruido gaussiano en el input.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val, ade_fde, cv_baseline, flip_x

OBS_LEN, PRED_LEN, N_FEATURES = 8, 12, 11
DROPOUT, REC_DROPOUT, NOISE = 0.3, 0.2, 0.05
REF_ADE, REF_FDE = 0.027315, 0.050732  # Resultados de LSTM con técnicas de generalización (versión 1)


class LSTMAug2HorizonTrainer:

    """
    LSTMAug2HorizonTrainer entrena modelo LSTM (aumento del horizonte) con técnicas de generalización:
    aumento de datos (reflejo horizontal) + aumento de dropout + recurrent dropout + ruido gaussiano en la entrada.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=DROPOUT,
                 rec_dropout=REC_DROPOUT, noise=NOISE, batch_size=256, epochs=140):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.noise = noise
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos (X_tr/Y_tr = train con reflejo), scaler, modelo e historial
        self.X_tr = self.Y_tr = None
        self.X_va = self.Y_va = None
        self.X_tr_s = self.X_va_s = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga el dataset submuestreado y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_ds.npy")
        Y = np.load(f"{self.data_dir}/Y_ds.npy")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)

        tm, vm = official_masks(subtrack_ids)
        self.X_tr, self.X_va, self.Y_tr, self.Y_va = X[tm], X[vm], Y[tm], Y[vm]

    def augment(self):
        # Data augmentation: añade las copias reflejadas solo al train
        Xf, Yf = flip_x(self.X_tr, self.Y_tr)
        self.X_tr = np.concatenate([self.X_tr, Xf]); self.Y_tr = np.concatenate([self.Y_tr, Yf])
        print(f"Train(aug): {self.X_tr.shape[0]}  Val: {self.X_va.shape[0]}")

    def scale_features(self):
        # Scaler
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(
            self.X_tr, self.X_va, self.n_features)

    def build_model(self):
        # Define el modelo -> igual que LSTM base pero con recurrent dropout y ruido gaussiano
        self.model = build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                                   units=self.lstm_units, dropout=self.dropout,
                                   rec_dropout=self.rec_dropout, noise=self.noise,
                                   optimizer="adam", metrics=["mae"], named=True)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)
        # Define los callbacks
        cbs = [
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_2d/best/ds_aug2_best.keras",
                            monitor="val_loss", save_best_only=True),
        ]
        # Entrena el modelo, midiendo el tiempo de entrenamiento
        t0 = time.time()
        self.history = self.model.fit(self.X_tr_s, self.Y_tr,
                                      validation_data=(self.X_va_s, self.Y_va),
                                      epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_ds_lstm_aug2.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
        plt.title("Curva de entrenamiento — LSTM + aug + reg + ruido")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_ds_lstm_aug2.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_ds_lstm_aug2.png, history/hist_ds_lstm_aug2.npy")

    def evaluate(self):
        # Predicción con el modelo LSTM + técnicas de generalización (versión 2)
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Evaluación: LSTM con generalización (versión 2) vs CV baseline
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, _ = cv_baseline(self.X_va, self.Y_va, self.pred_len)

        # Comparación: LSTM con generalización (versión 2) vs LSTM con generalización (versión 1) vs CV base
        print("\n===== SQUEEZE: + ruido gaussiano =====")
        print(f"{'CV baseline':<24} ADE {ade_cv:.6f}")
        print(f"{'LSTM+aug+reg (ref)':<24} ADE {REF_ADE:.6f}  FDE {REF_FDE:.6f}")
        print(f"{'+ ruido gaussiano':<24} ADE {ade:.6f}  FDE {fde:.6f}")
        print(f"\nMejora sobre LSTM+aug+reg: {(1-ade/REF_ADE)*100:+.1f}% ADE   |   vs CV: {(1-ade/ade_cv)*100:+.1f}%")

    def run(self):
        # 1. Carga de datos
        self.load_data()
        # 2. Aumento de datos
        self.augment()
        # 3. Scaler
        self.scale_features()
        # 4. Definición del modelo
        self.build_model()
        # 5. Entrenamiento del modelo
        self.train()
        # 6. Guardado de las salidas
        self.save_outputs()
        # 7. Evaluación
        self.evaluate()


if __name__ == "__main__":

    trainer = LSTMAug2HorizonTrainer()
    trainer.run()
