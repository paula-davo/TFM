"""
Modelo de múltiples predicciones: el modelo genera K=20 trayectorias futuras y se entrena con
pérdida best-of-K.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Reshape, Permute, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import scale_train_val, flip_x

OBS_LEN, PRED_LEN, N_FEATURES = 8, 12, 11
DROPOUT, REC_DROPOUT, NOISE = 0.3, 0.2, 0.05
K = 20            # número de hipótesis
EPS = 0.05        # WTA relajado (peso a las cabezas no-ganadoras)
REF_DET = 0.026951  # Resultados modelo determinista (con generalización y aumento de horizonte)
REF_FDE = 0.049851 


def wta_loss(y_true, y_pred):
    # y_true (b,12,2)   y_pred (b,K,12,2)
    yt = tf.expand_dims(y_true, 1)                       # (b,1,12,2)
    err = tf.reduce_mean(tf.square(y_pred - yt), axis=[2, 3])  # (b,K)
    best = tf.reduce_min(err, axis=1)                    # (b,)
    mean = tf.reduce_mean(err, axis=1)                   # (b,)
    return tf.reduce_mean((1.0 - EPS) * best + EPS * mean)


class LSTMMultimodalHorizonTrainer:

    """
    LSTMMultimodalHorizonTrainer entrena un LSTM con múltiples predicciones: genera K=20 hipótesis de futuro y
    se entrena con pérdida best-of-K. Tiene las técnicas de generalización del paso 13 y el aumento del horizonte.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=DROPOUT,
                 rec_dropout=REC_DROPOUT, noise=NOISE, k=K, batch_size=256, epochs=120):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.noise = noise
        self.k = k
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
        # Entrada: (8, 11)
        inp = Input(shape=(self.obs_len, self.n_features))
        # Ruido gaussiano
        x = GaussianNoise(self.noise)(inp)
        # Encoder LSTM con recurrent dropout
        eo, h, c = LSTM(self.lstm_units, return_state=True, recurrent_dropout=self.rec_dropout)(x)
        # Dropout
        eo = Dropout(self.dropout)(eo)
        # Decoder LSTM con recurrent dropout
        d = RepeatVector(self.pred_len)(eo)
        d = LSTM(self.lstm_units, return_sequences=True, recurrent_dropout=self.rec_dropout)(d, initial_state=[h, c])
        # Dropout
        d = Dropout(self.dropout)(d)
        d = TimeDistributed(Dense(64, activation="relu"))(d)
        d = TimeDistributed(Dense(self.k * 2))(d)        # (b,12,K*2)
        d = Reshape((self.pred_len, self.k, 2))(d)       # (b,12,K,2)
        # Salida: (b, K, 12, 2) siendo K el número de predicciones
        out = Permute((2, 1, 3))(d)
        # Compila el modelo
        self.model = Model(inp, out)
        self.model.compile(optimizer="adam", loss=wta_loss)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)
        # Define los callbacks
        cbs = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_2d/best/ds_multimodal_best.keras",
                            monitor="val_loss", save_best_only=True),
        ]
        # Entrena el modelo, midiendo el tiempo de entrenamiento
        t0 = time.time()
        self.history = self.model.fit(self.X_tr_s, self.Y_tr,
                                      validation_data=(self.X_va_s, self.Y_va),
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      callbacks=cbs, verbose=2)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_ds_multimodal.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (WTA, MSE best-of-K)")
        plt.title("Curva de entrenamiento — LSTM multimodal (K=20)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_ds_multimodal.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_ds_multimodal.png, history/hist_ds_multimodal.npy")

    def evaluate(self):
        # Predicción con el modelo de múltiples predicciones
        pred = self.model.predict(self.X_va_s, batch_size=256, verbose=0)   # (Nv,K,12,2)
        Yv = self.Y_va[:, None, :, :]                                       # (Nv,1,12,2)

        # Distancia en cada paso entre predicción y desplazamiento real
        disp = np.linalg.norm(pred - Yv, axis=3)                            # (Nv,K,12)
        # ADE de cada hipótesis
        ade_k = disp.mean(axis=2)                                           # (Nv,K)
        # ADE de la mejor hipótesis
        min_ade = ade_k.min(axis=1).mean()                                 # minADE_K
        # FDE de cada hipótesis
        fde_k = np.linalg.norm(pred[:, :, -1, :] - self.Y_va[:, None, -1, :], axis=2)  # (Nv,K)
        # FDE de la mejor hipótesis
        min_fde = fde_k.min(axis=1).mean()

        # Promedio de predicciones
        mean_pred = pred.mean(axis=1)
        # ADE y FDE del promedio de predicciones
        ade_mean = np.mean(np.linalg.norm(mean_pred - self.Y_va, axis=2))
        fde_mean = np.mean(np.linalg.norm(mean_pred[:, -1, :] - self.Y_va[:, -1, :], axis=1))

        # Comparación: modelo determinista vs promedio de modelo múltiples predicciones vs mejor hipótesis múltiples predicciones
        print(f"\n===== MULTIMODAL (K={self.k}) — NORMALIZADO =====")
        print(f"{'Modelo':<26} {'ADE':>9} {'FDE':>9} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*65}")
        print(f"{'Determinista (ref)':<26} {REF_DET:>9.5f} {REF_FDE:>9.5f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'Multimodal media-cabezas':<26} {ade_mean:>9.5f} {fde_mean:>9.5f} "
              f"{(1-ade_mean/REF_DET)*100:>8.1f}% {(1-fde_mean/REF_FDE)*100:>8.1f}%")
        print(f"{'Multimodal minADE_'+str(self.k):<26} {min_ade:>9.5f} {min_fde:>9.5f} "
              f"{(1-min_ade/REF_DET)*100:>8.1f}% {(1-min_fde/REF_FDE)*100:>8.1f}%")

    def run(self):
        # 1. Carga de los datos
        self.load_data()
        # 2. Aumento de los datos
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

    trainer = LSTMMultimodalHorizonTrainer()
    trainer.run()
