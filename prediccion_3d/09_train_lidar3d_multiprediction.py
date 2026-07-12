"""
Modelo de múltiples predicciones: el modelo genera K=20 trayectorias futuras y se entrena con
pérdida best-of-K.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Reshape, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import scale_train_val, cv_baseline

OBS_LEN, PRED_LEN, N_FEATURES = 8, 12, 9
DROPOUT, REC_DROPOUT = 0.3, 0.2
K = 20            # número de hipótesis
EPS = 0.05        # WTA relajado (peso a las cabezas no-ganadoras)
REF_DET = 0.2730  # 3D detecciones solo-movimiento, determinista (metros, split oficial)
REF_FDE = 0.4481  # FDE del 3D det solo-movimiento determinista (metros)


def wta_loss(y_true, y_pred):
    # y_true (b,12,2)   y_pred (b,K,12,2)
    yt = tf.expand_dims(y_true, 1)                            # (b,1,12,2)
    err = tf.reduce_mean(tf.square(y_pred - yt), axis=[2, 3])  # (b,K)
    best = tf.reduce_min(err, axis=1)                         # (b,)
    mean = tf.reduce_mean(err, axis=1)                        # (b,)
    return tf.reduce_mean((1.0 - EPS) * best + EPS * mean)


class Lidar3DMultimodalTrainer:

    """
    Lidar3DMultimodalTrainer entrena un LSTM con múltiples predicciones: genera k=20 hipótesis de futuro y se
    entrena con pérdida best-of-K. 
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=DROPOUT,
                 rec_dropout=REC_DROPOUT, k=K, ref_det=REF_DET, ref_fde=REF_FDE,
                 batch_size=256, epochs=120):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.k = k
        self.ref_det = ref_det
        self.ref_fde = ref_fde
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos, scaler, modelo e historial
        self.X_tr = self.X_va = self.Y_tr = self.Y_va = None
        self.X_tr_s = self.X_va_s = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga los datos: entrada detecciones 3D y salida desplazamiento relativo en metros
        X = np.load(f"{self.data_dir}/X3d_det_motion.npy")   # (N,8,9) metros, detecciones
        Y = np.load(f"{self.data_dir}/Y3d_det.npy")          # (N,12,2) metros
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"X(3D detecciones): {X.shape}  Y(metros): {Y.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va, self.Y_tr, self.Y_va = X[tm], X[vm], Y[tm], Y[vm]
        print(f"Train: {self.X_tr.shape[0]}  Val: {self.X_va.shape[0]}")

    def scale_features(self):
        # Scaler
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(self.X_tr, self.X_va, self.n_features)

    def build_model(self):
        # Se crea el modelo
        # Entrada: (8 frames, 9 características)
        inp = Input(shape=(self.obs_len, self.n_features))
        # Encoder LSTM con recurrent dropout
        eo, h, c = LSTM(self.lstm_units, return_state=True, recurrent_dropout=self.rec_dropout)(inp)
        # Dropout
        eo = Dropout(self.dropout)(eo)
        d = RepeatVector(self.pred_len)(eo)
        # Decoder LSTM con recurrent dropout
        d = LSTM(self.lstm_units, return_sequences=True, recurrent_dropout=self.rec_dropout)(d, initial_state=[h, c])
        # Dropout
        d = Dropout(self.dropout)(d)
        d = TimeDistributed(Dense(64, activation="relu"))(d)
        d = TimeDistributed(Dense(self.k * 2))(d)        # (b,12,K*2)
        d = Reshape((self.pred_len, self.k, 2))(d)       # (b,12,K,2)
        # La salida son: k trayectorias, con 12 frames y 2 características (desplazamiento relativo X e Y en metros)
        out = Permute((2, 1, 3))(d)                      # (b,K,12,2)
        self.model = Model(inp, out)
        # Compila el modelo
        self.model.compile(optimizer="adam", loss=wta_loss)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_3d/best", exist_ok=True)
         # Se definen los callbacks
        cbs = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
               ModelCheckpoint(f"{self.data_dir}/prediccion_3d/best/lidar3d_multimodal_best.keras", monitor="val_loss", save_best_only=True)]
        # Se entrena el modelo
        self.history = self.model.fit(self.X_tr_s, self.Y_tr, validation_data=(self.X_va_s, self.Y_va),
            epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs, verbose=2)

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_3d/figuras", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_3d/history", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_3d/history/hist_lidar3d_multimodal.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (WTA, MSE best-of-K, metros)")
        plt.title("Curva de entrenamiento — LiDAR 3D multimodal (K=20)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/loss_lidar3d_multimodal.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_lidar3d_multimodal.png, hist_lidar3d_multimodal.npy")

    def evaluate(self):
        # Predicción: k trayectorias, 12 frames, 2 características predichas
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)   # (Nv,K,12,2)

        # Cálulo del ADE para cada hipótesis predicha
        disp = np.linalg.norm(pred - self.Y_va[:, None, :, :], axis=3)  # (Nv,K,12)
        ade_k = disp.mean(axis=2)                                       # (Nv,K)
        # ADE de la mejor hipótesis
        min_ade = ade_k.min(axis=1).mean()                             # minADE_K

        # FDE de cada hipótesis predicha
        fde_k = np.linalg.norm(pred[:, :, -1, :] - self.Y_va[:, None, -1, :], axis=2)  # (Nv,K)
        # FDE de la mejor hipótesis
        min_fde = fde_k.min(axis=1).mean()

        # Evaluación de CV baseline (en metros)
        ade_cv, fde_cv = cv_baseline(self.X_va, self.Y_va, self.pred_len)

        # Comparación: cv vs LSTM determinista vs LSTM múltiples predicciones
        print(f"\n===== MULTIMODAL en METROS (K={self.k}) — LiDAR 3D detecciones =====")
        print(f"{'Modelo':<32} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*70}")
        print(f"{'CV':<32} {ade_cv:>8.4f} {fde_cv:>8.4f} {'':>9} {'':>9}")
        print(f"{'3D det determinista (ref)':<32} {self.ref_det:>8.4f} {self.ref_fde:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'3D det multimodal minADE_'+str(self.k):<32} {min_ade:>8.4f} {min_fde:>8.4f} "
              f"{(1-min_ade/self.ref_det)*100:>7.1f}% {(1-min_fde/self.ref_fde)*100:>7.1f}%")
        print(f"\nminADE_{self.k} vs CV: {(1-min_ade/ade_cv)*100:+.1f}% ADE")

    def run(self):
        # 1. Carga de datos
        self.load_data()
        # 2. Scaler
        self.scale_features()
        # 3. Modelo
        self.build_model()
        # 4. Entrenamiento
        self.train()
        # 5. Salidas
        self.save_outputs()
        # 6. Evaluación
        self.evaluate()


if __name__ == "__main__":

    trainer = Lidar3DMultimodalTrainer()
    trainer.run()
