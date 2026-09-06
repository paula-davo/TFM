"""
Modelo multitrayectoria sobre la fusión temprana: parte del mejor determinista (07). 
Entrada 20 características (11 imagen + 9 detección 3D).
Salida K=20 hipótesis.

Evaluación con franja densa.
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Reshape, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
sys.path.insert(0, f"{DATA_DIR}/evaluate_predictions")   # dense_band_pred
from split_utils import official_masks
from lstm_common import scale_train_val, load_cv_ref
from metrics_utils import dense_band_pred

OBS_LEN, PRED_LEN = 8, 8
N_IMG, N_3D = 11, 9
N_FEATURES = N_IMG + N_3D   # 20
DROPOUT, REC_DROPOUT = 0.3, 0.2
K = 20            # Número de hipótesis
EPS = 0.05        # Peso de la pérdida a las hipótesis no-ganadoras
BAND_RADIUS = 0.5 # Radio (m) de la franja densa
REF_EARLY = 0.2021  # Fusión temprana a 8/8 con MAE (07)
REF_EARLY_FDE = 0.3029
REF_DET = 0.2067    # LSTM 3D detecciones 3D (04e)
REF_DET_FDE = 0.3102


def wta_loss(y_true, y_pred):
    # WTA best-of-K con MAE.
    yt = tf.expand_dims(y_true, 1)
    # Error (MAE) de cada predicción (N,K,T,2) -> (N,K)
    err = tf.reduce_mean(tf.abs(y_pred - yt), axis=[2, 3])
    # Mejor predicción (N,K) -> (N,)
    best = tf.reduce_min(err, axis=1)
    # Error medio de todas las predicciones (N,K) -> (N,)
    mean = tf.reduce_mean(err, axis=1)
    # Pérdida = 95% best + 5% mean
    return tf.reduce_mean((1.0 - EPS) * best + EPS * mean)


class Combined3DMultimodalTrainer:

    """Combined3DMultimodalTrainer entrena el modelo multitrayectoria (K=20)
    sobre la fusión temprana imagen+3D."""

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=DROPOUT,
                 rec_dropout=REC_DROPOUT, k=K, band_radius=BAND_RADIUS,
                 ref_early=REF_EARLY, ref_early_fde=REF_EARLY_FDE,
                 ref_det=REF_DET, ref_det_fde=REF_DET_FDE, batch_size=256, epochs=120):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.k = k
        self.band_radius = band_radius
        self.ref_early = ref_early
        self.ref_early_fde = ref_early_fde
        self.ref_det = ref_det
        self.ref_det_fde = ref_det_fde
        self.batch_size = batch_size
        self.epochs = epochs

        self.X_tr = self.X_va = self.Y_tr = self.Y_va = None
        self.X_tr_s = self.X_va_s = None
        self.scaler = self.model = self.history = None

    def load_data(self):
        # Carga de datos: entrada de imagen 2D (11) + detecciones 3D (9) -> 20
        # Salida = desplazamiento GT (metros)
        X_img = np.load(f"{self.data_dir}/data/X_ds.npy")            # (N,8,11) imagen 2D
        X_3d = np.load(f"{self.data_dir}/data/X3d_det_motion.npy")   # (N,8,9) detecciones 3D
        Y = np.load(f"{self.data_dir}/data/Y3d_det.npy")[:, :self.pred_len, :]   # (N,8,2) metros (8/8)
        subtrack_ids = np.load(f"{self.data_dir}/data/subtrack_ids_ds.npy", allow_pickle=True)
        X = np.concatenate([X_img, X_3d], axis=2)              # (N,8,20)
        print(f"X(imagen+3D): {X.shape}  Y(metros): {Y.shape}")

        # Partición oficial
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)
        self.X_tr, self.X_va, self.Y_tr, self.Y_va = X[tm], X[vm], Y[tm], Y[vm]
        print(f"Train: {self.X_tr.shape[0]}  Val: {self.X_va.shape[0]}")

    def scale_features(self):
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(self.X_tr, self.X_va, self.n_features)

    def build_model(self):
        # Arquitectura multitrayectoria con 20 características de entrada
        inp = Input(shape=(self.obs_len, self.n_features))
        eo, h, c = LSTM(self.lstm_units, return_state=True, recurrent_dropout=self.rec_dropout)(inp)
        eo = Dropout(self.dropout)(eo)
        d = RepeatVector(self.pred_len)(eo)
        d = LSTM(self.lstm_units, return_sequences=True, recurrent_dropout=self.rec_dropout)(d, initial_state=[h, c])
        d = Dropout(self.dropout)(d)
        d = TimeDistributed(Dense(64, activation="relu"))(d)
        d = TimeDistributed(Dense(self.k * 2))(d)        # (b,8,K*2)
        d = Reshape((self.pred_len, self.k, 2))(d)       # (b,8,K,2)
        out = Permute((2, 1, 3))(d)                      # (b,K,8,2)
        self.model = Model(inp, out)
        self.model.compile(optimizer="adam", loss=wta_loss)
        self.model.summary()
        return self.model

    def train(self):
        # Se definen los callbacks
        os.makedirs(f"{self.data_dir}/prediccion_3d/best", exist_ok=True)
        cbs = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
               ModelCheckpoint(f"{self.data_dir}/prediccion_3d/best/combined3d_multimodal_best.keras",
                               monitor="val_loss", save_best_only=True)]
        # Entrena el modelo
        t0 = time.time()
        self.history = self.model.fit(self.X_tr_s, self.Y_tr, validation_data=(self.X_va_s, self.Y_va),
            epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs, verbose=2)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        os.makedirs(f"{self.data_dir}/prediccion_3d/figuras", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_3d/history", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_3d/history/hist_combined3d_multimodal.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MAE, metros)")
        plt.title("Curva de entrenamiento — Fusión temprana multitrayectoria")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/loss_combined3d_multimodal.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_combined3d_multimodal.png, hist_combined3d_multimodal.npy")

    def evaluate(self):
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)   # (Nv,K,8,2)

        # Métrica: franja densa
        rep = dense_band_pred(pred, self.band_radius)
        mode_ade = np.linalg.norm(rep - self.Y_va, axis=2).mean(axis=1).mean()
        mode_fde = np.linalg.norm(rep[:, -1, :] - self.Y_va[:, -1, :], axis=1).mean()

        # CV del 00
        ade_cv, fde_cv = load_cv_ref(self.data_dir, "8_8")

        print(f"\n===== Fusión temprana multitrayectoria (K={self.k}) =====")
        print(f"{'Modelo':<30} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*68}")
        print(f"{'CV':<30} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'LSTM con fusión temprana (07)':<30} {self.ref_early:>8.4f} {self.ref_early_fde:>8.4f} "
              f"{(1-self.ref_early/ade_cv)*100:>7.1f}% {(1-self.ref_early_fde/fde_cv)*100:>7.1f}%")
        print(f"{'LSTM: multitrayectoria':<30} {mode_ade:>8.4f} {mode_fde:>8.4f} "
              f"{(1-mode_ade/ade_cv)*100:>7.1f}% {(1-mode_fde/fde_cv)*100:>7.1f}%")
        
        # Análisis de sensibilidad al radio
        print(f"\n--- Sensibilidad al radio de la franja densa (nº medio de hipótesis en la franja) ---")
        print(f"{'radio (m)':>10} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'hip./franja':>12}")
        end = pred[:, :, -1, :]
        for r in (0.2, 0.3, 0.4, 0.5, 0.8, 1.2):
            rep_r = dense_band_pred(pred, r)
            ade_r = np.linalg.norm(rep_r - self.Y_va, axis=2).mean(axis=1).mean()
            fde_r = np.linalg.norm(rep_r[:, -1, :] - self.Y_va[:, -1, :], axis=1).mean()
            band_sz = (np.linalg.norm(end[:, :, None, :] - end[:, None, :, :], axis=3) <= r).sum(axis=2).max(axis=1).mean()
            mark = "  <-- tabla" if abs(r - self.band_radius) < 1e-9 else ""
            print(f"{r:>10.1f} {ade_r:>8.4f} {fde_r:>8.4f} {(1-ade_r/self.ref_early)*100:>7.1f}% {band_sz:>12.1f}{mark}")

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

    trainer = Combined3DMultimodalTrainer()
    trainer.run()
