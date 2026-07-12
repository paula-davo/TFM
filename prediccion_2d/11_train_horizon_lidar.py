"""
Fusión LiDAR a horizonte largo. Motion (X_ds) + rama MLP LiDAR (lidar_feats_ds).
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import scale_train_val, ade_fde, cv_baseline

OBS_LEN, PRED_LEN, N_FEATURES, N_LIDAR = 8, 12, 11, 8
LSTM_LARGO_ADE, LSTM_LARGO_FDE = 0.028178, 0.051457   # Resultados del modelo LSTM base (con aumento de horizonte)


class LSTMLidarFusionTrainer:

    """
    LSTMLidarFusionTrainer entrena el modelo LSTM con información lidar y aumento del horizonte.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, n_lidar=N_LIDAR, lstm_units=128, dropout=0.2,
                 batch_size=256, epochs=100):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_lidar = n_lidar
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos de train/val: trayectoria (X), salida (Y), features LiDAR (li) y flag (fl)
        self.X_tr = self.X_va = None
        self.Y_tr = self.Y_va = None
        self.li_tr = self.li_va = None
        self.fl_tr = self.fl_va = None
        self.X_tr_s = self.X_va_s = None
        self.li_tr_s = self.li_va_s = None
        self.scaler_traj = None
        self.scaler_lidar = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga trayectoria + LiDAR + flags y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_ds.npy")
        Y = np.load(f"{self.data_dir}/Y_ds.npy")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        lidar = np.load(f"{self.data_dir}/lidar_feats_ds.npy")
        flags = np.load(f"{self.data_dir}/lidar_flags_ds.npy")
        print(f"X: {X.shape}  lidar cobertura: {100*flags.mean():.1f}%")

        tm, vm = official_masks(subtrack_ids)
        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]
        self.li_tr, self.li_va = lidar[tm], lidar[vm]
        self.fl_tr, self.fl_va = flags[tm], flags[vm]
        print(f"Train: {self.X_tr.shape[0]}  Val: {self.X_va.shape[0]}")

    def _prep_lidar(self, f, fl):
        # Aplica scaler a datos lidar, y asigna 0.0 a las muestras que no tienen lidar
        s = self.scaler_lidar.transform(f)
        s[~fl] = 0.0
        # Concatena las flags
        return np.concatenate([s, fl.astype(np.float32)[:, None]], axis=1)

    def scale_features(self):
        # Scaler de la trayectoria (reutiliza scale_train_val del base)
        self.X_tr_s, self.X_va_s, self.scaler_traj = scale_train_val(
            self.X_tr, self.X_va, self.n_features)
        # Scaler del lidar (fit solo en train data)
        self.scaler_lidar = StandardScaler()
        self.scaler_lidar.fit(self.li_tr[self.fl_tr])
        self.li_tr_s = self._prep_lidar(self.li_tr, self.fl_tr)
        self.li_va_s = self._prep_lidar(self.li_va, self.fl_va)

    def build_model(self):
        # Rama trayectoria (encoder LSTM) + rama LiDAR (MLP) -> fusión -> decoder
        # Entrada de datos 2D: (8, 11)
        traj_in = Input(shape=(self.obs_len, self.n_features))
        # Encoder LSTM
        enc_out, h, c = LSTM(self.lstm_units, return_state=True, name="encoder_lstm")(traj_in)
        # Dropout
        enc_out = Dropout(self.dropout)(enc_out)

        # Entrada lidar: 9 features en el último frame observado
        lidar_in = Input(shape=(self.n_lidar + 1,))
        # MLP: dense -> dropout -> dense
        l = Dropout(self.dropout)(Dense(64, activation="relu")(lidar_in))
        emb_l = Dense(32, activation="relu")(l)

        # Fusión
        fused = Dense(self.lstm_units, activation="relu", name="fusion")(Concatenate()([enc_out, emb_l]))
        # Decoder LSTM
        dec = RepeatVector(self.pred_len)(fused)
        dec = LSTM(self.lstm_units, return_sequences=True, name="decoder_lstm")(dec, initial_state=[fused, c])
        dec = Dropout(self.dropout)(dec)
        dec = TimeDistributed(Dense(64, activation="relu"))(dec)
        out = TimeDistributed(Dense(2))(dec)

        # Crea el modelo con ambas entradas y lo compila
        self.model = Model([traj_in, lidar_in], out)
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)
        # Define los callbacks
        cbs = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_2d/best/ds_lidar_fusion_best.keras",
                            monitor="val_loss", save_best_only=True),
        ]
        # Entrena el modelo, mide el tiempo de entrenameinto
        t0 = time.time()
        self.history = self.model.fit([self.X_tr_s, self.li_tr_s], self.Y_tr,
                                      validation_data=([self.X_va_s, self.li_va_s], self.Y_va),
                                      epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_ds_lidar.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
        plt.title("Curva de entrenamiento — LSTM + LiDAR")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_ds_lidar.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_ds_lidar.png, history/hist_ds_lidar.npy")

    def evaluate(self):
        # Predice con el modelo LSTM con información lidar
        pred = self.model.predict([self.X_va_s, self.li_va_s], batch_size=512, verbose=0)
        # Evaluación: LSTM con info lidar vs CV baseline
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, _ = cv_baseline(self.X_va, self.Y_va, self.pred_len)

        # Comparación: LSTM con info lidar vs LSTM base vs CV baseline
        print("\n===== HORIZONTE LARGO: LSTM+LiDAR vs LSTM vs CV =====")
        print(f"{'Modelo':<22}  {'ADE':>10}  {'FDE':>10}  {'Mejora vs CV':>14}")
        print(f"{'-'*62}")
        print(f"{'CV baseline':<22}  {ade_cv:>10.6f}  {'':>10}  {'(ref)':>14}")
        print(f"{'LSTM (solo mov)':<22}  {LSTM_LARGO_ADE:>10.6f}  {LSTM_LARGO_FDE:>10.6f}  {(1-LSTM_LARGO_ADE/ade_cv)*100:>13.1f}%")
        print(f"{'LSTM + LiDAR':<22}  {ade:>10.6f}  {fde:>10.6f}  {(1-ade/ade_cv)*100:>13.1f}%")
        print(f"\nMejora LiDAR sobre LSTM-solo largo: {(1-ade/LSTM_LARGO_ADE)*100:+.1f}% ADE")

    def run(self):
        # 1. Carga de los datos
        self.load_data()
        # 2. Scaler
        self.scale_features()
        # 3. Definición del modelo
        self.build_model()
        # 4. Entrenamiento del modelo
        self.train()
        # 5. Guardado resultados
        self.save_outputs()
        # 6. Evaluación
        self.evaluate()


if __name__ == "__main__":

    trainer = LSTMLidarFusionTrainer()
    trainer.run()
