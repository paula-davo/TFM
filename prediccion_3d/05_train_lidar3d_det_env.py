"""
Modelo LSTM con información lidar 3D: detecciones reales del sensor (9 características)
e información del entorno obtenida de la nube de puntos (8 características). Predice el
desplazamiento relativo en metros.
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
sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val, ade_fde, cv_baseline

OBS_LEN, PRED_LEN = 8, 12
N_MOTION, N_ENV = 9, 8
N_FEATURES = N_MOTION + N_ENV   # 17
DROPOUT, REC_DROPOUT = 0.3, 0.2


class Lidar3DDetEnvTrainer:

    """
    Lidar3DDetEnvTrainer entrena el modelo LSTM con entrada de detecciones 3D + información del entorno.
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

        # Datos, scaler, modelo e historial
        self.X_tr = self.X_va = self.Y_tr = self.Y_va = None
        self.X_tr_s = self.X_va_s = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga los datos de entrada: detecciones 3D + datos de entorno + flags
        Xm = np.load(f"{self.data_dir}/X3d_det_motion.npy")   # (N,8,9) metros, detecciones
        Xe = np.load(f"{self.data_dir}/X_lidar_seq.npy")      # (N,8,8) entorno
        fe = np.load(f"{self.data_dir}/lidar_seq_flags.npy")  # (N,8)
        # Carga los datos de salida: desplazamiento relativo en metros
        Y = np.load(f"{self.data_dir}/Y3d_det.npy")           # (N,12,2) metros
        # Carga de subtrack IDs para partición
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)

        # Concatena las características de detecciones y entorno: (N,8,9) + (N,8,8) -> (N,8,17)
        X = np.concatenate([Xm, Xe], axis=2)
        print(f"X(3D det+entorno): {X.shape}  Y(metros): {Y.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]

    def scale_features(self):
        # Scaler
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(self.X_tr, self.X_va, self.n_features)
        # Guarda el scaler
        joblib.dump(self.scaler, f"{self.data_dir}/lidar3d_det_env_scaler.pkl")

    def build_model(self):
        # Define el modelo -> el mismo LSTM que con detecciones y etiquetas pero con 17 características de entrada 
        # (9 detecciones 3D + 8 de entorno), con recurrent dropout
        self.model = build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                                   units=self.lstm_units, dropout=self.dropout,
                                   rec_dropout=self.rec_dropout, optimizer="adam",
                                   metrics=["mae"], named=True)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        # Se definen los callbacks
        os.makedirs(f"{self.data_dir}/prediccion_3d/best", exist_ok=True)
        cbs = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
               ModelCheckpoint(f"{self.data_dir}/prediccion_3d/best/lidar3d_det_env_best.keras", monitor="val_loss", save_best_only=True)]
        # Se entrena el modelo midiendo el tiempo de entrenamiento
        t0 = time.time()
        self.history = self.model.fit(
            self.X_tr_s, self.Y_tr,
            validation_data=(self.X_va_s, self.Y_va),
            epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_3d/figuras", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_3d/history", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_3d/history/hist_lidar3d_det_env.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE, metros)")
        plt.title("Curva de entrenamiento — LiDAR 3D detecciones (mov+entorno)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/loss_lidar3d_det_env.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_lidar3d_det_env.png, hist_lidar3d_det_env.npy")

    def evaluate(self):
        # Evaluación del modelo
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Métricas: modelo de detecciones 3D + información entorno vs CV baseline vs baseline quieto (desplazamiento futuro nulo)
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = cv_baseline(self.X_va, self.Y_va, self.pred_len)
        ade_zero = np.mean(np.linalg.norm(self.Y_va, axis=2))
        fde_zero = np.mean(np.linalg.norm(self.Y_va[:, -1, :], axis=1))

        # Comparación: LSTM detecciones 3D + info entorno vs quieto vs CV
        print("\n===== LiDAR 3D DETECCIONES (trayectoria + entorno) — METROS =====")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'Quieto':<28} {ade_zero:>8.4f} {fde_zero:>8.4f} {'':>9} {'':>9}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'3D det (mov+entorno)':<28} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")
        print("\nNOTA: metros, NO comparable con los modelos de imagen (normalizados).")

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

    trainer = Lidar3DDetEnvTrainer()
    trainer.run()
