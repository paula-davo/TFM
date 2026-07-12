"""
Techo ideal: modelo que recibe etiquetas 3D (posiciones GT) en metros -> entrada de 9 características.
Sirve para medir en el futuro la robustez del predictor al ruido del detector.
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
sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val, ade_fde, cv_baseline

OBS_LEN, PRED_LEN, N_FEATURES = 8, 12, 9
DROPOUT, REC_DROPOUT = 0.3, 0.2


class Lidar3DMotionTrainer:

    """
    Lidar3DMotionTrainer entrena el modelo LSTM con etiquetas 3D (techo ideal).
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
        # Carga los datos -> etiquetas 3D en metros y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X3d_motion.npy")   # (N,8,9)
        Y = np.load(f"{self.data_dir}/Y3d.npy")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"X(3D motion only): {X.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]

    def scale_features(self):
        # Scaler
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(
            self.X_tr, self.X_va, self.n_features)

    def build_model(self):
        # Se define el modelo -> igual que el LSTM utilizado en 2D pero distintas entradas,
        # con recurrent dropout
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
               ModelCheckpoint(f"{self.data_dir}/prediccion_3d/best/lidar3d_motiononly_best.keras", monitor="val_loss", save_best_only=True)]
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
        np.save(f"{self.data_dir}/prediccion_3d/history/hist_lidar3d_motiononly.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE, metros)")
        plt.title("Curva de entrenamiento — LiDAR 3D GT solo-movimiento (techo)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/loss_lidar3d_motiononly.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_lidar3d_motiononly.png, hist_lidar3d_motiononly.npy")

    def evaluate(self):
        # Predicción de desplazamientos relativos (en metros)
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Métricas: modelo LSTM con etiquetas 3D vs CV baseline vs baseline quieto (desplazamiento futuro nulo)
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = cv_baseline(self.X_va, self.Y_va, self.pred_len)
        ade_zero = np.mean(np.linalg.norm(self.Y_va, axis=2))
        fde_zero = np.mean(np.linalg.norm(self.Y_va[:, -1, :], axis=1))

        # Comparación: LSTM con etiquetas 3D vs quieto vs CV
        print("\n===== TECHO IDEAL: LiDAR 3D GT solo-movimiento (METROS) =====")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'Quieto':<28} {ade_zero:>8.4f} {fde_zero:>8.4f} {'':>9} {'':>9}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'LiDAR 3D GT solo-mov':<28} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")

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

    trainer = Lidar3DMotionTrainer()
    trainer.run()
