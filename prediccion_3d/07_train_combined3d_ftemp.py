"""
Modelo que combina datos de imagen y de detecciones 3D con fusión temprana.
Entrada: 11 imagen + 9 detecciones = 20 características. Predice el desplazamiento relativo en metros.
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

OBS_LEN, PRED_LEN = 8, 12
N_IMG, N_3D = 11, 9
N_FEATURES = N_IMG + N_3D   # 20
DROPOUT, REC_DROPOUT = 0.3, 0.2
REF_3D_DET = 0.2730   # Resultados del modelo LSTM con detecciones 3D
REF_3D_FDE = 0.4481 


class Combined3DFTempTrainer:

    """
    Combined3DFTempTrainer entrena el modelo con información de entrada 2D + 3D con fusión temprana.
    Se concatenan las 11 características de imagen con las 9 de detección 3D, creando 20 características
    de entrada. Predice el desplazamiento futuro en metros.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=DROPOUT,
                 rec_dropout=REC_DROPOUT, ref_3d_det=REF_3D_DET, ref_3d_fde=REF_3D_FDE,
                 batch_size=256, epochs=120):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.ref_3d_det = ref_3d_det
        self.ref_3d_fde = ref_3d_fde
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos, scaler, modelo e historial
        self.X_tr = self.X_va = self.Y_tr = self.Y_va = None
        self.X3d_va = None          # parte 3D de val (metros) para CV y métricas
        self.X_tr_s = self.X_va_s = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga datos -> entrada de imagen 2D (11 características) + entrada de detecciones 3D (9 características) + salida 3D (en metros)
        X_img = np.load(f"{self.data_dir}/X_ds.npy")            # (N,8,11) imagen 2D
        X_3d = np.load(f"{self.data_dir}/X3d_det_motion.npy")   # (N,8,9) detecciones 3D
        Y = np.load(f"{self.data_dir}/Y3d_det.npy")             # (N,12,2) metros
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)

        # Concatena las características: (N,8,11) + (N,8,9) -> (N,8,20)
        X = np.concatenate([X_img, X_3d], axis=2)
        print(f"X(imagen+3D): {X.shape}  Y(metros): {Y.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]
       
        # Guarda las características 3D por separado para poder calcular el CV baseline más tarde con la
        # velocidad 3D (ya que la salida es en metros)
        self.X3d_va = X_3d[vm]

    def scale_features(self):
        # Scaler
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(self.X_tr, self.X_va, self.n_features)

    def build_model(self):
        # Se define el modelo -> el mismo que los anteriores pero con 20 características de entrada
        # y recurrent dropout -> 11 imagen + 9 detección
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
               ModelCheckpoint(f"{self.data_dir}/prediccion_3d/best/combined3d_best.keras", monitor="val_loss", save_best_only=True)]
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
        np.save(f"{self.data_dir}/prediccion_3d/history/hist_combined3d.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE, metros)")
        plt.title("Curva de entrenamiento — Fusión temprana imagen+3D")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/loss_combined3d.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_combined3d.png, hist_combined3d.npy")

    def evaluate(self):
        # Predicción de desplazamientos relativos (en metros)
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Evaluación: modelos con detecciones 3D e info 2D vs CV baseline (metros)
        ade, fde = ade_fde(pred, self.X3d_va, self.Y_va)
        ade_cv, fde_cv = cv_baseline(self.X3d_va, self.Y_va, self.pred_len)

        # Comparación: LSTM con detecciones 3D e info 2D vs LSTM con detecciones 3D vs CV baseline
        print("\n===== COMBINADO imagen + 3D (detecciones), FUSIÓN TEMPRANA — METROS =====")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'':>9} {'':>9}")
        print(f"{'detecciones-3D':<28} {self.ref_3d_det:>8.4f} {self.ref_3d_fde:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'Imagen + 3D (temprana)':<28} {ade:>8.4f} {fde:>8.4f} "
              f"{(1-ade/self.ref_3d_det)*100:>7.1f}% {(1-fde/self.ref_3d_fde)*100:>7.1f}%")

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

    trainer = Combined3DFTempTrainer()
    trainer.run()
