"""
Modelo que combina datos de imagen y de detecciones 3D con fusión tardía.
Modelo 1: entrada 11 características de imagen. Predice el desplazamiento relativo en metros.
Modelo 2: entrada 9 características de detección 3D. Predice el desplazamiento relativo en metros.
Se promedia sus predicciones.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val, ade_fde, load_cv_ref

OBS_LEN, PRED_LEN = 8, 8
DROPOUT, REC_DROPOUT = 0.3, 0.2
REF_EARLY, REF_EARLY_FDE = 0.2021, 0.3029   # Fusión temprana (07) a 8/8 con MAE
REF_DET, REF_FDE = 0.2067, 0.3102           # Base: LSTM 3D detecciones a 8/8 con MAE (04e)


class Combined3DFTardTrainer:

    """
    Combined3DFTardTrainer entrena el modelo con información de entrada 2D + 3D con fusión tardía.
    Se entrena un modelo con información de la imagen 2D (11 características), salida en metros.
    Se entrena un segundo modelo con información de detecciones 3D (9 características), salida en metros.
    Se hace la media de las predicciones -> fusión tardía.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 lstm_units=128, dropout=DROPOUT, rec_dropout=REC_DROPOUT,
                 ref_early=REF_EARLY, ref_early_fde=REF_EARLY_FDE,
                 ref_det=REF_DET, ref_fde=REF_FDE, batch_size=256, epochs=120):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.ref_early = ref_early
        self.ref_early_fde = ref_early_fde
        self.ref_det = ref_det
        self.ref_fde = ref_fde
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos y máscaras del split
        self.X_img = self.X_3d = self.Y = None
        self.tm = self.vm = None
        self.Y_va = None            # Y de val (metros)
        self.X3d_va = None          # parte 3D de val (metros) para CV y métricas

    def load_data(self):
        # Carga de datos: entrada de características de imagen 2D + entrada de características de detecciones 3D +
        # salida en metros.
        self.X_img = np.load(f"{self.data_dir}/data/X_ds.npy")            # (N,8,11)
        self.X_3d = np.load(f"{self.data_dir}/data/X3d_det_motion.npy")   # (N,8,9)
        self.Y = np.load(f"{self.data_dir}/data/Y3d_det.npy")[:, :self.pred_len, :]   # metros (8/8)
        subtrack_ids = np.load(f"{self.data_dir}/data/subtrack_ids_ds.npy", allow_pickle=True)

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        self.tm, self.vm = official_masks(sck)

        self.Y_va = self.Y[self.vm]
        # Guarda las características 3D para la evaluación
        self.X3d_va = self.X_3d[self.vm]

    def build_model(self, n_features):
        # Define el modelo -> el mismo que anteriormente con recurrent dropout y pérdida MAE
        return build_seq2seq(self.obs_len, self.pred_len, n_features,
                             units=self.lstm_units, dropout=self.dropout,
                             rec_dropout=self.rec_dropout, optimizer="adam",
                             loss="mae", metrics=["mse"])

    def train_one(self, X, name, fig_name, save_as=None):
        # La entrada puede ser 2D o 3D. El método entrena el modelo y devuelve las predicciones
        # Entrada con partición oficial
        Xtr, Xva = X[self.tm], X[self.vm]
        # Número de características
        nf = X.shape[2]
        # Scaler
        Xtr_s, Xva_s, _ = scale_train_val(Xtr, Xva, nf)
        # Define el modelo
        m = self.build_model(nf)
        # Define los callbacks
        cbs = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)]
        # Entrena el modelo
        print(f"\n--- Entrenando modelo {name} ---")
        history = m.fit(Xtr_s, self.Y[self.tm], validation_data=(Xva_s, self.Y[self.vm]),
                        epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs, verbose=2)

        # Curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_3d/figuras", exist_ok=True)
        plt.figure(figsize=(7, 5))
        plt.plot(history.history["loss"], label="Entrenamiento")
        plt.plot(history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MAE, metros)")
        plt.title(f"Curva de entrenamiento — {name}")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/{fig_name}", dpi=130); plt.close()
        print(f"Guardado: figuras/{fig_name}")
        # Guarda el checkpoint del sub-modelo
        if save_as is not None:
            os.makedirs(f"{self.data_dir}/prediccion_3d/best", exist_ok=True)
            m.save(f"{self.data_dir}/prediccion_3d/best/{save_as}")
            print(f"Guardado checkpoint: {save_as}")
        # Devuelve las predicciones
        return m.predict(Xva_s, batch_size=512, verbose=0)

    def evaluate(self, pred_img, pred_3d):
        # La predicción final será la media entre la predicción del modelo con info 2D y la del
        # modelo con info 3D
        pred_ens = 0.5 * pred_img + 0.5 * pred_3d

        # Métricas: modelo LSTM 2D + modelo LSTM 3D + modelo combinado (fusión tardia) + CV
        ade_img, fde_img = ade_fde(pred_img, self.X3d_va, self.Y_va)
        ade_3d, fde_3d = ade_fde(pred_3d, self.X3d_va, self.Y_va)
        ade_ens, fde_ens = ade_fde(pred_ens, self.X3d_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, "8_8")   # CV del 00

        # Combinación: CV vs LSTM 2D-metros vs LSTM 3D-metros (base) vs fusión temprana vs fusión tardía.
        # Todas las mejoras se miden respecto a CV.
        print("\n========== Fusión Tardía =========="  )
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*66}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'LSTM 2D-metros':<28} {ade_img:>8.4f} {fde_img:>8.4f} "
              f"{(1-ade_img/ade_cv)*100:>7.1f}% {(1-fde_img/fde_cv)*100:>7.1f}%")
        print(f"{'LSTM 3D-metros (04e)*':<28} {self.ref_det:>8.4f} {self.ref_fde:>8.4f} "
              f"{(1-self.ref_det/ade_cv)*100:>7.1f}% {(1-self.ref_fde/fde_cv)*100:>7.1f}%")
        if self.ref_early is not None:
            print(f"{'Fusión temprana (07)':<28} {self.ref_early:>8.4f} {self.ref_early_fde:>8.4f} "
                  f"{(1-self.ref_early/ade_cv)*100:>7.1f}% {(1-self.ref_early_fde/fde_cv)*100:>7.1f}%")
        else:
            print(f"{'Fusión temprana (07)':<28} {'(rellenar 07 a 8/8)':>8}")
        print(f"{'Fusión tardía':<28} {ade_ens:>8.4f} {fde_ens:>8.4f} "
              f"{(1-ade_ens/ade_cv)*100:>7.1f}% {(1-fde_ens/fde_cv)*100:>7.1f}%")
        print("* Base = LSTM 3D detecciones a 8/8 con MAE. El 'LSTM 3D-metros' entrenado "
              f"aquí dio {ade_3d:.4f}/{fde_3d:.4f}.")

    def run(self):
        # 1. Carga de datos
        self.load_data()
        # 2. Entrena ambos sub-modelos
        pred_img = self.train_one(self.X_img, "IMAGEN->metros",
                                  fig_name="loss_ensemble3d_imagen_a_metros.png", save_as="img2meters_best.keras")
        pred_3d = self.train_one(self.X_3d, "3D-detecciones->metros",
                                 fig_name="loss_ensemble3d_3ddetecciones_a_metros.png", save_as="ensemble3d_3d_best.keras")
        # 3. Evaluación: fusión tardía
        self.evaluate(pred_img, pred_3d)


if __name__ == "__main__":

    trainer = Combined3DFTardTrainer()
    trainer.run()
