"""
Fusión temprana imagen + 3D regularizada.
En la curva de pérdida observamos sobreajuste. En este caso se añade una tasa
de aprendizaje más pequeña y dropout más alto para reducirlo.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lidar3d_trainer import Lidar3DTrainer, DATA_DIR

sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import ade_fde, load_cv_ref

N_IMG, N_3D = 11, 9
N_FEATURES = N_IMG + N_3D   # 20
LEARNING_RATE = 3e-4        # LR
REF_3D_DET, REF_3D_FDE = 0.2067, 0.3102    # LSTM 3D detecciones MAE (04e)
REF_EARLY, REF_EARLY_FDE = 0.2021, 0.3029  # Fusión temprana LSTM (07)


class Combined3DFTempRegTrainer(Lidar3DTrainer):
    """Combined3DFTempRegTrainer entrena un modelo con fusión temprana 
    (imagen + 3D) con regularización para reducir el sobreajuste."""

    def __init__(self, lr=LEARNING_RATE, **kw):
        super().__init__(
            x_file="X_ds.npy", y_file="Y3d_det.npy",
            ckpt_name="combined3d_reg_best.keras",
            hist_name="hist_combined3d_reg.npy",
            fig_name="loss_combined3d_reg.png",
            fig_title="Curva de entrenamiento — Fusión temprana imagen+3D (regularizada)",
            cv_key="8_8",
            x_print="X(imagen+3D):",
            eval_header="===== Fusión temprana regularizada =====",
            model_label="LSTM con reducción de sobreajuste",
            obs_len=8, pred_len=8, n_features=N_FEATURES,
            # dropout 0.5; Adam con LR bajo; más paciencia para el descenso gradual
            dropout=0.5, loss="mae", metrics=("mse",), loss_label="MAE",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            epochs=200, es_patience=15, rlr_patience=7, **kw)
        # Parte 3D de val (metros)
        self.X3d_va = None

    def load_data(self):
        # Carga de datos: entrada de imagen 2D (11) + detección 3D (9) = 20
        # Salida en metros (8/8)
        X_img = np.load(f"{self.data_dir}/data/X_ds.npy")            # (N,8,11)
        X_3d = np.load(f"{self.data_dir}/data/X3d_det_motion.npy")   # (N,8,9)
        Y = np.load(f"{self.data_dir}/data/Y3d_det.npy")[:, :self.pred_len, :]   # (N,8,2) metros
        ids = np.load(f"{self.data_dir}/data/subtrack_ids_ds.npy", allow_pickle=True)

        # Concatena las características: (N,8,11) + (N,8,9) -> (N,8,20)
        X = np.concatenate([X_img, X_3d], axis=2)
        print(f"{self.x_print} {X.shape}  Y(metros): {Y.shape}")

        # Partición oficial
        sck = np.array([s.rsplit("_", 2)[0] for s in ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]
        # Guarda las características 3D para la evaluación
        self.X3d_va = X_3d[vm]

    def save_outputs(self):
        os.makedirs(f"{self.data_dir}/prediccion_3d/figuras", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_3d/history", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_3d/history/{self.hist_name}", self.history.history)
        # Hueco final train-val
        h = self.history.history
        gap = h["val_loss"][-1] - h["loss"][-1]
        plt.figure(figsize=(7, 5))
        plt.plot(h["loss"], label="Entrenamiento")
        plt.plot(h["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel(f"Loss ({self.loss_label}, metros)")
        plt.title(self.fig_title)
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/{self.fig_name}", dpi=130); plt.close()
        print(f"Guardado: figuras/{self.fig_name}, {self.hist_name}  | hueco train-val final: {gap:.4f}")

    def evaluate(self):
        # Predicción de desplazamientos relativos (en metros)
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Evaluación
        ade, fde = ade_fde(pred, self.X3d_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV del 00

        # Comparación con respecto a CV
        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<38} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*74}")
        print(f"{'CV':<38} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'LSTM con fusión temprana (07)':<38} {REF_EARLY:>8.4f} {REF_EARLY_FDE:>8.4f} "
              f"{(1-REF_EARLY/ade_cv)*100:>7.1f}% {(1-REF_EARLY_FDE/fde_cv)*100:>7.1f}%")
        print(f"{self.model_label:<38} {ade:>8.4f} {fde:>8.4f} "
              f"{(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")


if __name__ == "__main__":
    trainer = Combined3DFTempRegTrainer()
    trainer.run()
