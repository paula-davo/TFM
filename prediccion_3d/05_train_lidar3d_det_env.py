"""
Modelo LSTM con información lidar 3D: detecciones reales del sensor (9 características)
e información del entorno obtenida de la nube de puntos (7 características). Predice el
desplazamiento relativo en metros.
"""

import os
import sys
import numpy as np
import joblib
from lidar3d_trainer import Lidar3DTrainer, DATA_DIR

sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import ade_fde, load_cv_ref

N_MOTION, N_ENV = 9, 7
N_FEATURES = N_MOTION + N_ENV   # 16
REF_DET, REF_FDE = 0.2067, 0.3102   # LSTM 3D detecciones MAE (04e), mejor modelo hasta el momento


class Lidar3DDetEnvTrainer(Lidar3DTrainer):

    """
    Lidar3DDetEnvTrainer entrena el modelo LSTM con entrada de detecciones 3D + información del entorno.
    """

    def __init__(self, **kw):
        super().__init__(
            x_file="X3d_det_8obs_8pred.npy", y_file="Y3d_det_8obs_8pred.npy",   # (load_data propio los concatena con el entorno)
            ckpt_name="lidar3d_det_env_best.keras",
            hist_name="hist_lidar3d_det_env.npy",
            fig_name="loss_lidar3d_det_env.png",
            fig_title="Curva de entrenamiento — LiDAR 3D detecciones (mov+entorno)",
            cv_key="8_8",
            x_print="X(3D det+entorno):",
            eval_header="===== LiDAR 3D detecciones (trayectoria + entorno) =====",
            model_label="3D det (mov+entorno)",
            obs_len=8, pred_len=8, n_features=N_FEATURES,
            loss="mae", metrics=("mse",), loss_label="MAE", **kw)

    def load_data(self):
        # Carga los datos de entrada: detecciones 3D 8/8 (del 02b) + datos de entorno + flags
        Xm = np.load(f"{self.data_dir}/data/X3d_det_8obs_8pred.npy")   # (N,8,9) metros, detecciones (02b)
        Xe = np.load(f"{self.data_dir}/data/X_lidar_seq.npy")          # (N,8,7) entorno
        fe = np.load(f"{self.data_dir}/data/lidar_seq_flags.npy")      # (N,8)
        # Carga los datos de salida: desplazamiento relativo en metros (8/8, del 02b)
        Y = np.load(f"{self.data_dir}/data/Y3d_det_8obs_8pred.npy")    # (N,8,2) metros (8/8)
        # Carga de subtrack IDs para partición
        subtrack_ids = np.load(f"{self.data_dir}/data/subtrack_ids_ds.npy", allow_pickle=True)

        # Concatena las características de detecciones y entorno: (N,8,9) + (N,8,7) -> (N,8,16)
        X = np.concatenate([Xm, Xe], axis=2)
        print(f"{self.x_print} {X.shape}  Y(metros): {Y.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]

    def scale_features(self):
        # Scaler (reutiliza el de la base) y lo guarda para la inferencia posterior
        super().scale_features()
        os.makedirs(f"{self.data_dir}/scalers", exist_ok=True)
        joblib.dump(self.scaler, f"{self.data_dir}/scalers/lidar3d_det_env_scaler.pkl")

    def evaluate(self):
        # Evaluación del modelo
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Métricas: modelo de detecciones 3D + información entorno vs base MAE (04e) vs CV
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV del 00

        # Comparación: CV vs LSTM det MAE, 04e vs LSTM det + info entorno
        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'LSTM con MAE (04e)*':<28} {REF_DET:>8.4f} {REF_FDE:>8.4f} "
              f"{(1-REF_DET/ade_cv)*100:>7.1f}% {(1-REF_FDE/fde_cv)*100:>7.1f}%")
        print(f"{self.model_label:<28} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")


if __name__ == "__main__":

    trainer = Lidar3DDetEnvTrainer()
    trainer.run()
