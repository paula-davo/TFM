"""
Técnicas de generalización: aplicación de ruido gaussiano en la entrada.
LSTM de detecciones con pérdida MAE (04e) y 8/8 al cual se le añade ruido.
"""

import os
import sys
import numpy as np
import joblib
from lidar3d_trainer import Lidar3DTrainer, DATA_DIR

sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import ade_fde, load_cv_ref

NOISE = 0.05
REF_DET = 0.2067   # LSTM con detecciones 3D a 8/8, pérdida MAE (04e)
REF_FDE = 0.3102


class Lidar3DDetNoiseTrainer(Lidar3DTrainer):

    """Lidar3DDetNoiseTrainer entrena el LSTM con detecciones 3D aplicando ruido gaussiano en la entrada."""
    def __init__(self, ref_det=REF_DET, ref_fde=REF_FDE, **kw):
        super().__init__(
            x_file="X3d_det_motion.npy", y_file="Y3d_det.npy",
            ckpt_name="lidar3d_det_noise_best.keras",
            hist_name="hist_lidar3d_det_noise.npy",
            fig_name="loss_lidar3d_det_noise.png",
            fig_title="Curva de entrenamiento — LiDAR 3D det + ruido gaussiano",
            cv_key="8_8",
            x_print="X(3D det):",
            eval_header="===== LiDAR 3D detecciones + ruido gaussiano =====",
            model_label="3D det + ruido",
            obs_len=8, pred_len=8, n_features=9,
            loss="mae", metrics=("mse",), loss_label="MAE", noise=NOISE, **kw)
        # Referencia del 04e para comparar en la tabla
        self.ref_det = ref_det
        self.ref_fde = ref_fde

    def load_data(self):
        # Detecciones 3D en metros (entrada) y desplazamiento GT desde la última detección (salida)
        X = np.load(f"{self.data_dir}/data/X3d_det_motion.npy")   # (N,8,9) metros, detecciones
        Y = np.load(f"{self.data_dir}/data/Y3d_det.npy")[:, :self.pred_len, :]   # (N,8,2) metros (8/8)
        subtrack_ids = np.load(f"{self.data_dir}/data/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"{self.x_print} {X.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)
        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]
        print(f"Train: {self.X_tr.shape[0]} (el ruido no aumenta muestras)   Val: {self.X_va.shape[0]}")

    def scale_features(self):
        # Scaler (reutiliza el de la base) y lo guarda para la inferencia posterior
        super().scale_features()
        os.makedirs(f"{self.data_dir}/scalers", exist_ok=True)
        joblib.dump(self.scaler, f"{self.data_dir}/scalers/lidar3d_det_noise_scaler.pkl")

    def evaluate(self):
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV (detecciones) fijo del 00; no se recalcula

        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'3D det (04e)*':<28} {self.ref_det:>8.4f} {self.ref_fde:>8.4f} "
              f"{(1-self.ref_det/ade_cv)*100:>7.1f}% {(1-self.ref_fde/fde_cv)*100:>7.1f}%")
        print(f"{self.model_label:<28} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")
        print(f"\nMejora del ruido gaussiano sobre el CV: {(1-ade/ade_cv)*100:+.1f}% ADE  {(1-fde/fde_cv)*100:+.1f}% FDE")


if __name__ == "__main__":

    trainer = Lidar3DDetNoiseTrainer()
    trainer.run()
