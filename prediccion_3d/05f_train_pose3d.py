"""
LSTM 3D con información de pose (17 características = 9 detecciones 3D y 8 de pose).
"""

import sys
import numpy as np
from lidar3d_trainer import Lidar3DTrainer, DATA_DIR

sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import ade_fde, load_cv_ref

# Resultado del LSTM de detecciones 3D (04e, MAE) para comparar.
REF_3D_ADE, REF_3D_FDE = 0.2067, 0.3102


class Lidar3DPoseTrainer(Lidar3DTrainer):

    """
    Lidar3DPoseTrainer entrena el LSTM con detecciones 3D (metros) + pose corporal.
    """
    def __init__(self, **kw):
        super().__init__(
            x_file="X_3d_pose_ds.npy", y_file="Y3d_det.npy",
            ckpt_name="pose3d_best.keras",
            hist_name="hist_pose3d.npy",
            fig_name="loss_pose3d.png",
            fig_title="Curva de entrenamiento — LSTM 3D + pose",
            cv_key="8_8",
            x_print="X(3D + pose):",
            eval_header="===== LiDAR 3D detecciones + POSE =====",
            model_label="3D det + pose",
            obs_len=8, pred_len=8, n_features=17,
            loss="mae", metrics=("mse",), loss_label="MAE", **kw)

    def load_data(self):
        # Carga los datos (detecciones 3D + pose) y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_3d_pose_ds.npy")
        # Y3d_det: GT desde la última posición detectada -> pareja de X detecciones
        Y = np.load(f"{self.data_dir}/Y3d_det.npy")[:, :self.pred_len, :]   # (N,8,2) metros (8/8)
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"{self.x_print} {X.shape}")

        # Partición oficial (el subtrack_id lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]

    def evaluate(self):
        # Predicción de desplazamientos relativos (en metros)
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Métricas: LSTM 3D + pose vs CV vs base MAE (04e)
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV (detecciones) fijo del 00; no se recalcula

        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        if REF_3D_ADE is not None:
            print(f"{'LSTM con MAE (04e)*':<28} {REF_3D_ADE:>8.4f} {REF_3D_FDE:>8.4f} {(1-REF_3D_ADE/ade_cv)*100:>7.1f}% {(1-REF_3D_FDE/fde_cv)*100:>7.1f}%")
        print(f"{self.model_label:<28} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")
        if REF_3D_ADE is not None:
            print(f"\nMejora de la pose sobre el CV: {(1-ade/ade_cv)*100:+.1f}% ADE  {(1-fde/fde_cv)*100:+.1f}% FDE")

if __name__ == "__main__":

    trainer = Lidar3DPoseTrainer()
    trainer.run()
