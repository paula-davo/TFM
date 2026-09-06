"""
Entrenamiento LSTM 8 observaciones / 8 predicciones, con detecciones 3D.
Se cambia la función de pérdida a MAE - mejores resultados. Análogo al 04b con este cambio.
"""

import sys
from lidar3d_trainer import Lidar3DTrainer, DATA_DIR

sys.path.insert(0, DATA_DIR)
from lstm_common import ade_fde, load_cv_ref

REF_DET = 0.2118   # ADE del 04b
REF_FDE = 0.3173   # FDE del 04b


class Lidar3DDetMaeTrainer(Lidar3DTrainer):

    """Lidar3DDetMaeTrainer entrena el LSTM 3D con detecciones (8/8) usando pérdida MAE."""
    def __init__(self, ref_det=REF_DET, ref_fde=REF_FDE, **kw):
        super().__init__(
            x_file="X3d_det_8obs_8pred.npy", y_file="Y3d_det_8obs_8pred.npy",
            ckpt_name="lidar3d_det_mae_best.keras",
            hist_name="hist_lidar3d_det_mae.npy",
            fig_name="loss_lidar3d_det_mae.png",
            fig_title="Curva de entrenamiento — LiDAR 3D detecciones 8/8, pérdida MAE",
            cv_key="8_8",
            x_print="X(3D detecciones 8/8):",
            eval_header="===== LiDAR 3D detecciones 8/8 con pérdida MAE =====",
            model_label="3D det 8/8 (MAE)",
            obs_len=8, pred_len=8,
            loss="mae", metrics=("mse",), loss_label="MAE", **kw)
        # Referencias del 04b (MSE) para comparar la mejora de la pérdida en la tabla
        self.ref_det = ref_det
        self.ref_fde = ref_fde

    def evaluate(self):
        # Evaluación
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV del 00

        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'3D det 8/8 (MSE, 04b)*':<28} {self.ref_det:>8.4f} {self.ref_fde:>8.4f} "
              f"{(1-self.ref_det/ade_cv)*100:>7.1f}% {(1-self.ref_fde/fde_cv)*100:>7.1f}%")
        print(f"{self.model_label:<28} {ade:>8.4f} {fde:>8.4f} "
              f"{(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")
        print(f"\nMejora de la pérdida MAE sobre el CV: {(1-ade/ade_cv)*100:+.1f}% ADE  {(1-fde/fde_cv)*100:+.1f}% FDE")


if __name__ == "__main__":

    trainer = Lidar3DDetMaeTrainer()
    trainer.run()
