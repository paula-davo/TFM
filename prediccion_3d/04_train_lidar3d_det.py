"""
Modelo que recibe deteccciones 3D en metros -> entrada de 9 características.
"""

from lidar3d_trainer import Lidar3DTrainer


class Lidar3DDetTrainer(Lidar3DTrainer):

    """
    Lidar3DDetTrainer entrena el modelo LSTM con detecciones 3D.
    """

    def __init__(self, **kw):
        super().__init__(
            x_file="X3d_det_motion.npy", y_file="Y3d_det.npy",
            ckpt_name="lidar3d_det_best.keras",
            hist_name="hist_lidar3d_det.npy",
            fig_name="loss_lidar3d_det.png",
            fig_title="Curva de entrenamiento — LiDAR 3D detecciones",
            cv_key="8_12",
            x_print="X(3D detecciones):",
            eval_header="===== LiDAR 3D con DETECCIONES 8/12 =====",
            model_label="3D det 8/12",
            obs_len=8, pred_len=12, **kw)


if __name__ == "__main__":

    trainer = Lidar3DDetTrainer()
    trainer.run()
