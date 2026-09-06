"""
Techo ideal: modelo que recibe etiquetas 3D en metros -> entrada de 9 características.
8 OBS / 12 PRED
"""

from lidar3d_trainer import Lidar3DTrainer


class Lidar3DMotionTrainer(Lidar3DTrainer):

    """
    Lidar3DMotionTrainer entrena el modelo LSTM con etiquetas 3D.
    """
    def __init__(self, **kw):
        super().__init__(
            x_file="X3d_motion.npy", y_file="Y3d.npy",
            ckpt_name="lidar3d_motiononly_best.keras",
            hist_name="hist_lidar3d_motiononly.npy",
            fig_name="loss_lidar3d_motiononly.png",
            fig_title="Curva de entrenamiento — LiDAR 3D GT",
            cv_key="8_12",
            x_print="X(3D motion only):",
            eval_header="===== LiDAR 3D GT 8/12 =====",
            model_label="LiDAR 3D GT 8/12",
            obs_len=8, pred_len=12, **kw)


if __name__ == "__main__":

    trainer = Lidar3DMotionTrainer()
    trainer.run()
