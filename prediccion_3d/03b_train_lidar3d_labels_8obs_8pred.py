"""
Techo ideal: modelo que recibe etiquetas 3D en metros -> entrada de 9 características.
8 OBS / 8 PRED
"""

from lidar3d_trainer import Lidar3DTrainer


class Lidar3DLabels8obs8predTrainer(Lidar3DTrainer):

    """Lidar3DLabels8obs8predTrainer entrena el modelo LSTM con etiquetas 3D."""
    def __init__(self, **kw):
        super().__init__(
            x_file="X3d_motion_8obs_8pred.npy", y_file="Y3d_8obs_8pred.npy",
            ckpt_name="lidar3d_labels_8obs_8pred_best.keras",
            hist_name="hist_lidar3d_labels_8obs_8pred.npy",
            fig_name="loss_lidar3d_labels_8obs_8pred.png",
            fig_title="Curva de entrenamiento — LiDAR 3D etiquetas (8 obs / 8 pred)",
            cv_key="8_8",
            x_print="X(3D etiquetas 8/8):",
            eval_header="===== LiDAR 3D GT 8/8 =====",
            model_label="LiDAR 3D GT 8/8",
            obs_len=8, pred_len=8, **kw)


if __name__ == "__main__":

    trainer = Lidar3DLabels8obs8predTrainer()
    trainer.run()
