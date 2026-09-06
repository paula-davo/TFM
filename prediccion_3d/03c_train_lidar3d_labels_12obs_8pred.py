"""
Techo ideal: modelo que recibe etiquetas 3D en metros -> entrada de 9 características.
12 OBS / 8 PRED
"""

from lidar3d_trainer import Lidar3DTrainer


class Lidar3DLabels12obs8predTrainer(Lidar3DTrainer):

    """Lidar3DLabels12obs8predTrainer entrena el modelo LSTM con etiquetas 3D."""
    def __init__(self, **kw):
        super().__init__(
            x_file="X3d_motion_12obs_8pred.npy", y_file="Y3d_12obs_8pred.npy",
            ckpt_name="lidar3d_labels_12obs_8pred_best.keras",
            hist_name="hist_lidar3d_labels_12obs_8pred.npy",
            fig_name="loss_lidar3d_labels_12obs_8pred.png",
            fig_title="Curva de entrenamiento — LiDAR 3D etiquetas (12 obs / 8 pred)",
            cv_key="12_8",
            x_print="X(3D etiquetas 12/8):",
            eval_header="===== LiDAR 3D GT 12/8 =====",
            model_label="LiDAR 3D GT 12/8",
            obs_len=12, pred_len=8, **kw)


if __name__ == "__main__":

    trainer = Lidar3DLabels12obs8predTrainer()
    trainer.run()
