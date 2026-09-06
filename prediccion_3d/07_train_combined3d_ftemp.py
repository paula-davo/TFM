"""
Modelo que combina datos de imagen y de detecciones 3D con fusión temprana.
Entrada: 11 imagen + 9 detecciones = 20 características. Predice el desplazamiento relativo en metros.
"""

import sys
import numpy as np
from lidar3d_trainer import Lidar3DTrainer, DATA_DIR

sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import ade_fde, load_cv_ref

N_IMG, N_3D = 11, 9
N_FEATURES = N_IMG + N_3D   # 20
REF_3D_DET = 0.2067   # LSTM con detecciones 3D a 8/8, pérdida MAE (04e)
REF_3D_FDE = 0.3102


class Combined3DFTempTrainer(Lidar3DTrainer):

    """
    Combined3DFTempTrainer entrena el modelo con información de entrada 2D + 3D con fusión temprana.
    Se concatenan las 11 características de imagen con las 9 de detección 3D, creando 20 características
    de entrada. Predice el desplazamiento futuro en metros.
    """

    def __init__(self, ref_3d_det=REF_3D_DET, ref_3d_fde=REF_3D_FDE, **kw):
        super().__init__(
            x_file="X_ds.npy", y_file="Y3d_det.npy",   # (load_data propio concatena imagen + 3D)
            ckpt_name="combined3d_best.keras",
            hist_name="hist_combined3d.npy",
            fig_name="loss_combined3d.png",
            fig_title="Curva de entrenamiento — Fusión temprana imagen+3D",
            cv_key="8_8",
            x_print="X(imagen+3D):",
            eval_header="===== Combinado imagen + detecciones 3D, Fusión temprana =====",
            model_label="Imagen + 3D (temprana)",
            obs_len=8, pred_len=8, n_features=N_FEATURES,
            loss="mae", metrics=("mse",), loss_label="MAE", **kw)
        # Referencia del modelo de detecciones 3D (04e) para comparar en la tabla
        self.ref_3d_det = ref_3d_det
        self.ref_3d_fde = ref_3d_fde
        # Parte 3D de val (metros), para el CV baseline y las métricas
        self.X3d_va = None

    def load_data(self):
        # Carga datos -> entrada de imagen 2D (11 características) + entrada de detecciones 3D (9 características) + salida 3D (en metros)
        X_img = np.load(f"{self.data_dir}/data/X_ds.npy")            # (N,8,11) imagen 2D
        X_3d = np.load(f"{self.data_dir}/data/X3d_det_motion.npy")   # (N,8,9) detecciones 3D
        Y = np.load(f"{self.data_dir}/data/Y3d_det.npy")[:, :self.pred_len, :]   # (N,8,2) metros (8/8)
        subtrack_ids = np.load(f"{self.data_dir}/data/subtrack_ids_ds.npy", allow_pickle=True)

        # Concatena las características: (N,8,11) + (N,8,9) -> (N,8,20)
        X = np.concatenate([X_img, X_3d], axis=2)
        print(f"{self.x_print} {X.shape}  Y(metros): {Y.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)

        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]

        # Guarda las características 3D para la evaluación
        self.X3d_va = X_3d[vm]

    def evaluate(self):
        # Predicción de desplazamientos relativos (en metros)
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Evaluación: modelos con detecciones 3D e info 2D vs CV (metros)
        ade, fde = ade_fde(pred, self.X3d_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV del 00

        # Comparación con respecto a CV
        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'detecciones-3D (base)':<28} {self.ref_3d_det:>8.4f} {self.ref_3d_fde:>8.4f} "
              f"{(1-self.ref_3d_det/ade_cv)*100:>7.1f}% {(1-self.ref_3d_fde/fde_cv)*100:>7.1f}%")
        print(f"{self.model_label:<28} {ade:>8.4f} {fde:>8.4f} "
              f"{(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")


if __name__ == "__main__":

    trainer = Combined3DFTempTrainer()
    trainer.run()
