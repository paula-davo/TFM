"""
Técnicas de generalización: reflejo lateral.
Parte del LSTM con detecciones 3D a 8/8 con pérdida MAE (04e) y duplica el conjunto de
entrenamiento con el reflejo lateral de los datos.
"""

import sys
import numpy as np
import joblib
from lidar3d_trainer import Lidar3DTrainer, DATA_DIR

sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import ade_fde, load_cv_ref, flip_lat

REF_DET = 0.2067   # LSTM con detecciones 3D a 8/8, pérdida MAE (04e)
REF_FDE = 0.3102


class Lidar3DDetFlipTrainer(Lidar3DTrainer):

    """Lidar3DDetFlipTrainer entrenar el LSTM con detecciones 3D aplicando aumento por reflejo lateral."""

    def __init__(self, ref_det=REF_DET, ref_fde=REF_FDE, **kw):
        super().__init__(
            x_file="X3d_det_motion.npy", y_file="Y3d_det.npy",
            ckpt_name="lidar3d_det_flip_best.keras",
            hist_name="hist_lidar3d_det_flip.npy",
            fig_name="loss_lidar3d_det_flip.png",
            fig_title="Curva de entrenamiento — LiDAR 3D det + reflejo lateral",
            cv_key="8_8",
            x_print="X(3D det):",
            eval_header="===== LiDAR 3D detecciones + reflejo lateral =====",
            model_label="3D det + reflejo",
            obs_len=8, pred_len=8, n_features=9,
            loss="mae", metrics=("mse",), loss_label="MAE", **kw)
        # Referencia del 04e para comparar en la tabla
        self.ref_det = ref_det
        self.ref_fde = ref_fde

    def load_data(self):
        # Detecciones 3D en metros (entrada) y desplazamiento GT desde la última detección (salida)
        X = np.load(f"{self.data_dir}/X3d_det_motion.npy")   # (N,8,9) metros, detecciones
        Y = np.load(f"{self.data_dir}/Y3d_det.npy")[:, :self.pred_len, :]   # (N,8,2) metros (8/8)
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"{self.x_print} {X.shape}")

        # Media lateral: si se aleja de 0 hay un sesgo izquierda/derecha que el reflejo eliminaría
        print(f"media cx (profundidad): {X[:, :, 0].mean():+.3f} m   |   media cy (lateral): {X[:, :, 1].mean():+.3f} m")

        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)
        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]

    def augment(self):
        # Aumento por reflejo lateral: duplica el conjunto de entrenameinto
        n0 = self.X_tr.shape[0]
        Xf, Yf = flip_lat(self.X_tr, self.Y_tr)
        self.X_tr = np.concatenate([self.X_tr, Xf], axis=0)
        self.Y_tr = np.concatenate([self.Y_tr, Yf], axis=0)
        print(f"Train: {n0} -> {self.X_tr.shape[0]} (con reflejo)   Val: {self.X_va.shape[0]}")

    def scale_features(self):
        # Scaler (reutiliza el de la base) y lo guarda para la inferencia posterior
        super().scale_features()
        joblib.dump(self.scaler, f"{self.data_dir}/lidar3d_det_flip_scaler.pkl")

    def evaluate(self):
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV del 00

        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'3D det (04e)*':<28} {self.ref_det:>8.4f} {self.ref_fde:>8.4f} "
              f"{(1-self.ref_det/ade_cv)*100:>7.1f}% {(1-self.ref_fde/fde_cv)*100:>7.1f}%")
        print(f"{self.model_label:<28} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")
        print(f"\nMejora del reflejo lateral sobre el CV: {(1-ade/ade_cv)*100:+.1f}% ADE  {(1-fde/fde_cv)*100:+.1f}% FDE")

    def run(self):
        # 1. Carga de datos
        self.load_data()
        # 2. Aumento por reflejo lateral
        self.augment()
        # 3. Scaler
        self.scale_features()
        # 4. Modelo
        self.build_model()
        # 5. Entrenamiento
        self.train()
        # 6. Salidas
        self.save_outputs()
        # 7. Evaluación
        self.evaluate()


if __name__ == "__main__":

    trainer = Lidar3DDetFlipTrainer()
    trainer.run()
