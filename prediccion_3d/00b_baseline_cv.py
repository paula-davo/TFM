"""
Paso 00 - Referencia de Velocidad Constante (CV) para la predicción 3D, en los tres horizontes.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks           # noqa: E402
from lstm_common import ade_fde, cv_baseline      # noqa: E402

HORIZONS = {
    "8_8":  ("X3d_det_8obs_8pred.npy",  "Y3d_det_8obs_8pred.npy",  8,  "8 obs / 8 pred"),
    "12_8": ("X3d_det_12obs_8pred.npy", "Y3d_det_12obs_8pred.npy", 8,  "12 obs / 8 pred"),
    "8_12": ("X3d_det_motion.npy",      "Y3d_det.npy",             12, "8 obs / 12 pred"),
}

class CVBaseline3D:

    """
    CVBaseline3D calcula la referencia de Velocidad Constante (CV) 3D en los tres horizontes.

    1) Calcula el CV para 8/8, 12/8 y 8/12
    2) Guarda los resultados en cv_baselines.json para que el resto de scripts no recalculen el CV
    """

    def __init__(self, data_dir=DATA_DIR, horizons=HORIZONS):
        # Inicializa atributos con directorio de datos y horizontes de predicción a calcular
        self.data_dir = data_dir
        self.horizons = horizons

        # Resultados
        self.results = {}

    def val_split(self, xf, yf):
        # Validación oficial del dataset de detecciones 3D
        X = np.load(f"{self.data_dir}/data/{xf}")
        Y = np.load(f"{self.data_dir}/data/{yf}")
        ids = np.load(f"{self.data_dir}/data/subtrack_ids_ds.npy", allow_pickle=True)
        sck = np.array([s.rsplit("_", 2)[0] for s in ids])
        _, vm = official_masks(sck)
        return X[vm], Y[vm]

    def compute_baselines(self):
        self.results = {}
        print("\nBASELINE CV 3D por horizonte")
        print(f"{'Horizonte':<18} {'ADE':>8} {'FDE':>8}")
        print("-" * 38)
        # Para cada horizonte, obtiene el split de validación, calcula el CV
        # y guarda los resultados
        for key, (xf, yf, pred_len, label) in self.horizons.items():
            X_va, Y_va = self.val_split(xf, yf)
            ade, fde = cv_baseline(X_va, Y_va, pred_len)
            self.results[key] = {"ade": float(ade), "fde": float(fde)}
            print(f"{label:<18} {ade:>8.4f} {fde:>8.4f}")
        return self.results

    def save_results(self):
        # Guarda los resultados en un archivo JSON
        out_json = f"{self.data_dir}/prediccion_3d/cv_baselines.json"
        with open(out_json, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nGuardado: prediccion_3d/cv_baselines.json")

    def run(self):
        # 1. CV para los 3 horizontes
        self.compute_baselines()
        # 2. Guarda los resultados en cv_baselines.json
        self.save_results()


if __name__ == "__main__":

    baseline = CVBaseline3D()
    baseline.run()
