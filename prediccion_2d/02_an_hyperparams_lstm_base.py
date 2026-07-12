"""
Estudio de hiperparámetros del LSTM base (apartado 12.a — optimización de
hiperparámetros). Se parte de la configuración base y se varía UN parámetro cada vez
(unidades LSTM, dropout, optimizador, batch size), entrenando con el split oficial y
comparando ADE/FDE sobre validación.

Salida: tabla por consola + hparam_search_base.csv. El CV baseline se calcula una vez
como referencia (fila aparte).
"""
import csv
import time
import sys
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val, ade_fde, cv_baseline

OBS_LEN, PRED_LEN, N_FEATURES = 8, 12, 11


class AnalysisHyperparamLSTMBase:

    """
    AnalysisHyperparamLSTMBase estudia los hiperparámetros del LSTM base variando un
    parámetro cada vez.
    """

    # Configuraciones: base + una variación por parámetro
    CONFIGS = [
        dict(name="base (u128,d0.2,adam,b256)", units=128, dropout=0.2, optimizer="adam", batch=256),
        dict(name="units=64",                   units=64,  dropout=0.2, optimizer="adam", batch=256),
        dict(name="units=256",                  units=256, dropout=0.2, optimizer="adam", batch=256),
        dict(name="dropout=0.4",                units=128, dropout=0.4, optimizer="adam", batch=256),
        dict(name="optim=rmsprop",              units=128, dropout=0.2, optimizer="rmsprop", batch=256),
        dict(name="batch=64",                   units=128, dropout=0.2, optimizer="adam", batch=64),
    ]

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, epochs=60):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.epochs = epochs

        # Datos y resultados
        self.X_va = self.Y_va = None
        self.X_tr_s = self.X_va_s = self.Y_tr = None
        self.results = []
        self.ade_cv = self.fde_cv = None

    def load_data(self):
        # Carga, aplica el split oficial y estandariza (fit solo en train)
        X = np.load(f"{self.data_dir}/X_train.npy")
        Y = np.load(f"{self.data_dir}/Y_train.npy")
        ids = np.load(f"{self.data_dir}/subtrack_ids.npy", allow_pickle=True)
        tm, vm = official_masks(ids)
        X_tr, self.X_va, self.Y_tr, self.Y_va = X[tm], X[vm], Y[tm], Y[vm]

        self.X_tr_s, self.X_va_s, _ = scale_train_val(X_tr, self.X_va, self.n_features)

    def build_model(self, units, dropout, optimizer):
        # Arquitectura definida en lstm_common.py
        return build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                             units=units, dropout=dropout, optimizer=optimizer)

    def run_search(self):
        # Entrena y evalúa cada configuración
        self.results = []
        # Para cada configuración
        for i, cfg in enumerate(self.CONFIGS, 1):
            print(f"\n[{i}/{len(self.CONFIGS)}] {cfg['name']}")
            # Define el modelo y los callbacks
            model = self.build_model(cfg["units"], cfg["dropout"], cfg["optimizer"])
            cbs = [EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                   ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)]
            # Entrena
            t0 = time.time()
            h = model.fit(self.X_tr_s, self.Y_tr, validation_data=(self.X_va_s, self.Y_va),
                          epochs=self.epochs, batch_size=cfg["batch"], callbacks=cbs, verbose=2)
            t_train = time.time() - t0
            # Evalua
            pred = model.predict(self.X_va_s, batch_size=512, verbose=0)
            ade, fde = ade_fde(pred, self.X_va, self.Y_va)
            # Guarda resultados
            self.results.append({**cfg, "epocas": len(h.history["loss"]),
                                 "params": model.count_params(), "ade": ade, "fde": fde, "t_s": t_train})
            print(f"  -> ADE {ade:.5f}  FDE {fde:.5f}  ({len(h.history['loss'])} épocas, {t_train:.0f}s)")

    def report(self):
        # Tabla comparativa de modelos
        print("\n\n========== ESTUDIO DE HIPERPARÁMETROS (LSTM base, val oficial) ==========")
        print(f"{'Configuración':<28} {'params':>8} {'épocas':>7} {'ADE':>9} {'FDE':>9} {'mej.ADE':>8}")
        print("-" * 74)
        for r in self.results:
            print(f"{r['name']:<28} {r['params']:>8} {r['epocas']:>7} {r['ade']:>9.5f} {r['fde']:>9.5f} "
                  f"{(1-r['ade']/self.ade_cv)*100:>7.1f}%")
        print("-" * 74)
        print(f"{'CV baseline (referencia)':<28} {'-':>8} {'-':>7} {self.ade_cv:>9.5f} {self.fde_cv:>9.5f} {'ref':>8}")

    def save_csv(self):
        # Guarda los resultados en CSV
        path = f"{self.data_dir}/hparam_search_base.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "units", "dropout", "optimizer", "batch",
                                              "params", "epocas", "ade", "fde", "t_s"])
            w.writeheader(); w.writerows(self.results)
        print(f"\nGuardado: {path}")

    def run(self):
        # Flujo completo: carga -> búsqueda -> CV -> tabla -> guardado
        # 1. Carga de datos
        self.load_data()
        # 2. Entrenamiento y evaluación de todas las opciones
        self.run_search()
        # 3. Resultados de CV baseline
        self.ade_cv, self.fde_cv = cv_baseline(self.X_va, self.Y_va, self.pred_len)
        # 4. Reporte de resultados
        self.report()
        # 5. Guardado en CSV
        self.save_csv()


if __name__ == "__main__":

    search = AnalysisHyperparamLSTMBase()
    search.run()
