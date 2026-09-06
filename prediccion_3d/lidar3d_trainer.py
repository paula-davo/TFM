"""
Entrenamiento LSTM con etiquetas o con detecciones 3D: 03, 03b, 03c, 04, 04b y 04c.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val, ade_fde, load_cv_ref

N_FEATURES = 9
DROPOUT, REC_DROPOUT = 0.3, 0.2


class Lidar3DTrainer:

    """
    Lidar3DTrainer entrena el LSTM 3D (con etiquetas o con detecciones). La configuración concreta
    (ficheros, horizonte y textos) la fija cada subclase.
    """

    def __init__(self, x_file, y_file, ckpt_name, hist_name, fig_name, fig_title,
        cv_key, x_print, eval_header, model_label,
        obs_len, pred_len, data_dir=DATA_DIR, n_features=N_FEATURES,
        lstm_units=128, dropout=DROPOUT, rec_dropout=REC_DROPOUT,
        batch_size=256, epochs=120, loss="mse", metrics=("mae",), loss_label="MSE",
        noise=0.0, optimizer="adam", es_patience=10, rlr_patience=5):
        # Configuración del horizonte y del experimento
        self.x_file = x_file
        self.y_file = y_file
        self.ckpt_name = ckpt_name
        self.hist_name = hist_name
        self.fig_name = fig_name
        self.fig_title = fig_title
        self.cv_key = cv_key
        self.x_print = x_print
        self.eval_header = eval_header
        self.model_label = model_label
        # Pérdida que se optimiza y métrica
        self.loss = loss
        self.metrics = list(metrics)
        self.loss_label = loss_label
        # Ruido gaussiano en la entrada (0.0 -> sin capa)
        self.noise = noise
        # Optimizador y paciencias de los callbacks
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.rlr_patience = rlr_patience

        # Hiperparámetros
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos, scaler, modelo e historial
        self.X_tr = self.X_va = self.Y_tr = self.Y_va = None
        self.X_tr_s = self.X_va_s = None
        self.scaler = self.model = self.history = None

    def load_data(self):
        # Carga los datos (X, Y) en metros y aplica la partición oficial
        X = np.load(f"{self.data_dir}/{self.x_file}")
        Y = np.load(f"{self.data_dir}/{self.y_file}")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids_ds.npy", allow_pickle=True)
        print(f"{self.x_print} {X.shape}")

        # Partición oficial (el subtrack_id 3D lleva sufijo _cam_track: se recorta al id de secuencia)
        sck = np.array([s.rsplit("_", 2)[0] for s in subtrack_ids])
        tm, vm = official_masks(sck)
        self.X_tr, self.X_va = X[tm], X[vm]
        self.Y_tr, self.Y_va = Y[tm], Y[vm]

    def scale_features(self):
        # Scaler
        self.X_tr_s, self.X_va_s, self.scaler = scale_train_val(self.X_tr, self.X_va, self.n_features)

    def build_model(self):
        # Se define el modelo -> LSTM con recurrent dropout
        self.model = build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                        units=self.lstm_units, dropout=self.dropout,
                        rec_dropout=self.rec_dropout, noise=self.noise, optimizer=self.optimizer,
                        loss=self.loss, metrics=self.metrics, named=True)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        # Se definen los callbacks
        os.makedirs(f"{self.data_dir}/prediccion_3d/best", exist_ok=True)
        cbs = [EarlyStopping(monitor="val_loss", patience=self.es_patience, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=self.rlr_patience, min_lr=1e-6),
               ModelCheckpoint(f"{self.data_dir}/prediccion_3d/best/{self.ckpt_name}",
                               monitor="val_loss", save_best_only=True)]
        # Se entrena el modelo midiendo el tiempo de entrenamiento
        t0 = time.time()
        self.history = self.model.fit(self.X_tr_s, self.Y_tr,
                                      validation_data=(self.X_va_s, self.Y_va),
                                      epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_3d/figuras", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_3d/history", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_3d/history/{self.hist_name}", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel(f"Loss ({self.loss_label}, metros)")
        plt.title(self.fig_title)
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/{self.fig_name}", dpi=130); plt.close()
        print(f"Guardado: figuras/{self.fig_name}, {self.hist_name}")

    def evaluate(self):
        # Predicción de desplazamientos relativos (en metros)
        pred = self.model.predict(self.X_va_s, batch_size=512, verbose=0)
        # Métricas: modelo LSTM 3D vs CV
        ade, fde = ade_fde(pred, self.X_va, self.Y_va)
        ade_cv, fde_cv = load_cv_ref(self.data_dir, self.cv_key)   # CV del 00

        # Comparación: LSTM 3D vs CV
        print(f"\n{self.eval_header}")
        print(f"{'Modelo':<28} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*64}")
        print(f"{'CV':<28} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{self.model_label:<28} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {(1-fde/fde_cv)*100:>7.1f}%")

    def run(self):
        # 1. Carga de datos
        self.load_data()
        # 2. Scaler
        self.scale_features()
        # 3. Modelo
        self.build_model()
        # 4. Entrenamiento
        self.train()
        # 5. Salidas
        self.save_outputs()
        # 6. Evaluación
        self.evaluate()
