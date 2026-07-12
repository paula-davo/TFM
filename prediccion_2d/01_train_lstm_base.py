import os
import sys
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)             # split_utils y lstm_common viven en la raíz del TFM
from split_utils import official_masks
from lstm_common import build_seq2seq, scale_train_val

OBS_LEN = 8
PRED_LEN = 12
N_FEATURES = 11


class LSTMBaseTrainer:

    """
    LSTMBaseTrainer entrena el modelo LSTM base de predicción 2D.

    1) Carga X_train/Y_train y aplica la partición oficial (train/val)
    2) Estandariza las características (fit solo en train)
    3) Define y entrena un encoder-decoder LSTM
    4) GUARDA modelo, scaler, curva de pérdida y tiempos

    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 n_features=N_FEATURES, lstm_units=128, dropout=0.2,
                 batch_size=256, epochs=100):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos, scaler, modelo e historial
        self.X_train = self.Y_train = None
        self.X_val = self.Y_val = None
        self.X_train_scaled = self.X_val_scaled = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga las muestras
        X = np.load(f"{self.data_dir}/X_train.npy")
        Y = np.load(f"{self.data_dir}/Y_train.npy")
        subtrack_ids = np.load(f"{self.data_dir}/subtrack_ids.npy", allow_pickle=True)
        print("X:", X.shape)
        print("Y:", Y.shape)

        # Aplica partición oficial
        train_mask, val_mask = official_masks(subtrack_ids)

        self.X_train, self.Y_train = X[train_mask], Y[train_mask]
        self.X_val, self.Y_val = X[val_mask], Y[val_mask]
        print(f"\nTrain: {self.X_train.shape[0]} muestras")
        print(f"Val:   {self.X_val.shape[0]} muestras")

    def scale_features(self):
        # Estandariza (fit solo en train) usando lstm_common y guarda el scaler
        self.X_train_scaled, self.X_val_scaled, self.scaler = scale_train_val(
            self.X_train, self.X_val, self.n_features)

        joblib.dump(self.scaler, f"{self.data_dir}/feature_scaler.pkl")
        print("\nScaler guardado.")

    def build_model(self):
        # Arquitectura seq2seq (definida en lstm_common), con los nombres del modelo base
        self.model = build_seq2seq(self.obs_len, self.pred_len, self.n_features,
                                   units=self.lstm_units, dropout=self.dropout,
                                   optimizer="adam", metrics=["mae"], named=True)
        self.model.summary()
        return self.model

    def train(self):
        # Entrena con EarlyStopping, ReduceLROnPlateau y ModelCheckpoint
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint(filepath=f"{self.data_dir}/prediccion_2d/best/lstm_best.keras", monitor="val_loss", save_best_only=True),
        ]

        # Entrenamiento        
        t0 = time.time()
        self.history = self.model.fit(
            self.X_train_scaled, self.Y_train,
            validation_data=(self.X_val_scaled, self.Y_val),
            epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])

        # Tiempo de inferencia (se descarta la primera predicción)
        _ = self.model.predict(self.X_val_scaled[:256], verbose=0)
        t0 = time.time()
        _ = self.model.predict(self.X_val_scaled, batch_size=512, verbose=0)
        t_inf_ms = (time.time() - t0) * 1000 / len(self.X_val_scaled)

        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")
        print(f"Inferencia: {t_inf_ms:.3f} ms/muestra")

    def save_outputs(self):
        # Guarda la curva de pérdida, el historial y el modelo final
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/models", exist_ok=True)
        # Historia
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_base.npy", self.history.history)
        # Figura
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
        plt.title("Curva de entrenamiento — LSTM base")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_base.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_base.png, hist_base.npy")

        # Modelo
        self.model.save(f"{self.data_dir}/models/lstm_baseline.keras")
        print("\nModelo guardado.")

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


if __name__ == "__main__":

    trainer = LSTMBaseTrainer()
    trainer.run()
