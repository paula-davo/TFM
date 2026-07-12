import os
import sys
import time
import numpy as np
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import scale_train_val, ade_fde, cv_baseline
from social_lstm_model import SocialLSTM

OBS_LEN, PRED_LEN, MAX_PEDS, N_FEATURES = 8, 12, 10, 11
LSTM_UNITS, SOCIAL_RADIUS = 128, 2.0
LSTM_LARGO_ADE, LSTM_LARGO_FDE = 0.028178, 0.051457   # Resultados LSTM con aumento del Horizonte


class SocialLSTMHorizonTrainer:

    """
    SocialLSTMHorizonTrainer entrena el Social LSTM sobre el dataset social submuestreado
    (horizonte largo) y lo compara con el LSTM base con aumento de horizonte y el baseline CV.
    """

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 max_peds=MAX_PEDS, n_features=N_FEATURES, lstm_units=LSTM_UNITS,
                 social_radius=SOCIAL_RADIUS, batch_size=64, epochs=100):
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_peds = max_peds
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.social_radius = social_radius
        self.batch_size = batch_size
        self.epochs = epochs

        # Datos de train/val (X entrada, Y salida, M máscara, P posición 3D)
        self.Xtr = self.Xva = None
        self.Ytr = self.Yva = None
        self.Mtr = self.Mva = None
        self.Ptr = self.Pva = None
        self.Xtr_s = self.Xva_s = None
        self.scaler = None
        self.model = None
        self.history = None

    def load_data(self):
        # Carga el dataset social submuestreado y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_social_ds.npy")
        Y = np.load(f"{self.data_dir}/Y_social_ds.npy")
        mask = np.load(f"{self.data_dir}/mask_social_ds.npy")
        pos3d = np.load(f"{self.data_dir}/pos3d_social_ds.npy")
        keys = np.load(f"{self.data_dir}/scene_keys_ds.npy", allow_pickle=True)
        print(f"X: {X.shape}")

        # Split oficial por escena (scene_keys -> secuencia)
        train_mask, val_mask = official_masks(keys)
        self.Xtr, self.Xva = X[train_mask], X[val_mask]
        self.Ytr, self.Yva = Y[train_mask], Y[val_mask]
        self.Mtr, self.Mva = mask[train_mask], mask[val_mask]
        self.Ptr, self.Pva = pos3d[train_mask], pos3d[val_mask]
        print(f"Train: {self.Xtr.shape[0]}  Val: {self.Xva.shape[0]}")

    def scale_features(self):
        # Aplica el scaler y lo guarda
        self.Xtr_s, self.Xva_s, self.scaler = scale_train_val(
            self.Xtr, self.Xva, self.n_features)
        joblib.dump(self.scaler, f"{self.data_dir}/ds_social_scaler.pkl")

    def build_model(self):
        # Crea el modelo SocialLSTM
        self.model = SocialLSTM(self.max_peds, self.obs_len, self.pred_len,
                                self.n_features, self.lstm_units, self.social_radius)
        # Compila el modelo
        self.model.compile(optimizer=tf.keras.optimizers.Adam())
        # Construye los pesos con un batch de prueba (entrada = (X, pos3d, máscara))
        _ = self.model((self.Xtr_s[:2].astype(np.float32),
                        self.Ptr[:2].astype(np.float32),
                        self.Mtr[:2]), training=False)
        # Summary
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_2d/best", exist_ok=True)

        # Datasets tf: la entrada del modelo es (X escalada, pos3d, máscara)
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((self.Xtr_s.astype(np.float32), self.Ptr.astype(np.float32), self.Mtr),
             self.Ytr.astype(np.float32))
        ).shuffle(8000, seed=42).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(
            ((self.Xva_s.astype(np.float32), self.Pva.astype(np.float32), self.Mva),
             self.Yva.astype(np.float32))
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Define los callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_2d/best/ds_social_best.weights.h5",
                            monitor="val_loss", save_best_only=True, save_weights_only=True),
        ]
        # Entrena midiendo el tiempo de entrenamiento
        t0 = time.time()
        self.history = self.model.fit(train_ds, validation_data=val_ds,
                                      epochs=self.epochs, callbacks=callbacks)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_2d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_2d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_2d/history/hist_ds_social.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MSE)")
        plt.title("Curva de entrenamiento — Social LSTM")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_2d/figuras/loss_ds_social.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_ds_social.png, history/hist_ds_social.npy")

    def evaluate(self):
        # Predicción por lotes: predice por lotes de 128 escenas
        pred = []
        for i in range(0, self.Xva.shape[0], 128):
            pred.append(self.model((self.Xva_s[i:i+128].astype(np.float32),
                                    self.Pva[i:i+128].astype(np.float32),
                                    self.Mva[i:i+128]), training=False).numpy())
        pred = np.concatenate(pred, axis=0)

        # Aplana los datos
        # (n, max_peds, 8, 11) -> (n*max_peds, 8, 11)
        mflat = self.Mva.reshape(-1).astype(bool)
        Xflat = self.Xva.reshape(-1, self.obs_len, self.n_features)[mflat]
        Yflat = self.Yva.reshape(-1, self.pred_len, 2)[mflat]
        pflat = pred.reshape(-1, self.pred_len, 2)[mflat]

        # Calcula métricas de modelo social y del cv baseline
        ade, fde = ade_fde(pflat, Xflat, Yflat)
        ade_cv, fde_cv = cv_baseline(Xflat, Yflat, self.pred_len)

        print("\n===== HORIZONTE LARGO: Social vs LSTM vs CV =====")
        print(f"{'Modelo':<22}  {'ADE':>10}  {'FDE':>10}  {'Mejora vs CV':>14}")
        print(f"{'-'*62}")
        print(f"{'CV baseline':<22}  {ade_cv:>10.6f}  {fde_cv:>10.6f}  {'(ref)':>14}")
        print(f"{'LSTM (solo mov)*':<22}  {LSTM_LARGO_ADE:>10.6f}  {LSTM_LARGO_FDE:>10.6f}  {(1-LSTM_LARGO_ADE/ade_cv)*100:>13.1f}%")
        print(f"{'Social LSTM':<22}  {ade:>10.6f}  {fde:>10.6f}  {(1-ade/ade_cv)*100:>13.1f}%")
        print("\n* LSTM-solo usa muestreo per-subtrack; el social usa rejilla global → CV puede diferir algo.")

    def run(self):
        # 1. Carga los datos
        self.load_data()
        # 2. Aplica scaler
        self.scale_features()
        # 3. Construye el modelo
        self.build_model()
        # 4. Entrena el modelo
        self.train()
        # 5. Guarda las salidas
        self.save_outputs()
        # 6. Evalúa el modelo
        self.evaluate()


if __name__ == "__main__":

    trainer = SocialLSTMHorizonTrainer()
    trainer.run()
