"""
LSTM Social con detecciones 3D. Entrada de 9 características 3D.
"""

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
sys.path.insert(0, DATA_DIR)                        # split_utils y lstm_common (raíz)
from split_utils import official_masks
from lstm_common import scale_train_val, ade_fde, cv_baseline, load_cv_ref
from social_lstm_model import SocialLSTM

OBS_LEN, PRED_LEN, MAX_PEDS, N_FEATURES = 8, 8, 10, 9
LSTM_UNITS, SOCIAL_RADIUS = 128, 2.0    # radio social en metros (pos3d)

# Resultado del LSTM con detecciones 3D (04e, MAE) para comparar.
REF_DET_ADE, REF_DET_FDE = 0.2067, 0.3102


class SocialLSTM3DDetTrainer:

    """
    SocialLSTM3DDetTrainer entrena el LSTM Social sobre el dataset social 3D de detecciones (metros).
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
        # Carga el dataset social 3D de detecciones y aplica la partición oficial
        X = np.load(f"{self.data_dir}/X_social3d_det_ds.npy")
        Y = np.load(f"{self.data_dir}/Y_social3d_det_ds.npy")[:, :, :self.pred_len, :]   # (esc,peds,8,2) 8/8
        mask = np.load(f"{self.data_dir}/mask_social3d_det_ds.npy")
        pos3d = np.load(f"{self.data_dir}/pos3d_social3d_det_ds.npy")
        keys = np.load(f"{self.data_dir}/scene_keys3d_det_ds.npy", allow_pickle=True)
        print(f"X: {X.shape}")

        # Split oficial por escena (scene_keys -> secuencia)
        train_mask, val_mask = official_masks(keys)
        self.Xtr, self.Xva = X[train_mask], X[val_mask]
        self.Ytr, self.Yva = Y[train_mask], Y[val_mask]
        self.Mtr, self.Mva = mask[train_mask], mask[val_mask]
        self.Ptr, self.Pva = pos3d[train_mask], pos3d[val_mask]
        print(f"Train: {self.Xtr.shape[0]}  Val: {self.Xva.shape[0]}")

    def scale_features(self):
        # Aplica el scaler (9 features) y lo guarda
        self.Xtr_s, self.Xva_s, self.scaler = scale_train_val(
            self.Xtr, self.Xva, self.n_features)
        joblib.dump(self.scaler, f"{self.data_dir}/social3d_det_scaler.pkl")

    def build_model(self):
        # Crea el modelo SocialLSTM, con 9 features de entrada y pérdida MAE
        self.model = SocialLSTM(self.max_peds, self.obs_len, self.pred_len,
                                self.n_features, self.lstm_units, self.social_radius,
                                loss="mae")
        # Compila el modelo
        self.model.compile(optimizer=tf.keras.optimizers.Adam())
        # Construye los pesos con un batch de prueba (entrada = (X, pos3d, máscara))
        _ = self.model((self.Xtr_s[:2].astype(np.float32),
                        self.Ptr[:2].astype(np.float32),
                        self.Mtr[:2]), training=False)
        self.model.summary()
        return self.model

    def train(self):
        os.makedirs(f"{self.data_dir}/prediccion_3d/best", exist_ok=True)

        # Datasets: la entrada del modelo es (X escalada, pos3d, máscara)
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((self.Xtr_s.astype(np.float32), self.Ptr.astype(np.float32), self.Mtr),
             self.Ytr.astype(np.float32))
        ).shuffle(8000, seed=42).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(
            ((self.Xva_s.astype(np.float32), self.Pva.astype(np.float32), self.Mva),
             self.Yva.astype(np.float32))
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Callbacks: EarlyStopping, ReduceLROnPlateau y ModelCheckpoint
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(f"{self.data_dir}/prediccion_3d/best/social3d_det_best.weights.h5",
                            monitor="val_loss", save_best_only=True, save_weights_only=True),
        ]
        # Se entrena el modelo
        t0 = time.time()
        self.history = self.model.fit(train_ds, validation_data=val_ds,
                                      epochs=self.epochs, callbacks=callbacks)
        t_train = time.time() - t0
        n_epochs = len(self.history.history["loss"])
        print(f"\nEntrenamiento: {n_epochs} épocas, {t_train:.0f}s total ({t_train/n_epochs:.1f}s/época)")

    def save_outputs(self):
        # Guarda el historial y la curva de pérdida
        os.makedirs(f"{self.data_dir}/prediccion_3d/history", exist_ok=True)
        os.makedirs(f"{self.data_dir}/prediccion_3d/figuras", exist_ok=True)
        np.save(f"{self.data_dir}/prediccion_3d/history/hist_social3d_det.npy", self.history.history)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.history["loss"], label="Entrenamiento")
        plt.plot(self.history.history["val_loss"], label="Validación")
        plt.xlabel("Época"); plt.ylabel("Loss (MAE, metros)")
        plt.title("Curva de entrenamiento — Social LSTM 3D")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{self.data_dir}/prediccion_3d/figuras/loss_social3d_det.png", dpi=130); plt.close()
        print("Guardado: figuras/loss_social3d_det.png, history/hist_social3d_det.npy")

    def evaluate(self):
        # Predicción por lotes de 128 escenas
        pred = []
        for i in range(0, self.Xva.shape[0], 128):
            pred.append(self.model((self.Xva_s[i:i+128].astype(np.float32),
                                    self.Pva[i:i+128].astype(np.float32),
                                    self.Mva[i:i+128]), training=False).numpy())
        pred = np.concatenate(pred, axis=0)

        # Aplana: (n, max_peds, ...) -> (n*max_peds, ...) y filtra por la máscara
        mflat = self.Mva.reshape(-1).astype(bool)
        Xflat = self.Xva.reshape(-1, self.obs_len, self.n_features)[mflat]
        Yflat = self.Yva.reshape(-1, self.pred_len, 2)[mflat]
        pflat = pred.reshape(-1, self.pred_len, 2)[mflat]

        # Métricas: cada modelo contra el CV de su población. El social usa rejilla por escena, así que su
        # CV se recalcula sobre ese subconjunto; el base MAE (04e) es por subtrack (CV del 00).
        ade, fde = ade_fde(pflat, Xflat, Yflat)
        ade_cv_soc, fde_cv_soc = cv_baseline(Xflat, Yflat, self.pred_len)      # CV sobre la población social
        ade_cv, fde_cv = load_cv_ref(self.data_dir, "8_8")                     # CV del 00

        print("\n===== Social 3D detecciones vs LSTM 3D det vs CV =====")
        print(f"{'Modelo':<38} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'mej.FDE':>9}")
        print(f"{'-'*74}")
        print(f"{'CV 8/8':<38} {ade_cv:>8.4f} {fde_cv:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'LSTM con MAE (04e)':<38} {REF_DET_ADE:>8.4f} {REF_DET_FDE:>8.4f} "
              f"{(1-REF_DET_ADE/ade_cv)*100:>7.1f}% {(1-REF_DET_FDE/fde_cv)*100:>7.1f}%")
        print(f"{'CV subconjunto social':<38} {ade_cv_soc:>8.4f} {fde_cv_soc:>8.4f} {'(ref)':>9} {'(ref)':>9}")
        print(f"{'LSTM Social':<38} {ade:>8.4f} {fde:>8.4f} "
              f"{(1-ade/ade_cv_soc)*100:>7.1f}% {(1-fde/fde_cv_soc)*100:>7.1f}%")

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

    trainer = SocialLSTM3DDetTrainer()
    trainer.run()
