"""
Barrido de arquitecturas/hiperparámetros sobre el modelo de detecciones 8/8.
"""
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.stdout.reconfigure(line_buffering=True)   # progreso en vivo aunque la salida vaya a un pipe (tee)

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from lstm_common import scale_train_val, ade_fde, cv_baseline
from arch_variants import build_seq2seq_variant

OBS_LEN, PRED_LEN, N_FEATURES = 8, 8, 9

def make_optimizer(lr, wd):
    # Adam, o AdamW si hay weight decay
    if wd:
        try:
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        except (AttributeError, TypeError):
            return tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=wd)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def make_loss(name):
    # MSE / MAE / Huber
    return tf.keras.losses.Huber() if name == "huber" else name


# Configuraciones a probar.
CONFIGS = [
    dict(name="128 d0.3 (04b)",   units=128, dropout=0.3, rec_dropout=0.2, n_layers=1, cell="lstm"),
    dict(name="128 d0.5",         units=128, dropout=0.5, rec_dropout=0.2, n_layers=1, cell="lstm"),
    dict(name="128 d0.3 rec0.0",  units=128, dropout=0.3, rec_dropout=0.0, n_layers=1, cell="lstm"),
    dict(name="64 d0.3",          units=64,  dropout=0.3, rec_dropout=0.2, n_layers=1, cell="lstm"),
    dict(name="64 d0.4",          units=64,  dropout=0.4, rec_dropout=0.2, n_layers=1, cell="lstm"),
    dict(name="256 d0.5",         units=256, dropout=0.5, rec_dropout=0.2, n_layers=1, cell="lstm"),
    dict(name="128 d0.3 x2capas", units=128, dropout=0.3, rec_dropout=0.2, n_layers=2, cell="lstm"),
    dict(name="128 AdamW wd1e-4", units=128, dropout=0.3, rec_dropout=0.2, n_layers=1, cell="lstm", weight_decay=1e-4),
    dict(name="128 loss=MAE",     units=128, dropout=0.3, rec_dropout=0.2, n_layers=1, cell="lstm", loss="mae"),
    dict(name="128 loss=Huber",   units=128, dropout=0.3, rec_dropout=0.2, n_layers=1, cell="lstm", loss="huber"),
    dict(name="128 batch=64",     units=128, dropout=0.3, rec_dropout=0.2, n_layers=1, cell="lstm", batch_size=64),
    dict(name="128 lr=3e-4",      units=128, dropout=0.3, rec_dropout=0.2, n_layers=1, cell="lstm", learning_rate=3e-4),
]


def main():
    # Detecciones 3D 8/8 y partición oficial
    X = np.load(f"{DATA_DIR}/data/X3d_det_8obs_8pred.npy")
    Y = np.load(f"{DATA_DIR}/data/Y3d_det_8obs_8pred.npy")
    ids = np.load(f"{DATA_DIR}/data/subtrack_ids_ds.npy", allow_pickle=True)
    sck = np.array([s.rsplit("_", 2)[0] for s in ids])
    tm, vm = official_masks(sck)
    Xtr, Xva, Ytr, Yva = X[tm], X[vm], Y[tm], Y[vm]
    Xtr_s, Xva_s, _ = scale_train_val(Xtr, Xva, N_FEATURES)
    ade_cv, fde_cv = cv_baseline(Xva, Yva, PRED_LEN)
    print(f"CV: ADE {ade_cv:.4f}  FDE {fde_cv:.4f}   (val: {Xva.shape[0]} muestras)")

    results = []
    for cfg in CONFIGS:
        print(f"\n### {cfg['name']} ###")
        # Optimizador
        opt = make_optimizer(cfg.get("learning_rate", 1e-3), cfg.get("weight_decay", 0.0))
        # Modelo
        model = build_seq2seq_variant(OBS_LEN, PRED_LEN, N_FEATURES,
                                      units=cfg["units"], dropout=cfg["dropout"],
                                      rec_dropout=cfg["rec_dropout"], n_layers=cfg["n_layers"],
                                      cell=cfg["cell"], optimizer=opt,
                                      loss=make_loss(cfg.get("loss", "mse")), metrics=["mae"])
        # Callbacks
        cbs = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)]
        # Se entrena el modelo midiendo el tiempo de entrenamiento
        t0 = time.time()
        h = model.fit(Xtr_s, Ytr, validation_data=(Xva_s, Yva),
                      epochs=120, batch_size=cfg.get("batch_size", 256), callbacks=cbs, verbose=0)
        # Predicción
        pred = model.predict(Xva_s, batch_size=512, verbose=0)
        # Métricas
        ade, fde = ade_fde(pred, Xva, Yva)
        ep, dt = len(h.history["loss"]), time.time() - t0
        results.append((cfg["name"], ade, fde, ep, dt))
        print(f"  ADE {ade:.4f}  FDE {fde:.4f}  ({ep} épocas, {dt:.0f}s)")

    # Tabla final ordenada por ADE
    results.sort(key=lambda r: r[1])
    print(f"\n===== BARRIDO det 8/8 — ordenado por ADE (CV {ade_cv:.4f}/{fde_cv:.4f}) =====")
    print(f"{'Config':<22} {'ADE':>8} {'FDE':>8} {'mej.ADE':>9} {'épocas':>7}")
    print(f"{'-'*58}")
    for name, ade, fde, ep, dt in results:
        print(f"{name:<22} {ade:>8.4f} {fde:>8.4f} {(1-ade/ade_cv)*100:>7.1f}% {ep:>7}")


if __name__ == "__main__":
    main()
