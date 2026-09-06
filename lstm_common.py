"""
Métodos comunes de modelos LSTM de predicción 3D
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, GaussianNoise
from tensorflow.keras.models import Model

def load_cv_ref(data_dir, horizon):
    # Carga los resultados de la referencia de Velocidad Constante (ade, fde) del horizonte
    # dado ('8_8', '12_8', '8_12'), guardado por el paso 00b en prediccion_3d/cv_baselines.json. 
    # Evita recalcular el CV en cada script de entrenamiento.
    import json
    path = f"{data_dir}/prediccion_3d/cv_baselines.json"
    try:
        with open(path) as f:
            d = json.load(f)[horizon]
    except FileNotFoundError:
        raise SystemExit(f"No existe {path}. Ejecuta antes: python prediccion_3d/00b_baseline_cv.py")
    return d["ade"], d["fde"]


def build_seq2seq(obs_len, pred_len, n_features, units=128, dropout=0.2,
                  rec_dropout=0.0, noise=0.0, optimizer="adam", loss="mse",
                  metrics=None, named=False):
    # Construye y compila el modelo LSTM
    # Nombres de capa del modelo base (o None -> Keras los asigna automáticamente)
    if named:
        nm = dict(inp="encoder_input", enc="encoder_lstm", encd="encoder_dropout",
                  rep="repeat", dec="decoder_lstm", decd="decoder_dropout",
                  hid="dense_hidden", out="output")
    else:
        nm = dict(inp=None, enc=None, encd=None, rep=None, dec=None, decd=None, hid=None, out=None)

    # Encoder
    encoder_input = Input(shape=(obs_len, n_features), name=nm["inp"])
    # Ruido gaussiano (opcional). noise = 0 -> sin capa, comportamiento base.
    x = GaussianNoise(noise)(encoder_input) if noise > 0 else encoder_input
    # Recurrent dropout (opcional)
    encoder_out, state_h, state_c = LSTM(units, return_state=True,
                                         recurrent_dropout=rec_dropout, name=nm["enc"])(x)
    encoder_out = Dropout(dropout, name=nm["encd"])(encoder_out)

    # Decoder: el estado del encoder inicializa el decoder
    decoder_input = RepeatVector(pred_len, name=nm["rep"])(encoder_out)
    # Recurrent dropout (opcional)
    decoder_lstm = LSTM(units, return_sequences=True, recurrent_dropout=rec_dropout, name=nm["dec"])(
        decoder_input, initial_state=[state_h, state_c])
    decoder_lstm = Dropout(dropout, name=nm["decd"])(decoder_lstm)
    decoder_dense1 = TimeDistributed(Dense(64, activation="relu"), name=nm["hid"])(decoder_lstm)
    decoder_output = TimeDistributed(Dense(2), name=nm["out"])(decoder_dense1)

    model = Model(encoder_input, decoder_output)
    # loss="mse" por defecto -> comportamiento base; "mae" para la variante robusta al ruido
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def scale_train_val(X_tr, X_va, n_features):
    # Scaler: estandariza las entradas. Fit solo con train. Transform con train y val. 
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, n_features)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, n_features)).reshape(X_va.shape)
    return X_tr_s, X_va_s, scaler


def ade_fde(pred, X_va, Y_va):
    # Calculo de las métricas
    last = X_va[:, -1, 0:2]
    # Predicción: última posición observada + desplazamiento predicho
    pa = pred + last[:, None, :]
    # GT: última posición observada + desplazamiento real
    ya = Y_va + last[:, None, :]
    # Cálculo de ADE y FDE
    ade = np.mean(np.linalg.norm(pa - ya, axis=2))
    fde = np.mean(np.linalg.norm(pa[:, -1, :] - ya[:, -1, :], axis=1))
    return ade, fde


def cv_disp(X_va, pred_len):
    # Desplazamiento relativo del CV: p_{t+k} - p_t = k * v_t
    # Velocidad del último punto observado
    vx, vy = X_va[:, -1, 2], X_va[:, -1, 3]
    # Pasos futuros
    steps = np.arange(1, pred_len + 1, dtype=np.float32)
    # Desplazamiento relativo del CV
    return np.stack([np.outer(vx, steps), np.outer(vy, steps)], axis=2)


def cv_baseline(X_va, Y_va, pred_len):
    # Cálculo de métricas resultantes del Modelo de Velocidad Constante
    return ade_fde(cv_disp(X_va, pred_len), X_va, Y_va)


def flip_lat(X, Y):
    # Data augmentation 3D: reflejo del eje lateral (cy) en el marco del robot.
    # cy, vy, ay, sin_dir -> negados; dy (salida) -> negado. (cx, vx, ax, speed, cos_dir intactos)
    # Features 3D (N,8,9): cx, cy, vx, vy, ax, ay, speed, sin_dir, cos_dir (metros).
    Xf = X.copy()
    Xf[:, :, 1] *= -1.0   # cy
    Xf[:, :, 3] *= -1.0   # vy
    Xf[:, :, 5] *= -1.0   # ay
    Xf[:, :, 7] *= -1.0   # sin_dir
    Yf = Y.copy()
    Yf[:, :, 1] *= -1.0   # dy
    return Xf, Yf
