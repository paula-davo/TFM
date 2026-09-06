"""
Constructor flexible del modelo para el barrido de arquitecturas e
hiperparámetros.
"""

from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model


def _make(cell, units, return_sequences, return_state, rec_dropout, name):
    # Crear el tipo de capa solicitado con la configuración dada
    Cell = LSTM if cell == "lstm" else GRU
    return Cell(units, return_sequences=return_sequences, return_state=return_state,
                recurrent_dropout=rec_dropout, name=name)


def build_seq2seq_variant(obs_len, pred_len, n_features, units=128, dropout=0.3,
    rec_dropout=0.2, n_layers=1, cell="lstm", optimizer="adam", 
    loss="mse", metrics=None):
    inp = Input(shape=(obs_len, n_features), name="encoder_input")

    # Encoder
    x = inp
    for i in range(n_layers - 1):
        x = _make(cell, units, True, False, rec_dropout, f"enc_{i}")(x)
        x = Dropout(dropout)(x)
    enc = _make(cell, units, False, True, rec_dropout, "encoder_lstm")(x)
    if cell == "lstm":
        enc_out, h, c = enc
        states = [h, c]
    else:
        enc_out, h = enc
        states = [h]
    enc_out = Dropout(dropout)(enc_out)

    # Decoder
    dec = RepeatVector(pred_len)(enc_out)
    dec = _make(cell, units, True, False, rec_dropout, "decoder_lstm")(dec, initial_state=states)
    dec = Dropout(dropout)(dec)
    for i in range(n_layers - 1):
        dec = _make(cell, units, True, False, rec_dropout, f"dec_{i}")(dec)
        dec = Dropout(dropout)(dec)

    hidden = TimeDistributed(Dense(64, activation="relu"))(dec)
    out = TimeDistributed(Dense(2))(hidden)
    # Se compila el modelo
    model = Model(inp, out)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics or [])
    return model
