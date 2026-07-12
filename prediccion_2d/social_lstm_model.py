import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, GaussianNoise


class SocialLSTM(tf.keras.Model):
    """
    SocialLSTM define el modelo Social LSTM.
    """
    def __init__(self, max_peds, obs_len, pred_len, n_features, lstm_units, social_radius,
                 dropout=0.2, rec_dropout=0.0, noise=0.0):
        super().__init__()
        self.max_peds = max_peds
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.social_radius = social_radius

        # Define el modelo
        self.input_noise = GaussianNoise(noise)
        self.encoder_lstm = LSTM(lstm_units, return_state=True, recurrent_dropout=rec_dropout, name="encoder_lstm")
        self.enc_dropout = Dropout(dropout)
        self.combine_dense = Dense(lstm_units, activation="relu", name="combine")
        self.repeat = RepeatVector(pred_len)
        self.decoder_lstm = LSTM(lstm_units, return_sequences=True, recurrent_dropout=rec_dropout, name="decoder_lstm")
        self.dec_dropout = Dropout(dropout)
        self.dense_hidden = TimeDistributed(Dense(64, activation="relu"))
        self.dense_out = TimeDistributed(Dense(2))

    def call(self, inputs, training=False):
        # Características escaladas, posición 3D del peatón en último frame observado, máscara
        X_scaled, last_pos, mask = inputs
        # Tamaño de lote
        batch_size = tf.shape(X_scaled)[0]

        # 1. Encoder
        # Ruido gaussiano
        X_scaled = self.input_noise(X_scaled, training=training)
        # Aplana peatones: (batch, max_peds, 8, 11) -> (batch*max_peds, 8, 11)
        x_flat = tf.reshape(X_scaled, (batch_size * self.max_peds, self.obs_len, self.n_features))
        # Encoder
        enc_out, _h, c = self.encoder_lstm(x_flat)
        # Dropout
        enc_out = self.enc_dropout(enc_out, training=training)
        # Deshace el aplanado: (batch*max_peds, 128) -> (batch, max_peds, 128)
        H = tf.reshape(enc_out, (batch_size, self.max_peds, self.lstm_units))
        c_r = tf.reshape(c, (batch_size, self.max_peds, self.lstm_units))
        
        # 2. Social pooling: posición de cada par de peatones, y distancia entre ellos
        pos_i = tf.expand_dims(last_pos, 2)
        pos_j = tf.expand_dims(last_pos, 1)
        dist = tf.norm(pos_i - pos_j, axis=-1)
        # Filtros: in_radius (1 si está en el radio de vecindad), diag_mask (máscara de vecinos: uno no es vecino de si mismo),
        # neighbor_valid (máscara de peatones activos)
        in_radius = tf.cast(dist < self.social_radius, tf.float32)
        diag_mask = 1.0 - tf.eye(self.max_peds)[tf.newaxis]
        neighbor_valid = tf.cast(tf.expand_dims(mask, 1), tf.float32)
        # Pesos: es 1 cuando el vecino está en el radio, existe, y no es uno mismo.
        # Normaliza cada fila por su suma -> media de vecinos
        weights = in_radius * diag_mask * neighbor_valid
        weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-8)
        # Calcula, para cada peatón, el promedio de estados ocultos de sus vecinos.
        social_pool = tf.matmul(weights, H)

        # 3. Combinación
        # Concatena el estado H del peatón con el de sus vecinos (social_pool).
        combined = self.combine_dense(tf.concat([H, social_pool], axis=-1))

        # 4. Decoder
        # Aplana
        combined_flat = tf.reshape(combined, (batch_size * self.max_peds, self.lstm_units))
        c_flat = tf.reshape(c_r, (batch_size * self.max_peds, self.lstm_units))

        # Decodifica
        repeated = self.repeat(combined_flat)
        decoded = self.decoder_lstm(repeated, initial_state=[combined_flat, c_flat])
        decoded = self.dec_dropout(decoded, training=training)
        decoded = self.dense_hidden(decoded)
        out_flat = self.dense_out(decoded)

        # Recupera forma y devuelve salida
        return tf.reshape(out_flat, (batch_size, self.max_peds, self.pred_len, 2))

    def _masked_mse(self, y_true, y_pred, mask):
        # Calcula el MSE teniendo en cuenta la máscara de peatones reales en la escena
        # Error cuadrático medio por peatón
        sq_err = tf.reduce_mean(tf.square(y_true - y_pred), axis=[2, 3])
        # Convierte la máscara a float, y calcula la media del error aplicando la máscara:
        # solo tiene en cuenta los peatones reales
        mask_f = tf.cast(mask, tf.float32)
        return tf.reduce_sum(sq_err * mask_f) / (tf.reduce_sum(mask_f) + 1e-8)

    def train_step(self, data):
        # Carga los datos: características escaladas, última posición 3D observada, máscara
        # de peatones reales, y salida
        (X_s, lp, mk), Y = data
        # Se ejecuta el modelo (pred) y se mide el error (loss)
        with tf.GradientTape() as tape:
            pred = self((X_s, lp, mk), training=True)
            loss = self._masked_mse(Y, pred, mk)
        # Se calcula los gradientes (cuanto influye cada peso en la pérdida)
        grads = tape.gradient(loss, self.trainable_variables)
        # Se actualizan los pesos
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        # Carga los datos: características escaladas, última posición 3D observada, máscara
        # de peatones reales, y salida
        (X_s, lp, mk), Y = data
        # Se ejecuta el modelo (pred) -> predicción (training=False) y se mide el error (loss)
        pred = self((X_s, lp, mk), training=False)
        loss = self._masked_mse(Y, pred, mk)
        return {"loss": loss}
