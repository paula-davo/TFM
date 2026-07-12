import pandas as pd
import numpy as np

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

OBS_LEN = 8
PRED_LEN = 12


class JRDBDatasetLoader:

    """
    JRDBDatasetLoader carga el CSV, prepara entradas y salida, y las guarda en formato .npy.
    1) Carga el CSV
    2) Ventana deslizante: 8 frames de entrada, 12 de salida
    3) Y como desplazamiento relativo al último frame observado
    4) GUARDADO en .npy
    """

    # 11 entradas - modelo base
    FEATURES = ["x_norm", "y_norm", "vx", "vy", "ax", "ay", "speed", "sin_dir", "cos_dir", "bbox_w_norm", "bbox_h_norm"]

    def __init__(self, data_dir=DATA_DIR, obs_len=OBS_LEN, pred_len=PRED_LEN):
        # Directorio 
        self.data_dir = data_dir

        # Longitud de entrada y de salida
        self.obs_len = obs_len
        self.pred_len = pred_len

        # Resultados
        self.df = None
        self.X = None
        self.Y = None
        self.subtrack_ids = None
        self.splits = None

    def load_dataset(self) -> pd.DataFrame:
        # Carga el CSV en DataFrame
        self.df = pd.read_csv(f"{self.data_dir}/jrdb_clean_trajectories.csv")
        return self.df

    @staticmethod
    def create_sequences(df, features, obs_len=OBS_LEN, pred_len=PRED_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Ventana deslizante sobre cada subtrayectoria -> muestras (X, Y)
        X = []
        Y = []
        subtrack_ids = []
        splits = []

        total_tracks = 0
        valid_tracks = 0
        # Para cada trayectoria
        for subtrack_id, traj in df.groupby("subtrack_id"):
            total_tracks += 1
            # Ordena por frames
            traj = traj.sort_values("frame")
            traj_len = len(traj)

            # Descarta subtrayectorias cortas
            if traj_len < obs_len + pred_len:
                continue

            valid_tracks += 1
            # Extrae split y características
            split = traj["split"].iloc[0]
            feats = traj[features].values

            # Para cada subtrayectoria -> ventana deslizante
            # Recorre cada posible inicio
            for start in range(traj_len - obs_len - pred_len + 1):
                # Calcula fin de observación y de predicción
                obs_end = start + obs_len
                pred_end = obs_end + pred_len
                # Entrada
                x_seq = feats[start:obs_end]
                # Posición del último frame observado (entrada)
                last_obs_pos = feats[obs_end - 1, 0:2]      # x_norm, y_norm
                # Posiciones absolutas de salida
                y_abs = feats[obs_end:pred_end, 0:2]
                # Desplazamientos relativos al último frame observado
                y_seq = y_abs - last_obs_pos

                # Guarda la secuencia en entrada y salida. 
                X.append(x_seq)
                Y.append(y_seq)
                subtrack_ids.append(subtrack_id)
                splits.append(split)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        subtrack_ids = np.array(subtrack_ids)
        splits = np.array(splits)

        print("DATASET")
        print(f"Subtrayectorias totales: {total_tracks}")
        print(f"Subtrayectorias válidas: {valid_tracks}")

        print("Shapes:")
        print("X:", X.shape)
        print("Y:", Y.shape)

        print("Muestras por split:")
        print(f"  train: {(splits == 'train').sum()}")
        print(f"  val:   {(splits == 'val').sum()}")

        print("Y (desplazamientos relativos) — estadísticas:")
        print(f"  media: {Y.mean():.6f}  (debe ser ~0)")
        print(f"  std:   {Y.std():.6f}")
        print(f"  min:   {Y.min():.6f}")
        print(f"  max:   {Y.max():.6f}")

        return X, Y, subtrack_ids, splits

    def build(self):
        # Devuelve X, Y, subtrack_ids y splits
        self.X, self.Y, self.subtrack_ids, self.splits = self.create_sequences(self.df, self.FEATURES, self.obs_len, self.pred_len)
        return self.X, self.Y, self.subtrack_ids, self.splits

    def save_numpy(self):
        # Guarda en formato .npy
        np.save(f"{self.data_dir}/X_train.npy", self.X)
        np.save(f"{self.data_dir}/Y_train.npy", self.Y)
        np.save(f"{self.data_dir}/subtrack_ids.npy", self.subtrack_ids)
        np.save(f"{self.data_dir}/splits.npy", self.splits)

        print("Archivos guardados.")

    def run(self):
        # 1. Carga el CSV
        self.load_dataset()
        # 2. Genera entradas y salidas con ventana deslizante
        self.build()
        # 3. Guarda en .npy
        self.save_numpy()


if __name__ == "__main__":

    builder = JRDBDatasetLoader()
    builder.run()
