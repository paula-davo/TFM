import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"


class JRDBDatasetJsonAnalysis:

    """
    JRDBDatasetJsonAnalysis crea un CSV con las trayectorias 2D a partir de las particiones
    oficiales de JRDB.

    1) Carga
    2) Filtrado
    3) Dinámica
    4) GUARDADO
    """
    def __init__(self, data_dir=DATA_DIR, min_visible=5, max_gap=1, min_len=20):

        # Directorio del proyecto
        self.data_dir = data_dir

        # Particiones de JRDB
        self.train_json = f"{data_dir}/jrdb_mmpose_train/jrdb_mmpose_train/train_individual.json"
        self.val_json = f"{data_dir}/jrdb_mmpose_train/jrdb_mmpose_train/val_individual.json"

        # Mínimo número de keypoints visibles
        self.min_visible = min_visible
        # Máximo hueco temporal permitido en una subtrayectoria
        self.max_gap = max_gap
        # Longitud mínima de una subtrayectoria
        self.min_len = min_len

        # DataFrame resultante
        self.df = None

    @staticmethod
    def open_json_file(path) -> dict:
        # Carga el archivo JSON
        with open(path, "r") as f:
            data = json.load(f)

        return data

    @staticmethod
    def mostrar_caracteristicas_dataset(data):
        # Imprimir información
        print(data.keys())

        print("\nNúmero de imágenes:")
        print(len(data["images"]))

        print("\nNúmero de anotaciones:")
        print(len(data["annotations"]))

        print("\nPrimera anotación:")
        print(data["annotations"][0])

    @staticmethod
    def preprocess_dataset_format(data) -> pd.DataFrame:
        image_info = {}

        # Guarda información de las imágenes
        for img in data["images"]:

            image_info[img["id"]] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"]
            }

        rows = []

        # Guarda información de las anotaciones
        for ann in data["annotations"]:

            img = image_info[ann["image_id"]]

            # path:
            # image_0/bytes-cafe-2019-02-07_0/000000.jpg
            file_name = img["file_name"]
            parts = file_name.split("/")
            camera = parts[0]
            sequence = parts[1]
            frame_name = parts[2]

            # frame temporal
            frame_idx = int(frame_name.replace(".jpg", ""))

            # Bounding box
            # bbox = [x, y, width, height]
            bbox = ann["bbox"]

            # Centro de la bounding box
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2

            # Keypoints
            keypoints = ann["keypoints"]

            # Convierte:
            # [x1,y1,v1,x2,y2,v2,...] -> [[x1,y1,v1], [x2,y2,v2], ...]
            kp = [keypoints[i:i+3] for i in range(0, len(keypoints), 3)]

            # Número de keypoints visibles
            visible = sum(1 for k in kp if k[2] > 0)

            rows.append({
                "sequence": sequence,
                "camera": camera,
                "frame": frame_idx,
                "track_id": ann["track_id"],
                "x": cx,
                "y": cy,
                "w": img["width"],
                "h": img["height"],
                "visible_keypoints": visible,
                "bbox_w": bbox[2],
                "bbox_h": bbox[3]
            })

        # Crear DataFrame
        df = pd.DataFrame(rows)

        # Crear ID global único
        # track_id se reutiliza entre secuencias/cámaras.
        df["global_id"] = (df["sequence"] + "_" + df["camera"] + "_" + df["track_id"].astype(str))

        # Orden temporal
        df = df.sort_values(["global_id", "frame"])

        return df


    @staticmethod
    def split_tracks_by_gaps(df, max_gap=1) -> pd.DataFrame:
        # Divide trayectorias discontinuas en subtrayectorias continuas.
        split_trajectories = []

        # Para cada trayectoria
        for gid, traj in df.groupby("global_id"):
            # Orden temporal
            traj = traj.sort_values("frame")
            frames = traj["frame"].values

            start_idx = 0
            sub_id = 0

            # Detectar discontinuidades temporales
            for i in range(1, len(frames)):
                gap = frames[i] - frames[i - 1]

                # Si es grande, guarda subtrayectoria
                if gap > max_gap:
                    subtraj = traj.iloc[start_idx:i].copy()
                    subtraj["subtrack_id"] = f"{gid}_{sub_id}"
                    split_trajectories.append(subtraj)
                    start_idx = i
                    sub_id += 1

            # Añadir último segmento
            subtraj = traj.iloc[start_idx:].copy()
            subtraj["subtrack_id"] = f"{gid}_{sub_id}"
            split_trajectories.append(subtraj)

        df_split = pd.concat(split_trajectories)

        return df_split

    @staticmethod
    def filter_by_keypoint_quality(df, min_visible=5) -> pd.DataFrame:
        # Si una detección tiene menos de min_visible keypoints visibles, se elimina.
        df_filtered = df[df["visible_keypoints"] >= min_visible].copy()

        removed = len(df) - len(df_filtered)
        print(f"Filas eliminadas por baja calidad de detección: {removed}")
        print(f"Filas restantes: {len(df_filtered)}")

        return df_filtered

    @staticmethod
    def filter_short_tracks(df, min_len=20) -> pd.DataFrame:
        # Se quiere 8 frames de entrada y 12 de salida -> mínimo 20 frames por subtrayectoria.
        # Longitud
        subtrack_lengths = df.groupby("subtrack_id").size()

        print("Longitud de subtrayectorias:")
        print(subtrack_lengths.describe())

        # Selecciona las válidas
        valid_subtracks = subtrack_lengths[subtrack_lengths >= min_len].index
        # Filtra
        df_filtered = df[df["subtrack_id"].isin(valid_subtracks)]

        return df_filtered

    @staticmethod
    def add_motion_features(df) -> pd.DataFrame:
        # Añade posición normalizada
        df["x_norm"] = df["x"] / df["w"]
        df["y_norm"] = df["y"] / df["h"]

        # Añade ancho y alto del bbox normalizado
        df["bbox_w_norm"] = df["bbox_w"] / df["w"]
        df["bbox_h_norm"] = df["bbox_h"] / df["h"]

        # Orden temporal y agrupa por subtrayectorias
        # Calcula la velocidad -> diferencia entre pos anterior y actual
        df = df.sort_values(["subtrack_id", "frame"])
        df["vx"] = df.groupby("subtrack_id")["x_norm"].diff()
        df["vy"] = df.groupby("subtrack_id")["y_norm"].diff()
        df["vx"] = df["vx"].fillna(0)
        df["vy"] = df["vy"].fillna(0)
        # Velocidad lineal
        df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
        # De la misma forma, calcula aceleración -> diferencia entre vel anterior y actual
        df["ax"] = df.groupby("subtrack_id")["vx"].diff().fillna(0)
        df["ay"] = df.groupby("subtrack_id")["vy"].diff().fillna(0)

        # Calcula la dirección
        direction = np.arctan2(df["vy"], df["vx"])
        df["sin_dir"] = np.sin(direction)
        df["cos_dir"] = np.cos(direction)

        return df


    @staticmethod
    def compute_statistics(df):
        print("\nNúmero total de subtrayectorias:")
        print(df["subtrack_id"].nunique())

        if "split" in df.columns:
            print("\nSubtrayectorias por split (tras la limpieza):")
            print(df.groupby("split")["subtrack_id"].nunique())

        # Peatones simultáneos por frame
        people_per_frame = df.groupby(["sequence", "camera", "frame"])["subtrack_id"].nunique()
        print("\nPeatones simultáneos por frame:")
        print(people_per_frame.describe())

    @staticmethod
    def visualize_trajectories(df, sequence_name):
        # Se selecciona la secuencia
        seq_df = df[df["sequence"] == sequence_name]

        # Se visualizan las trayectorias ordenadas por frame
        plt.figure(figsize=(10, 8))
        for _, traj in seq_df.groupby("subtrack_id"):

            traj = traj.sort_values("frame")

            plt.plot(traj["x"], traj["y"], linewidth=1)

        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Trajectories - {sequence_name}")
        plt.show()


    def load(self) -> pd.DataFrame:
        # 1. Carga de datos
        data_train = self.open_json_file(self.train_json)
        data_val = self.open_json_file(self.val_json)

        # 2. Muestra información general
        self.mostrar_caracteristicas_dataset(data_train)

        # 3. Preprocesa los datos y los divide en train y val
        df_train = self.preprocess_dataset_format(data_train)
        df_train["split"] = "train"
        df_val = self.preprocess_dataset_format(data_val)
        df_val["split"] = "val"
        df = pd.concat([df_train, df_val], ignore_index=True)

        print("Secuencias por split (partición oficial JRDB):")
        print(f"  train: {df_train['sequence'].nunique()} secuencias, {len(df_train)} detecciones")
        print(f"  val:   {df_val['sequence'].nunique()} secuencias, {len(df_val)} detecciones")

        # 4. Guarda y devuelve el DataFrame
        self.df = df
        return df

    def build(self) -> pd.DataFrame:
        df = self.df

        # 1. Filtra por visibilidad
        df = self.filter_by_keypoint_quality(df, self.min_visible)

        # 2. Divide por huecos temporales
        df = self.split_tracks_by_gaps(df, self.max_gap)

        # 3. Filtra por longitud mínima de subtrayectoria
        df = self.filter_short_tracks(df, self.min_len)

        # 4. Añade las dinámicas
        df = self.add_motion_features(df)

        # 5. Guarda y devuelve el DataFrame
        self.df = df
        return df

    def save(self, path=None):
        # Guarda el DataFrame en un CSV
        path = path or f"{self.data_dir}/jrdb_clean_trajectories.csv"
        self.df.to_csv(path, index=False)
        print("\nCSV limpio guardado correctamente.")

    def print_example_annotations(self):
        # Muestra un ejemplo de anotación (frame intermedio, para que se observen datos y v/a diferente a 0)
        with pd.option_context("display.max_columns", None, "display.width", None):
            for s in ["train", "val"]:
                sub = self.df[self.df["split"] == s]
                if len(sub) == 0:
                    continue
                # subtrayectoria más larga y un frame intermedio de ella
                sid = sub.groupby("subtrack_id").size().idxmax()
                traj = sub[sub["subtrack_id"] == sid].sort_values("frame")
                ejemplo = traj.iloc[len(traj) // 2]
                print(f"Ejemplo de anotación ({s}):")
                print(ejemplo.to_string())

    def run(self) -> pd.DataFrame:
        # Carga el DataFrame con los dos splits preprocesados
        self.load()
        # Termina de construir el DataFrame: filtrado y dinámicas
        self.build()
        # Muestra estadísticas
        self.compute_statistics(self.df)
        # Guarda el DataFrame en un CSV
        self.save()
        # Muestra ejemplo de anotación
        self.print_example_annotations()
        # Visualiza
        try:
            self.visualize_trajectories(self.df, "bytes-cafe-2019-02-07_0")
        except Exception as e:
            print(f"(Error durante la visualización: {e})")

        return self.df


if __name__ == "__main__":
    dataset = JRDBDatasetJsonAnalysis()
    dataset.run()
