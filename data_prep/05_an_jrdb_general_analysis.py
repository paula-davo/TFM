from pathlib import Path
import yaml
from pprint import pprint

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"

class JRDBGeneralAnalysis:

    """
    JRDBGeneralAnalysis realiza el análisis general de la estructura del dataset JRDB:
    1) Correspondencia imágenes <-> nubes de puntos (mismo frame)
    2) Número de PCDs por secuencia
    3) Ficheros de calibración y claves de sus YAML

    Conclusiones de este análisis:
    - número de imágenes = número de PCDs = 1726
    - 000000.jpg -> 000000.pcd (frame N de imagen = frame N de nube de puntos)
    - hay 27 secuencias con entre 430 y 1735 frames, unas 28000 nubes de puntos
    - claves de cameras.yaml: stitching y cameras
    - claves de defaults.yaml: calibrated, image y frames
    - claves de lidars.yaml: lidar, sensor_0, sensor_2, sensor_4, sensor_6, sensor_8
    """

    def __init__(self, data_dir=DATA_DIR, sequence="bytes-cafe-2019-02-07_0"):
        # Directorio del dataset (train con etiquetas)
        self.train_dir = Path(data_dir) / "train_dataset_with_activity" / "train_dataset_with_activity"

        # Subcarpetas
        self.img_dir = self.train_dir / "images" / "image_0"
        self.pcd_dir = self.train_dir / "pointclouds" / "lower_velodyne"
        self.calib = self.train_dir / "calibration"

        # Secuencia de ejemplo para la correspondencia imagen<->PCD
        self.sequence = sequence

    def count_frames(self):
        # Compara número de imágenes y de PCDs de la secuencia de ejemplo
        imgs = sorted((self.img_dir / self.sequence).glob("*.jpg"))
        pcds = sorted((self.pcd_dir / self.sequence).glob("*.pcd"))

        print("Número imágenes:", len(imgs))
        print("Número PCDs:", len(pcds))

        print("\nPrimeras imágenes:")
        for p in imgs[:5]:
            print(p.name)

        print("\nPrimeros PCDs:")
        for p in pcds[:5]:
            print(p.name)

    def count_pcds_per_sequence(self):
        # Número de nubes de puntos por secuencia
        print("\nPCDs por secuencia:")
        for seq in sorted(self.pcd_dir.iterdir()):
            if seq.is_dir():
                n = len(list(seq.glob("*.pcd")))
                print(f"{seq.name}: {n}")

    def list_calibration_files(self):
        # Lista todos los ficheros de calibración
        print("\nFicheros de calibración:")
        for file in self.calib.rglob("*"):
            print(file)

    def inspect_yaml(self, name):
        # Carga un YAML de calibración e imprime sus claves
        with open(self.calib / name, "r") as f:
            data = yaml.safe_load(f)

        print(f"\n{name}:")
        print(type(data))
        if isinstance(data, dict):
            print("Keys:")
            pprint(list(data.keys()))

    def run(self):
        # 1. Cuenta imágenes y PCDs de la secuencia de ejemplo
        self.count_frames()
        # 2. Cuenta PCDs por secuencia
        self.count_pcds_per_sequence()
        # 3. Lista ficheros de calibración y claves de sus YAML
        self.list_calibration_files()
        for name in ["cameras.yaml", "defaults.yaml", "lidars.yaml"]:
            self.inspect_yaml(name)


if __name__ == "__main__":

    analysis = JRDBGeneralAnalysis()
    analysis.run()
