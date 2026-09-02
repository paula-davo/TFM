import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"


class JRDBDatasetVisualizer:

    """
    JRDBDatasetVisualizer genera figuras de análisis del dataset
    1) Histograma de longitudes de subtrayectoria
    2) Histograma de peatones simultáneos por escena
    3) Ejemplo de trayectorias de una secuencia de train y otra de val
    """

    # Límite del eje x del histograma de longitudes
    XMAX = 500

    def __init__(self, data_dir=DATA_DIR):
        # Directorio y carpeta de salida
        self.data_dir = data_dir
        self.out = f"{data_dir}/data_prep/figuras"
        os.makedirs(self.out, exist_ok=True)

        # DataFrame y estadísticas
        self.df = None
        self.lengths = None
        self.len_tr = None
        self.len_va = None
        self.peds = None
        self.peds_tr = None
        self.peds_va = None

    def load_dataset(self) -> pd.DataFrame:
        # Carga el CSV
        self.df = pd.read_csv(f"{self.data_dir}/jrdb_clean_trajectories.csv")
        return self.df

    def compute_stats(self):
        # Longitudes de subtrayectoria
        g = self.df.groupby("subtrack_id")
        self.lengths = g.size()
        split_of = g["split"].first()
        self.len_tr = self.lengths[split_of == "train"].values
        self.len_va = self.lengths[split_of == "val"].values

        # Peatones simultáneos por escena
        scene = self.df.groupby(["sequence", "camera", "frame"])
        self.peds = scene["subtrack_id"].nunique()
        peds_split = scene["split"].first()
        self.peds_tr = self.peds[peds_split == "train"].values
        self.peds_va = self.peds[peds_split == "val"].values

    def plot_longitudes(self):
        # Figura: Longitudes de subtrayectoria
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist([self.len_tr, self.len_va], bins=50, range=(0, self.XMAX), stacked=True,
                color=["tab:blue", "tab:orange"], label=["Entrenamiento", "Validación"],
                edgecolor="white")
        ax.axvline(20, color="red", ls="--", label="mínimo (OBS+PRED=20)")
        ax.set_xlim(0, self.XMAX)
        ax.set_xlabel("Longitud de subtrayectoria (frames)", fontsize=16); ax.set_ylabel("Número de subtrayectorias", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.set_title("Longitudes de subtrayectorias", fontsize=18)
        ax.legend(fontsize=14)
        fig.tight_layout(); fig.savefig(f"{self.out}/dataset_longitudes.png", dpi=130); plt.close()

    def plot_densidad(self):
        # Figura: Peatones simultáneos por escena
        fig, ax = plt.subplots(figsize=(7, 5))
        bins = range(1, int(self.peds.max()) + 2)
        ax.hist([self.peds_tr, self.peds_va], bins=bins, stacked=True,
                color=["tab:blue", "tab:orange"], label=["Entrenamiento", "Validación"],
                edgecolor="white", align="left")
        ax.set_xlabel("Peatones simultáneos por escena", fontsize=16); ax.set_ylabel("Número de frames", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.set_title("Peatones por escena", fontsize=18)
        ax.legend(fontsize=14)
        fig.tight_layout(); fig.savefig(f"{self.out}/dataset_densidad.png", dpi=130); plt.close()

    def biggest_seq(self, split) -> str:
        # Secuencia con más filas dentro de un split
        sub = self.df[self.df["split"] == split]
        return sub["sequence"].value_counts().index[0]

    def plot_trayectorias(self, nrows, ncols, figsize, fname):
        # Figura: Ejemplo de trayectorias de una secuencia de train y otra de val
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for ax, split in zip(axes, ["train", "val"]):
            seq = self.biggest_seq(split)
            sub = self.df[(self.df["sequence"] == seq) & (self.df["camera"] == "image_0")]
            for sid, t in sub.groupby("subtrack_id"):
                t = t.sort_values("frame")
                ax.plot(t["x"], t["y"], lw=1)
            ax.invert_yaxis(); ax.set_xlabel("x (píxeles)", fontsize=15); ax.set_ylabel("y (píxeles)", fontsize=15)
            ax.tick_params(labelsize=13)
            ax.set_title(f"{split}: {seq}", fontsize=16)
        fig.suptitle("Ejemplo de distribución de trayectorias en una escena", fontsize=18)
        fig.tight_layout(); fig.savefig(f"{self.out}/{fname}", dpi=130); plt.close()

    def print_summary(self):
        # Resumen
        print(f"Guardado: {self.out}/dataset_longitudes.png, {self.out}/dataset_densidad.png y "
              f"{self.out}/dataset_trayectorias_h.png + dataset_trayectorias_v.png")
        print(f"Subtrayectorias: entrenamiento {len(self.len_tr)} y validación {len(self.len_va)}"
              f"media {self.lengths.mean():.1f}, mediana {np.median(self.lengths):.0f}, máxima {self.lengths.max()}")
        print(f"Peatones/escena: media {self.peds.mean():.2f}, máxima {self.peds.max()}")

    def run(self):
        # 1. Carga el dataset del CSV
        self.load_dataset()
        # 2. Calcula estadísticas
        self.compute_stats()
        # 3. Figura de Longitudes de subtrayectorias
        self.plot_longitudes()
        # 4. Figura de Peatones simultáneos por escena
        self.plot_densidad()
        # 5. Figuras de ejemplo de trayectorias de una secuencia de train y otra de val
        self.plot_trayectorias(1, 2, (14, 6), "dataset_trayectorias_h.png")
        self.plot_trayectorias(2, 1, (7, 11), "dataset_trayectorias_v.png")
        # 6. Resumen
        self.print_summary()


if __name__ == "__main__":
    viz = JRDBDatasetVisualizer()
    viz.run()
