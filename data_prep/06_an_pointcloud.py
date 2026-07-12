import os
import open3d as o3d
import numpy as np
from pathlib import Path


# Extraer features de la nube para introducirla como entrada a la red
# Se extrae el contexto espacial alrededor del peaton (puntos cercanos)

# Luego podemos hacer
# Opcion a) una sola lstm ->
#    - point could -> procesado
#    - trajectory -> trajectory features
#    - concat -> LSTM -> prediction
# Opcion b) dos lstm separadas ->
#    - trajectory -> LSTM encoder
#    - point cloud-> procesado -> MLP
#    - concat (features) -> Dense fusion -> decoder LSTM -> prediction

TRAIN_DIR = Path("/mnt/c/Users/paula/Desktop/TFM/train_dataset_with_activity/train_dataset_with_activity")

pcd_path = TRAIN_DIR / "pointclouds/lower_velodyne/bytes-cafe-2019-02-07_0/000006.pcd"
print(pcd_path)

# Carga nube de puntos
pcd = o3d.io.read_point_cloud(str(pcd_path))
# Imprime información de la nube
print(pcd)
# Visualiza la nube de puntos solo si hay pantalla disponible
# (evita que se quede bloqueado en un entorno sin display, p. ej. WSL headless)
if os.environ.get("DISPLAY"):
    try:
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(f"(Visualización omitida: {e})")
else:
    print("(Visualización omitida: no hay pantalla disponible / DISPLAY no definido)")

# Extrae características de la nube de puntos
len(pcd.points)
points = np.asarray(pcd.points)
print(points.min(axis=0))
print(points.max(axis=0))