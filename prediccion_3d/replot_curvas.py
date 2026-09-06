"""
Re-dibuja las curvas de pérdida de los modelos ajustando los textos mediante los historiales guardados.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
HIST_DIR = Path(DATA_DIR) / "prediccion_3d" / "history"
FIG_DIR = Path(DATA_DIR) / "prediccion_3d" / "figuras_replot"; FIG_DIR.mkdir(parents=True, exist_ok=True)

# Tamaños de texto
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 23, 21, 17, 21

# (nombre historia, título, etiqueta del eje Y)
CURVAS = [
    ("hist_lidar3d_motiononly", "Curva de entrenamiento — LiDAR 3D GT", "Loss (MSE, metros)"),
    ("hist_lidar3d_labels_8obs_8pred", "Curva de entrenamiento — LiDAR 3D etiquetas (8 obs / 8 pred)", "Loss (MSE, metros)"),
    ("hist_lidar3d_labels_12obs_8pred","Curva de entrenamiento — LiDAR 3D etiquetas (12 obs / 8 pred)", "Loss (MSE, metros)"),
    ("hist_lidar3d_det", "Curva de entrenamiento — LiDAR 3D detecciones", "Loss (MSE, metros)"),
    ("hist_lidar3d_det_8obs_8pred", "Curva de entrenamiento — LiDAR 3D detecciones (8 obs / 8 pred)","Loss (MSE, metros)"),
    ("hist_lidar3d_det_12obs_8pred", "Curva de entrenamiento — LiDAR 3D detecciones (12 obs / 8 pred)","Loss (MSE, metros)"),
    ("hist_lidar3d_det_mae", "Curva de entrenamiento — LiDAR 3D detecciones 8/8, pérdida MAE","Loss (MAE, metros)"),
    ("hist_lidar3d_det_env", "Curva de entrenamiento — LiDAR 3D detecciones (mov+entorno)",  "Loss (MAE, metros)"),
    ("hist_social3d_det", "Curva de entrenamiento — Social LSTM 3D", "Loss (MAE, metros)"),
    ("hist_pose3d", "Curva de entrenamiento — LSTM 3D + pose", "Loss (MAE, metros)"),
    ("hist_lidar3d_det_noise", "Curva de entrenamiento — LiDAR 3D det + ruido gaussiano", "Loss (MAE, metros)"),
    ("hist_lidar3d_det_flip", "Curva de entrenamiento — LiDAR 3D det + reflejo lateral", "Loss (MAE, metros)"),
    ("hist_combined3d", "Curva de entrenamiento — Fusión temprana imagen+3D", "Loss (MAE, metros)"),
    ("hist_combined3d_multimodal", "Curva de entrenamiento — Fusión temprana multitrayectoria (K=20)", "Loss (MAE, metros)"),
    ("hist_combined3d_reg", "Curva de entrenamiento — Fusión temprana imagen+3D (regularizada)","Loss (MAE, metros)"),
]


def replot(name, title, ylabel):
    # HISTORIA
    p = HIST_DIR / f"{name}.npy"
    if not p.exists():
        print(f"  (falta) {p.name} -> se omite")
        return
    h = np.load(p, allow_pickle=True).item()
    # Los títulos largos se parten en dos líneas
    if len(title) > 46:
        title = title.replace(" — ", "\n", 1)
    # Plot
    plt.figure(figsize=(11.5, 5))
    plt.plot(h["loss"], label="Entrenamiento", lw=2)
    plt.plot(h["val_loss"], label="Validación", lw=2)
    plt.xlabel("Época", fontsize=FS_LABEL)
    plt.ylabel(ylabel, fontsize=FS_LABEL)
    plt.title(title, fontsize=FS_TITLE)
    plt.xticks(fontsize=FS_TICK); plt.yticks(fontsize=FS_TICK)
    plt.legend(fontsize=FS_LEG)
    plt.grid(alpha=0.3); plt.tight_layout()
    # Se guarda la imagen
    out = FIG_DIR / f"loss_{name.replace('hist_', '')}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Guardado: {out.name}")


if __name__ == "__main__":
    print(f"Re-dibujando {len(CURVAS)} curvas desde {HIST_DIR} ...")
    for name, title, ylabel in CURVAS:
        replot(name, title, ylabel)
    print("Hecho.")
