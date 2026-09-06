"""
Carga el número de épocas de cada entrenamiento a partir de los historiales
guardados y los imprime.
"""
import numpy as np
from pathlib import Path

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
DIRS = [Path(DATA_DIR) / "prediccion_3d" / "history"]

print(f"{'Modelo (hist_*)':<40} {'épocas':>8}  {'loss final':>12} {'val_loss final':>15}")
print("-" * 78)
for d in DIRS:
    if not d.exists():
        continue
    print(f"[{d.parent.name}]")
    for p in sorted(d.glob("hist_*.npy")):
        h = np.load(p, allow_pickle=True).item()
        loss = h.get("loss", [])
        vloss = h.get("val_loss", [])
        n = len(loss)
        lf = f"{loss[-1]:.4f}" if n else "-"
        vf = f"{vloss[-1]:.4f}" if len(vloss) else "-"
        print(f"  {p.stem:<38} {n:>8}  {lf:>12} {vf:>15}")
