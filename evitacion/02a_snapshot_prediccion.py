"""
Fase 2 - Paso 2A: snapshot de predicción en una escena reconstruida.

Para un instante de una secuencia de validación:
- Reconstruye la observación de los peatones a partir de las detecciones 3D.
- Predice con el modelo de múltiples trayectorias 3D -> 20 hipótesis.
- Dibuja en marco mundo las observaciones y predicciones.

Uso:
    python evitacion/02a_snapshot_prediccion.py [secuencia] [frame_actual]
Por defecto: huang-2-2019-01-25_0, frame 300.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LAB = TRAIN_DIR / "labels" / "labels_3d"
DET = TRAIN_DIR / "detections" / "detections_3d"
RECON = Path(DATA_DIR) / "evitacion" / "reconstruccion"
OUTF = Path(DATA_DIR) / "evitacion" / "figuras"; OUTF.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks          # noqa: E402
from evitacion_common import (fill, nearest_det, load_gt, load_det,   # noqa: E402
                              feats as build_features)

OBS, PRED, STRIDE = 8, 12, 3

SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0"
C = int(sys.argv[2]) if len(sys.argv) > 2 else 300  # Frame actual

# 1. Poses
frames8 = [C - STRIDE * (OBS - 1 - i) for i in range(OBS)]   # 8 frames observados (..., C)
need = set(frames8)
poses = None
# Para cada npz de la secuencia seleccionada, se carga el fichero y los frames que lo cubren
for npz in sorted(RECON.glob(f"{SEQ}_*.npz")):
    d = np.load(npz)
    fr = set(int(x) for x in d["frames"])
    # Si todos los frames necesitados están en este conjunto de frames, se construye el diccionario:
    # frame -> matriz de pose 4x4
    if need <= fr:
        poses = {int(f): P for f, P in zip(d["frames"], d["poses"])}
        print(f"Poses cargadas de {npz.name}")
        break
if poses is None:
    raise SystemExit(f"No hay .npz que cubra los frames {min(frames8)}..{C}. "
                     f"Ejecuta antes: python evitacion/01_reconstruir_escena.py {SEQ} "
                     f"{max(0, min(frames8))} {C}")

# 2. Carga de datos: etiquetas y detecciones 3D
gt = load_gt(LAB, SEQ)
det = load_det(DET, SEQ)


# 3. Preparación de los datos
# Peatones existentes en los 8 frames observados
tids = [t for t in gt[C] if all(t in gt[f] for f in frames8)]
print(f"Peatones con observación completa en frame {C}: {len(tids)}")

# Carga los datos de detecciones 3D con partición oficial
X3d = np.load(f"{DATA_DIR}/X3d_det_motion.npy")
ids = np.load(f"{DATA_DIR}/subtrack_ids_ds.npy", allow_pickle=True)
tm, _ = official_masks(ids)
# Ajusta scaler
scaler = StandardScaler().fit(X3d[tm].reshape(-1, 9))
# Carga el modelo
model = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_multimodal_best.keras", compile=False)

# 4. Predicción

def to_world(fr, xy):
    # Transforma un punto (x, y) del marco robot al marco mundo
    w = poses[fr] @ np.array([xy[0], xy[1], 0.0, 1.0])
    return w[:2]


# Por cada peatón
obs_world = {}     # tid -> (8,2) mundo
pred_world = {}    # tid -> (20,12,2) mundo
feats_batch, order = [], []
for t in tids:
    # Trayectoria de detecciones en frames observados (rellena los huecos)
    od = [nearest_det(gt[f][t], det.get(f, np.zeros((0, 2), np.float32))) for f in frames8]
    od = np.array(fill(od), np.float32)
    # Transforma estas detecciones al marco mundo
    obs_world[t] = np.array([to_world(frames8[i], od[i]) for i in range(OBS)])
    # Genera las características (marco robot)
    feats_batch.append(scaler.transform(build_features(od)))
    order.append((t, od[-1]))                        # última detección (ancla del desplazamiento)


if feats_batch:
    # Características de todos los peatones
    Xb = np.array(feats_batch, np.float32)           # (Np,8,9)
    # Predice con el modelo
    predK = model.predict(Xb, batch_size=64, verbose=0)   # (Np,20,12,2) desplazamiento marco robot
    # Por cada peatón
    for (t, last_det), pk in zip(order, predK):
        # Suma la última predicción al desplazamiento: desplazamiento absoluto en el marco robot
        abs_ego = last_det[None, None, :] + pk        # (20,12,2) abs marco robot
        # Transforma todos los puntos al marco mundo
        pw = np.array([[to_world(C, p) for p in hyp] for hyp in abs_ego])  # (20,12,2) mundo
        # Guarda las 20 hipótesis en mundo
        pred_world[t] = pw

# 5. Dibujo
fig, ax = plt.subplots(figsize=(11, 11))
cmap = plt.get_cmap("tab20")
# Trayectoria del robot
robxy = np.array([poses[f][:2, 3] for f in sorted(poses)])
ax.plot(robxy[:, 0], robxy[:, 1], color="0.6", lw=1.5, ls=":", label="Camino del robot")
# Posición actual del robot
ax.scatter(*poses[C][:2, 3], c="k", marker="^", s=170, zorder=6, edgecolors="white",
           label="Robot (ahora)")

# Por cada peatón
for i, t in enumerate(tids):
    col = cmap(i % 20)
    # Trayectoria observada
    o = obs_world[t]
    ax.plot(o[:, 0], o[:, 1], "-", color=col, lw=2.2, zorder=4)
    ax.scatter(o[-1, 0], o[-1, 1], color=col, s=25, zorder=5)
    # Si se ha predicho 
    if t in pred_world:
        # Se dibuja las predicciones
        for hyp in pred_world[t]:
            seg = np.vstack([o[-1], hyp])                              # une últ. obs con predicción
            ax.plot(seg[:, 0], seg[:, 1], "--", color=col, lw=0.7, alpha=0.35, zorder=2)

ax.plot([], [], "-", color="0.3", lw=2.2, label="Observación (detecciones)")
ax.plot([], [], "--", color="0.3", lw=1.0, label="Predicción multimodal (20)")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Snapshot de predicción - {SEQ}, frame {C}\n({len(tids)} peatones)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
# Se guarda la figura
out = OUTF / f"fase2_snapshot_{SEQ}_{C}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
