"""
Fase 2 - Paso 2B: bucle temporal en modo REPLAY (sin evitar todavía).

GIF y tira de figuras donde el robot continúa su camino grabado, los peatones su trayectoria
real (etiquetas 3D) y se muestran las predicciones (cada K pasos se re-predice) con el modelo
de múltiples trayectorias.

Uso:
    python evitacion/02b_bucle_replay.py [secuencia] [frame_ini] [frame_fin] [K]
Por defecto: huang-2-2019-01-25_0, 120, 360, K=3.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
                              load_poses, feats as build_features)

OBS, PRED, STRIDE = 8, 12, 3
TRAIL = 8        # nº de pasos de estela del peatón (observación)

SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0"  # Secuencia
F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 120                 # Frame inicial
F1 = int(sys.argv[3]) if len(sys.argv) > 3 else 360                 # Frame final
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3          # re-planificar cada K pasos

CS = list(range(F0, F1 + 1, STRIDE))                       # frames "actuales" (submuestreados)

# 1. Cargar poses de robot (marco robot a marco mundo)
# Frames para los que se necesita pose
need_lo, need_hi = min(CS) - STRIDE * (OBS - 1), max(CS)
poses = load_poses(RECON, SEQ, range(need_lo, need_hi + 1))
if poses is None:
    raise SystemExit(f"No hay .npz que cubra {need_lo}..{need_hi}. Ejecuta antes 01 con ese rango.")

# 2. Carga de datos: etiquetas y detecciones 3D
gt, det = load_gt(LAB, SEQ), load_det(DET, SEQ)

# 3. Preparación de los datos

def to_world(fr, xy):
    # Transforma un punto (x, y) del marco robot al marco mundo
    return (poses[fr] @ np.array([xy[0], xy[1], 0.0, 1.0]))[:2]


# Carga los datos de detecciones 3D con partición oficial
X3d = np.load(f"{DATA_DIR}/X3d_det_motion.npy")
ids = np.load(f"{DATA_DIR}/subtrack_ids_ds.npy", allow_pickle=True)
tm, _ = official_masks(ids)
# Ajusta scaler
scaler = StandardScaler().fit(X3d[tm].reshape(-1, 9))
# Carga el modelo
model = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_multimodal_best.keras", compile=False)

# IDs -> peatones -> obtenidos de los frames presentes en las etiquetas de la secuencia y sin repetir
all_tids = sorted({t for c in CS for t in gt.get(c, {})})
# Asigna un color a cada peatón
color = {t: plt.get_cmap("tab20")(i % 20) for i, t in enumerate(all_tids)}

# 4. Predicciones y Figuras

def replan(c):
    # Predicciones por peatón con observación completa en el frame actual.
    # 8 frames observados
    fr8 = [c - STRIDE * (OBS - 1 - i) for i in range(OBS)]
    # Peatones válidos -> etiqueta en los 8 frames de la ventana
    tids = [t for t in gt[c] if all(t in gt.get(f, {}) for f in fr8)]
    fans, feats, anchors, names = {}, [], [], []
    # Para cada peatón
    for t in tids:
        # Detección asociada por frame
        od = [nearest_det(gt[f][t], det.get(f, np.zeros((0, 2), np.float32))) for f in fr8]
        # Rellena huecos
        od = np.array(fill(od), np.float32)
        # 9 características - marco robot
        feats.append(scaler.transform(build_features(od)))
        # última detección - frame 8 (C)
        anchors.append((c, od[-1])) # (frame actual, última detección)
        names.append(t)             # label ID del peatón
    # Si se han extraido características
    if feats:
        # Se predicen trayectorias - marco robot
        predK = model.predict(np.array(feats, np.float32), batch_size=64, verbose=0)
        # Para cada predicción, ID, frame actual y última detección
        for (cc, last), pk, t in zip(anchors, predK, names):
            # Posición absoluta predicha - marco robot
            abs_ego = last[None, None, :] + pk
            # Se pasa a marco mundo y se guarda
            fans[t] = np.array([[to_world(cc, p) for p in hyp] for hyp in abs_ego])
    return fans


def ped_trail(c, t):
    # Calcula la estela real en los frames de observación -> etiquetas 3D
    # Frames necesarios (con stride y trail)
    fr = [c - STRIDE * (TRAIL - 1 - i) for i in range(TRAIL)]
    # De cada uno, comprueba que exista pose del robot para ese frame y que el peatón tenga
    # etiqueta. Después convierte a marco mundo, la posición de etiqueta 3D (cx, cy) (marco robot)
    pts = [to_world(f, gt[f][t]) for f in fr if f in poses and t in gt.get(f, {})]
    return np.array(pts) if pts else None


# Recorre todos los frames que se van a animar y guarda las posiciones de los peatones en cada
# frame (marco mundo)
allw = []
for c in CS:
    for t in gt.get(c, {}):
        allw.append(to_world(c, gt[c][t]))
# Añade el recorrido del robot
allw = np.array(allw + [poses[c][:2, 3] for c in CS])
# Límites de la figura
xlim = (allw[:, 0].min() - 2, allw[:, 0].max() + 2)
ylim = (allw[:, 1].min() - 2, allw[:, 1].max() + 2)


def draw(ax, c, fans):
    # Se comienza la figura
    ax.clear()
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    # Dibuja poses del robot hasta frame actual, marcando la posición actual
    up_to = [poses[cc][:2, 3] for cc in CS if cc <= c]
    rp = np.array(up_to)
    ax.plot(rp[:, 0], rp[:, 1], color="0.6", lw=1.5, ls=":")
    ax.scatter(*poses[c][:2, 3], c="k", marker="^", s=150, zorder=6, edgecolors="white")
    # Recorre los peatones presentes en el frame actual
    for t in gt.get(c, {}):
        # Comprueba que tenga estela (observación pasada) y la dibuja
        tr = ped_trail(c, t)
        if tr is None or len(tr) < 2:
            continue
        col = color[t]
        ax.plot(tr[:, 0], tr[:, 1], "-", color=col, lw=2.0, zorder=4)       # observación
        ax.scatter(tr[-1, 0], tr[-1, 1], color=col, s=22, zorder=5)
        # Si el peatón tiene predicciones -> las dibuja
        if t in fans:
            for hyp in fans[t]:
                seg = np.vstack([tr[-1], hyp])
                ax.plot(seg[:, 0], seg[:, 1], "--", color=col, lw=0.6, alpha=0.30, zorder=2)


# Figura 1. GIF
state = {"fans": {}}
fig, ax = plt.subplots(figsize=(9, 9))


def update(i):
    # Dibuja la escena
    c = CS[i]
    if i % K == 0:
        state["fans"] = replan(c)
    draw(ax, c, state["fans"])
    ax.set_title(f"Predicción multimodal en {SEQ}  (frame {c})", fontsize=11)


# Genera y guarda la animación
print(f"Generando animación ({len(CS)} pasos)...")
anim = FuncAnimation(fig, update, frames=len(CS), interval=200)
gif = OUTF / f"fase2_replay_{SEQ}_{F0}-{F1}.gif"
anim.save(gif, writer=PillowWriter(fps=5))
plt.close(fig)
print(f"Guardado: {gif}")

# Figura 2. Capturas en distintos frames de una escena
# Selecciona 5 frames y crea 5 subplots
idxs = np.linspace(0, len(CS) - 1, 5).astype(int)
fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
# Dibuja esos 5 instantes
for ax, i in zip(axes, idxs):
    draw(ax, CS[i], replan(CS[i]))
    ax.set_title(f"frame {CS[i]}", fontsize=10)
axes[0].plot([], [], "-", color="0.3", lw=2, label="Observación")
axes[0].plot([], [], "--", color="0.3", lw=1, label="Predicción (20)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle(f"Predicción multimodal en {SEQ}", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
# Guarda la figura
png = OUTF / f"fase2_replay_{SEQ}_{F0}-{F1}.png"
fig.savefig(png, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {png}")
