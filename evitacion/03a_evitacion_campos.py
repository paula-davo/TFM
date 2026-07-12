"""
Fase 2 - Paso 3A: evitación con CAMPOS POTENCIALES.

El robot se mueve en marco mundo. Algoritmo:
- Campo atractivo hacia el objetivo -> posición final real del robot.
- Campo repulsivo -> predicciones del modelo y posiciones actuales de los peatones.
- Parada de emergencia.

Se re-predice cada K pasos.

Uso: python evitacion/03a_evitacion_campos.py [secuencia] [frame_ini] [frame_fin] [K]
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
from evitacion_common import (fill, feats, nearest_det, load_gt, load_det,   # noqa: E402
                              load_poses)

OBS, PRED, STRIDE = 8, 12, 3
TRAIL = 8       # nº de pasos de estela del peatón (observación)

# Parámetros del controlador
DT = 0.2            # s por paso
V_MAX = 1.2         # m/s
W_MAX = 1.5         # rad/s
K_ATT = 1.0         # ganancia atractiva
K_REP = 0.5         # ganancia repulsiva
R_INFL = 2.5        # m, radio de influencia repulsivo
F_REP_MAX = 3.0     # cap de la fuerza repulsiva
D_SAFE = 0.6        # m, parada de emergencia
GOAL_TOL = 0.7      # m, objetivo alcanzado
COLLISION_R = 0.4   # m, radio de colisión (para métrica)

SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0"  # Secuencia
F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 120                 # Frame inicial
F1 = int(sys.argv[3]) if len(sys.argv) > 3 else 360                 # Frame final
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3                    # re-planificar cada K pasos
CS = list(range(F0, F1 + 1, STRIDE))                                # frames "actuales" (submuestreados)

# 1. Cargar poses de robot (marco robot a marco mundo)
# Frames para los que se necesita pose
need_lo, need_hi = min(CS) - STRIDE * (OBS - 1), max(CS)
poses = load_poses(RECON, SEQ, range(need_lo, need_hi + 1))
if poses is None:
    raise SystemExit(f"No hay .npz que cubra {need_lo}..{need_hi}. Ejecuta antes 01.")

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
# Asigna un color a cada peatón
color = {t: plt.get_cmap("tab20")(i % 20) for i, t in enumerate(sorted({t for c in CS for t in gt.get(c, {})}))}


def replan(c):
    # Predicciones por peatón con observación completa en el frame actual.
    # 8 frames observados
    fr8 = [c - STRIDE * (OBS - 1 - i) for i in range(OBS)]
    # Peatones válidos -> etiqueta en los 8 frames de la ventana
    tids = [t for t in gt[c] if all(t in gt.get(f, {}) for f in fr8)]
    fb, anc, nm = [], [], []
    # Para cada peatón
    for t in tids:
        # Detección asociada por frame, rellena huecos
        od = np.array(fill([nearest_det(gt[f][t], det.get(f, np.zeros((0, 2), np.float32))) for f in fr8]), np.float32)
        # 9 características - marco robot
        fb.append(scaler.transform(feats(od)))
        # última detección - frame 8 (C)
        anc.append(od[-1])
        nm.append(t)    # label ID del peatón
    # Si se han extraido características
    fans = {}
    if fb:
        # Se predicen trayectorias - marco robot
        pk = model.predict(np.array(fb, np.float32), batch_size=64, verbose=0)
        # Para cada predicción, ID y última detección
        for last, p, t in zip(anc, pk, nm):
            # Posición absoluta predicha - marco robot - se pasa a marco mundo
            fans[t] = np.array([[to_world(c, q) for q in (last[None, :] + h)] for h in p])  # (20,12,2)
    return fans


def peds_world(c):
    # Devuelve las posiciones actuales de todos los peatones en el frame dado
    return {t: to_world(c, gt[c][t]) for t in gt.get(c, {})}


def ped_trail(c, t):
    # Calcula la estela real en los frames de observación -> etiquetas 3D
    # Frames necesarios (con stride y trail)
    fr = [c - STRIDE * (TRAIL - 1 - i) for i in range(TRAIL)]
    # De cada uno, comprueba que exista pose del robot para ese frame y que el peatón tenga
    # etiqueta. Después convierte a marco mundo, la posición de etiqueta 3D (cx, cy) (marco robot)
    pts = [to_world(f, gt[f][t]) for f in fr if f in poses and t in gt.get(f, {})]
    return np.array(pts) if len(pts) >= 2 else None


# 4. Controlador de campos potenciales
def control(pos, th, goal, fans, pcur):
    # Recibe posición y orientación actual del robot, objetivo, predicciones y posiciones 
    # actuales de los peatones.
    # Campo atractivo
    dg = goal - pos                 # Vector que apunta robot-objetivo
    dist = np.linalg.norm(dg)       # Distancia al objetivo
    F = K_ATT * dg / max(dist, 1.0) # Fuerza atractiva
    # Campo repulsivo
    # Repelen: posiciones actuales de peatones + predicciones
    rep_pts = list(pcur.values())
    for hyp in fans.values():
        rep_pts.extend(hyp.reshape(-1, 2))
    if rep_pts:
        P = np.array(rep_pts)
        diff = pos - P                       # Vector obstáculo-robot
        d = np.linalg.norm(diff, axis=1)     # Distancia robot-obstáculo
        m = (d < R_INFL) & (d > 1e-3)        # Máscara: radio de influencia 
        if m.any():
            # Potencial repulsivo
            contrib = (diff[m] / d[m, None]) * (K_REP * (1.0 / d[m] - 1.0 / R_INFL))[:, None]
            # Suma potencial repulsivo de todos los obstáculos
            Frep = contrib.sum(0)
            # Limita esta suma a un máximo
            nrm = np.linalg.norm(Frep)
            if nrm > F_REP_MAX:
                Frep *= F_REP_MAX / nrm
            # Suma atractivo + repulsivo
            F = F + Frep
    # fuerza -> (v, w)
    desired = np.arctan2(F[1], F[0])                    # Ángulo hacia el que apunta la fuerza (deseado)
    err = (desired - th + np.pi) % (2 * np.pi) - np.pi  # Diferencia entre ángulo deseado y orientación actual
    w = float(np.clip(2.0 * err, -W_MAX, W_MAX))        # Velocidad angular proporcional al error angular
    v = float(np.clip(V_MAX * max(0.0, np.cos(err)), 0.0, V_MAX))   # Velocidad lineal 
    # Si el peatón está a menos de D_SAFE -> parada de emergencia
    if pcur:
        dmin = min(np.linalg.norm(pos - p) for p in pcur.values())
        if dmin < D_SAFE:
            v = 0.0
    return v, w


# 5. Simulación
goal = poses[max(CS)][:2, 3]                    # Posición del robot objetivo (último frame)
pos = poses[F0][:2, 3].astype(float).copy()     # Posición inicial 
R0 = poses[F0][:3, :3]
th = float(np.arctan2(R0[1, 0], R0[0, 0]))      # Orientación inicial
robot_path = [pos.copy()]                       # Lista con recorrido (pos inicial)
fans = {}
min_dist = np.inf; collisions = 0; reached = False

# Para cada frame de la escena
for i, c in enumerate(CS):
    # Cada K pasos, repredice trayectorias
    if i % K == 0:
        fans = replan(c)
    # Posiciones actuales de los peatones en ese frame
    pcur = peds_world(c)
    if pcur:
        # Si la distancia del peatón más cercano es menor a COLLISION_R -> se considera colisión
        dmin = min(np.linalg.norm(pos - p) for p in pcur.values())
        min_dist = min(min_dist, dmin)
        if dmin < COLLISION_R:
            collisions += 1
    # En cambio si está a menos de GOAL_TOL -> se considera objetivo alcanzado
    if np.linalg.norm(goal - pos) < GOAL_TOL:
        reached = True
        robot_path.append(pos.copy()); continue
    # Se aplica el control
    v, w = control(pos, th, goal, fans, pcur)
    th += w * DT
    pos = pos + v * DT * np.array([np.cos(th), np.sin(th)])
    # Se suma nueva posición
    robot_path.append(pos.copy())

robot_path = np.array(robot_path)
path_len = float(np.sum(np.linalg.norm(np.diff(robot_path, axis=0), axis=1)))

# Baseline: robot grabado
rec_path = np.array([poses[c][:2, 3] for c in CS])  # Camino real grabado del robot
rec_min = np.inf                                    # Distancia mínima a peatón del robot real
# Se compara
for i, c in enumerate(CS):
    for t, p in peds_world(c).items():
        rec_min = min(rec_min, np.linalg.norm(rec_path[i] - p))

print("\n===== MÉTRICAS =====")
print(f"{'':22}{'Original':>10}{'Evitación':>12}")
print(f"{'Dist. mínima peatón (m)':22}{rec_min:>10.2f}{min_dist:>12.2f}")
print(f"{'Colisiones (<%.1fm)' % COLLISION_R:22}{'-':>10}{collisions:>12}")
print(f"{'Llega al objetivo':22}{'-':>10}{str(reached):>12}")
print(f"{'Longitud camino (m)':22}{'-':>10}{path_len:>12.2f}")

# 6. Dibujo
# Recorre todos los frames que se van a animar y guarda las posiciones de los peatones en cada
# frame (marco mundo)
allw = []
for c in CS:
    for t in gt.get(c, {}):
        allw.append(to_world(c, gt[c][t]))
# Añade el recorrido del robot
allw = np.array(allw + list(rec_path) + list(robot_path) + [goal])
# Límites de la figura
xlim = (allw[:, 0].min() - 2, allw[:, 0].max() + 2)
ylim = (allw[:, 1].min() - 2, allw[:, 1].max() + 2)


def draw(ax, i, fans_i):
    # Se comienza la figura
    ax.clear()
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    # Frame del paso i -> dibuja el camino original grabado y el de evitación
    c = CS[i]
    ax.plot(rec_path[:, 0], rec_path[:, 1], color="0.7", lw=1.5, ls=":", zorder=1)   # original
    ax.plot(robot_path[:i + 2, 0], robot_path[:i + 2, 1], "k-", lw=2.5, zorder=5)     # evitación
    ax.scatter(*robot_path[min(i + 1, len(robot_path) - 1)], c="k", marker="^", s=150,
               zorder=6, edgecolors="white")    # Resalta la pos actual del robot
    ax.scatter(*goal, marker="*", s=260, c="gold", edgecolors="k", zorder=6)
    # Recorre los peatones presentes en el frame
    for t in gt.get(c, {}):
        # Comprueba que tenga estela (observación pasada) y la dibuja
        tr = ped_trail(c, t)
        if tr is None:
            continue
        col = color[t]
        ax.plot(tr[:, 0], tr[:, 1], "-", color=col, lw=2.0, zorder=3)   # observación
        ax.scatter(tr[-1, 0], tr[-1, 1], color=col, s=22, zorder=4)
        # Si el peatón tiene predicciones -> las dibuja
        if t in fans_i:
            for hyp in fans_i[t]:
                seg = np.vstack([tr[-1], hyp])
                ax.plot(seg[:, 0], seg[:, 1], "--", color=col, lw=0.6, alpha=0.28, zorder=2)


# Figura 1. GIF
state = {"fans": {}}
fig, ax = plt.subplots(figsize=(9, 9))


def upd(i):
    # Dibuja la escena
    if i % K == 0:
        state["fans"] = replan(CS[i])
    draw(ax, i, state["fans"])
    ax.set_title(f"Evitación (campos potenciales) - {SEQ}  (frame {CS[i]})", fontsize=11)

# Genera y guarda la animación
print(f"\nGenerando animación ({len(CS)} pasos)...")
anim = FuncAnimation(fig, upd, frames=len(CS), interval=200)
gif = OUTF / f"fase3_evitacion_{SEQ}_{F0}-{F1}.gif"
anim.save(gif, writer=PillowWriter(fps=5)); plt.close(fig)
print(f"Guardado: {gif}")

# Figura 2. Capturas en distintos frames de una escena
# Selecciona 5 frames y crea 5 subplots
idxs = np.linspace(0, len(CS) - 1, 5).astype(int)
fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
# Dibuja esos 5 instantes
for ax, i in zip(axes, idxs):
    draw(ax, i, replan(CS[i]))
    ax.set_title(f"frame {CS[i]}", fontsize=10)
axes[0].plot([], [], color="0.7", ls=":", label="Camino original")
axes[0].plot([], [], "k-", lw=2, label="Evitación")
axes[0].scatter([], [], marker="*", c="gold", edgecolors="k", label="Objetivo")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle(f"Evitación con campos potenciales - {SEQ}", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
# Guarda la figura
png = OUTF / f"fase3_evitacion_{SEQ}_{F0}-{F1}.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"Guardado: {png}")
