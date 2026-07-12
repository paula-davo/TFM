"""
Fase 2 - Paso 5: evitacion con MAPA DE RIESGO GAUSSIANO (mejora del 04).

Mismo que 04 pero el campo repulsivo ya no trata cada hipótesis como obstáculo independiente. Sino
que cada punto predicho aporta una gaussiana, donde la fuerza repulsiva es el gradiente de la suma. Los
puntos de cada peatón se normalizan por el nº de modos. Densidad de probabilidad de ocupación. 
Y las hipótesis más lejanas en el tiempo pesan menos.

Uso: python evitacion/05_riesgo_gauss.py [secuencia] [frame_ini] [frame_fin] [K] [gx] [gy]
"""
import sys
from pathlib import Path
from collections import defaultdict
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
OUTF = Path(DATA_DIR) / "evitacion" / "figuras"
OUTF.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks          # noqa: E402
from evitacion_common import (fill, feats, nearest_det, load_gt, load_det,   # noqa: E402
                              load_poses)

OBS, PRED, STRIDE = 8, 12, 3
DT, V_MAX, W_MAX = 0.2, 1.2, 1.5
# Parámetros del controlador
K_ATT = 1.0         # tiron al objetivo
K_REP = 3.0         # ganancia del riesgo (escala distinta a 04: formulacion gaussiana)
SIGMA = 0.7         # ancho de la gaussiana de cada hipotesis (m) ~ espacio personal
TAU = 6.0           # descuento temporal: peso del paso k = exp(-k/TAU), k=0..11
F_REP_MAX = 4.0     # tope de la fuerza de riesgo
D_SAFE = 0.6        # parada de emergencia v=0 si un peaton real esta mas cerca
GOAL_TOL, COLLISION_R = 0.7, 0.4
CUTOFF = 3.0 * SIGMA  # ignora puntos mas lejanos que esto (gaussiana despreciable)
# ========================================================
TAG = f"fase5_s{SIGMA}_krep{K_REP}_katt{K_ATT}_dsafe{D_SAFE}"
OUTDIR = OUTF / TAG; OUTDIR.mkdir(exist_ok=True)
print(f"Params: K_ATT={K_ATT} K_REP={K_REP} SIGMA={SIGMA} TAU={TAU} D_SAFE={D_SAFE}  ->  {TAG}/")

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
# Carga modelo de múltiples predicciones y el determinista
M_MULTI = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_multimodal_best.keras", compile=False)
M_DET = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_det_best.keras", compile=False)

WK = np.exp(-np.arange(PRED) / TAU)            # (12,) descuento temporal por paso


def predict_points(c, mode):
    # Predicciones por peatón con observación completa en el frame actual.
    # 8 frames observados
    fr8 = [c - STRIDE * (OBS - 1 - i) for i in range(OBS)]
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
        nm.append(t)
    out = {}
    if not fb:
        return out
    Xb = np.array(fb, np.float32)
    # Si el modo es multipredicción
    if mode == "multi":
        # Se predicen trayectorias - marco robot
        pk = M_MULTI.predict(Xb, batch_size=64, verbose=0)            # (N,20,12,2)
        n_modes = pk.shape[1]
        for last, p, t in zip(anc, pk, nm):
            # Posición absoluta predicha - marco robot - se pasa a marco mundo
            world = np.array([[to_world(c, q) for q in hyp] for hyp in (last[None, None, :] + p)])  # (20,12,2)
            # Peso de cada punto: los más cercanos temporalmente pesan más, y se normalzia por modo (en multitrayectoria / 20)
            w = (np.tile(WK, (n_modes, 1)) / n_modes)                 # (20,12) normalizado por modos
            # Se concatena posición + peso
            out[t] = np.concatenate([world.reshape(-1, 2), w.reshape(-1, 1)], axis=1)  # (240,3)
    else:
        # Se predicen trayectorias - marco robot
        pk = M_DET.predict(Xb, batch_size=64, verbose=0)              # (N,12,2)
        for last, p, t in zip(anc, pk, nm):
            # Posición absoluta predicha - marco robot - se pasa a marco mundo
            world = np.array([to_world(c, q) for q in (last[None, :] + p)])  # (12,2)
            # Se concatena posición + peso
            out[t] = np.concatenate([world, WK.reshape(-1, 1)], axis=1)      # (12,3)
    return out


def peds_world(c):
    # Devuelve las posiciones actuales de todos los peatones en el frame dado
    return {t: to_world(c, gt[c][t]) for t in gt.get(c, {})}

# 4. Controlador de campos potenciales
def control(pos, th, goal, reppts, pcur):
    # Recibe posición y orientación actual del robot, objetivo, predicciones y posiciones 
    # actuales de los peatones.
    # Campo atractivo
    dg = goal - pos             # Vector que apunta robot-objetivo
    dist = np.linalg.norm(dg)   # Distancia al objetivo
    F = K_ATT * dg / max(dist, 1.0) # Fuerza atractiva
    # Campo repulsivo
    # Repelen: posiciones actuales de peatones + predicciones
    if len(reppts):
        P = reppts[:, :2]; wt = reppts[:, 2]
        diff = pos - P                          # Vector obstáculo-robot
        d2 = np.einsum("ij,ij->i", diff, diff)  # Distancia al cuadrado robot-obstáculo
        m = d2 < CUTOFF * CUTOFF                # Máscara: radio de influencia 
        if m.any():
            # Gaussiana ponderada: pesos
            g = wt[m] * np.exp(-d2[m] / (2.0 * SIGMA * SIGMA))
            # Potencial repulsivo - suma potencial repulsivo de todos los obstáculos
            Frep = K_REP * (diff[m] * (g / (SIGMA * SIGMA))[:, None]).sum(0)
            # Limita esta suma a un máximo
            nrm = np.linalg.norm(Frep)
            if nrm > F_REP_MAX:
                Frep *= F_REP_MAX / nrm
            # Suma atractivo + repulsivo
            F = F + Frep
    # fuerza -> (v, w)
    desired = np.arctan2(F[1], F[0])
    err = (desired - th + np.pi) % (2 * np.pi) - np.pi
    w = float(np.clip(2.0 * err, -W_MAX, W_MAX))
    v = float(np.clip(V_MAX * max(0.0, np.cos(err)), 0.0, V_MAX))
    # Si el peatón está a menos de D_SAFE -> parada de emergencia
    if pcur and min(np.linalg.norm(pos - p) for p in pcur.values()) < D_SAFE:
        v = 0.0
    return v, w


def run_avoidance(mode):
    goal = GOAL     # Posición del robot objetivo 
    pos = poses[F0][:2, 3].astype(float).copy() # Posición inicial 
    R0 = poses[F0][:3, :3]
    th = float(np.arctan2(R0[1, 0], R0[0, 0]))  # Orientación inicial
    path = [pos.copy()]                         # Lista con recorrido (pos inicial)
    held = {}
    min_dist = np.inf; coll = 0; reached = False
    # Para cada frame de la escena
    for i, c in enumerate(CS):
        # Cada K pasos, repredice trayectorias
        if i % K == 0:
            held = predict_points(c, mode)
        # Posiciones actuales de los peatones en ese frame
        pcur = peds_world(c)
        if pcur:
            # Si la distancia del peatón más cercano es menor a COLLISION_R -> se considera colisión
            dmin = min(np.linalg.norm(pos - p) for p in pcur.values())
            min_dist = min(min_dist, dmin)
            coll += int(dmin < COLLISION_R)
        # Si está a menos de GOAL_TOL -> se considera objetivo alcanzado
        if np.linalg.norm(goal - pos) < GOAL_TOL:
            reached = True; path.append(pos.copy()); continue
        cur = np.array([[p[0], p[1], 1.0] for p in pcur.values()]) if pcur else np.zeros((0, 3))
        pred = np.vstack(list(held.values())) if held else np.zeros((0, 3))
        reppts = np.vstack([cur, pred]) if (len(cur) + len(pred)) else np.zeros((0, 3))
        # Se aplica el control
        v, w = control(pos, th, goal, reppts, pcur)
        th += w * DT
        pos = pos + v * DT * np.array([np.cos(th), np.sin(th)])
        # Se suma nueva posición
        path.append(pos.copy())
    path = np.array(path)
    plen = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return path, dict(min_dist=min_dist, coll=coll, reached=reached, plen=plen)


# Objetivo
start = poses[F0][:2, 3]    # Posición inicial
disp_rec = np.linalg.norm(poses[max(CS)][:2, 3] - start)    # Desplazamiento neto grabado
# Meta: o pasada como parámetro o último punto de la trayectoria del robot
if len(sys.argv) > 6:   
    GOAL = np.array([float(sys.argv[5]), float(sys.argv[6])])
else:
    GOAL = poses[max(CS)][:2, 3]
    if disp_rec < 1.5:
        print("WARNING: el robot no se mueve. Llama dando un objetivo: ... [K] [gx] [gy]")
print(f"Objetivo: ({GOAL[0]:.1f}, {GOAL[1]:.1f})")

# Ejecutar
rec_path = np.array([poses[c][:2, 3] for c in CS])  # Camino real grabado del robot
rec_min = np.inf                                    # Distancia mínima a peatón del robot real
# Se compara
for i, c in enumerate(CS):
    for p in peds_world(c).values():
        rec_min = min(rec_min, np.linalg.norm(rec_path[i] - p))

print("Ejecutando determinista..."); det_path, det_m = run_avoidance("det")
print("Ejecutando multimodal...");   mul_path, mul_m = run_avoidance("multi")

print(f"\n===== COMPARATIVA (riesgo gaussiano) - {SEQ} {F0}-{F1} =====")
print(f"{'':24}{'Sin evitar':>12}{'Determinista':>14}{'Multimodal':>12}")
print(f"{'Dist. min. peaton (m)':24}{rec_min:>12.2f}{det_m['min_dist']:>14.2f}{mul_m['min_dist']:>12.2f}")
print(f"{'Colisiones (<%.1fm)' % COLLISION_R:24}{'-':>12}{det_m['coll']:>14}{mul_m['coll']:>12}")
print(f"{'Llega al objetivo':24}{'-':>12}{str(det_m['reached']):>14}{str(mul_m['reached']):>12}")
print(f"{'Longitud camino (m)':24}{'-':>12}{det_m['plen']:>14.2f}{mul_m['plen']:>12.2f}")

# Figura 1. comparativa
# Trayectoria completa para cada peatón.
ped_tracks = defaultdict(list)
for c in CS:
    for t in gt.get(c, {}):
        ped_tracks[t].append(to_world(c, gt[c][t]))
goal = GOAL
# Dibuja las trayectorias de los peatones
fig, ax = plt.subplots(figsize=(9, 9))
for t, pts in ped_tracks.items():
    p = np.array(pts)
    if len(p) >= 2:
        ax.plot(p[:, 0], p[:, 1], "-", color="0.8", lw=1.0, zorder=1)
# Dibuja los 3 caminos del robot: sin evitar, determinista, multitrayectoria
ax.plot(rec_path[:, 0], rec_path[:, 1], color="0.5", ls=":", lw=2, label="Sin evitar", zorder=3)
ax.plot(det_path[:, 0], det_path[:, 1], color="tab:blue", ls="--", lw=2.2, label="Determinista", zorder=4)
ax.plot(mul_path[:, 0], mul_path[:, 1], color="k", lw=2.6, label="Multimodal", zorder=5)
# Destaca el inicio y el objetivo
ax.scatter(*poses[F0][:2, 3], c="green", marker="o", s=90, zorder=6, edgecolors="white", label="Inicio")
ax.scatter(*goal, marker="x", s=120, c="red", linewidths=3, zorder=6, label="Objetivo")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Mapa de riesgo gaussiano en {SEQ}")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
# Guarda la figura
png = OUTDIR / f"fase5_comparativa_{SEQ}_{F0}-{F1}.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"Guardado: {png}")

# Figura 2: GIF comparativo
# 2 caminos del robot, inicio y objetivo
focus = np.vstack([det_path, mul_path, GOAL[None, :], start[None, :]])
# Límites con 3 m de margen
gx0, gx1 = focus[:, 0].min() - 3, focus[:, 0].max() + 3
gy0, gy1 = focus[:, 1].min() - 3, focus[:, 1].max() + 3
figg, axg = plt.subplots(figsize=(8, 8))


def updg(i):
    # Se comienza la figura
    axg.clear()
    axg.set_xlim(gx0, gx1); axg.set_ylim(gy0, gy1); axg.set_aspect("equal"); axg.grid(alpha=0.3)
    axg.set_xlabel("x (m)"); axg.set_ylabel("y (m)")
    # Frame real del paso i
    c = CS[i]
    # Dibuja la posición actual de los peatones
    pw = peds_world(c)
    if pw:
        P = np.array(list(pw.values()))
        axg.scatter(P[:, 0], P[:, 1], c="0.5", s=35, zorder=3)
    # Dibuja los caminos del robot (determinista y multimodal)
    j = min(i + 1, len(det_path))
    axg.plot(det_path[:j, 0], det_path[:j, 1], "--", color="tab:blue", lw=2, zorder=4)
    axg.plot(mul_path[:j, 0], mul_path[:j, 1], "k-", lw=2.5, zorder=5)
    axg.scatter(*det_path[j - 1], c="tab:blue", marker="s", s=70, zorder=6, edgecolors="white")
    axg.scatter(*mul_path[j - 1], c="k", marker="^", s=130, zorder=7, edgecolors="white")
    # Se dibuja el inicio y el objetivo
    axg.scatter(*start, c="green", marker="o", s=80, zorder=6, edgecolors="white")
    axg.scatter(*GOAL, marker="x", c="red", s=120, linewidths=3, zorder=6)
    axg.set_title(f"Det (azul) vs Multi (negro), peatones (gris) - frame {c}", fontsize=10)


animg = FuncAnimation(figg, updg, frames=len(CS), interval=200)
# Se guarda el GIF
gifc = OUTDIR / f"fase5_comparativa_{SEQ}_{F0}-{F1}.gif"
animg.save(gifc, writer=PillowWriter(fps=5)); plt.close(figg)
print(f"Guardado: {gifc}")
