"""
Evitación con campos potenciales.
- Campo atractivo hacia el objetivo.
- Campo repulsivo por predicciones del modelo y posiciones actuales de los peatones.
- Evitación de emergencia.
- Se re-predice cada K pasos.

Uso: python evitacion/03a_evitacion_campos.py [secuencia] [frame_ini] [frame_fin] [K] [gx] [gy]
Objetivo por defecto = pose final grabada del robot; en escenas donde el robot casi no se mueve
conviene dar un objetivo manual [gx gy] para que tenga a dónde ir.
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
from evitacion_common import (load_gt, load_det, load_poses, load_img2d,   # noqa: E402
                              nearest_det, fill, to_world, predict_scene)

OBS, PRED, STRIDE = 8, 8, 3
TRAIL = 8       # nº de pasos de estela del peatón (observación)

# Parámetros del controlador
DT = 0.2            # s por paso
V_MAX = 1.2         # velocidad lineal máxima
W_MAX = 1.5         # velocidad angular máxima
K_ATT = 1.0         # ganancia atractiva
K_REP = 0.5         # ganancia repulsiva
R_INFL = 2.5        # radio de influencia repulsivo
F_REP_MAX = 3.0     # cap de la fuerza repulsiva
D_SAFE = 0.6        # evitación de emergencia
GOAL_TOL = 0.7      # objetivo alcanzado
COLLISION_R = 0.4   # radio de colisión
VIEW_R = 3.0        # margen alrededor del robot para recortar la vista


def peds_world(c, poses, gt):
    # Posiciones GT (etiquetas) de todos los peatones en el frame dado
    return {t: to_world(poses, c, gt[c][t]) for t in gt.get(c, {})}


def peds_world_det(c, poses, det, gt):
    # Posiciones observadas por el robot (detección asociada a cada etiqueta) en el frame dado
    out = {}
    for t in gt.get(c, {}):
        d = nearest_det(gt[c][t], det.get(c, np.zeros((0, 2), np.float32)))
        if d is not None:
            out[t] = to_world(poses, c, d)
    return out


def ped_trail(c, t, poses, det, gt):
    # Detecciones 3D en los frames de observación
    fr = [c - STRIDE * (TRAIL - 1 - i) for i in range(TRAIL)]
    fr = [f for f in fr if f in poses and t in gt.get(f, {})]
    if len(fr) < 2:
        return None
    # Detección más cercana a la etiqueta en cada frame
    od = np.array(fill([nearest_det(gt[f][t], det.get(f, np.zeros((0, 2), np.float32))) for f in fr]),
                  np.float32)
    return np.array([to_world(poses, f, od[i]) for i, f in enumerate(fr)])


def control(pos, th, goal, fans, pcur):
    # Control por campos potenciales
    # Campo atractivo
    dg = goal - pos                 # Vector robot-objetivo
    dist = np.linalg.norm(dg)       # Distancia al objetivo
    F = K_ATT * dg / max(dist, 1.0) # Fuerza atractiva
    # Campo repulsivo
    # Repelen: posiciones detectadas de peatones + predicciones
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
    # Fuerza -> (v, w)
    desired = np.arctan2(F[1], F[0])                    # Ángulo hacia el que apunta la fuerza (deseado)
    err = (desired - th + np.pi) % (2 * np.pi) - np.pi  # Diferencia entre ángulo deseado y orientación actual
    w = float(np.clip(2.0 * err, -W_MAX, W_MAX))        # Velocidad angular proporcional al error angular
    v = float(np.clip(V_MAX * max(0.0, np.cos(err)), 0.0, V_MAX))   # Velocidad lineal
    # Evasión de emergencia: si un peatón entra en D_SAFE, el robot lo evade en dirección contraria.
    if pcur:
        # Posiciones detectadas de los peatones
        Pc = np.array(list(pcur.values()))
        # Distancias robot-peatón
        dd = np.linalg.norm(pos - Pc, axis=1)
        # Peatones a menos de D_SAFE
        close = (dd < D_SAFE) & (dd > 1e-3)
        # Si alguno está cerca
        if close.any():
            # Dirección de escape
            esc = ((pos - Pc[close]) / (dd[close][:, None] ** 2)).sum(0)
            if np.linalg.norm(esc) > 1e-6:
                # Ángulo de la dirección de escape
                e_ang = np.arctan2(esc[1], esc[0])
                # Error respecto a la orientación actual
                e_err = (e_ang - th + np.pi) % (2 * np.pi) - np.pi
                # Velocidad angular y lineal hacia la dirección de escape
                w = float(np.clip(2.0 * e_err, -W_MAX, W_MAX))
                v = float(np.clip(V_MAX * np.cos(e_err), -V_MAX, V_MAX))
            else:
                v = 0.0
    return v, w


def draw(ax, i, fans_i, poses, gt, det, color, CS, rec_path, robot_path, goal, view):
    # Vista fija en torno al robot
    VX0, VX1, VY0, VY1 = view
    ax.clear()
    c = CS[i]
    ax.set_xlim(VX0, VX1); ax.set_ylim(VY0, VY1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)", fontsize=15); ax.set_ylabel("y (m)", fontsize=15); ax.tick_params(labelsize=13)
    # Frame del paso i -> dibuja el camino original grabado y el de evitación
    ax.plot(rec_path[:, 0], rec_path[:, 1], color="0.7", lw=1.5, ls=":", zorder=1)
    ax.plot(robot_path[:i + 2, 0], robot_path[:i + 2, 1], "k-", lw=2.5, zorder=5)
    ax.scatter(*robot_path[min(i + 1, len(robot_path) - 1)], c="k", marker="^", s=150,
               zorder=6, edgecolors="white")    # Resalta la pos actual del robot
    ax.scatter(*goal, marker="*", s=260, c="gold", edgecolors="k", zorder=6)
    # Recorre los peatones presentes en el frame
    for t in gt.get(c, {}):
        # Peatones dentro de la ventana
        pw = to_world(poses, c, gt[c][t])
        if not (VX0 <= pw[0] <= VX1 and VY0 <= pw[1] <= VY1):
            continue
        # Dibuja observación pasada
        tr = ped_trail(c, t, poses, det, gt)
        if tr is None:
            continue
        col = color[t]
        ax.plot(tr[:, 0], tr[:, 1], "-", color=col, lw=2.0, zorder=3)
        ax.scatter(tr[-1, 0], tr[-1, 1], color=col, s=22, zorder=4)
        # Si existe predicción del peatón, la dibuja
        if t in fans_i:
            seg = np.vstack([tr[-1], fans_i[t]])
            ax.plot(seg[:, 0], seg[:, 1], "--", color=col, lw=1.8, alpha=0.9, zorder=2)


def main():
    SEQ = sys.argv[1] if len(sys.argv) > 1 else "huang-2-2019-01-25_0"  # Secuencia
    F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 120                 # Frame inicial
    F1 = int(sys.argv[3]) if len(sys.argv) > 3 else 360                 # Frame final
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 3                    # Re-predecir cada K pasos
    CS = list(range(F0, F1 + 1, STRIDE))                                # Frames actuales

    # 1. Cargar poses de robot (marco robot a marco mundo)
    need_lo, need_hi = min(CS) - STRIDE * (OBS - 1), max(CS)
    poses = load_poses(RECON, SEQ, range(need_lo, need_hi + 1))
    if poses is None:
        raise SystemExit(f"No hay .npz que cubra {need_lo}..{need_hi}. Ejecuta antes 01b.")

    # 2. Carga de datos: etiquetas, detecciones 3D y trayectorias 2D de imagen
    gt, det = load_gt(LAB, SEQ), load_det(DET, SEQ)
    traj2d = load_img2d(f"{DATA_DIR}/jrdb_clean_trajectories.csv", SEQ)

    # 3. Preparación de los datos
    # Carga imagen 2D (11) + detecciones 3D (9), con partición oficial
    Ximg = np.load(f"{DATA_DIR}/data/X_ds.npy")            # (N,8,11)
    X3d = np.load(f"{DATA_DIR}/data/X3d_det_motion.npy")   # (N,8,9)
    X20 = np.concatenate([Ximg, X3d], axis=2)         # (N,8,20)
    ids = np.load(f"{DATA_DIR}/data/subtrack_ids_ds.npy", allow_pickle=True)
    tm, _ = official_masks(ids)
    # Un scaler de 20 características (fusión) y otro de 9 (detecciones)
    scaler20 = StandardScaler().fit(X20[tm].reshape(-1, 20))
    scaler9 = StandardScaler().fit(X3d[tm].reshape(-1, 9))
    # Modelos: fusión temprana (2D+3D) y base (3D)
    model_fus = load_model(f"{DATA_DIR}/prediccion_3d/best/combined3d_reg_best.keras", compile=False)
    model_base = load_model(f"{DATA_DIR}/prediccion_3d/best/lidar3d_det_mae_best.keras", compile=False)
    # Asigna un color a cada peatón
    color = {t: plt.get_cmap("tab20")(i % 20) for i, t in enumerate(sorted({t for c in CS for t in gt.get(c, {})}))}

    # 4. Simulación
    # Objetivo: manual [gx gy] si se pasa; si no, la pose final grabada del robot
    if len(sys.argv) > 6:
        goal = np.array([float(sys.argv[5]), float(sys.argv[6])])
    else:
        # Último frame
        goal = poses[max(CS)][:2, 3]
        if np.linalg.norm(poses[max(CS)][:2, 3] - poses[F0][:2, 3]) < 1.5:
            print("WARNING: el robot grabado apenas se mueve. Da un objetivo: ... [K] [gx] [gy]")
    print(f"Objetivo: ({goal[0]:.1f}, {goal[1]:.1f})")
    # Posición inicial
    pos = poses[F0][:2, 3].astype(float).copy()
    R0 = poses[F0][:3, :3]
    # Orientación inicial
    th = float(np.arctan2(R0[1, 0], R0[0, 0]))
    robot_path = [pos.copy()]
    fans = {}
    min_dist = np.inf; collisions = 0; reached = False
    col_acerca = col_aleja = 0
    # Pos del robot en el frame anterior
    prev_pos = pos.copy()

    # Para cada frame de la escena
    for i, c in enumerate(CS):
        # Cada K pasos, repredice trayectorias
        if i % K == 0:
            fans = predict_scene(c, gt, det, traj2d, scaler20, scaler9, model_fus, model_base, poses)[0]
        # Posición actual: GT para métricas de colisión y detecciones para cálculo del campo repulsivo
        pcur_gt = peds_world(c, poses, gt)
        pcur_det = peds_world_det(c, poses, det, gt)
        # Pos del robot
        pos_here = pos.copy()
        if pcur_gt:
            # Peatón más cercano -> colisión si dmin < COLLISION_R
            dmin = np.inf; pnear = None
            for p in pcur_gt.values():
                dp = float(np.linalg.norm(pos_here - p))
                if dp < dmin:
                    dmin, pnear = dp, p
            min_dist = min(min_dist, dmin)
            if dmin < COLLISION_R:
                collisions += 1
                # El paso del robot lo acerca o aleja del peatón
                if dmin < float(np.linalg.norm(prev_pos - pnear)):
                    col_acerca += 1
                else:
                    col_aleja += 1
        # Si está a menos de GOAL_TOL -> objetivo alcanzado: se detiene la simulación
        if np.linalg.norm(goal - pos) < GOAL_TOL:
            reached = True
            robot_path.append(pos.copy())
            break
        # Se aplica el control
        v, w = control(pos, th, goal, fans, pcur_det)
        th += w * DT
        pos = pos + v * DT * np.array([np.cos(th), np.sin(th)])
        # Se suma nueva posición
        robot_path.append(pos.copy())
        prev_pos = pos_here

    robot_path = np.array(robot_path)
    path_len = float(np.sum(np.linalg.norm(np.diff(robot_path, axis=0), axis=1)))

    # Camino del robot del conjunto de datos y distancia mínima al peatón
    rec_path = np.array([poses[c][:2, 3] for c in CS])
    rec_min = np.inf
    # Se compara
    for i, c in enumerate(CS):
        for t, p in peds_world(c, poses, gt).items():
            rec_min = min(rec_min, np.linalg.norm(rec_path[i] - p))

    print("\n===== MÉTRICAS =====")
    print(f"{'':22}{'Original':>10}{'Evitación':>12}")
    print(f"{'Dist. mínima peatón (m)':22}{rec_min:>10.2f}{min_dist:>12.2f}")
    print(f"{'Colisiones (<%.1fm)' % COLLISION_R:22}{'-':>10}{collisions:>12}")
    if collisions:
        # En la colisión: el robot se aleja o se acerca del peatón
        print(f"{'   robot se acerca':22}{'-':>10}{col_acerca:>12}")
        print(f"{'   robot se aleja':22}{'-':>10}{col_aleja:>12}")
    print(f"{'Llega al objetivo':22}{'-':>10}{str(reached):>12}")
    print(f"{'Longitud camino (m)':22}{'-':>10}{path_len:>12.2f}")

    # 5. Dibujo
    r_ini, r_fin = robot_path[0], robot_path[-1]
    VX0, VX1 = min(r_ini[0], r_fin[0]) - VIEW_R, max(r_ini[0], r_fin[0]) + VIEW_R
    VY0, VY1 = min(r_ini[1], r_fin[1]) - VIEW_R, max(r_ini[1], r_fin[1]) + VIEW_R
    view = (VX0, VX1, VY0, VY1)

    # Figura 1. GIF
    state = {"fans": {}}
    fig, ax = plt.subplots(figsize=(9, 9))

    def upd(i):
        # Dibuja la escena
        if i % K == 0:
            state["fans"] = predict_scene(CS[i], gt, det, traj2d, scaler20, scaler9,
                                          model_fus, model_base, poses)[0]
        draw(ax, i, state["fans"], poses, gt, det, color, CS, rec_path, robot_path, goal, view)
        ax.set_title(f"Evitación (campos potenciales) - {SEQ}  (frame {CS[i]})", fontsize=11)

    # Genera y guarda la animación
    print(f"\nGenerando animación ({len(CS)} pasos)...")
    anim = FuncAnimation(fig, upd, frames=len(CS), interval=200)
    gif = OUTF / f"fase3_evitacion_{SEQ}_{F0}-{F1}.gif"
    anim.save(gif, writer=PillowWriter(fps=5)); plt.close(fig)
    print(f"Guardado: {gif}")

    # Figura 2. Capturas en distintos frames de una escena
    # Selecciona 3 frames y crea 3 subplots
    idxs = np.linspace(0, len(CS) - 1, 3).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    # Dibuja esos 3 instantes
    for ax, i in zip(axes, idxs):
        fans_i = predict_scene(CS[i], gt, det, traj2d, scaler20, scaler9, model_fus, model_base, poses)[0]
        draw(ax, i, fans_i, poses, gt, det, color, CS, rec_path, robot_path, goal, view)
        ax.set_title(f"frame {CS[i]}", fontsize=17)
    h1, = axes[0].plot([], [], color="0.7", ls=":", lw=1.8, label="Camino original")
    h2, = axes[0].plot([], [], "k-", lw=2, label="Evitación")
    h3, = axes[0].plot([], [], marker="^", color="k", linestyle="none", markersize=12,
                       markeredgecolor="white", label="Robot")
    h4, = axes[0].plot([], [], marker="*", color="gold", linestyle="none", markersize=16,
                       markeredgecolor="k", label="Objetivo")
    h5, = axes[0].plot([], [], "--", color="0.3", lw=1.8, label="Predicción")
    # Leyenda común
    fig.suptitle(f"Evitación con campos potenciales - {SEQ}", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.legend(handles=[h1, h2, h3, h4, h5], loc="lower center", ncol=5, fontsize=15, frameon=True,
               bbox_to_anchor=(0.5, 0.015))
    # Guarda la figura
    png = OUTF / f"fase3_evitacion_{SEQ}_{F0}-{F1}.png"
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Guardado: {png}")


if __name__ == "__main__":
    main()
