"""
Figuras cualitativas de la predicción de intención de trayectoria con el modelo de
fusión temprana regularizada (10). Se muestra: trayectoria observada de los peatones
 + futuro real (GT) + predicción del modelo.

Figura 1 (cualitativa_ejemplos.png): tres casos -> trayectoria recta, giro y parada.
Figura 2 (cualitativa_caso_uso.png): caso de uso -> varios peatones alrededor del robot.
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OUTPUT_DIR = f"{DATA_DIR}/evaluate_predictions/figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OBS_LEN, STRIDE, WIN = 8, 3, 20
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from metrics_utils import angle, build_meta


def lims(ax, pts, margin=0.03, square=True):
    # square=True (ejemplos): ventana cuadrada centrada
    # square=False (caso_uso): límites ajustados a cada eje con el mismo margen por lado
    pts = np.asarray(pts)
    if square:
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
        half = max(pts[:, 0].ptp(), pts[:, 1].ptp()) / 2 * 1.18 + margin
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
    else:
        ax.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
        ax.set_ylim(pts[:, 1].min() - margin, pts[:, 1].max() + margin)
        ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)", fontsize=17); ax.set_ylabel("y (m)", fontsize=17)
    ax.tick_params(labelsize=15)


def pick(mask, score, scene, used):
    # Elige la mejor muestra de la máscara, priorizando escenas nuevas
    # Obtiene los índices de muestras que cumplen la máscara
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    # Los ordena por score de mayor a menor
    idx = idx[np.argsort(-score[idx])]
    # Recorre los candidatos y devuelve el primero que no se haya usado todavía.
    for j in idx:
        if scene[j] not in used:
            used.add(scene[j])
            return int(j)
    # Si ya se han usado todos, devuelve el mejor
    used.add(scene[idx[0]]); return int(idx[0])


def draw_ex(ax, idx, title, obs_abs, gt_abs, det_abs):
    # Dibuja: observado + futuro real + predicción
    # Extrae observación y GT para la muestra dada
    o, g = obs_abs[idx], gt_abs[idx]
    full_g = np.vstack([o[-1], g])
    fd = np.vstack([o[-1], det_abs[idx]])
    # Dibuja observacion, gt y predicción
    h_obs, = ax.plot(o[:, 0], o[:, 1], "--o", color="black", lw=1.8, ms=3, zorder=5, label="Observado")
    h_gt, = ax.plot(full_g[:, 0], full_g[:, 1], "-o", color="tab:green", lw=2.4, ms=3, zorder=3,
                    label="Futuro real (GT)")
    h_pred, = ax.plot(fd[:, 0], fd[:, 1], "--s", color="tab:red", lw=2.4, ms=3, zorder=4,
                      label="Predicción (modelo adoptado)")
    # Dibuja el robot
    ax.scatter(0, 0, c="tab:blue", marker="^", s=130, zorder=6, edgecolors="white")
    pts = np.vstack([o, full_g, fd])
    lims(ax, pts)
    # Título
    ax.set_title(title, fontsize=19, fontweight="bold")
    return h_obs, h_gt, h_pred


def displayable(ks, tidv):
    # Peatones dibujables de un grupo: uno por track_id (evita duplicados de la misma persona)
    seen, chosen = set(), []
    # Recorre los candidatos
    for k in ks:
        # Si el candidato ya ha sido dibujado, se descarta
        if tidv[k] in seen:
            continue
        seen.add(tidv[k])
        chosen.append(k)
    return chosen


def draw_caso(members, seq_sel, fr_sel, out_name, obs_abs, gt_abs, det_abs, tidv):
    # Filtra los peatones que se pueden dibujar
    members = displayable(members, tidv)[:7]
    # Los ordena según su coordenada X
    members = sorted(members, key=lambda k: obs_abs[k][-1, 0])
    fig3, ax = plt.subplots(figsize=(9, 9))
    allpts = [np.zeros((1, 2))]
    # Para cada peatón
    for n, k in enumerate(members, start=1):
        # Obtiene su trayectoria observada y su GT
        o, g = obs_abs[k], gt_abs[k]
        full_g = np.vstack([o[-1], g])
        fd = np.vstack([o[-1], det_abs[k]])
        ax.plot(o[:, 0], o[:, 1], "--o", color="black", lw=1.5, ms=2, zorder=4)         # observado
        ax.plot(full_g[:, 0], full_g[:, 1], "-", color="tab:green", lw=2.0, zorder=3)   # futuro real
        ax.plot(fd[:, 0], fd[:, 1], "--", color="tab:red", lw=1.8, zorder=2)            # predicción
        ax.scatter(o[-1, 0], o[-1, 1], c="black", s=25, zorder=5)                       # pos actual
        # Etiqueta numérica junto a la posición actual del peatón
        ax.annotate(str(n), (o[-1, 0], o[-1, 1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=15, fontweight="bold", color="tab:blue", zorder=7)
        allpts += [o, full_g, fd]
    ax.scatter(0, 0, c="tab:blue", marker="^", s=180, zorder=6, edgecolors="white", label="Robot")
    ax.plot([], [], "--o", color="black", lw=1.5, ms=3, label="Observado")
    ax.plot([], [], "-", color="tab:green", lw=2.0, label="Futuro real (GT)")
    ax.plot([], [], "--", color="tab:red", lw=1.8, label="Predicción (modelo adoptado)")
    lims(ax, np.vstack(allpts), margin=1.0, square=False)
    ax.legend(fontsize=16, loc="best")
    ax.set_title(f"Predicción de intención de {len(members)} peatones alrededor del robot",
                 fontsize=20, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(f"{OUTPUT_DIR}/{out_name}", dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"Guardado: {out_name}  (seq={seq_sel} frame={fr_sel}, {len(members)} peatones)")


def main():
    # Carga los datos: fusión temprana (imagen 11 + detecciones 3D 9 = 20)
    Ximg = np.load(f"{DATA_DIR}/X_ds.npy")             # (N,8,11) imagen 2D
    X3d = np.load(f"{DATA_DIR}/X3d_det_motion.npy")    # (N,8,9) detección 3D
    X = np.concatenate([Ximg, X3d], axis=2)            # (N,8,20) fusión
    Y = np.load(f"{DATA_DIR}/Y3d_det.npy")[:, :8, :]   # (N,8,2) desplazamiento GT (8/8)
    # Carga la partición oficial
    ids = np.load(f"{DATA_DIR}/subtrack_ids_ds.npy", allow_pickle=True)
    tm, vm = official_masks(ids)

    # Ajusta el scaler (20 features) y carga el modelo de fusión (10)
    scaler = StandardScaler().fit(X[tm].reshape(-1, 20))
    dm = load_model(f"{DATA_DIR}/prediccion_3d/best/combined3d_reg_best.keras", compile=False)

    # Partición oficial de validación
    Xv, Yv, X3v = X[vm], Y[vm], X3d[vm]
    Xs = scaler.transform(Xv.reshape(-1, 20)).reshape(Xv.shape)
    predD = dm.predict(Xs, batch_size=512, verbose=0) # (Nv,8,2) desplazamiento

    # Trayectoria observada, última pos observada, futuro real y predicción del modelo
    last = X3v[:, -1, 0:2]                                      # última pos observada
    obs_abs = X3v[:, :, 0:2]                                    # (Nv,8,2)
    gt_abs = last[:, None, :] + Yv                              # (Nv,8,2)
    det_abs = last[:, None, :] + predD                          # (Nv,8,2)

    # Descarta muestras con algún paso de relleno (0,0) por detección faltante
    ok = ~np.any(np.all(obs_abs == 0, axis=2), axis=1)

    # Vectores de avance en observación y futuro, y sus longitudes
    obs_net = obs_abs[:, -1, :] - obs_abs[:, 0, :]
    fut_net = gt_abs[:, -1, :] - last
    on, fn = np.linalg.norm(obs_net, axis=1), np.linalg.norm(fut_net, axis=1)
    # Error del modelo adoptado (ADE)
    det_ade = np.linalg.norm(det_abs - gt_abs, axis=2).mean(1)
    on_med, fn_med = np.median(on[ok]), np.median(fn[ok])

    # 1. Casos representativos: recta, giro y parada (una escena distinta por caso)
    scene = np.array([s.rsplit("_", 2)[0] for s in ids[vm]])
    # Ángulo de observación-futuro
    turn = angle(obs_net, fut_net)
    # El modelo acierta
    good = det_ade < np.percentile(det_ade[ok], 40)
    # Valida que se mueva más que la mediana tanto en el pasado como en el futuro, y que el
    # futuro no sea ni mucho más grande ni mucho más pequeño que el pasado
    move = ok & (on > on_med) & (fn > fn_med) & (fn < 2.5 * on) & (fn > 0.4 * on)
    # Selecciona ejemplos: recto, giro y parada
    m_recto = move & good & (turn < 12)
    m_giro = move & good & (turn > 35)
    m_parada = ok & good & (on > on_med) & (fn < 0.5 * on_med)

    used = set()
    i_recto = pick(m_recto, fn, scene, used)         # Recta
    i_giro = pick(m_giro, turn, scene, used)         # Giro
    i_parada = pick(m_parada, on, scene, used)       # Parada

    casos = [("Trayectoria recta", i_recto), ("Giro", i_giro), ("Parada", i_parada)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    handles = None
    for ax, (tit, idx) in zip(axes, casos):
        if idx is None:
            ax.set_visible(False)
            print(f"[aviso] sin candidato para: {tit}"); continue
        handles = draw_ex(ax, idx, tit, obs_abs, gt_abs, det_abs)
    # Leyenda común
    fig.suptitle("Predicción con el modelo seleccionado — casos representativos",
                 fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    if handles is not None:
        fig.legend(handles=list(handles), loc="lower center", ncol=3, fontsize=16,
                   frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.savefig(f"{OUTPUT_DIR}/cualitativa_ejemplos.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Guardado: cualitativa_ejemplos.png")
    for tit, idx in casos:
        if idx is not None:
            print(f"  {tit:<18} idx {idx:6d}  giro {turn[idx]:3.0f}°  obs {on[idx]:.2f}m  "
                  f"fut {fn[idx]:.2f}m  det_ADE {det_ade[idx]:.2f}m")

    # 2. Caso de uso multipeatón
    # Se cargan los datos
    m_seq, m_fr, m_tid = build_meta(pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv"), OBS_LEN, STRIDE, WIN)
    assert len(m_seq) == X.shape[0], f"meta {len(m_seq)} != X {X.shape[0]}"
    seqv, frv, tidv = m_seq[vm], m_fr[vm], m_tid[vm]

    # Agrupa por secuencia y frame para encontrar una escena con varios peatones a la vez.
    # Se buscan con movimiento.
    grupos = defaultdict(list)
    for k in np.where(ok & (on > 0.4))[0]:
        grupos[(seqv[k], frv[k])].append(k)
    cands = sorted(grupos.items(), key=lambda kv: len(displayable(kv[1], tidv)), reverse=True)
    cands_ok = [c for c in cands if 4 <= len(displayable(c[1], tidv)) <= 7]
    fuente = cands_ok if cands_ok else cands[:1]

    # Elige dos escenas, priorizando escenas distintas
    escenas, used_seq = [], set()
    for (seq, fr), ks in fuente:
        if seq in used_seq:
            continue
        used_seq.add(seq); escenas.append(((seq, fr), ks))
        if len(escenas) == 2:
            break
    # Si no había dos escenas distintas
    for c in fuente:
        if len(escenas) == 2:
            break
        if c not in escenas:
            escenas.append(c)

    nombres = ["cualitativa_caso_uso.png", "cualitativa_caso_uso_2.png"]
    for ((seq_sel, fr_sel), ks), nombre in zip(escenas, nombres):
        draw_caso(ks, seq_sel, fr_sel, nombre, obs_abs, gt_abs, det_abs, tidv)


if __name__ == "__main__":
    main()
