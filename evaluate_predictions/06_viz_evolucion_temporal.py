"""
Evolución temporal de la predicción de intención (modelo de fusión 10). Sigue a un peatón
del conjunto de validación a lo largo del tiempo mostrando su trayectoria observada, su GT
y la predicción del modelo.

Uso:
    python evaluate_predictions/07_viz_evolucion_temporal.py [secuencia] [track_id] [n_instantes]
Sin argumentos: elige automáticamente un peatón con giro y suficientes instantes.
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
sys.path.insert(0, DATA_DIR)
from split_utils import official_masks
from metrics_utils import angle, build_meta

OBS_LEN, STRIDE, WIN = 8, 3, 20
FPS = 15.0     # Hz de grabación de JRDB (para etiquetar el tiempo)
MIN_RUN = 20   # Ventanas mínimas del tramo contiguo para que los instantes tengan variación temporal


def curvatura(ks, obs_net, fut_net):
    # Giro total entre la dirección observada al principio y al final del seguimiento
    return angle(obs_net[ks[0]], fut_net[ks[-1]])


def longest_run(ks, frv):
    best, cur = [], [ks[0]]
    # Recorre las ventanas de un peatón ordenado por frame
    for a, b in zip(ks, ks[1:]):
        # Si la diferencia entre frames es pequeña, se considera que es continua
        if frv[b] - frv[a] <= 2 * STRIDE:
            cur.append(b)
        # En caso contrario, se corta el tramo actual
        else:
            if len(cur) > len(best):
                best = cur
            cur = [b]
    # Devuelve el tramo continuo más largo
    return best if len(best) >= len(cur) else cur


def filtra(min_run, con_mov, runs, on, on_med):
    # Filtra por tramo corto (tiene que tener al menos min_run), y en caso de estar activo,
    # se exige que el peatón se mueva considerablemente en el tramo
    return [(key, run) for key, run in runs.items()
            if len(run) >= min_run and (np.median(on[run]) > on_med if con_mov else True)]


def lims(ax, pts, margin=0.7):
    # Ventana cuadrada centrada en la trayectoria.
    pts = np.asarray(pts)
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
    half = max(pts[:, 0].ptp(), pts[:, 1].ptp()) / 2 + margin
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)", fontsize=19); ax.set_ylabel("y (m)", fontsize=19)
    ax.tick_params(labelsize=16)


def draw_evolucion(sel_key, sel_ks, n_inst, frv, obs_abs, gt_abs, det_abs):
    seq_sel, tid_sel = sel_key
    sel_ks = np.array(sel_ks)
    # Se seleccionan N instantes repartidos en el tiempo
    pick = sel_ks[np.linspace(0, len(sel_ks) - 1, n_inst).astype(int)]
    # Frame del primer instante elegido
    frame0 = frv[pick[0]]
    # Subplots
    fig, axes = plt.subplots(1, n_inst, figsize=(5.2 * n_inst, 5.8))
    axes = np.atleast_1d(axes)
    h_obs = h_gt = h_pred = None
    # Para cada instante elegido
    for ax, k in zip(axes, pick):
        # Obtiene trayectoria observada y GT
        o, g = obs_abs[k], gt_abs[k]
        full_g = np.vstack([o[-1], g])
        fd = np.vstack([o[-1], det_abs[k]])
        # Dibuja las tres
        h_obs, = ax.plot(o[:, 0], o[:, 1], "--o", color="black", lw=1.8, ms=3, zorder=5, label="Observado")
        h_gt, = ax.plot(full_g[:, 0], full_g[:, 1], "-o", color="tab:green", lw=2.4, ms=3, zorder=3,
                        label="Futuro real (GT)")
        h_pred, = ax.plot(fd[:, 0], fd[:, 1], "--s", color="tab:red", lw=2.4, ms=3, zorder=4,
                          label="Predicción (modelo adoptado)")
        ax.scatter(0, 0, c="tab:blue", marker="^", s=130, zorder=6, edgecolors="white")
        pts = np.vstack([o, full_g, fd])
        span = max(pts[:, 0].ptp(), pts[:, 1].ptp())
        # Margen ajustado (zoom)
        lims(ax, pts, margin=max(0.08, 0.02 * span))
        t_rel = (frv[k] - frame0) / FPS
        ax.set_title(f"t = +{t_rel:.1f} s  (frame {frv[k]})", fontsize=20, fontweight="bold")
    # Leyenda común
    h_rob, = axes[0].plot([], [], marker="^", color="tab:blue", linestyle="none", markersize=12,
                          markeredgecolor="white", label="Robot")
    fig.suptitle(f"Evolución temporal de la predicción de intención — {seq_sel} (peatón {tid_sel})",
                 fontsize=23, fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.legend(handles=[h_obs, h_gt, h_pred, h_rob], loc="lower center", ncol=4, fontsize=18,
               frameon=True, bbox_to_anchor=(0.5, 0.02))
    out = f"{OUTPUT_DIR}/cualitativa_evolucion_{seq_sel}_{tid_sel}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}  (peatón {tid_sel}, {len(sel_ks)} instantes)")


if __name__ == "__main__":
    # Argumentos: [secuencia] [track_id] [n_instantes]
    seq_arg = sys.argv[1] if len(sys.argv) > 1 else None
    tid_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    n_inst = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    # Carga los datos: fusión temprana (imagen 11 + detecciones 3D 9 = 20)
    Ximg = np.load(f"{DATA_DIR}/X_ds.npy")             # (N,8,11)
    X3d = np.load(f"{DATA_DIR}/X3d_det_motion.npy")    # (N,8,9)
    X = np.concatenate([Ximg, X3d], axis=2)            # (N,8,20)
    Y = np.load(f"{DATA_DIR}/Y3d_det.npy")[:, :8, :]   # (N,8,2) desplazamiento GT (8/8)
    # Carga partición oficial
    ids = np.load(f"{DATA_DIR}/subtrack_ids_ds.npy", allow_pickle=True)
    tm, vm = official_masks(ids)

    # Modelo adoptado (10) y predicción sobre validación
    scaler = StandardScaler().fit(X[tm].reshape(-1, 20))
    dm = load_model(f"{DATA_DIR}/prediccion_3d/best/combined3d_reg_best.keras", compile=False)
    # Aplica partición oficial
    Xv, Yv, X3v = X[vm], Y[vm], X3d[vm]
    # Aplica scaler
    Xs = scaler.transform(Xv.reshape(-1, 20)).reshape(Xv.shape)
    # Predice
    predD = dm.predict(Xs, batch_size=512, verbose=0)          # (Nv,8,2)

    last = X3v[:, -1, 0:2]
    obs_abs = X3v[:, :, 0:2]                                    # (Nv,8,2)
    gt_abs = last[:, None, :] + Yv                              # (Nv,8,2)
    det_abs = last[:, None, :] + predD                          # (Nv,8,2)
    # Detecta las observaciones que están completas
    ok = ~np.any(np.all(obs_abs == 0, axis=2), axis=1)

    # Vectores de movimiento
    obs_net = obs_abs[:, -1, :] - obs_abs[:, 0, :]
    fut_net = gt_abs[:, -1, :] - last
    # Magnitud de movimiento del vector de observación
    on = np.linalg.norm(obs_net, axis=1)
    # Mediana para luego filtrar
    on_med = np.median(on[ok])

    m_seq, m_fr, m_tid = build_meta(pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv"), OBS_LEN, STRIDE, WIN)
    assert len(m_seq) == X.shape[0], f"meta {len(m_seq)} != X {X.shape[0]}"
    seqv, frv, tidv = m_seq[vm], m_fr[vm], m_tid[vm]

    # Agrupa las ventanas por peatón (secuencia, track_id) y las ordena por frame
    ped = defaultdict(list)
    for k in np.where(ok)[0]:
        ped[(seqv[k], tidv[k])].append(k)
    for key in ped:
        ped[key].sort(key=lambda k: frv[k])

    # Candidatos: por peatón, su tramo contiguo más largo; se prioriza el movimiento y, 
    # dentro de eso, el mayor giro. Si no hay suficientes, se relajan las condiciones.
    runs = {key: longest_run(ks, frv) for key, ks in ped.items()}

    cands = (filtra(MIN_RUN, True, runs, on, on_med)
             or filtra(MIN_RUN, False, runs, on, on_med)
             or filtra(n_inst, False, runs, on, on_med))
    if not cands:
        raise SystemExit(f"Ningún peatón de validación tiene un tramo contiguo de >= {n_inst} instantes.")
    cands.sort(key=lambda kk: curvatura(kk[1], obs_net, fut_net), reverse=True)

    # Selecciona cuatro peatones distintos (a ser posible de secuencias diferentes)
    seleccion, used_seq, used_key = [], set(), set()
    # Si se ha solicitado un peatón concreto, se obtiene el primero
    if seq_arg is not None and tid_arg is not None:
        key = (seq_arg, tid_arg)
        if key not in runs or len(runs[key]) < n_inst:
            raise SystemExit(f"El peatón {key} no está en validación o su tramo contiguo es < {n_inst}.")
        seleccion.append((key, runs[key])); used_seq.add(key[0]); used_key.add(key)
    # Completa evitando repetir peatón y secuencia
    for key, ks in cands:
        if len(seleccion) == 4:
            break
        if key in used_key or key[0] in used_seq:
            continue
        seleccion.append((key, ks)); used_seq.add(key[0]); used_key.add(key)
    # Si no es posible, permite repetir secuencia
    for key, ks in cands:
        if len(seleccion) == 4:
            break
        if key in used_key:
            continue
        seleccion.append((key, ks)); used_key.add(key)

    for sel_key, sel_ks in seleccion:
        draw_evolucion(sel_key, sel_ks, n_inst, frv, obs_abs, gt_abs, det_abs)
