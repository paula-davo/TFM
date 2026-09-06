"""
Obtención del dataset Social 3D con detecciones.
Se agrupan escenas por (secuencia, cámara), se constriuen ventanas submuestreadas
y se guarda una máscara de peatones activos, posición 3D.

Las características de entrada son las 9 3D: (cx, cy, vx, vy, ax, ay, speed, sin_dir, cos_dir)
Cada peatón se representa por su detección 3D emparejada.

Salidas: X_social3d_det_ds, Y_social3d_det_ds, mask_social3d_det_ds, pos3d_social3d_det_ds,
scene_keys3d_det_ds.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LABELS_3D_DIR = TRAIN_DIR / "labels" / "labels_3d"
DET_3D_DIR = TRAIN_DIR / "detections" / "detections_3d"

OBS_LEN, PRED_LEN, STRIDE, MAX_PEDS = 8, 12, 3, 10   # MAX_PEDS: máximo de peatones por escena
WIN = OBS_LEN + PRED_LEN
N_FEATURES = 9
MATCH_THRESH = 1.0   # umbral máximo para emparejar detección-etiqueta (m), igual que el builder individual


def load_gt3d():
    # gt[seq][frame][tid] = (cx, cy) en metros (etiquetas 3D) -> ancla para emparejar detecciones y salida Y
    gt = defaultdict(lambda: defaultdict(dict))
    for jf in LABELS_3D_DIR.glob("*.json"):
        seq = jf.stem
        with open(jf) as f:
            data = json.load(f)
        for pcd_key, anns in data["labels"].items():
            fr = int(pcd_key.replace(".pcd", ""))
            for ann in anns:
                tid = int(ann["label_id"].split(":")[1])
                gt[seq][fr][tid] = (ann["box"]["cx"], ann["box"]["cy"])
    return gt


def load_det3d():
    # det[seq][frame] = array (n, 2) de centros de detecciones 3D (sin identidad)
    det = defaultdict(dict)
    for jf in DET_3D_DIR.glob("*.json"):
        seq = jf.stem
        with open(jf) as f:
            data = json.load(f)["detections"]
        for pcd_key, dets in data.items():
            fr = int(pcd_key.replace(".pcd", ""))
            pts = [(d["box"]["cx"], d["box"]["cy"]) for d in dets]
            det[seq][fr] = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)
    return det


def nearest_det(gt_xy, det_arr, thresh=MATCH_THRESH):
    # Detección más cercana a la etiqueta si está dentro del umbral (si no, None)
    if det_arr.shape[0] == 0:
        return None
    d = np.linalg.norm(det_arr - np.array(gt_xy), axis=1)
    j = int(np.argmin(d))
    return tuple(det_arr[j]) if d[j] <= thresh else None


def fill_missing(seq_pos):
    # Rellena huecos (None): forward, luego backward, y (0,0) si sigue vacío (igual que el 01/builder)
    n = len(seq_pos)
    last = None
    for i in range(n):
        if seq_pos[i] is None:
            seq_pos[i] = last
        else:
            last = seq_pos[i]
    nxt = None
    for i in range(n - 1, -1, -1):
        if seq_pos[i] is None:
            seq_pos[i] = nxt
        else:
            nxt = seq_pos[i]
    return [p if p is not None else (0.0, 0.0) for p in seq_pos]


def feats_3d(obs):
    # obs: (8, 2) posiciones 3D en metros -> 9 características (metros)
    cx, cy = obs[:, 0], obs[:, 1]
    vx = np.diff(cx, prepend=cx[0])
    vy = np.diff(cy, prepend=cy[0])
    ax = np.diff(vx, prepend=vx[0])
    ay = np.diff(vy, prepend=vy[0])
    speed = np.sqrt(vx**2 + vy**2)
    ang = np.arctan2(vy, vx)
    return np.stack([cx, cy, vx, vy, ax, ay, speed, np.sin(ang), np.cos(ang)], axis=1)


def build(df, gt3d, det3d):
    sX, sY, sM, sP, keys = [], [], [], [], []
    matched = total = 0   # tasa de emparejamiento detección-peatón (frames observados)

    # Para cada escena (secuencia, cámara)
    for (seq, cam), group in df.groupby(["sequence", "camera"]):
        # Por subtrack: conjunto de frames disponibles y track_id (para localizar la posición)
        sub = {}
        for sid, traj in group.groupby("subtrack_id"):
            traj = traj.sort_values("frame")
            frames = traj["frame"].values.astype(int)
            tid = int(traj["track_id"].iloc[0])
            sub[sid] = (set(int(f) for f in frames), tid)
        if not sub:
            continue

        all_frames = sorted(group["frame"].unique().astype(int))
        gtb = gt3d.get(seq, {})
        detb = det3d.get(seq, {})
        # Ventanas submuestreadas sobre la escena
        for t0 in all_frames:
            win_frames = [t0 + k * STRIDE for k in range(WIN)]
            obs_frames = win_frames[:OBS_LEN]

            # Peatones presentes en los 20 frames de la ventana (criterio por GT)
            present = []
            for sid, (ffset, tid) in sub.items():
                if all(f in ffset for f in win_frames):
                    present.append((sid, tid))
            if not present:
                continue
            present.sort(key=lambda x: x[0])

            Xs = np.zeros((MAX_PEDS, OBS_LEN, N_FEATURES), dtype=np.float32)   # 9 features 3D (detecciones)
            Ys = np.zeros((MAX_PEDS, PRED_LEN, 2), dtype=np.float32)          # desplazamiento GT (m)
            Ms = np.zeros(MAX_PEDS, dtype=bool)                              # máscara de peatones
            Ps = np.zeros((MAX_PEDS, 2), dtype=np.float32)                   # posición detectada (pooling)

            for i, (sid, tid) in enumerate(present[:MAX_PEDS]):
                # OBS: detección más cercana a la etiqueta en cada frame observado (con relleno de huecos)
                obs_det = []
                for f in obs_frames:
                    gpos = gtb.get(int(f), {}).get(tid)
                    total += 1
                    if gpos is None:
                        obs_det.append(None); continue
                    dpos = nearest_det(gpos, detb.get(int(f), np.zeros((0, 2), np.float32)))
                    if dpos is not None:
                        matched += 1
                    obs_det.append(dpos)
                obs_det = np.array(fill_missing(obs_det), dtype=np.float32)   # (8, 2) metros (detección)

                # PRED: posiciones reales GT (con relleno)
                pred_gt = [gtb.get(int(f), {}).get(tid) for f in win_frames[OBS_LEN:]]
                pred_gt = np.array(fill_missing(pred_gt), dtype=np.float32)   # (12, 2) metros (GT)

                last_det = obs_det[OBS_LEN - 1]        # última posición DETECTADA observada
                Xs[i] = feats_3d(obs_det)
                Ys[i] = pred_gt - last_det              # desplazamiento GT desde la última detección
                Ms[i] = True
                Ps[i] = last_det                        # posición detectada para el pooling

            sX.append(Xs); sY.append(Ys); sM.append(Ms); sP.append(Ps)
            keys.append(f"{seq}_{cam}")

    match_rate = 100.0 * matched / total if total else 0.0
    return (np.stack(sX), np.stack(sY), np.stack(sM),
            np.stack(sP), np.array(keys), match_rate)


if __name__ == "__main__":
    # 1. Carga etiquetas 3D (ancla) y detecciones 3D
    gt3d = load_gt3d()
    det3d = load_det3d()
    # 2. Lee el CSV de datos (solo para la estructura de escenas y la presencia de peatones)
    df = pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv")
    # 3. Submuestreo global (rejilla)
    df = df[df["frame"] % STRIDE == 0].copy()
    # 4. Construye las escenas con características 3D de detecciones
    print(f"STRIDE={STRIDE}. Construyendo escenas 3D con detecciones...")
    X, Y, mask, pos, keys, match_rate = build(df, gt3d, det3d)

    peds = mask.sum(axis=1)
    print(f"\n========== SOCIAL 3D DETECCIONES ==========")
    print(f"Escenas: {len(X)}   peatones/escena media={peds.mean():.2f} max={peds.max()}")
    print(f"X: {X.shape}  Y: {Y.shape}")
    print(f"Tasa de emparejamiento detección-peatón: {match_rate:.1f}%")
    print(f"Y3d (metros) — desplazamiento medio: {np.linalg.norm(Y[mask], axis=2).mean():.3f} m")

    os.makedirs(f"{DATA_DIR}/data", exist_ok=True)
    np.save(f"{DATA_DIR}/data/X_social3d_det_ds.npy", X)         # entrada (escena, MAX_PEDS, 8, 9) detecciones
    np.save(f"{DATA_DIR}/data/Y_social3d_det_ds.npy", Y)         # salida  (escena, MAX_PEDS, 12, 2) GT
    np.save(f"{DATA_DIR}/data/mask_social3d_det_ds.npy", mask)   # máscara de peatones activos
    np.save(f"{DATA_DIR}/data/pos3d_social3d_det_ds.npy", pos)   # posición detectada último frame obs (pooling)
    np.save(f"{DATA_DIR}/data/scene_keys3d_det_ds.npy", keys)    # clave secuencia_cámara
    print("Guardado social 3D detecciones submuestreado.")
