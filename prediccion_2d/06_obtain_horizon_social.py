"""
Obtención del dataset empleado para LSTM Social. Contiene aumento de horizonte tras demostrar
su mejora.

Al ser el modelo un LSTM Social, la obtención de datos 2D es diferente a la realizada en el paso 04.
En el 04 -> se obtiene trayectorias individuales de peatones -> X(N, 8, 11), Y(N, 12, 2). Se guarda
subtrack ID, y el submuestreo se aplica a las trayectorias (trayectorias con saltos seleccionando 1 
de cada 3 frames)

En el 06 -> se obtienen las escenas completas, con los peatones que contienen -> X(escena, máximo de
peatones, 8, 11), Y(escena, máximo de peatones, 12, 2). Se guarda una máscara con peatones activos en 
la escena, etiquetas 3D, y escena y cámaras a la que pertenece. El submuestreo se aplica a la escena.
Es decir, antes cada peatón tenia su trayectoria y esta trayectoria se submuestreaba. Ahora se aplica
a la escena en términos absolutos.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
TRAIN_DIR = Path(DATA_DIR) / "train_dataset_with_activity" / "train_dataset_with_activity"
LABELS_3D_DIR = TRAIN_DIR / "labels" / "labels_3d"

OBS_LEN, PRED_LEN, STRIDE, MAX_PEDS = 8, 12, 3, 10 # MAX_PEDS es el máximo de peatones en una misma escena
WIN = OBS_LEN + PRED_LEN
N_FEATURES = 11

FEATURE_COLS = ["x_norm", "y_norm", "vx", "vy", "ax", "ay",
                "speed", "sin_dir", "cos_dir", "bbox_w_norm", "bbox_h_norm"]


def load_pos3d():
    pos = defaultdict(lambda: defaultdict(dict))
    # Recorre todos los archivos JSON de etiquetas
    for jf in LABELS_3D_DIR.glob("*.json"):
        # Itera por secuencia, frame y peatón, y almacena su posición (cx, cy)
        seq = jf.stem
        with open(jf) as f:
            data = json.load(f)
        for pcd_key, anns in data["labels"].items():
            fr = int(pcd_key.replace(".pcd", ""))
            for ann in anns:
                tid = int(ann["label_id"].split(":")[1])
                pos[seq][fr][tid] = (ann["box"]["cx"], ann["box"]["cy"])
    return pos


def recompute_motion_global_grid(df):
    # Obtiene las dinámicas con el stride definido
    # Filtra con stride
    df = df[df["frame"] % STRIDE == 0].copy()
    df = df.sort_values(["subtrack_id", "frame"])
    g = df.groupby("subtrack_id")
    # Calcula: vx, vy, ax, ay, speed, sin_dir, cos_dir (norm).
    df["vx"] = g["x_norm"].diff().fillna(0)
    df["vy"] = g["y_norm"].diff().fillna(0)
    df["ax"] = g["vx"].diff().fillna(0)
    df["ay"] = g["vy"].diff().fillna(0)
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    ang = np.arctan2(df["vy"], df["vx"])
    df["sin_dir"], df["cos_dir"] = np.sin(ang), np.cos(ang)
    return df


def build(df, pos3d):
    sX, sY, sM, sP, keys = [], [], [], [], []

    # Para cada secuencia y cámara de datos 2D
    for (seq, cam), group in df.groupby(["sequence", "camera"]):
        sub = {}
        # Para cada trayectoria, guarda primer y último frame, características, posición 2D,
        # track ID original (para localizar pos 3D)
        for sid, traj in group.groupby("subtrack_id"):
            traj = traj.sort_values("frame")
            frames = traj["frame"].values.astype(int)
            feats = traj[FEATURE_COLS].values.astype(np.float32)
            posn = traj[["x_norm", "y_norm"]].values.astype(np.float32)
            tid = int(traj["track_id"].iloc[0])
            sub[sid] = (int(frames[0]), int(frames[-1]),
                        {int(f): feats[i] for i, f in enumerate(frames)},
                        {int(f): posn[i] for i, f in enumerate(frames)}, tid)
            
        # Si queda vacio, se salta a la siguiente combinación secuencia-cámara
        if not sub:
            continue

        all_frames = sorted(group["frame"].unique().astype(int))
        # Crea ventanas sobre los frames (resultantes de submuestreo con stride)
        for t0 in all_frames:
            win_frames = [t0 + k * STRIDE for k in range(WIN)]
            obs_frames = win_frames[:OBS_LEN]
            pred_frames = win_frames[OBS_LEN:]
            t_last = obs_frames[-1]

            # Filtra: peatones que existan en los 20 frames de la ventana.
            # Ordena por subtrack ID
            present = []
            for sid, (_, _, ff, fp, tid) in sub.items():
                if all(f in ff for f in win_frames):
                    present.append((sid, ff, fp, tid))
            if not present:
                continue
            present.sort(key=lambda x: x[0])

            # Entradas
            Xs = np.zeros((MAX_PEDS, OBS_LEN, N_FEATURES), dtype=np.float32)
            # Salidas
            Ys = np.zeros((MAX_PEDS, PRED_LEN, 2), dtype=np.float32)
            # Máscara: indica los peatones existentes
            Ms = np.zeros(MAX_PEDS, dtype=bool)
            # Posiciones 3D (cx, cy) -> etiqueta 3D
            Ps = np.zeros((MAX_PEDS, 2), dtype=np.float32)

            # Guarda cada peatón detectado en la ventana
            for i, (sid, ff, fp, tid) in enumerate(present[:MAX_PEDS]):
                # 8 frames de entrada -> 11 características
                Xs[i] = np.stack([ff[f] for f in obs_frames])
                last = fp[t_last]
                # 12 frames de salida -> desplazamiento relativo normalizado
                Ys[i] = np.stack([fp[f] for f in pred_frames]) - last
                # Marca True en la máscara cuando lo rellena (en esa posición hay peatón)
                Ms[i] = True
                # Obtiene y guarda la etiqueta 3D para ese peatón
                p3 = pos3d.get(seq, {}).get(t_last, {}).get(tid)
                Ps[i] = p3 if p3 is not None else (0.0, 0.0)

            sX.append(Xs); sY.append(Ys); sM.append(Ms); sP.append(Ps)
            keys.append(f"{seq}_{cam}")

    return (np.stack(sX), np.stack(sY), np.stack(sM),
            np.stack(sP), np.array(keys))


if __name__ == "__main__":

    # 1. Carga posición 3D
    pos3d = load_pos3d()
    # 2. Lee el CSV de datos 2D
    df = pd.read_csv(f"{DATA_DIR}/jrdb_clean_trajectories.csv")
    # 3. Calcula dinámicas
    df = recompute_motion_global_grid(df)
    # 4. Genera datos 2D con stride + etiquetas 3D
    print(f"STRIDE={STRIDE}. Construyendo escenas...")
    X, Y, mask, pos, keys = build(df, pos3d)

    peds = mask.sum(axis=1)
    print(f"\n========== SOCIAL SUBMUESTREADO ==========")
    print(f"Escenas: {len(X)}   peatones/escena media={peds.mean():.2f} max={peds.max()}")
    print(f"X: {X.shape}  Y: {Y.shape}")
    # Entrada Social
    np.save(f"{DATA_DIR}/X_social_ds.npy", X)
    # Salida Social
    np.save(f"{DATA_DIR}/Y_social_ds.npy", Y)
    # Máscara de peatones
    np.save(f"{DATA_DIR}/mask_social_ds.npy", mask)
    # Etiqueta 3D
    np.save(f"{DATA_DIR}/pos3d_social_ds.npy", pos)
    # Escena y Cámara a la que pertenece
    np.save(f"{DATA_DIR}/scene_keys_ds.npy", keys)
    print("Guardado social submuestreado.")
