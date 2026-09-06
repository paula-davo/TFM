"""
Funciones comunes para la evitación.
"""
import json
from collections import defaultdict
import numpy as np
import pandas as pd

MATCH = 1.0   # Umbral de asociacion deteccion-peaton

# Calibración LiDAR lower -> marco del robot (ego)
_UPPER2EGO = np.array([
    [0.9963896745022904, -0.08489768280241602, 0.0, 0.0],
    [0.08489768280241602, 0.9963896745022904, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.33529],
    [0.0, 0.0, 0.0, 1.0]])
_LOWER2UPPER = np.array([
    [0.9967586199688887, 0.08033737338163417, 0.0, 0.0],
    [-0.0803373733816342, 0.9967586199688887, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.4720000013709068],
    [0.0, 0.0, 0.0, 1.0]])
LOWER2EGO = _UPPER2EGO @ _LOWER2UPPER


def load_cloud_ego(pcd_dir, seq, fr, voxel=0.25, crop_r=25.0):
    import open3d as o3d
    p = pcd_dir / seq / f"{fr:06d}.pcd"
    if not p.exists():
        return None
    # Carga los puntos de la nube -> (x, y, z)
    pts = np.asarray(o3d.io.read_point_cloud(str(p)).points)
    if pts.size == 0:
        return None
    # Transforma al marco del robot (ego)
    pe = (LOWER2EGO @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    # Recorte
    pe = pe[np.linalg.norm(pe[:, :2], axis=1) < crop_r]
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(pe)
    # Submuestreo
    c = c.voxel_down_sample(voxel)
    # Calcula las normales
    c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return c


def load_gt(lab_dir, seq):
    gt = defaultdict(dict)
    # Recorre el JSON de etiquetas 3D correspondiente a la secuencia
    for k, anns in json.load(open(lab_dir / f"{seq}.json"))["labels"].items():
        # Para cada anotación guarda: gt[frame][label_id] = (cx, cy)
        fr = int(k.replace(".pcd", ""))
        for a in anns:
            gt[fr][a["label_id"]] = (a["box"]["cx"], a["box"]["cy"])
    return gt


def load_det(det_dir, seq):
    # Recorre el JSON de detecciones 3D correspondientes a la secuencia
    det = {}
    for k, ds in json.load(open(det_dir / f"{seq}.json"))["detections"].items():
        # Para cada anotación guarda: det[frame] = (Nd, 2)
        fr = int(k.replace(".pcd", ""))
        det[fr] = np.array([(d["box"]["cx"], d["box"]["cy"]) for d in ds], np.float32) \
            if ds else np.zeros((0, 2), np.float32)
    return det


def nearest_det(g, darr):
    # Asocia peatón a detección
    # Comprueba que haya detecciones
    if darr.shape[0] == 0:
        return None
    # Distancia de cada detección a la etiqueta
    dd = np.linalg.norm(darr - np.array(g), axis=1)
    # Índice de la más cercana
    j = int(np.argmin(dd))
    # Devuelve la detección si está a menos de 1 m de la etiqueta
    return tuple(darr[j]) if dd[j] <= MATCH else None


def fill(s):
    # Rellena los huecos de observación
    # Hacia delante -> cada None toma la última posición observada válida
    n = len(s); last = None
    for i in range(n):
        s[i] = s[i] if s[i] is not None else last; last = s[i] if s[i] is not None else last
    # Hacia detrás -> cada None toma la próxima posición observada válida
    nxt = None
    for i in range(n - 1, -1, -1):
        s[i] = s[i] if s[i] is not None else nxt; nxt = s[i] if s[i] is not None else nxt
    # Si continua siendo None -> (0.0, 0.0)
    return [p if p is not None else (0.0, 0.0) for p in s]


def feats(od):
    # 9 caracteristicas (marco robot) a partir de la trayectoria observada
    # Posición X e Y
    cx, cy = od[:, 0], od[:, 1]
    # Velocidad X e Y
    vx = np.diff(cx, prepend=cx[0])
    vy = np.diff(cy, prepend=cy[0])
    # Aceleración X e Y
    ax = np.diff(vx, prepend=vx[0])
    ay = np.diff(vy, prepend=vy[0])
    # Velocidad lineal y dirección
    sp = np.sqrt(vx**2 + vy**2)
    an = np.arctan2(vy, vx)
    return np.stack([cx, cy, vx, vy, ax, ay, sp, np.sin(an), np.cos(an)], axis=1)


def load_img2d(csv_path, seq):
    # Datos 2D de la secuencia, por peatón y cámara
    # traj[track_id][camera] = {frame: (x_norm, y_norm, bbox_w_norm, bbox_h_norm)}.
    df = pd.read_csv(csv_path)
    df = df[df["sequence"] == seq]
    traj = defaultdict(lambda: defaultdict(dict))
    for r in df.itertuples(index=False):
        traj[int(r.track_id)][r.camera][int(r.frame)] = (
            float(r.x_norm), float(r.y_norm), float(r.bbox_w_norm), float(r.bbox_h_norm))
    return traj


def _fill4(seq):
    # Rellena con forward, backward y ceros pero para 4 componentes
    n = len(seq); last = None
    for i in range(n):
        if seq[i] is None: seq[i] = last
        else: last = seq[i]
    nxt = None
    for i in range(n - 1, -1, -1):
        if seq[i] is None: seq[i] = nxt
        else: nxt = seq[i]
    return [p if p is not None else (0.0, 0.0, 0.0, 0.0) for p in seq]


def img_feats(track_id, win_frames, traj2d, min_cov=1):
    # 11 características de imagen del peatón sobre la ventana observada. Elige la
    # cámara con más frames presentes.
    cams = traj2d.get(int(track_id))
    if not cams:
        return None
    best = max(cams, key=lambda c: sum(int(f) in cams[c] for f in win_frames))
    # Cobertura 2D -> si es insuficiente no se puede usar la fusión
    cov = sum(int(f) in cams[best] for f in win_frames)
    if cov < min_cov:
        return None
    d = cams[best]
    a = np.array(_fill4([d.get(int(f)) for f in win_frames]), np.float32)
    # Posición y medidas del bbox
    x, y, bw, bh = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    # Velocidad X e Y
    vx = np.diff(x, prepend=x[0])
    vy = np.diff(y, prepend=y[0])
    # Aceleración X e Y
    ax = np.diff(vx, prepend=vx[0])
    ay = np.diff(vy, prepend=vy[0])
    # Velocidad lineal y direción
    sp = np.sqrt(vx**2 + vy**2)
    an = np.arctan2(vy, vx)
    return np.stack([x, y, vx, vy, ax, ay, sp, np.sin(an), np.cos(an), bw, bh], axis=1)


def feats20(od, img11):
    # Concatena las 11 características de imagen y las 9 de detección 3D -> 20
    return np.concatenate([img11, feats(od)], axis=1)


def split_fusion_base(tids, frames8, gt, det, traj2d, scaler20, scaler9):
    # Si el peatón tiene datos 2D completos en la ventana de estudio -> se emplea el modelo de
    # fusión temprana -> 20 características escaladas de entrada.
    # Si no lo tiene pero sí que tiene detecciones 3D -> se emplea el modelo de detecciones 3D.
    # -> 9 características escaladas de entrada.
    # En ambos casos se devuelve la misma salida.
    Xf, of, Xb, ob = [], [], [], []
    for t in tids:
        od = np.array(fill([nearest_det(gt[f][t], det.get(f, np.zeros((0, 2), np.float32)))
                            for f in frames8]), np.float32)
        img11 = img_feats(int(t.split(":")[1]), frames8, traj2d, min_cov=len(frames8))
        if img11 is not None:
            Xf.append(scaler20.transform(feats20(od, img11)))
            of.append((t, od[-1]))
        else:
            Xb.append(scaler9.transform(feats(od)))
            ob.append((t, od[-1]))
    return Xf, of, Xb, ob


def to_world(poses, fr, xy):
    # Transforma un punto (x, y) del marco robot al marco mundo
    return (poses[fr] @ np.array([xy[0], xy[1], 0.0, 1.0]))[:2]


def predict_scene(c, gt, det, traj2d, scaler20, scaler9, model_fus, model_base, poses,
                  obs=8, stride=3):
    # Predice en el frame actual y devuelve:
    # preds[t] = (obs, 2)
    # of
    # ob
    # 8 frames observados
    fr8 = [c - stride * (obs - 1 - i) for i in range(obs)]
    # Peatones validos -> etiqueta en los frames de la ventana
    tids = [t for t in gt[c] if all(t in gt.get(f, {}) for f in fr8)]
    # Reparto hibrido: fusion (2D completa) vs base solo-LiDAR (sin imagen)
    Xf, of, Xb, ob = split_fusion_base(tids, fr8, gt, det, traj2d, scaler20, scaler9)
    preds = {}

    def _store(order, pred):
        # Suma el desplazamiento a la ultima deteccion (abs marco robot) y pasa a marco mundo
        for (t, last), pd_ in zip(order, pred):
            preds[t] = np.array([to_world(poses, c, p) for p in (last[None, :] + pd_)])   # (obs,2) mundo

    # Predice cada lote con su modelo
    if Xf:
        _store(of, model_fus.predict(np.array(Xf, np.float32), batch_size=64, verbose=0))
    if Xb:
        _store(ob, model_base.predict(np.array(Xb, np.float32), batch_size=64, verbose=0))
    return preds, of, ob


def load_poses(recon_dir, seq, frames_needed):
    # Filtra los frames que se solicitan
    fn = [f for f in frames_needed if f >= 0]
    # Recorre todos los npz de esa secuencia
    for npz in sorted(recon_dir.glob(f"{seq}_*.npz")):
        # Comprueba si todos los frames están presentes y los devuelve
        d = np.load(npz)
        fr = set(int(x) for x in d["frames"])
        if all(f in fr for f in fn):
            print(f"Poses cargadas de {npz.name}")
            return {int(f): P for f, P in zip(d["frames"], d["poses"])}
    return None
