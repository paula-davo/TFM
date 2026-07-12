"""
Utilidades compartidas de la fase 2 (evitacion).

Funciones puras y de carga reutilizadas por 02a/02b/03a/04/05. Las rutas se pasan como argumento para no acoplar el modulo a una
ubicacion concreta. La geometria dependiente de las poses (to_world, peds_world)
se deja local en cada script porque cierra sobre variables propias de la ejecucion.
"""
import json
from collections import defaultdict
import numpy as np

MATCH = 1.0   # m, umbral de asociacion deteccion-peaton


def load_gt(lab_dir, seq):
    gt = defaultdict(dict)
    # Recorre el JSON de labels 3D correspondiente a la secuencia
    for k, anns in json.load(open(lab_dir / f"{seq}.json"))["labels"].items():
        # Para cada anotación guarda: gt[frame][label_id] = (cx, cy) - posición del peatón en metros
        fr = int(k.replace(".pcd", ""))
        for a in anns:
            gt[fr][a["label_id"]] = (a["box"]["cx"], a["box"]["cy"])
    return gt


def load_det(det_dir, seq):
    # Recorre el JSON de detecciones 3D correspondientes a la secuencia
    det = {}
    for k, ds in json.load(open(det_dir / f"{seq}.json"))["detections"].items():
        # Para cada anotación guarda: det[frame] = (Nd, 2) - apila las detecciones 3D
        fr = int(k.replace(".pcd", ""))
        det[fr] = np.array([(d["box"]["cx"], d["box"]["cy"]) for d in ds], np.float32) \
            if ds else np.zeros((0, 2), np.float32)
    return det


def nearest_det(g, darr):
    # Asocia peatón a detección
    # Comprueba que haya detecciones
    if darr.shape[0] == 0:
        return None
    dd = np.linalg.norm(darr - np.array(g), axis=1)   # distancia de cada deteccion a la etiqueta
    j = int(np.argmin(dd))                            # indice de la mas cercana
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
    # 9 caracteristicas (marco robot) a partir de la trayectoria observada od
    cx, cy = od[:, 0], od[:, 1]                                  # posicion X, Y (m)
    vx = np.diff(cx, prepend=cx[0])
    vy = np.diff(cy, prepend=cy[0])   # velocidad X, Y
    ax = np.diff(vx, prepend=vx[0])
    ay = np.diff(vy, prepend=vy[0])   # aceleracion X, Y
    sp = np.sqrt(vx**2 + vy**2)
    an = np.arctan2(vy, vx)               # velocidad lineal y direccion
    return np.stack([cx, cy, vx, vy, ax, ay, sp, np.sin(an), np.cos(an)], axis=1)


def load_poses(recon_dir, seq, frames_needed):
    # Primer .npz de la secuencia que cubra todos los frames pedidos (>=0) -> {frame: pose 4x4}, o None
    fn = [f for f in frames_needed if f >= 0]
    for npz in sorted(recon_dir.glob(f"{seq}_*.npz")):
        d = np.load(npz)
        fr = set(int(x) for x in d["frames"])
        if all(f in fr for f in fn):
            print(f"Poses cargadas de {npz.name}")
            return {int(f): P for f, P in zip(d["frames"], d["poses"])}
    return None
