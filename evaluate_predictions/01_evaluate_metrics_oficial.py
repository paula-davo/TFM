"""
Evalúa las métricas (MAE, RMSE, MSE, MASE, ADE, FDE) de los modelos de detecciones/etiquetas 3D.
"""
import sys
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OBS, PRED = 8, 8
BAND_RADIUS = 0.5
# Metrics
MET = ["MAE", "RMSE", "MSE", "MASE", "ADE", "FDE"]
# Column weight
W = {"MAE": 9, "RMSE": 9, "MSE": 10, "MASE": 7, "ADE": 9, "FDE": 9}
HEAD = f"{'Modelo':<28} " + " ".join(f"{k:>{W[k]}}" for k in MET)
sys.path.insert(0, DATA_DIR)
sys.path.insert(0, f"{DATA_DIR}/prediccion_3d")   # social_lstm_model
from split_utils import official_masks
from metrics_utils import classic_metrics, dense_band_pred
from social_lstm_model import SocialLSTM
from lstm_common import flip_lat, cv_disp


def fit_scaler(Xtr, nf, flipfn=None, Ytr=None):
    # Si el modelo se entrenó con reflejo lateral, se replica en el ajuste del scaler
    if flipfn is not None:
        Xf, _ = flipfn(Xtr, Ytr)
        Xtr = np.concatenate([Xtr, Xf])
    return StandardScaler().fit(Xtr.reshape(-1, nf))


def ev_single(ckpt, X, Y, vm, scaler, nf):
    # Evalúa un modelo de predicción de una sola trayectoria
    # Filas de validación
    Xva, Yva = X[vm], Y[vm]
    # Aplica scaler
    Xs = scaler.transform(Xva.reshape(-1, nf)).reshape(Xva.shape)
    # Carga el modelo
    m = load_model(f"{DATA_DIR}/{ckpt}", compile=False)
    # Predice
    pred = m.predict(Xs, batch_size=512, verbose=0)
    # Obtiene las métricas clásicas
    return classic_metrics(Yva, pred, Xva[:, -1, 0:2])


def ev_ensemble(ckpt_img, ckpt_3d, Ximg, X3, Y, vm, sc_img, sc_3d):
    # Evalúa un modelo de predicción de una sola trayectoria con fusión tardía
    # Media de las predicciones de un modelo con imagen y de otro con detecciones 3D
    # Filas de validación y aplica scaler
    Yva = Y[vm]
    Xi = sc_img.transform(Ximg[vm].reshape(-1, 11)).reshape(Ximg[vm].shape)
    X3s = sc_3d.transform(X3[vm].reshape(-1, 9)).reshape(X3[vm].shape)
    # Carga el modelo 2D y el modelo 3D
    mi = load_model(f"{DATA_DIR}/{ckpt_img}", compile=False)
    m3 = load_model(f"{DATA_DIR}/{ckpt_3d}", compile=False)
    # Predice
    pred = (0.5 * mi.predict(Xi, batch_size=512, verbose=0)
            + 0.5 * m3.predict(X3s, batch_size=512, verbose=0))
    # Obtiene las métricas clásicas
    return classic_metrics(Yva, pred, X3[vm][:, -1, 0:2])


def ev_social3d(ckpt, Xs_, Ys_, Ms_, Ps_, vm, scaler):
    # Evalúa un modelo LSTM Social
    # Obtiene las filas de validación con el filtro vm
    Xva, Yva, Mva, Pva = Xs_[vm], Ys_[vm], Ms_[vm], Ps_[vm]
    # Aplica scaler
    Xn = scaler.transform(Xva.reshape(-1, 9)).reshape(Xva.shape)
    # El modelo LSTM se creó con un call para introducir el social pooling por lo que
    # no se pudo guardar tal cual, pero sí que se guardo sus pesos y su checkpoint
    # Por lo que se crea el modelo y se cargan sus pesos
    m = SocialLSTM(10, OBS, PRED, 9, 128, 2.0)
    _ = m((Xn[:2].astype(np.float32), Pva[:2].astype(np.float32), Mva[:2]), training=False)
    m.load_weights(f"{DATA_DIR}/{ckpt}")
    # Se predice
    pred = []
    for i in range(0, Xva.shape[0], 256):
        pred.append(m((Xn[i:i+256].astype(np.float32), Pva[i:i+256].astype(np.float32),
                       Mva[i:i+256]), training=False).numpy())
    pred = np.concatenate(pred)
    # Filtra para quedarse con los peatones reales
    mf = Mva.reshape(-1).astype(bool)
    pf = pred.reshape(-1, PRED, 2)[mf]
    yf = Yva.reshape(-1, PRED, 2)[mf]
    last = Xva[:, :, -1, 0:2].reshape(-1, 2)[mf]
    # Obtiene las métricas clásicas
    return classic_metrics(yf, pf, last)


def ev_multitrayectoria(ckpt, Ximg, X3, Y, vm, scaler):
    # Evalúa el modelo multitrayectoria sobre la fusión temprana (entrada 20 = 11 imagen + 9 detección 3D).
    # Franja densa
    # Concatena datos de imagen y de detecciones 3D y extrae conjunto de validación
    X20 = np.concatenate([Ximg, X3], axis=2)
    Xva, Yva, X3va = X20[vm], Y[vm], X3[vm]
    # Aplica el scaler
    Xs = scaler.transform(Xva.reshape(-1, 20)).reshape(Xva.shape)
    # Carga el modelo
    m = load_model(f"{DATA_DIR}/{ckpt}", compile=False)
    # Predice 20 trayectorias
    predK = m.predict(Xs, batch_size=256, verbose=0)        # (N,K,T,2)
    # Guarda la última posición observada de las detecciones 3D
    last = X3va[:, -1, 0:2]
    # Obtiene la trayectoria representativa de la franja densa
    rep = dense_band_pred(predK, BAND_RADIUS)
    # Obtiene las métricas clásicas
    return classic_metrics(Yva, rep, last)


def safe(fn, *a, **k):
    # Ejecuta una evaluación. Si falla, avisa.
    try:
        return fn(*a, **k)
    except Exception as e:
        print(f"  [omitido] {e}")
        return None


def cv_metrics(X, Y, vm):
    # Métricas de la referencia de Velocidad Constante.
    # Obtiene conjunto de validación.
    Xva, Yva = X[vm], Y[vm]
    # obtiene las métricas clásicas
    return classic_metrics(Yva, cv_disp(Xva, PRED), Xva[:, -1, 0:2])


def fmt_val(n, m):
    # Formateo de una fila de métricas
    cells = {"MAE": f"{m['MAE']:.5f}", "RMSE": f"{m['RMSE']:.5f}", "MSE": f"{m['MSE']:.2e}",
             "MASE": f"{m['MASE']:.3f}", "ADE": f"{m['ADE']:.5f}", "FDE": f"{m['FDE']:.5f}"}
    return f"{n:<28} " + " ".join(f"{cells[k]:>{W[k]}}" for k in MET)


def fmt_pct(m, ref):
    # Formateo de los pocentajes
    cells = {k: f"{(1 - m[k] / ref[k]) * 100:+.1f}%" for k in MET}
    return f"{'   Δ% vs CV':<28} " + " ".join(f"{cells[k]:>{W[k]}}" for k in MET)


def main():
    results = []

    Xm = np.load(f"{DATA_DIR}/data/X3d_motion.npy")            # etiquetas 3D (GT)
    Y3 = np.load(f"{DATA_DIR}/data/Y3d.npy")[:, :PRED, :]
    Xd = np.load(f"{DATA_DIR}/data/X3d_det_motion.npy")        # detecciones 3D
    Yd = np.load(f"{DATA_DIR}/data/Y3d_det.npy")[:, :PRED, :]
    Xe = np.load(f"{DATA_DIR}/data/X_lidar_seq.npy")           # entorno (7 features)
    Ximg = np.load(f"{DATA_DIR}/data/X_ds.npy")                # imagen 2D (11 features)
    ids3 = np.load(f"{DATA_DIR}/data/subtrack_ids_ds.npy", allow_pickle=True)
    tm, vm = official_masks(ids3)

    # Modelo de velocidad constante 8/8. Referencia común
    cv_ref = cv_metrics(Xd, Yd, vm)

    # 1. LSTM con detecciones 3D (04e)
    results.append(("Detecciones 3D (04e)",
                    safe(ev_single, "prediccion_3d/best/lidar3d_det_mae_best.keras", Xd, Yd, vm, fit_scaler(Xd[tm], 9), 9)))

    # 2. GT con etiquetas 3D
    results.append(("Techo GT (etiquetas)",
                    safe(ev_single, "prediccion_3d/best/lidar3d_labels_8obs_8pred_best.keras", Xm, Y3, vm, fit_scaler(Xm[tm], 9), 9)))

    # 3. Det + entorno (05)
    X16 = np.concatenate([Xd, Xe], axis=2)
    results.append(("Det + entorno",
                    safe(ev_single, "prediccion_3d/best/lidar3d_det_env_best.keras", X16, Yd, vm, fit_scaler(X16[tm], 16), 16)))

    # 4. Det + social (05c)
    Xsa = np.load(f"{DATA_DIR}/data/X_social3d_det_ds.npy")
    Ysa = np.load(f"{DATA_DIR}/data/Y_social3d_det_ds.npy")[:, :, :PRED, :]
    Msa = np.load(f"{DATA_DIR}/data/mask_social3d_det_ds.npy")
    Psa = np.load(f"{DATA_DIR}/data/pos3d_social3d_det_ds.npy")
    Ksa = np.load(f"{DATA_DIR}/data/scene_keys3d_det_ds.npy", allow_pickle=True)
    tms, vms = official_masks(Ksa)
    scs = StandardScaler().fit(Xsa[tms].reshape(-1, 9))
    results.append(("Det + social *",
                    safe(ev_social3d, "prediccion_3d/best/social3d_det_best.weights.h5", Xsa, Ysa, Msa, Psa, vms, scs)))

    # 5. Det + pose (05e)
    Xpo = np.load(f"{DATA_DIR}/data/X_3d_pose_ds.npy")
    results.append(("Det + pose",
                    safe(ev_single, "prediccion_3d/best/pose3d_best.keras", Xpo, Yd, vm, fit_scaler(Xpo[tm], 17), 17)))

    # 6/7. Det + generalización (ruido 06 / reflejo 06b)
    results.append(("Det + gener. (ruido)",
                    safe(ev_single, "prediccion_3d/best/lidar3d_det_noise_best.keras", Xd, Yd, vm, fit_scaler(Xd[tm], 9), 9)))
    results.append(("Det + gener. (reflejo)",
                    safe(ev_single, "prediccion_3d/best/lidar3d_det_flip_best.keras", Xd, Yd, vm, fit_scaler(Xd[tm], 9, flip_lat, Yd[tm]), 9)))

    # 8. Fusión temprana imagen+3D (07)
    X20 = np.concatenate([Ximg, Xd], axis=2)
    results.append(("Fusión temprana 2D+3D",
                    safe(ev_single, "prediccion_3d/best/combined3d_best.keras", X20, Yd, vm, fit_scaler(X20[tm], 20), 20)))

    # 8b. Fusión temprana regularizada (10)
    results.append(("Fusión temprana reg. (10)",
                    safe(ev_single, "prediccion_3d/best/combined3d_reg_best.keras", X20, Yd, vm, fit_scaler(X20[tm], 20), 20)))

    # 9. Fusión tardía imagen+3D (08)
    results.append(("Fusión tardía 2D+3D",
                    safe(ev_ensemble, "prediccion_3d/best/img2meters_best.keras", "prediccion_3d/best/ensemble3d_3d_best.keras",
                         Ximg, Xd, Yd, vm, fit_scaler(Ximg[tm], 11), fit_scaler(Xd[tm], 9))))

    # 10. Multitrayectoria sobre la fusión temprana (09) — franja densa
    results.append(("Multitrayectoria (franja densa)",
                    safe(ev_multitrayectoria, "prediccion_3d/best/combined3d_multimodal_best.keras", Ximg, Xd, Yd, vm, fit_scaler(X20[tm], 20))))

    # Se imprime
    # Δ% = mejora relativa respecto al CV (velocidad constante)
    rows = [(n, m) for n, m in results if m is not None]
    print(f"\n############### ESPACIO FÍSICO 3D (METROS) ###############")
    print(f"(referencia %: Velocidad Constante 8/8 detecciones)")
    print(HEAD)
    print("-" * len(HEAD))
    # Fila de referencia: CV (velocidad constante); Δ% = --- (es la propia referencia)
    print(fmt_val("Velocidad Constante", cv_ref))
    print(f"{'   Δ% vs CV':<28} " + " ".join(f"{'-':>{W[k]}}" for k in MET))
    for n, m in rows:
        print(fmt_val(n, m))
        print(fmt_pct(m, cv_ref))

    print("\n* Social LSTM: muestreo escena-céntrico (val ligeramente distinto).")
    print("Δ% vs CV: mejora respecto al baseline de velocidad constante, positivo = menor error.")


if __name__ == "__main__":
    main()
