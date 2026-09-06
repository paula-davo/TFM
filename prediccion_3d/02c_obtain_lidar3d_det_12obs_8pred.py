"""
Obtención de datos de entorno lidar y de las detecciones 3D -> 12 frames observados, 8 predichos.

Se regenera con el builder de detecciones (data_prep/11_gen_jrdb_lidar3d_det_builder.py),
llamándolo con obs_len=12, pred_len=8.
"""

import importlib.util
import numpy as np

DATA_DIR = "/mnt/c/Users/paula/Desktop/TFM"
OBS_LEN, PRED_LEN = 12, 8

if __name__ == "__main__":
    # Carga el builder de detecciones
    spec = importlib.util.spec_from_file_location(
        "det_builder_mod", f"{DATA_DIR}/data_prep/11_gen_jrdb_lidar3d_det_builder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Re-genera con el nuevo horizonte
    builder = mod.JRDBLidar3DDetBuilder(obs_len=OBS_LEN, pred_len=PRED_LEN)
    builder.load_gt_xy()
    builder.load_det_xy()
    builder.load_dataset()
    builder.build()

    print(f"X: {builder.X3d.shape}  Y: {builder.Y3d.shape}  "
          f"emparejamiento={100*builder.matched/builder.total:.1f}%")
    np.save(f"{DATA_DIR}/X3d_det_12obs_8pred.npy", builder.X3d)
    np.save(f"{DATA_DIR}/Y3d_det_12obs_8pred.npy", builder.Y3d)
    print("Guardado: X3d_det_12obs_8pred.npy, Y3d_det_12obs_8pred.npy")
