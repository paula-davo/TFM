# Predicción en el espacio 3D
El objetivo de esta sección es la generación de un modelo que, a partir de las detecciones 3D del sensor LiDAR, prediga la intención de desplazamiento futuro de un peatón.

- La arquitectura se define en `lstm_common.py`.
- Muchos de los desarrollos comparten una clase base común, `lidar3d_trainer.py`.

---

## Preparación de datos

### `00a_obtain_data_2d.py`

> *Genera el dataset 2D submuestreado: `X_ds`, `Y_ds` y `subtrack_ids_ds`.*

La partición es la oficial de JRDB. Aplica una ventana deslizante submuestreada (1 de cada 3 frames) sobre el CSV generado en `data_prep`. Genera una entrada con 11 características obtenidas de las detecciones mediante las cámaras del robot. Por otro lado, la salida es el desplazamiento en base al último punto observado.

---

## Modelo de Velocidad Constante

### `00b_baseline_cv.py`

> *Referencia de Velocidad Constante (CVM): el peatón sigue una línea recta con la última velocidad observada.*

Ecuación: `p_{t+k} = p_t + k·v_t`. Es la referencia con la que se comparan todos los modelos desarrollados.

---

## Generación de datasets 3D

### `01_obtain_lidar3d.py`

> *Obtención de datos de etiquetas 3D del lidar. Horizonte: 8 frames observados y 12 frames predichos. Ventana submuestreada (1 de cada 3 frames).*

- Entrada: 9 características en metros: cx, cy, vx, vy, ax, ay, speed, sin_dir, cos_dir. `X3d_motion.npy` (N, 8, 9).
- Salida: desplazamiento relativo al último frame observado en metros. `Y3d.npy` (N, 12, 2).

### `01b_obtain_lidar3d_8obs_8pred.py`

> *Reprocesa el dataset de etiquetas al horizonte: 8 frames observados y 8 frames predichos.*

- Mantiene la misma observación (`X3d_motion.npy`) y recorta la salida a los 8 primeros pasos de `Y3d.npy`. Salida: `X3d_motion_8obs_8pred.npy`,
`Y3d_8obs_8pred.npy`.

### `01c_obtain_lidar3d_12obs_8pred.py`

> *Reprocesa el dataset de etiquetas al horizonte: 12 frames observados y 8 frames predichos.*

- Reconstruye las 20 posiciones GT de cada ventana a partir `X3d_motion.npy` y
`Y3d.npy`, recalculando las 9 características sobre 12 frames de observación. Salida:
`X3d_motion_12obs_8pred.npy`, `Y3d_12obs_8pred.npy`.

### `02_obtain_lidar3d_seq.py`

> *Obtención de datos de características del entorno. Horizonte: 8 frames observados. Ventana submuestreada (1 de cada 3 frames).*

- Entrada: 7 características de entorno: número de puntos sobre el peatón, obstáculo más cercano, densidad total del recorte y obstáculo más cercano en las 4 direcciones.

### `02b_obtain_lidar3d_det_8obs_8pred.py`

> *Reprocesa el dataset de detecciones 3D al horizonte: 8 frames observados y 8 frames predichos.*

- Mantiene la misma observación y recorta la salida a los 8 primeros pasos. Salida:
`X3d_det_8obs_8pred.npy`, `Y3d_det_8obs_8pred.npy`.
- La entrada es la de detecciones 3D generada en `data_prep`.

### `02c_obtain_lidar3d_det_12obs_8pred.py`

> *Reprocesa el dataset de detecciones 3D al horizonte: 12 frames observados y 8 frames predichos.*

- Regenera las 20 posiciones de detecciones de cada ventana. Salida: `X3d_det_12obs_8pred.npy`, `Y3d_det_12obs_8pred.npy`.
- La entrada es la de detecciones 3D generada en `data_prep`.

---

## Experimentos con información LiDAR y horizonte de predicción

A continuación se desarrollan modelos con etiquetas 3D (`03`) y con detecciones 3D (`04`).

### `03_train_lidar3d_labels.py`

> *GT: modelo que recibe etiquetas 3D en metros. Horizonte: 8 frames de observación y 12 frames de predicción.*

- Entrada: 9 características generadas en `01`.
- Referencia para medir el impacto del detector.

### `03b_train_lidar3d_labels_8obs_8pred.py`

> *GT: modelo que recibe etiquetas 3D en metros. Horizonte: 8 frames de observación y 8 frames de predicción.*

### `03c_train_lidar3d_labels_12obs_8pred.py`

> *GT: modelo que recibe etiquetas 3D en metros. Horizonte: 12 frames de observación y 8 frames de predicción.*

### `04_train_lidar3d_det.py`

> *Modelo LSTM que recibe detecciones 3D del lidar en metros. Horizonte: 8 frames de observación y 12 frames de predicción.*

- Detecciones reales del sensor. Mide la robustez del predictor en condiciones realistas. La diferencia con el `03` muestra la pérdida por el ruido del detector.
- Entrada: 9 características generadas en `data_prep`.

### `04b_train_lidar3d_det_8obs_8pred.py`

> *Modelo LSTM que recibe detecciones 3D del lidar en metros. Horizonte: 8 frames de observación y 8 frames de predicción.*

- Entrada: 9 características generadas en `02b`.

### `04c_train_lidar3d_det_12obs_8pred.py`

> *Modelo LSTM que recibe detecciones 3D del lidar en metros. Horizonte: 12 frames de observación y 8 frames de predicción.*

- Entrada: 9 características generadas en `02c`.

**Conclusión del estudio de horizonte:** más observación **no** mejora la predicción. Se adopta **8 obs / 8 pred** como horizonte del pipeline 3D.

---

## Experimentos de arquitectura

*Se comprueba si cambiar la arquitectura o los hiperparámetros mejora sobre el LSTM base, en el horizonte elegido (8 obs / 8 pred, `04b`).*

### `04d_sweep_arch_det_8obs_8pred.py`

> Barrido de arquitecturas/hiperparámetros sobre las detecciones 3D 8/8, ordenado por ADE.

- Entrena y compara varias configuraciones definidas en `arch_variants.py`.
- La única variación que afecta de manera positiva es el cambio en la función de pérdida a **MAE**.

### `04e_train_lidar3d_det_mae.py`

> *LSTM con detecciones 3D (8/8) entrenado con pérdida MAE en lugar de MSE.*

- Igual que el `04b` cambiando la función de pérdida a MAE.
- A partir de este punto, en todos los experimentos se emplea la función de pérdida MAE. Nueva base de referencia.

---

## Información del entorno
Se aplican tres técnicas de adición de información del entorno:
- Información del entorno obtenida de las nubes de puntos del sensor LiDAR.
- Información social de los vecinos.
- Información de pose.

### `05_train_lidar3d_det_env.py`

> *Modelo LSTM con información lidar 3D: detecciones reales (9) + información del entorno (7).*

- Se fusiona en 16 características (detecciones reales obtenidas del `02b` e información del entorno obtenida del `02`).

### `05b_obtain_social3d_det.py`

> *Obtención del dataset social con detecciones 3D.*

- Genera el dataset social: por cada escena (secuencia, cámara) y ventana, agrupa los peatones con sus 9 características, una máscara de peatones activos y su posición 3D.
- Cada peatón se representa con su detección más cercana (a menos de 1 metro de su etiqueta).
- La salida continúa siendo GT.

### `05c_train_social3d_det.py`

> *LSTM Social con detecciones 3D.*

- Hace pooling de los estados de los peatones vecinos.
- Utiliza el script: `social_lstm_model.py`.

### `05d_extract_pose.py`

> *Obtención del dataset con pose: 11 características base 2D + 8 de pose corporal (19 en total).*

- Salida: `X_ds_pose.npy` a partir del CSV generado en `data_prep`.

### `05e_obtain_pose3d.py`

> *Obtención del dataset de pose 3D: 9 movimiento 3D (detecciones) + 8 de pose corporal (17 características en total).*

### `05f_train_pose3d.py`

> *LSTM 3D con información de pose (17 características).*

---

## Técnicas de generalización
Se aplican dos técnicas distintas de generalización para tratar de mejorar los resultados.
- Adición de ruido gaussiano a los datos.
- Reflejo lateral de los datos.

### `06_train_lidar3d_det_noise.py`

> *Ruido gaussiano en la entrada (σ = 0.05).*

- Sólo está activa en el entrenamiento.

### `06b_train_lidar3d_det_flip.py`

> *Aumento de datos por reflejo lateral.*

- Duplica los datos de entrada para el entrenamiento.
- Se niegan `cy`, `vy`, `ay`, `sin_dir` y la salida `dy`.

---

## Fusión con datos 2D

### `07_train_combined3d_ftemp.py`

> *Modelo que combina datos de imagen 2D y de detecciones 3D con fusión temprana (20 características).*

- **Fusión temprana**: concatena las 11 características de imagen 2D con las 9 de detección 3D, resultando en 20 características de entrada para un único LSTM.

### `08_train_combined3d_ftard.py`

> *Modelo que combina datos de imagen 2D y de detecciones 3D con fusión tardía (promedio de dos modelos).*

- **Fusión tardía**: se entrenan dos modelos independientes (uno con las 11 características de imagen 2D y otro con las 9 de detección 3D) y promedia sus predicciones.

### `09_train_combined3d_multiprediction.py`

> *Modelo de múltiples predicciones sobre la fusión temprana: genera K=20 trayectorias.*

- En lugar de una trayectoria, genera 20 predicciones por peatón.
- Se obtiene la franja densa, que es la zona donde más predicciones se generan, y se promedia la franja obteniendo una sola predicción, con un radio de 0,5 metros.

**Conclusión.** El mejor modelo hasta el momento es la fusión temprana de datos de imagen y detecciones.

### `10_train_combined3d_ftemp_reg.py`

> *Fusión temprana imagen2D+deteccion3D regularizada.*

- Se aplican cambios para obtener una curva de entrenamiento/validación más sana.
- Se reduce la tasa de aprendizaje.
- Se aumenta el Dropout.

---

## Módulos comunes y utilidades
*Scripts que se importan en otros de los ejecutados, anteriormente explicados.*

### `lidar3d_trainer.py`

> Clase base común de entrenamiento (`Lidar3DTrainer`).

- Implementa el flujo que comparten muchos de los modelos desarrollados: `load_data → scale_features → build_model → train → save_outputs → evaluate → run`.

### `arch_variants.py`

> Constructor flexible que define los cambios de arquitectura evaluados.

- Generaliza `lstm_common.py` para poder realizar un barrido de configuraciones, empleado en `04d`.

### `social_lstm_model.py`

> Implementación del modelo `SocialLSTM`.

- Define la arquitectura del LSTM Social, empleado por `05c`.

### `listar_epocas.py`

> Lista el nº de épocas y las pérdidas finales de cada entrenamiento.

- Lee los historiales de `prediccion_3d/history/` e imprime, por modelo, cuántas épocas entrenó y sus `loss`/`val_loss` finales.

### `replot_curvas.py`

> Re-dibuja las curvas de pérdida de los historiales guardados con texto más grande.

- Recorre los historiales de `prediccion_3d/history/` y regenera cada curva de entrenamiento (train vs validación) con tamaños de fuente y figura mayores, en `figuras_replot/`.
