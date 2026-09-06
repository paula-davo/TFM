# Fase 2 — Evitación de peatones
A partir de las trayectorias predichas por los modelos, el robot navega evitando a los peatones.

Los scripts se ejecutan en orden. Todos trabajan sobre la partición oficial de
JRDB (`split_utils.official_masks`) y cargan los modelos 3D desde
`prediccion_3d/best/`.

---

## `01a_ilustracion_icp.py`

> *Ilustración del registro ICP.*

Se trata de una figura de muestra para la memoria. Toma dos barridos LiDAR consecutivos y los dibuja antes de la aplicación de ICP (desalineadas) y después: correctamente superpuestas.

## `01b_reconstruir_escena.py`

> *Reconstrucción de una escena en marco mundo.*

Las posiciones de los peatones se dan en el marco del robot, que en algunas escenas está en movimiento. Se estima el movimiento del robot mediante SLAM: aplicando ICP a las nubes LiDAR consecutivas, y se encadenan las transformaciones obteniendo la pose acumulada. Se transforman las posiciones de los peatones al marco mundo. Se dibuja el recorrido del robot y de los peatones y se guardan las poses en un `.npz` para ser reutilizadas en las siguientes etapas.

## `02a_snapshot_prediccion.py`

> *Snapshot de predicción en una escena reconstruida.*

Para un fotograma dado, se reconstruye la observación de cada peatón a partir de las detecciones 3D y se predice la intención de trayectoria futura de cada peatón. Se emplea fusión temprana en caso de que el robot disponga de datos 2D y 3D del peatón, y el modelo de detecciones 3D en caso de sólo disponer de estas segundas. Se dibujan en marco mundo las observaciones y las predicciones.

## `02b_bucle_replay.py`

> *Bucle temporal en modo replay (sin evitar).*

Predicciones a lo largo del tiempo, sin evitación. El robot realiza su camino definido en el conjunto de datos y cada `K` pasos re-predice la intención futura de los peatones que detecta. Según los datos disponibles, utiliza fusión temprana o modelo de detecciones 3D.

## `03a_evitacion_campos.py`

> *Evitación con campos potenciales.*

Simulación de la evitación. El robot ya no sigue el camino grabado, sino que emplea un controlador basado en campos potenciales para evitar a los peatones según sus predicciones de intención de trayectoria para estos.

Campos potenciales:
- Campo atractivo hacia el objetivo.
- Campo repulsivo generado por las posiciones actuales de los peatones y por las predicciones del modelo (fusión temprana en caso de disponer de todos los datos y modelo de detecciones 3D en caso contrario).
- Evasión de emergencia en caso de tener un peatón muy cercano.

Por otro lado, re-predice cada `K` pasos. Reporta distancia mínima a peatón, colisiones, si alcanza el objetivo y la longitud del camino realizado.

Importante: tanto para la predicción como para las figuras se emplean las detecciones 3D como trayectoria observada de los peatones. Sin embargo, para conocer en las métricas si el robot ha colisionado o no con los peatones, se emplean las etiquetas 3D. Al alcanzar el objetivo, el robot se detiene y la simulación finaliza. 

Comandos usados para las figuras de referencia:

```
python evitacion/03a_evitacion_campos.py clark-center-2019-02-28_1 60 540
python evitacion/03a_evitacion_campos.py gates-ai-lab-2019-02-08_0 30 420 3 5 4
python evitacion/03a_evitacion_campos.py huang-2-2019-01-25_0 120 360
python evitacion/03a_evitacion_campos.py meyer-green-2019-03-16_0 60 420
```

---

## `evitacion_common.py`

Módulo de **utilidades compartidas**:

- **Calibración LiDAR y nubes** — `LOWER2EGO` y `load_cloud_ego`: llevan la nube del
  LiDAR inferior al marco ego, recortan, submuestrean por vóxel y calculan normales.
  Las usan `01a`/`01b`.
- **Carga de datos** — `load_gt`/`load_det` (etiquetas y detecciones 3D desde JSON),
  `load_img2d` (trayectorias 2D del CSV, por peatón y cámara) y `load_poses`
  (poses del `.npz` de la reconstrucción que cubran el rango de frames pedido).
- **Asociación y relleno** — `nearest_det` (detección más cercana a una etiqueta,
  dentro de `MATCH`=1 m) y `fill`/`_fill4` (relleno de huecos hacia delante/atrás).
- **Características** — `feats` (9, detecciones 3D), `img_feats` (11, imagen 2D a
  partir de los datos de `load_img2d`) y `feats20` (20, concatena ambas).
- **Predicción híbrida** — `split_fusion_base`: fusión (20 características) si el
  peatón tiene cobertura 2D completa en la ventana, o detecciones 3D (9) si no.
- **Geometría y predicción compartidas** — `to_world` (marco robot -> mundo) y
  `predict_scene` (predicción híbrida de 1 trayectoria por peatón en un frame,
  usada por `02a`/`02b`/`03a`).

