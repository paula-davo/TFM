# Estudio del conjunto de datos empleado - Carpeta `data_prep`

Scripts de preparación y análisis del dataset JRDB. Cada entrada indica lo que hace el script y los ficheros/figuras que genera.

## Orden de ejecución del pipeline

`01` genera el CSV que consumen `02`, `11` y `12`. El resto (figuras y análisis) son independientes.

---

## 1. `01_gen_jrdb_dataset_json_analysis.py`
> JRDBDatasetJsonAnalysis crea un CSV con las trayectorias 2D a partir de las particiones oficiales de JRDB.

**Salida:** `jrdb_clean_trajectories.csv`

Cada fila es un peatón en un frame concreto (de una secuencia y cámara). Junto a
(`sequence`, `camera`, `frame`, `track_id`, `global_id`, `subtrack_id`, `split`,
`visible_keypoints`) se guardan las **11 características** que usará el modelo:

| Característica | Definición | Obtención |
|---|---|---|
| `x_norm`, `y_norm` | Posición del centro del bbox | Centro del bbox dividido por el ancho/alto de la imagen |
| `bbox_w_norm`, `bbox_h_norm` | Tamaño del bbox | Ancho y alto del bbox normalizados por la imagen |
| `vx`, `vy` | Velocidad | Diferencia de posición entre frames consecutivos |
| `speed` | Velocidad lineal | Módulo `sqrt(vx² + vy²)` |
| `ax`, `ay` | Aceleración | Diferencia de la velocidad entre frames consecutivos |
| `sin_dir`, `cos_dir` | Dirección del movimiento | Seno y coseno de `arctan2(vy, vx)` |

Todo se calcula por **subtrayectoria** (trayectoria continua sin saltos temporales), tras filtrar
las detecciones con pose de baja calidad (< 5 keypoints visibles) y las trayectorias de menos de 20 frames.

## 2. `02_gen_jrdb_dataset_loader.py`
> JRDBDatasetLoader carga el CSV, prepara entradas y salida, y las guarda en formato .npy.

**Salida:** `X_train.npy`, `Y_train.npy`, `subtrack_ids.npy`, `splits.npy`

Aplica una **ventana deslizante** sobre cada subtrayectoria: por cada tramo de 20 frames toma
8 de entrada y 12 de salida.

- **`X_train` → forma `(N, 8, 11)`**: N muestras, 8 frames observados y las **11 características 2D**
  del CSV (`x_norm`, `y_norm`, `vx`, `vy`, `ax`, `ay`, `speed`, `sin_dir`, `cos_dir`, `bbox_w_norm`,
  `bbox_h_norm`), en el marco de la imagen.
- **`Y_train` → forma `(N, 12, 2)`**: 12 frames futuros y 2 valores (Δx, Δy), el **desplazamiento
  relativo** al último frame observado.

`subtrack_ids` y `splits` guardan, por muestra, de qué subtrayectoria viene y si es train o val
(evita mezclar ventanas de la misma trayectoria entre train y val).

## 3. `03_viz_jrdb_dataset_visualizer.py`
> JRDBDatasetVisualizer genera figuras de análisis del dataset.

**Salida:**
- `dataset_longitudes.png`: histograma de la longitud de las subtrayectorias (train vs val), con la línea del mínimo de 20 frames.
- `dataset_densidad.png`: histograma del número de peatones simultáneos por escena (train vs val).
- `dataset_trayectorias_h.png` / `_v.png`: ejemplo de las trayectorias de los peatones en una escena de train y otra de val.

## 4. `04_viz_scene_boxes.py`
> Figura de una escena completa con todos los peatones anotados y sus cuadros delimitadores 2D dibujados.

**Salida:**
- `escena_cajas.png`: una imagen real de la escena con el cuadro delimitador 2D de cada peatón dibujado encima.

## 5. `05_an_jrdb_general_analysis.py`
> JRDBGeneralAnalysis realiza el análisis general de la estructura del dataset JRDB.

## 6. `06_an_pointcloud.py`
> Extraer features de la nube para introducirla como entrada a la red.

## 7. `07_viz_pointcloud_bev.py`
> Figura de nube de puntos del lidar en vista desde arriba, con la caja 3D de un peatón y el radio de recorte de 3,5 m (recorte de entorno).

**Salida:**
- `pointcloud_bev.png`: nube de puntos LiDAR vista desde arriba, con la caja 3D de un peatón y el círculo de 3,5 m que marca el entorno del que se extraen las características.

## 8. `08_viz_lidar_distribution.py`
> Distribución de los peatones en el espacio físico.

**Salida:**
- `dataset_lidar_posiciones.png`: mapa de densidad de las posiciones `(cx, cy)` de los peatones en metros alrededor del robot, que muestra la cobertura de 360° del sensor.
- `dataset_lidar_distancia.png`: histograma de la distancia de cada peatón al robot.

## 9. `09_viz_detection_error.py`
> Figura que comprueba la relación detección-etiqueta 3D, midiendo el error de localización de las detecciones y la tasa de emparejamiento.

**Salida:**
- `dataset_deteccion_error.png`: histograma del error de localización entre detección y etiqueta y la tasa de emparejamiento; representa el ruido del detector LiDAR sobre todo el conjunto.

## 10. `10_viz_detection_error_filtrado.py`
> Figura que comprueba la relación detección-etiqueta 3D, midiendo el error de localización y tasa de emparejamiento -> sobre el conjunto filtrado de subtrayectorias, submuestreadas (STRIDE=3), y solo sobre frames observados.

**Salida:**
- `dataset_deteccion_error_filtrado.png`: mismo histograma de error y tasa de emparejamiento que el anterior, pero sobre el subconjunto curado que realmente entrena el modelo (representa el ruido del detector que el modelo debe tolerar).

## 11. `11_gen_jrdb_lidar3d_det_builder.py`
> Datos 3D a partir de las detecciones del lidar.

**Salida:** `X3d_det_motion.npy`, `Y3d_det.npy`

Igual que el `02` pero en el espacio físico 3D y con submuestreo temporal (`STRIDE=3`, horizonte
de 2,4 s). La posición observada del peatón es la **detección 3D del LiDAR más cercana a su etiqueta**
(umbral de 1 m); si no hay detección, el hueco se rellena con la posición conocida más próxima.

- **`X3d_det_motion` → forma `(N, 8, 9)`**: 8 frames observados y **9 características 3D** obtenidas de
  las detecciones (`cx`, `cy`, `vx`, `vy`, `ax`, `ay`, `speed`, `sin_dir`, `cos_dir`), en **metros** y
  en el marco del robot.
- **`Y3d_det` → forma `(N, 12, 2)`**: 12 frames futuros y el **desplazamiento relativo** en
  metros.

## 12. `12_val_proyeccion_2d3d.py`
> Validación de la correspondencia 2D–3D que entra al modelo de fusión temprana.

**Salida:**
- `val_proyeccion_2d3d_{secuencia}_{frame}_{camara}.png` (una por cámara): sobre la imagen real se
  dibuja la caja 2D anotada (verde) y, encima, la caja 3D del mismo peatón **proyectada** con la
  calibración de JRDB (roja) más su centro 3D proyectado (aspa roja).

Para un frame se proyecta la caja 3D de cada peatón sobre la imagen de la cámara y se comprueba que
el **centro 3D proyectado cae dentro de la caja 2D** anotada.

Reporta el **% de centros dentro** de la caja 2D y el **offset medio**. Un % alto con offset bajo (< 0,5) confirma que la detección 2D
y la 3D son **el mismo peatón en el mismo frame**.

---

## Resumen de figuras (carpeta `data_prep/figuras/`)

| Figura | Generada por |
|---|---|
| `escena_cajas.png` | 04 |
| `dataset_longitudes.png` | 03 |
| `dataset_densidad.png` | 03 |
| `dataset_trayectorias_h.png` / `_v.png` | 03 |
| `pointcloud_bev.png` | 07 |
| `dataset_lidar_posiciones.png` | 08 |
| `dataset_lidar_distancia.png` | 08 |
| `dataset_deteccion_error.png` | 09 |
| `dataset_deteccion_error_filtrado.png` | 10 |
| `val_proyeccion_2d3d_{secuencia}_{frame}_{camara}.png` | 12 |
