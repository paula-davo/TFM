# Fase 2 — Evitación de peatones

Segunda fase del TFM: a partir de las trayectorias predichas por los modelos 3D,
el robot navega evitando a los peatones. El *pipeline* va de la **reconstrucción
de la escena** (SLAM) a la **evitación** propiamente dicha (campos potenciales),
pasando por una etapa intermedia de **visualización** de las predicciones.

Los scripts se ejecutan en orden. Todos trabajan sobre la partición oficial de
JRDB (`split_utils.official_masks`) y cargan los modelos 3D desde
`prediccion_3d/best/`. La etapa 01 debe ejecutarse **antes** que las demás, ya
que genera las poses (`.npz`) que el resto reutiliza.

---

## `01_reconstruir_escena.py`

> *Reconstrucción de una escena en MARCO MUNDO.*

Primera etapa del *pipeline*. Como el robot se desplaza, tanto sus nubes LiDAR
como las posiciones de los peatones están expresadas en el marco (móvil) del
robot. Este script estima el movimiento del robot mediante **SLAM**: aplica
**ICP** (variante punto-a-plano) entre nubes LiDAR consecutivas —previa
calibración *lower → ego*, recorte radial y submuestreo por vóxel— y encadena las
transformaciones relativas para obtener la **pose acumulada** en cada fotograma,
tomando como marco mundo el del primer fotograma. Con esas poses transforma las
trayectorias de los peatones al marco mundo, dibuja el recorrido del robot y de
los peatones, y **guarda las poses en un `.npz`** para reutilizarlas en las
etapas siguientes (evita recalcular el SLAM).

## `02a_snapshot_prediccion.py`

> *Snapshot de predicción en una escena reconstruida.*

Visualización **estática** de un único instante. Para un fotograma dado,
reconstruye la observación de cada peatón a partir de las **detecciones 3D**
(asociadas por vecino más cercano y con relleno de huecos), predice con el modelo
multimodal 3D (**20 hipótesis** por peatón) y dibuja en marco mundo tanto las
observaciones como el abanico de predicciones. Sirve para comprobar visualmente
que la predicción encaja con la escena reconstruida antes de llevarla a un bucle
temporal.

## `02b_bucle_replay.py`

> *Bucle temporal en modo REPLAY (sin evitar todavía).*

Versión **animada** del 02a a lo largo del tiempo, aún **sin evitación**. El robot
sigue su camino grabado y los peatones su trayectoria real; cada `K` pasos se
re-predice con el modelo multimodal (**horizonte deslizante**). Genera un GIF y
una tira de *snapshots*. Es el paso intermedio que valida el bucle de
re-predicción antes de dar al robot control propio.

## `03a_evitacion_campos.py`

> *Evitación con CAMPOS POTENCIALES.*

Primera versión de la **evitación real**. El robot deja de seguir el camino
grabado y se mueve como **agente libre** (cinemática diferencial) guiado por
campos potenciales: un campo **atractivo** hacia el objetivo y un campo
**repulsivo** generado por las posiciones actuales de los peatones y por las
**predicciones** del modelo, más una **parada de emergencia** de seguridad.
Re-predice cada `K` pasos y compara el camino de evitación con el grabado,
reportando distancia mínima a peatón, colisiones, si alcanza el objetivo y
longitud del camino.

## `04_comparar_det_multi.py`

> *Comparación DETERMINISTA vs MULTIMODAL como fuente del campo repulsivo.*

Mismo controlador de campos potenciales que el 03a, pero planteado como
**experimento controlado**: ejecuta la evitación dos veces cambiando **solo** el
origen de las predicciones —modelo **determinista** (1 trayectoria) frente a
**multimodal** (20 hipótesis)— y las compara entre sí y con el *baseline* «sin
evitar». Imprime una tabla de métricas y genera una figura y un GIF con los tres
caminos superpuestos. Aísla la pregunta central: ¿evita mejor el robot usando
predicción multimodal?

## `05_riesgo_gauss.py`

> *Evitación con MAPA DE RIESGO GAUSSIANO (mejora del 04).*

Refinamiento del campo repulsivo. En lugar de tratar cada punto predicho como un
obstáculo duro (`1/d`), cada punto aporta una **gaussiana** ponderada por su
**probabilidad** (normalización ÷ nº de modos) y su **inminencia temporal**
(descuento `exp(-k/τ)`); la fuerza repulsiva es el **gradiente de la suma de
gaussianas** (superficie de riesgo suave). Así el multimodal deja de «bloquear de
más»: donde las hipótesis coinciden el riesgo se concentra y donde se abren se
diluye. Mantiene la comparación *det* vs *multi* vs «sin evitar» del 04.

---

## `evitacion_common.py`

Módulo de **utilidades compartidas** por 02a–05: carga de etiquetas y detecciones
3D, asociación detección–peatón por vecino más cercano, relleno de huecos,
cálculo de las 9 características de movimiento y carga de poses del `.npz`.
Centraliza el código duplicado sin alterar el comportamiento de los scripts (la
geometría dependiente de las poses —`to_world`, `peds_world`— se mantiene local
en cada script).
