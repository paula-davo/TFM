"""
Partición oficial de JRDB-Pose: 20 secuencias de train / 7 de val.
"""
import numpy as np

VAL_SEQUENCES = (
    "clark-center-2019-02-28_1",
    "gates-ai-lab-2019-02-08_0",
    "huang-2-2019-01-25_0",
    "meyer-green-2019-03-16_0",
    "nvidia-aud-2019-04-18_0",
    "tressider-2019-03-16_1",
    "tressider-2019-04-26_2",
)


def is_val(subtrack_id):
    # Devuelve true si el subtrack ID pertenece a una secuencia de validación oficial
    return any(subtrack_id.startswith(v + "_") for v in VAL_SEQUENCES)


def official_masks(subtrack_ids):
    # Devuelve la máscara de entrenamiento y de validación booleanas según la partición oficial de JRDB.
    vm = np.array([is_val(s) for s in subtrack_ids], dtype=bool)
    return ~vm, vm
