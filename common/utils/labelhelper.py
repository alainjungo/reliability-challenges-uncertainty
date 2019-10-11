import scipy.ndimage.morphology as morphology
import numpy as np


def to_one_hot(label_arr: np.ndarray, nb_classes=None) -> np.ndarray:
    if nb_classes is None:
        nb_classes = label_arr.max() + 1

    return np.eye(nb_classes, dtype=label_arr.dtype)[label_arr]


def boarder_mask(binary_label_map: np.ndarray, distance_in: int, distance_out: int):
    binary_label_map = binary_label_map.astype(np.bool)

    dist_in = morphology.distance_transform_edt(binary_label_map)
    dist_out = morphology.distance_transform_edt(~binary_label_map)

    mask = (dist_in <= distance_in) * (dist_out <= distance_out)

    return dist_in + dist_out, mask
