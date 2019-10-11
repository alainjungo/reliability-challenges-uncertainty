import warnings
import csv

import numpy as np


def uncertainty_to_foreground_probabilities(uncertainty_np: np.ndarray, prediction_np: np.ndarray):
    if prediction_np.shape != uncertainty_np.shape:
        raise ValueError('shapes must agree. Found {} and {}'.format(uncertainty_np.shape, prediction_np.shape))
    check_min_max(uncertainty_np)
    if prediction_np.max() > 1:
        raise ValueError('Found class larger than 1. Only works for binary problems')

    foreground_probabilities = uncertainty_np*0.5  # range (0,0.5)
    foreground_probabilities[prediction_np == 1] = 1 - foreground_probabilities[prediction_np == 1]
    return foreground_probabilities


def rescale_uncertainties(uncertainty_np: np.ndarray, min_, max_, epsilon=1e-5):
    rescaled_np = (uncertainty_np - min_)/(max_ - min_)  # range [0,1]
    rescaled_np = rescaled_np*(1 - 2*epsilon) + epsilon  # range [0+eps, 1-eps]
    return rescaled_np


def add_background_probability(probability_np: np.ndarray):
    check_min_max(probability_np)
    probability_with_background = np.stack([1 - probability_np, probability_np], axis=-1)
    return probability_with_background


def check_min_max(arr: np.ndarray, min_=0, max_=1, only_warn=False):
    # note: has to be in range (0,1)
    probability_max = arr.max()
    if probability_max > max_:
        msg = 'Found value larger than {}: "{}"'.format(max_, probability_max)
        if not only_warn:
            raise ValueError(msg)
        else:
            warnings.warn(msg)

    probability_min = arr.min()
    if probability_min < min_:
        msg = 'Found value smaller than {}: "{}"'.format(min_, probability_min)
        if not only_warn:
            raise ValueError(msg)
        else:
            warnings.warn(msg)


def read_min_max(min_max_file: str):
    with open(min_max_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        confidence_entry, min_, max_ = next(reader)
    return float(min_), float(max_)


def pandas_error_recall(fp, fn, fpu, fnu):
    undef_mask = ((fnu + fpu) == 0) & ((fn + fp) == 0)
    result = (fnu + fpu) / (fn + fp)
    if undef_mask.any():
        result[undef_mask] = 1.
    return result


def pandas_error_precision(tpu, tnu, fpu, fnu):
    undef_mask = ((fnu + fpu) == 0) & ((fnu + fpu + tpu + tnu) == 0)
    result = (fnu + fpu) / (fnu + fpu + tpu + tnu)
    if undef_mask.any():
        result[undef_mask] = 1.
    return result




