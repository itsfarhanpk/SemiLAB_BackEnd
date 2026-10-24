"""Ideal normalized diode current model."""

import numpy as np

from .constants import Q, K_B


def normalized_current(voltage_v, temperature_k: float, ideality: float = 1.0):
    exponent = np.clip(Q * np.asarray(voltage_v) / (ideality * K_B * temperature_k), -100.0, 100.0)
    return np.expm1(exponent)
