import math

import numpy as np
from .fourier import fourier


def frcoef(ndata: int, t: np.ndarray, y: np.ndarray, nh: int, nterm: int) -> np.ndarray:
    """Estimate the coefficients of Fourier harmonics

    Args:
        ndata (int): number of data
        t (np.ndarray): time
        y (np.ndarray): value
        nh (int): number of harmonics
        nterm (int): number of terms for fourier transformation
    Returns:
        np.ndarray: coefficients of Fourier harmonics
    """

    if nh == 0 and nh == nterm:
        return np.zeros(nterm, dtype=np.float64)

    f = np.zeros((nterm, ndata), dtype=np.float64)

    for j in range(0, nh):
        if j % 2 == 0:
            f[j, :] = np.sin(2.0 * math.pi * float(int((j) / 2) + 1) * t[:ndata])
        else:
            f[j, :] = np.cos(2.0 * math.pi * float(int((j) / 2) + 1) * t[:ndata])

    for j in range(nh, nterm):
        if j == nh:
            f[j, :] = 1.0
        else:
            f[j, :] = t[:ndata]

    return fourier(f, y, nterm, ndata)
