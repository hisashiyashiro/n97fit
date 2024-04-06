import math
import numpy as np


def harmonic(ti: float, a: np.ndarray, nh: int) -> float:
    """calculate the seasonal cycle

    Args:
        ti (float): time
        a (np.ndarray): coefficients
        nh (int): number of harmonics

    Returns:
        float: seasonal cycle
    """

    ys = 0.0

    for i in range(0, nh):
        if i % 2 == 0:
            ys += a[i] * math.sin(math.pi * ti * float(i + 2))
        else:
            ys += a[i] * math.cos(math.pi * ti * float(i + 1))

    return ys
