import numpy as np


def harmonic(ti, a: np.ndarray, nh: int):
    """calculate the seasonal cycle

    Args:
        ti (float or np.ndarray): time
        a (np.ndarray): coefficients
        nh (int): number of harmonics

    Returns:
        float or np.ndarray: seasonal cycle
    """

    ti = np.asarray(ti, dtype=np.float64)
    ys = np.zeros_like(ti)

    for i in range(0, nh):
        if i % 2 == 0:
            ys += a[i] * np.sin(np.pi * ti * float(i + 2))
        else:
            ys += a[i] * np.cos(np.pi * ti * float(i + 1))

    return ys if ys.ndim else float(ys)
