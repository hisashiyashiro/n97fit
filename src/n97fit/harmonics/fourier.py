import numpy as np

from .invmat import invmat


def fourier(f: np.ndarray, y: np.ndarray, n_t: int, n_d: int) -> np.ndarray:
    """least square fitting with fourier harmonics

    Args:
        f (np.ndarray): matrix
        y (np.ndarray): value
        n_t (int): number of terms
        n_d (int): number of values

    Returns:
        np.ndarray:
    """

    # generate normal matrix c
    c = np.dot(f, f.T)

    # generate right-side matrix b
    b = np.dot(f[:n_t, :n_d], y[:n_d])

    # invert normal matrix c
    c, d = invmat(n_t, c)

    # matrix d is inverse of normal matrix
    # form matrix product of inverse and right-side matrix
    a = np.dot(d[:n_t, :n_t], b[:n_t])

    return a
