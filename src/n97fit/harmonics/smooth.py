from typing import Tuple
import numpy as np

from scipy.interpolate import make_smoothing_spline


def smooth(
    ndata: int, x1: np.ndarray, y1: np.ndarray, rm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """smoothing of data points by cubic spline

    Args:
        ndata (int): number of data
        x1 (np.ndarray): x
        y1 (np.ndarray): y
        rm (float): P=1.0/(1.0+RM)

    Returns:
        (np.ndarray, np.ndarray): coefficients of cubic spline
    """

    spline = make_smoothing_spline(x1[:ndata], y1[:ndata], lam=rm)
    return spline
