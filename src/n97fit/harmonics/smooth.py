import numpy as np

from scipy.interpolate import PPoly
from scipy.linalg import solveh_banded


def smooth(ndata: int, x1: np.ndarray, y1: np.ndarray, rm: float) -> PPoly:
    """smoothing of data points by cubic spline (Reinsch)

    The natural cubic spline g that minimizes
        sum (y - g(x))**2 + rm * integral g''(x)**2 dx,
    the same as scipy.interpolate.make_smoothing_spline(x, y, lam=rm).
    It is solved by the Reinsch algorithm (Green and Silverman, 1994):
        (R + rm * Q^T Q) gamma = Q^T y,  g = y - rm * Q gamma,
    where gamma is g'' at the inner knots and R, Q^T Q are banded.

    Args:
        ndata (int): number of data
        x1 (np.ndarray): x (strictly increasing)
        y1 (np.ndarray): y
        rm (float): P=1.0/(1.0+RM)

    Returns:
        PPoly: cubic spline (callable, extrapolated with the end polynomials)
    """

    x = np.asarray(x1[:ndata], dtype=np.float64)
    y = np.asarray(y1[:ndata], dtype=np.float64)
    if ndata < 3:
        raise ValueError("smooth requires at least 3 data points")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")

    # The smoothing spline of y - c is that of y minus c. Subtracting the
    # mean makes the rounding error much smaller.
    ymean = np.mean(y)
    y = y - ymean

    # Q (n x n-2): column j has 1/h[j], -1/h[j]-1/h[j+1], 1/h[j+1]
    # at rows j, j+1, j+2
    ih = 1.0 / h
    q0 = ih[:-1]
    q1 = -(ih[:-1] + ih[1:])
    q2 = ih[1:]

    # A = R + rm * Q^T Q (symmetric pentadiagonal, upper banded form)
    ab = np.zeros((3, ndata - 2))
    ab[2] = (h[:-1] + h[1:]) / 3.0 + rm * (q0 * q0 + q1 * q1 + q2 * q2)
    ab[1, 1:] = h[1:-1] / 6.0 + rm * (q1[:-1] * q0[1:] + q2[:-1] * q1[1:])
    ab[0, 2:] = rm * (q2[:-2] * q0[2:])

    # second derivatives at the inner knots
    gamma = solveh_banded(ab, q0 * y[:-2] + q1 * y[1:-1] + q2 * y[2:])

    # values at the knots
    qg = np.zeros(ndata)
    qg[:-2] += q0 * gamma
    qg[1:-1] += q1 * gamma
    qg[2:] += q2 * gamma
    g = y - rm * qg

    # cubic polynomial on each interval (natural: g'' = 0 at both ends)
    g2 = np.concatenate(([0.0], gamma, [0.0]))
    c = np.empty((4, ndata - 1))
    c[0] = (g2[1:] - g2[:-1]) / (6.0 * h)
    c[1] = g2[:-1] / 2.0
    c[2] = (g[1:] - g[:-1]) / h - h * (2.0 * g2[:-1] + g2[1:]) / 6.0
    c[3] = g[:-1] + ymean

    return PPoly(c, x, extrapolate=True)
