import numpy as np

from scipy.interpolate import BSpline, PPoly, make_smoothing_spline


def smooth(ndata: int, x1: np.ndarray, y1: np.ndarray, rm: float) -> PPoly:
    """smoothing of data points by cubic spline

    Args:
        ndata (int): number of data
        x1 (np.ndarray): x
        y1 (np.ndarray): y
        rm (float): P=1.0/(1.0+RM)

    Returns:
        PPoly: cubic spline (callable, extrapolated with the end polynomials)
    """

    # The smoothing spline of y - c is that of y minus c. Subtracting the
    # mean makes the rounding error of the solver much smaller.
    ymean = np.mean(y1[:ndata])
    spline = make_smoothing_spline(x1[:ndata], y1[:ndata] - ymean, lam=rm)

    # add the mean back to the coefficients (B-splines sum to one)
    spline = BSpline(
        spline.t, spline.c + ymean, spline.k, extrapolate=spline.extrapolate
    )

    # Far outside the data, the B-spline basis takes large values that
    # cancel each other. The piecewise polynomial form keeps the
    # extrapolation accurate.
    return PPoly.from_spline(spline)
