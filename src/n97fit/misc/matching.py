import numpy as np


def matching(
    n1: int, x1: np.ndarray, n2: int, x2: np.ndarray, y2: np.ndarray, dx: float
) -> np.ndarray:
    """matching x1 to x2 and return y1

    For each x1, the last x2 within dx is used (x1 and x2 must be sorted).
    Unmatched points are set to 0.

    Args:
        n1 (int): length of x1
        x1 (np.ndarray): x1
        n2 (int): length of x2
        x2 (np.ndarray): x2
        y2 (np.ndarray): y2
        dx (float): tolerance

    Returns:
        np.ndarray: y1
    """

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)[:n2]

    y1 = np.zeros_like(x1)

    # last index of x2 with x2 <= x1 + dx
    k = np.searchsorted(x2, x1[:n1] + dx, side="right") - 1
    ok = (k >= 0) & (np.abs(x1[:n1] - x2[np.clip(k, 0, None)]) <= dx)
    y1[:n1][ok] = np.asarray(y2)[k[ok]]

    return y1
