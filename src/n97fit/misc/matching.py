import numpy as np


def matching(
    n1: int, x1: np.ndarray, n2: int, x2: np.ndarray, y2: np.ndarray, dx: float
) -> np.ndarray:
    """matching x1 to x2 and return y1

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

    def matching_index(x, start):
        m = np.argwhere(np.abs(x - x2[start:n2]) <= dx)
        if m.size != 0:
            return m.max() + start
        else:
            return -1

    y1 = np.zeros_like(x1)

    j = 0
    for i in range(0, n1):
        j = matching_index(x1[i], j)

        if j != -1:
            y1[i] = y2[j]
        else:
            # マッチしなかった場合は0から再検索
            j = 0

    return y1
