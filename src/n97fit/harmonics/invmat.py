from typing import Tuple

import numpy as np


def invmat(n: int, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    matrix inversion by elimination with partial piviting
    original matrix=a, inverse matrix=b

    Args:
        n (int): size of matrix
        a (np.ndarray): matrix

    Returns:
        (np.ndarray, np.ndarray): matrix, inverse matrix
    """

    esp = 0.0000001

    a = a.copy()
    b = np.zeros_like(a)

    # construct identity matrix b(i,j)=i
    b[:n, :n] = np.identity(n, dtype=np.float64)

    # locate maximum magnitude a(i,k) on or below main diagonal
    d_el = 1.0
    for k in range(0, n):
        if k < n - 1:
            imax = k
            amax = np.abs(a[k, k])
            kp1 = k + 1

            for i in range(kp1, n):
                if amax < np.abs(a[i, k]):
                    imax = i
                    amax = np.abs(a[i, k])

            # interchange rows imax and k if imax not equal to k
            if imax != k:
                a[k, :], a[imax, :] = a[imax, :], a[k, :].copy()
                b[k, :], b[imax, :] = b[imax, :], b[k, :].copy()

            d_el = -d_el

        # test for singular matrix
        if np.abs(a[k, k]) - esp > 0:
            d_el *= a[k, k]

            # divide pivot row by its main diagonal element
            div = a[k, k]
            a[k, :n] = a[k, :n] / div
            b[k, :n] = b[k, :n] / div

            for i in range(0, n):
                amult = a[i, k]
                if i != k:
                    a[i, :] = a[i, :] - amult * a[k, :]
                    b[i, :] = b[i, :] - amult * b[k, :]

        else:
            raise RuntimeError(f"singular matrix for k={k:d}")

    return a, b
