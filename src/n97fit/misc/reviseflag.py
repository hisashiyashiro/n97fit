from typing import Tuple

import numpy as np


def reviseflag(y: float, flag: int, sigma: float) -> Tuple[int, int]:
    """revise flag

    Args:
        y (float): data
        flag (int): flag
        sigma (float): threshold

    Returns:
        Tuple[int, int]: revised flag and count
    """

    cnt = 0
    if flag == 0 and abs(y) >= sigma:
        flag = 2
        cnt = 1

    return flag, cnt


def reviseflags(
    flag: np.ndarray,
    index: np.ndarray,
    dev: np.ndarray,
    sigma,
    mask: np.ndarray = None,
) -> int:
    """revise flags at once (same as reviseflag for each data)

    flag[index] is set to 2 if it is 0 and abs(dev) >= sigma.
    Each index must appear only once.

    Args:
        flag (np.ndarray): flags (modified in place)
        index (np.ndarray): indices of the data to judge
        dev (np.ndarray): deviation of each data
        sigma (float or np.ndarray): threshold of each data
        mask (np.ndarray, optional): judge only the data with True

    Returns:
        int: number of the revised flags
    """

    sel = (flag[index] == 0) & (np.abs(dev) >= sigma)
    if mask is not None:
        sel &= mask
    flag[index[sel]] = 2

    return int(np.count_nonzero(sel))
