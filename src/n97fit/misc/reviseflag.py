from typing import Tuple


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
