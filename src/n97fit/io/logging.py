import logging

import numpy as np

logging.basicConfig(format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("n97fit")
logger.setLevel("INFO")


def reportrej(ndata: int, ymd: np.ndarray, y: np.ndarray, flag: np.ndarray, vflag: int):
    """report flagged data

    Args:
        ndata (int): number of data
        ymd (np.ndarray): yyyymmdd
        y (np.ndarray): data
        flag (np.ndarray): flag
        vflag (int): flag value to be reported

    """

    logger.info("")
    logger.info("***Rejected data (flag = %2d)***", vflag)

    sumcnt = 0

    logger.info(" YYYYMMDD    value")

    for i in range(0, ndata):
        if flag[i] == vflag:
            logger.info(" %8d %9.3f", ymd[i], y[i])
            sumcnt += 1

    logger.info(" -Number of rejected data : %5d", sumcnt)
