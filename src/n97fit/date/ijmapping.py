import numpy as np
import pandas as pd

from ..io import logger
from ..misc.groupmean import group_mean


def ijmapping(
    avgday: int,
    t: np.ndarray = None,
    d: np.ndarray = None,
    y: np.ndarray = None,
    flag: np.ndarray = None,
    data: pd.DataFrame = None,
    asarray: bool = False,
):
    """Allocate to 2D Array

    Args:
        avgday (int): The average day.
        ndata (int): The number of data.
        d (np.ndarray): The day.
        t (np.ndarray): The time.
        y (np.ndarray): The value.
        flag (np.ndarray): The flag.

    Returns:
        m_ndata (int):
        m_dt (float): The average interval of t.
        m_d (np.ndarray): The day.
        m_t (np.ndarray): The time.
        m_yavg (np.ndarray): The average value.
        m_ysdev (np.ndarray): The stddev of value.
        m_ijdata (np.ndarray): The number of data.
        m_ijmap (np.ndarray): The index of data.

    """

    if data is None:
        label = None
    else:
        d = data["d"].to_numpy()
        t = data["t"].to_numpy()
        y = data["y"].to_numpy()
        flag = data["qf"].to_numpy()
        label = data.index.to_numpy()

    d = np.asarray(d)
    t = np.asarray(t)
    y = np.asarray(y)

    # [Range of day] ごとにgroupbyするためのインデックスを作成
    d_group = (d - d.min()) // avgday

    # group number of the effective data (flag 0), in the order of d_group
    eff = np.flatnonzero(np.asarray(flag) == 0)
    keys, codes = np.unique(d_group[eff], return_inverse=True)
    codes = codes.ravel()
    ngroups = len(keys)

    # indices of the data in each group, in the order of the data
    ijdata = np.bincount(codes, minlength=ngroups)
    members = eff[np.argsort(codes, kind="stable")]
    if label is not None:
        members = label[members]
    ijmap = np.empty(ngroups, dtype=object)
    for i, m in enumerate(np.split(members, np.cumsum(ijdata)[:-1])):
        ijmap[i] = m.tolist()

    # [Range of day] ごとの平均値を計算
    avgdata = pd.DataFrame(
        {
            "d": group_mean(d[eff], codes, ngroups).astype(int),
            "t": group_mean(t[eff], codes, ngroups),
            "y": group_mean(y[eff], codes, ngroups),
            "ysdev": pd.Series(y[eff]).groupby(codes).std(ddof=0).to_numpy(),
            "ijdata": ijdata,
            "ijmap": ijmap,
        },
        index=pd.Index(keys, name="d_group"),
    )

    ndata = len(avgdata)
    avgdt = (avgdata["t"].max() - avgdata["t"].min()) / float(ndata)

    logger.info("")
    logger.info(" =>Allocate to 2D Array")
    logger.info("     New number of data          : %7d", ndata)
    logger.info("     Average interval of t       : %7.4f", avgdt)

    if asarray:
        return (
            ndata,
            avgdt,
            avgdata["d"].values,
            avgdata["t"].values,
            avgdata["y"].values,
            avgdata["ysdev"].values,
            avgdata["ijdata"].values,
            avgdata["ijmap"].values,
        )
    else:
        return avgdata


def flatten_ijmap(ijmap: np.ndarray, ijdata: np.ndarray):
    """Flatten the index of data in each group

    Args:
        ijmap (np.ndarray): indices of data in each group (from ijmapping)
        ijdata (np.ndarray): number of data in each group

    Returns:
        (np.ndarray, np.ndarray): indices of data, and group of each data
    """

    ij = np.fromiter(
        (k for m in ijmap for k in m), dtype=np.int64, count=int(np.sum(ijdata))
    )
    i = np.repeat(np.arange(len(ijdata)), ijdata)
    return ij, i
