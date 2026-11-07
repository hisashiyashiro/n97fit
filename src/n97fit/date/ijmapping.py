import numpy as np
import pandas as pd

from ..io import logger
from ..misc import groupmean


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
        data = pd.DataFrame(
            {
                "d": d,
                "t": t,
                "y": y,
                "qf": flag,
            }
        )
    else:
        data = data.copy()

    # [Range of day] ごとにgroupbyするためのインデックスを作成
    data["d_group"] = (data["d"].values - data["d"].min()) // avgday

    effdata = data[data["qf"] == 0]
    group = effdata.groupby("d_group")
    mean = groupmean(effdata, "d_group", ["d", "t", "y"])

    # [Range of day] ごとの平均値を計算
    avgdata = pd.DataFrame(
        {
            "d": mean["d"].astype(int),
            "t": mean["t"],
            "y": mean["y"],
            "ysdev": group["y"].std(ddof=0),
            "ijdata": group.size(),
            "ijmap": pd.Series({k: v.to_list() for k, v in group.groups.items()}),
        }
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
