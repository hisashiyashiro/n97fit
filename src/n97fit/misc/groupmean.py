from typing import List

import numpy as np
import pandas as pd


def _by_position(codes: np.ndarray, ngroups: int):
    """Order the data to process the k-th values of all groups at once

    Returns:
        order (np.ndarray): indices of the data, sorted by the position in
            the group (in the order of the data) and then by the group
        bounds (np.ndarray): end of each position in order
        nobs (np.ndarray): number of data in each group
    """
    by_group = np.argsort(codes, kind="stable")
    g = codes[by_group]
    nobs = np.bincount(g, minlength=ngroups)
    pos = np.arange(len(g)) - (np.cumsum(nobs) - nobs)[g]

    by_pos = np.argsort(pos, kind="stable")
    return by_group[by_pos], np.cumsum(np.bincount(pos)), nobs


def group_mean(values: np.ndarray, codes: np.ndarray, ngroups: int) -> np.ndarray:
    """Mean of each group with Kahan summation

    Same as groupby().mean() of pandas 2.x, which adds the values in the
    order of the data with Kahan summation. NaN is skipped.

    Args:
        values (np.ndarray): values
        codes (np.ndarray): group number of each value (0 to ngroups-1,
            negative for no group)
        ngroups (int): number of groups

    Returns:
        np.ndarray: mean of each group (NaN if no data)
    """

    values = np.asarray(values, dtype=np.float64)
    codes = np.asarray(codes)
    valid = (codes >= 0) & ~np.isnan(values)
    v = values[valid]
    g = codes[valid]
    order, bounds, nobs = _by_position(g, ngroups)
    v = v[order]
    g = g[order]

    total = np.zeros(ngroups)
    comp = np.zeros(ngroups)
    start = 0
    for end in bounds:
        gk = g[start:end]
        y = v[start:end] - comp[gk]
        t = total[gk] + y
        c = (t - total[gk]) - y
        comp[gk] = np.where(np.isnan(c), 0.0, c)  # avoid NaN with inf
        total[gk] = t
        start = end

    with np.errstate(invalid="ignore"):
        return total / nobs


def groupmean(data: pd.DataFrame, by, columns: List[str]) -> pd.DataFrame:
    """Mean of each group with Kahan summation

    Same as data.groupby(by)[columns].mean() of pandas 2.x, which adds the
    values in the order of the data with Kahan summation. pandas 3.x adds
    them in a different way, and the last bits of the means differ. The
    difference grows through the outlier rejection, so the means are
    computed here to keep the results independent of the pandas version.

    Args:
        data (pd.DataFrame): data
        by: key(s) of groupby
        columns (List[str]): columns to average (NaN is skipped)

    Returns:
        pd.DataFrame: mean of each group, indexed by the group keys
    """

    grouped = data.groupby(by)
    index = grouped.size().index
    codes = grouped.ngroup().to_numpy()

    means = {
        col: group_mean(data[col].to_numpy(dtype=np.float64), codes, len(index))
        for col in columns
    }
    return pd.DataFrame(means, index=index)
