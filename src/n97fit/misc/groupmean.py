from typing import List

import numpy as np
import pandas as pd


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
    ngroups = len(index)
    codes = grouped.ngroup().to_numpy()

    means = {}
    for col in columns:
        values = data[col].to_numpy(dtype=np.float64)
        valid = (codes >= 0) & ~np.isnan(values)
        g = codes[valid]
        v = values[valid]

        # position of each value in its group, in the order of the data
        order = np.argsort(g, kind="stable")
        g = g[order]
        v = v[order]
        nobs = np.bincount(g, minlength=ngroups)
        pos = np.arange(len(g)) - (np.cumsum(nobs) - nobs)[g]

        # add the k-th values of all groups at once
        order = np.argsort(pos, kind="stable")
        g = g[order]
        v = v[order]
        bounds = np.cumsum(np.bincount(pos))

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
            means[col] = total / nobs  # NaN if no data

    return pd.DataFrame(means, index=index)
