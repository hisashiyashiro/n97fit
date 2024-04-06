from typing import Union, IO, Any

import os
from pathlib import Path

import pandas as pd

from .logging import logger


def read_n97format(file: Union[str, os.PathLike, IO[Any]]) -> pd.DataFrame:
    """Read data file and return a numpy array.

    Parameters:
        file (str, os.PathLike, IO[Any]): File path or file-like object.

    Returns:
        pd.DataFrame: Data. Contains the following columns:
            - date: Date. [dtype=datetime64]
            - obsid: Observation ID. [dtype=int64]
            - lat: Latitude. [dtype=float64]
            - lon: Longitude. [dtype=float64]
            - hgt: Height. [dtype=float64]
            - value: Value. [dtype=float64]
            - anl: Analysis date. [dtype=datetime64]
            - qf: Flag. [dtype=int64]

    input file format:
    ---
    123456789012345678901234567890123456789012345678901234
    ------------------------------------------------------
    YYYYMMDD obsID   Lat   Lon   Hgt    Value  Anl.Date QF
    19891011  2135    13   144     2   131.17  19891023 -1
    """

    if isinstance(file, str):
        file = Path(file)

    dtypes = {
        "date": object,
        "obsid": int,
        "lat": float,
        "lon": float,
        "hgt": float,
        "value": float,
        "anl": object,
        "qf": int,
    }
    date_format = {
        "date": "%Y%m%d",
        "anl": "%Y%m%d",
    }

    data = pd.read_table(
        file,
        sep="\s+",
        skiprows=0,
        header=None,
        names=dtypes.keys(),
        dtype=dtypes,
        date_format=date_format,
        parse_dates=[*date_format.keys()],
    )

    for col, fmt in date_format.items():
        data[col] = pd.to_datetime(data[col], format=fmt)

    if isinstance(file, os.PathLike):
        name = os.fspath(file)
    else:
        name = file.name if hasattr(file, "name") else type(file).__name__

    logger.info("")
    logger.info(" Input file          : %s", name)
    logger.info(" +Number of data     : %d", len(data))

    return data
