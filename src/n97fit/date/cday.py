from typing import Union
from datetime import date, timedelta

import pandas as pd

# start date
ys = 1904
sdate = date(ys, 1, 1)

# cache dictionaries
cday_dict = dict()
date_dict = dict()


def date2cday(dt: Union[date, pd.Timestamp]) -> int:
    """Converts a date to a day count from 1904-01-01

    Args:
        year (int): year
        month (int): month
        day (int): day

    Returns:
        int: day count
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()

    if dt in cday_dict:
        return cday_dict[dt]

    cday = dt.toordinal() - sdate.toordinal()
    cday_dict[dt] = cday

    return cday


def cday2date(cday: int) -> date:
    """Converts a day count from 1904-01-01 to a date

    Args:
        cday (int): day count

    Returns:
        date: date
    """

    if cday in date_dict:
        return date_dict[cday]

    cdate = sdate + timedelta(days=int(cday))
    date_dict[cday] = cdate

    return cdate
