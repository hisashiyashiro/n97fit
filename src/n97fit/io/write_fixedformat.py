from typing import Dict, List, Union, Any, IO

import os
import re
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)

from .logging import logger
from ..date import date2cday


def write_fixedformat(
    data: pd.DataFrame,
    file: Union[str, os.PathLike, IO[Any]],
    format: Union[Dict[str, str], List[str], str] = None,
    int_format: str = "%6d",
    float_format: str = "%13.7f",
    date_format: str = "%(excel)6d",
    fillna: Union[Dict[str, Any], Any] = -999.99999,
    **kwargs,
):
    """Write data to file.

    [Format specification]

    The format specification supports both str.format() style like
    `{:12.5f}`, `{:6d} and printf style like `%12.5f`, `%6d`.
        ex)
        - `{:12.5f}`    : Floating point number with 12 digits wide
                          and 5 decimal places.
        - `%12.5f`      : Same as above.
        - `{:6d}`       : Integer with 6 digits wide.
        - `%6d`         : Same as above.

    If the column's dtype is datetime64, you can specify the format
    in strftime style.
    Additionally, if the column's dtype is datetime64, you can specify
    `excel` as the field_name to output the value as Microsoft Excel
    serial value (1904 date system).
        ex)
        - `%Y-%m-%d`    : Date formating by strftime style. YYYY-MM-DD.
        - `{:%Y-%m-%d}` : Same as above.
        - `%(excel)6d`  : Excel serial value with 6 digits.
        - `{excel:6d}`  : Same as above.

    int_format, float_format, and date_format specify the default formats
    for integers, floats, and dates, respectively. These arguments will
    be overridden by the formats specified in the format argument.

    Args:
        data (pd.DataFrame): Data to write.
        file (str, os.PathLike, IO[Any]): File path or file-like object.
        format (Union[Dict[str, str], List[str], str], optional): Format.
            Dict[str, str]: Format for each column by name.
            List[str]: Format for each column by order.
            str: Format for all columns.
        int_format (str, optional): Default integer format. Overridden by format.
        float_format (str, optional): Default float format. Overridden by format.
        date_format (str, optional): Default date format. Overridden by format.
        fillna (Union[Dict[str, Any], Any], optional): Value to replace NaN.
            Dict[str, str]: Value for each column by name.
            str: Value for all columns.
        **kwargs: Additional arguments for pd.DataFrame.to_string.
    """

    if isinstance(file, str):
        file = Path(file)

    formatters = make_formatters(data, format, int_format, float_format, date_format)

    if not isinstance(fillna, dict):
        fillna = {col: fillna for col in data.columns}

    data = data.copy()
    for col, value in fillna.items():
        if is_object_dtype(data[col].dtype):
            logger.warn("Warning: %s is object dtype. Fillna is skipped.", col)
            continue

        data[col] = data[col].fillna(value)

    data.to_string(file, index=False, formatters=formatters, **kwargs)

    name = file.name if hasattr(file, "name") else type(file).__name__
    logger.info(" +Output %-24s : %5d lines", name, len(data))


class DefaultFormatter:
    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, value):
        return str(value)


class StringFormatter:
    match_regex = re.compile(r"\{.*?\}")

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, value) -> str:
        if isinstance(value, dict):
            return self.fmt.format(**value)
        else:
            return self.fmt.format(value)

    @classmethod
    def match(cls, fmt: str, dtype: Any) -> bool:
        """Check if the format string is in str.format() style.

        Conditions:
            - Match the regex pattern
        """

        return bool(re.search(cls.match_regex, fmt))


class PrintfFormatter:
    match_regex = re.compile(r"%.*?[diouxXeEfFgGcrs]{1}")

    def __init__(self, fmt: str):
        self.fmt = fmt

    def __call__(self, value: Any) -> str:
        return self.fmt % value

    @classmethod
    def match(cls, fmt: str, dtype: Any) -> bool:
        """Check if the format string is in printf style.

        Conditions:
            - Not datetime64 dtype
            - Not str.format() style
            - Match the regex pattern
        """

        return (
            not is_datetime64_any_dtype(dtype)
            and not StringFormatter.match(fmt, dtype)
            and re.search(cls.match_regex, fmt)
        )


class StrftimeFormatter:
    match_regex = re.compile(r"(%[aAbBcdHIjmMpSUwWxXyYz:Z]{1,2})+")

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, value) -> str:
        return value.strftime(self.fmt)

    @classmethod
    def match(cls, fmt: str, dtype: Any) -> bool:
        """Check if the format string is in strftime style.

        Conditions:
            - datetime64 dtype
            - Not str.format() style
            - Match the regex pattern
        """

        return (
            is_datetime64_any_dtype(dtype)
            and not StringFormatter.match(fmt, dtype)
            and re.search(cls.match_regex, fmt)
        )


class ExcelDateFormatter:
    match_regex = re.compile(r"%\(excel\)|%\(cday\)|{excel:|{cday:")

    def __init__(self, fmt: str):
        if StringFormatter.match(fmt, int):
            self.formatter = StringFormatter(fmt)
        else:
            self.formatter = PrintfFormatter(fmt)

    def __call__(self, value: Any) -> str:
        val = date2cday(value)
        return self.formatter({"excel": val, "cday": val})

    @classmethod
    def match(cls, fmt: str, dtype: Any) -> bool:
        """Check if the format string is for Excel date.

        Conditions:
            - datetime64 dtype
            - Match the regex pattern
            - str.format() style or printf style
        """

        return (
            is_datetime64_any_dtype(dtype)
            and re.match(cls.match_regex, fmt)
            and (StringFormatter.match(fmt, int) or PrintfFormatter.match(fmt, int))
        )


def formatter_factory(dtype: Any, fmt: str):
    if not fmt:
        return DefaultFormatter(fmt)

    if ExcelDateFormatter.match(fmt, dtype):
        return ExcelDateFormatter(fmt)
    elif StrftimeFormatter.match(fmt, dtype):
        return StrftimeFormatter(fmt)
    elif PrintfFormatter.match(fmt, dtype):
        return PrintfFormatter(fmt)
    elif StringFormatter.match(fmt, dtype):
        return StringFormatter(fmt)

    return DefaultFormatter(fmt)


def make_formatters(
    df: pd.DataFrame,
    format: Union[Dict[str, str], List[str], str],
    int_format: str,
    float_format: str,
    date_format: str,
):
    columns = df.columns
    dtypes = df.dtypes

    if format is None:
        format = {}
    elif isinstance(format, str):
        format = {col: format for col in columns}
    elif isinstance(format, list):
        format = {col: fmt for col, fmt in zip(columns, format)}

    formatters = {}
    for name, dtype in zip(columns, dtypes):
        if name in format:
            f = format[name]
        else:
            if is_datetime64_any_dtype(dtype):
                f = date_format
            elif is_float_dtype(dtype):
                f = float_format
            elif is_integer_dtype(dtype):
                f = int_format
            else:
                f = None

        formatters[name] = formatter_factory(dtype, f)

    return formatters
