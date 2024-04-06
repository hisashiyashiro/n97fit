from typing import List
from pathlib import Path

from pandas import read_fwf

from .io import read_n97format, write_fixedformat, logger
from . import N97Fit


def readcfg(file, nrows: int = None) -> List:
    """
    Read configuration file and return a numpy array.

    Parameters:
      n (int): Number of rows to read from the file.
      filename (os.PathLike): Path to the configuration file.

    Returns:
      np.ndarray: Numpy array containing the configuration data.
    """

    logger.info("")
    logger.info(" Config file         : %s", file)
    logger.info(" +Number of list     : %d", nrows)

    dtypes = {
        "label": str,
        "value": int,
    }

    conf = read_fwf(
        file,
        widths=[34, 4],
        dtype=dtypes,
        names=dtypes.keys(),
        nrows=nrows,
        skiprows=0,
        header=None,
    )

    for i, row in conf.iterrows():
        logger.info(" +%-34s %3d", row["label"], row["value"])

    return conf["value"].values.tolist()


def getargs():
    import argparse

    parser = argparse.ArgumentParser(
        prog="n97fit",
        description="A program of data selection and curve fitting.",
    )
    parser.add_argument("input", type=Path, help="input data file")
    parser.add_argument("config", type=Path, help="config file")
    parser.add_argument("outdir", type=Path, help="output directory")
    args, _ = parser.parse_known_args()

    return args.input, args.config, args.outdir


def cli_main():
    input, config, outdir = getargs()

    [avgday, sigma, outday, nhm, cutlo, orderlo, orderhi, sw_iteration] = readcfg(
        file=config, nrows=8
    )

    # Read input data
    data = read_n97format(file=input)

    # create N97Fit instance
    n97fit = N97Fit(
        avgday=avgday,
        sigma=sigma,
        nhm=nhm,
        cutlo=cutlo,
        orderlo=orderlo,
        orderhi=orderhi,
        sw_iteration=sw_iteration,
        loglevel="INFO",
    )

    # exec fitting
    n97fit.fit(data)

    # fetch results
    all_data = n97fit.data_all()
    rej_data = n97fit.data_rej()
    eff_ave = n97fit.data_eff_ave()
    eq_ave = n97fit.data_eq_ave()
    fit_all = n97fit.all(interval=outday)
    fit_annual = n97fit.annual()
    fit_monthly = n97fit.monthly()
    fit_multimonth = n97fit.multimonth()
    fit_period = n97fit.period()
    fit_onemonth = n97fit.onemonth()

    # write output
    logger.info("")
    logger.info(" * * * * * START Output * * * * *")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(" Output directory    : %s", outdir)

    write_fixedformat(
        all_data,
        Path(outdir, "data_all.txt"),
        format={"flg": "%3d"},
        float_format="%12.5f",
    )

    write_fixedformat(
        rej_data,
        Path(outdir, "data_rej.txt"),
        format={"flg": "%3d"},
        float_format="%12.5f",
    )

    write_fixedformat(
        eff_ave,
        Path(outdir, "data_eff_ave.txt"),
        float_format="%12.5f",
        date_format="%(excel)11d",
    )

    write_fixedformat(
        eq_ave,
        Path(outdir, "data_eq_ave.txt"),
        float_format="%12.5f",
        date_format="%(excel)11d",
    )

    write_fixedformat(
        fit_all,
        Path(outdir, "fit_all.txt"),
        format={"year": "%9.5f"},
        float_format="%13.7f",
        date_format="%(excel)7d",
    )

    write_fixedformat(
        fit_annual,
        Path(outdir, "fit_annual.txt"),
        format={"year": "%4d"},
        float_format="%13.7f",
        fillna={
            "ave_fit": -999.9999999,
            "stdev_fit": -99.9999999,
            "ave_trend": -999.9999999,
            "stdev_trend": -99.9999999,
        },
    )

    write_fixedformat(
        fit_monthly,
        Path(outdir, "fit_monthly.txt"),
        format={"year": "%4d", "mn": "%2d"},
        float_format="%13.7f",
        fillna={
            "ave_fit": -999.9999999,
            "stdev_fit": -99.9999999,
            "ave_trend": -999.9999999,
            "stdev_trend": -99.9999999,
        },
    )

    write_fixedformat(
        fit_onemonth,
        Path(outdir, "fit_onemonth.txt"),
        float_format="%13.7f",
    )

    write_fixedformat(
        fit_multimonth,
        Path(outdir, "fit_multimonth.txt"),
        format={"period": "%10s", "mn": "%2d"},
        float_format="%13.7f",
    )

    write_fixedformat(
        fit_period,
        Path(outdir, "fit_period.txt"),
        format={"period": "%10s"},
        float_format="%13.7f",
    )

    logger.info("")
    logger.info(" ************** E N D ***************")


if __name__ == "__main__":
    cli_main()
