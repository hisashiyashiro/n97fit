from typing import Union
from datetime import date
from math import nan as NaN
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal

from .date import ijmapping, flatten_ijmap, date2cday
from .io import logger, reportrej
from .misc import reviseflags, matching, groupmean, FiltFilt
from .misc import make_rng, jackknife_scale, check_exclude_frac, exclude_units
from .misc.groupmean import group_mean
from .harmonics import frcoef, harmonic, smooth


TRUE = 1

label = [
    "sin(t)  :",
    "con(t)  :",
    "sin(t/2):",
    "con(t/2):",
    "sin(t/3):",
    "con(t/3):",
    "sin(t/4):",
    "con(t/4):",
    "sin(t/5):",
    "con(t/5):",
    "sin(t/6):",
    "con(t/6):",
]


class N97Fit:
    avgday: int
    sigma: int
    nhm: int
    cutlo: int
    orderlo: int
    orderhi: int
    sw_iteration: bool

    syear: int
    trange: tuple
    nh: int
    hm_coef: np.ndarray
    raw_data: pd.DataFrame = None
    eff_data: pd.DataFrame = None
    fit_result: pd.DataFrame = None
    input_data: pd.DataFrame = None
    boot_result: pd.DataFrame = None
    boot_smooth: np.ndarray = None
    boot_trend: np.ndarray = None
    boot_growthrate: np.ndarray = None
    boot_info: dict = None

    extrap_days: int = 30
    day_interval: float = 0.0027
    days_in_year: float = 365.25

    def __init__(
        self,
        avgday: int = 1,
        sigma: int = 3,
        nhm: int = 2,
        cutlo: int = 24,
        orderlo: int = 26,
        orderhi: int = 16,
        sw_iteration: bool = True,
        loglevel: Union[str, int] = "INFO",
    ):
        """Initialize N97Fit

        Args:
            avgday (int): The range of days for average
            sigma (int): The number of sigma
            nhm (int): The number of harmonics
            cutlo (int): The cutoff month for low path filter
            orderlo (int): The order for low path filter
            orderhi (int): The order for high path filter
            sw_iteration (bool): Switch for iteration
            loglevel (Union[str, int]): Log level.
        """

        self.avgday = avgday
        self.sigma = sigma
        self.nhm = nhm
        self.cutlo = cutlo
        self.orderlo = orderlo
        self.orderhi = orderhi
        self.sw_iteration = sw_iteration

        logger.setLevel(loglevel)

    def fit(self, data: pd.DataFrame):
        """Execute fitting

        Args:
            data (pd.DataFrame): Input data
        Returns:
            pd.DataFrame: Fitting result
        """

        logger.info(" * * * * * Start Program * * * * *")

        pl_avgday = self.avgday
        pl_sigma = self.sigma
        pl_nhm = self.nhm
        pl_cutlo = self.cutlo
        pl_orderlo = self.orderlo
        pl_orderhi = self.orderhi
        sw_iteration = self.sw_iteration

        ## Preprocess
        rawndata, rawt, rawd, rawymd, rawy, flag, syear = self.preprocess(
            data, asarray=True
        )

        ## Stage 1-3 : Allocate to 2D Array

        logger.info("")
        logger.info(" *** Allocate to 2D Array ***")
        logger.info(" The range of days for average: %5d", pl_avgday)

        ndata, dt, d, t, y, ysdev, ijdata, ijmap = ijmapping(
            pl_avgday, rawt, rawd, rawy, flag, asarray=True
        )

        # if individual value is out of sdev*pl_sigma, set to 2
        ij, i = flatten_ijmap(ijmap, ijdata)
        sgm_allow = ysdev[i] * pl_sigma
        sumcnt = reviseflags(
            flag, ij, rawy[ij] - y[i], sgm_allow, mask=ysdev[i] != 0
        )

        logger.info("")
        logger.info(" *** Reject irregular data in same day ***")
        logger.info(" =>Reject data (sigma*%2d): %5d", pl_sigma, sumcnt)

        reportrej(rawndata, rawymd, rawy, flag, -1)
        reportrej(rawndata, rawymd, rawy, flag, 1)
        reportrej(rawndata, rawymd, rawy, flag, 2)

        # Re-Allocate 2D Array
        ndata, dt, d, t, y, ysdev, ijdata, ijmap = ijmapping(
            pl_avgday, rawt, rawd, rawy, flag, asarray=True
        )

        logger.info("")
        logger.info(" Average interval of the day : %8d", int(dt * self.days_in_year))
        logger.info(" Start day                   : %8d", d[0])
        logger.info(" End day                     : %8d", d[ndata - 1])
        logger.info(" Start t                     : %8.4f", t[0])
        logger.info(" End t                       : %8.4f", t[ndata - 1])

        # *** Stage 1-4 : Prepare extended (and interpolated) t (= x) *** !

        xstart = d[0] - 910
        xend = d[ndata - 1] + 2920
        xdata = xend - xstart + 1

        x_d = np.zeros(xdata, dtype=np.int64)
        x_t = np.zeros(xdata, dtype=np.float64)

        tstart = 0
        tend = 0

        sdate = date2cday(date(syear, 1, 1))
        x_d[:] = np.arange(xstart, xend + 1)
        # assume 12:00, same as t of the data (see preprocess)
        x_t[:] = ((x_d - sdate).astype(np.float64) + 0.5) / self.days_in_year

        # last grid point within day_interval
        idx = np.nonzero(np.abs(x_t - t[0]) <= self.day_interval)[0]
        if idx.size:
            tstart = idx[-1]
        idx = np.nonzero(np.abs(x_t - t[ndata - 1]) <= self.day_interval)[0]
        if idx.size:
            tend = idx[-1]

        logger.info("")
        logger.info("***Prepare ideal wave (1-day resolution)***")
        logger.info(" Number of data              : %8d", xdata)
        logger.info(" Interval of the day         : %8d", 1)
        logger.info(" Start d                     : %8d", x_d[0])
        logger.info(" End d                       : %8d", x_d[xdata - 1])
        logger.info(" Interval x                  : %8.4f", 1 / self.days_in_year)
        logger.info(" Start x                     : %8.4f", x_t[0])
        logger.info(" End x                       : %8.4f", x_t[xdata - 1])
        logger.info("")

        # memo: tstart, tendの表示は1-based indexなので+1する
        logger.info(" Start i(Location of start t): %8d", tstart + 1)
        logger.info(" End i  (Location of end t)  : %8d", tend + 1)
        logger.info(" x(Start i)                  : %8.4f", x_t[tstart])
        logger.info(" x(End i)                    : %8.4f", x_t[tend])

        # *** Stage 1 : END ***

        if sw_iteration == 1:
            itelim = 5
        else:
            itelim = 1

        for iteration in range(0, itelim):
            logger.info("")
            logger.info("* * * * * * * * * * * * * * * * * * * * * * * * * * * *")
            logger.info(
                "* * *              START FITTING No. %1d             * * *", iteration + 1
            )
            logger.info("* * * * * * * * * * * * * * * * * * * * * * * * * * * *")

            # * * *   Stage 2 : fit of Spline + fourier to the data

            logger.info("")
            logger.info(" *** START Fourier&Spline FITTING ***")

            ite_i = -1
            while TRUE == 1:
                ite_i += 1

                logger.info("")
                logger.info(" +ITERATION fourier: %02d", ite_i + 1)

                # *** Stage 2-1 : fit of linear eq. + fourier to the data *** !

                # Calc fourier coefficient
                nh = pl_nhm * 2
                nterm = nh + 2
                hm_coef = frcoef(ndata, t, y, nh, nterm)

                lineb = hm_coef[nh + 0]
                linea = hm_coef[nh + 1]

                logger.info("  +Number of Harmonics %3d", pl_nhm)
                logger.info("  +COEFFICIENTS A(K)=")
                logger.info("  Const.  : %10.2f", lineb)
                logger.info("  t       : %10.2f", linea)
                for i in range(0, nh):
                    logger.info("  %s %10.4E", label[i], hm_coef[i])

                t_ln = linea * t + lineb
                t_hm = harmonic(t, hm_coef, nh)

                # Calc deviation (y - harmonic - linear)
                de_hm = y - t_hm
                de_hmln = y - t_hm - t_ln
                sgm_hmln = np.std(de_hmln)

                logger.info("  +Sigma(Y - harmonic - line)   = %12.7E", sgm_hmln)
                logger.info("")

                # *** Stage 2-2 : fit of spline + fourier(revised) to the data ***
                pi = np.arctan(1.0) * 4.0
                rm = (5.0 / (2.0 * pi)) ** 4 / dt
                nterm = nh

                ite_j = 0 - 1
                sgm_temp = 0

                while TRUE == 1:
                    ite_j += 1
                    logger.info("  +ITERATION spline: %02d-%02d", ite_i + 1, ite_j + 1)

                    # Smooth spline
                    spline = smooth(ndata, t, de_hm, rm)
                    t_sp = spline(t)

                    de_sp = y - t_sp

                    # Re-calc fourier coefficient from detrend
                    hm_coef = frcoef(ndata, t, de_sp, nh, nterm)
                    t_hm = harmonic(t, hm_coef, nh)

                    # Calc deviation (y - new harmonic - spline)
                    de_hm = y - t_hm
                    de_hmsp = y - t_hm - t_sp

                    sgm_hmsp = np.std(de_hmsp)
                    logger.info("  +Sigma(Y - harmonic - spline) = %12.7E", sgm_hmsp)

                    # Judge
                    if ite_j != 0 and sgm_temp != 0:
                        if 1.0 - sgm_hmsp / sgm_temp <= 0.001:
                            break
                    sgm_temp = sgm_hmsp

                # *** Stage 2-3 : Judge the flags and re-allocate 2D data *** !
                if iteration >= 1:
                    logger.info("")
                    logger.info(" =>Skip Judgement in this Stage...")
                    break

                # if value is out of sigma*pl_sigma, set to 2
                sgm_allow = sgm_hmsp * pl_sigma
                ij, i = flatten_ijmap(ijmap, ijdata)
                sumcnt = reviseflags(flag, ij, de_hmsp[i], sgm_allow)

                logger.info("")
                logger.info(" =>Reject data (sigma*%2d): %5d", pl_sigma, sumcnt)

                if sumcnt == 0:
                    logger.info(" =>Go to Stage3")
                    break

                # Re-Allocate 2D Array (reject flag 1 and 2)
                ndata, dt, d, t, y, ysdev, ijdata, ijmap = ijmapping(
                    pl_avgday, rawt, rawd, rawy, flag, asarray=True
                )

                # END of fourier iteration

            # *** Stage 2 : END ***

            # * * * Stage 3 : Apply the butterworth filter (hi and low) to the remained deviation * * *  !

            # Smooth spline
            spline = smooth(ndata, t, de_hm, rm)
            t_sp = spline(t)
            x_sp = spline(x_t)

            de_sp = y - t_sp
            de_hmsp = y - t_sp - t_hm

            # *** Stage 3-1 : interpolation and extrapolation for bwf *** !

            spt = float(self.extrap_days) / self.days_in_year / dt

            # extrapolate left side
            lpad_pdata = int((x_t[tstart] - x_t[0]) / dt - spt) - 1
            # extrapolate right side
            rpad_pdata = int((x_t[xdata - 1] - x_t[tend]) / dt - spt) - 1

            # interpolate large interval ( >= half year)
            gap_t = t[1:ndata] - t[: ndata - 1]
            gap_pdata = np.zeros(ndata, dtype=np.int64)
            gap_pdata[:-1] = np.where(gap_t >= 0.5, (gap_t / dt).astype(np.int64) - 1, 0)

            # each data point followed by the interpolated points
            nblock = gap_pdata + 1
            pdata = lpad_pdata + int(nblock.sum()) + rpad_pdata

            p = np.zeros(pdata, dtype=np.float64)
            q = np.zeros_like(p)
            r = np.zeros_like(p, dtype=np.int64)

            # left side (ideal, Q = 0)
            i = np.arange(1, lpad_pdata + 1, dtype=np.float64)
            p[:lpad_pdata] = x_t[0] + dt * i

            # data (real, Q = Y - sp - hm) and large intervals (ideal, Q = 0)
            j = np.repeat(np.arange(ndata), nblock)
            k = np.arange(len(j)) - np.repeat(np.cumsum(nblock) - nblock, nblock)
            real = lpad_pdata + np.flatnonzero(k == 0)
            ideal = lpad_pdata + np.flatnonzero(k > 0)
            p[real] = t[:ndata]
            q[real] = de_hmsp[:ndata]
            r[real] = 1
            p[ideal] = t[j[k > 0]] + dt * k[k > 0].astype(np.float64)

            # right side (ideal, Q = 0)
            i = np.arange(1, rpad_pdata + 1, dtype=np.float64)
            p[pdata - rpad_pdata :] = x_t[tend] + dt * i + dt * spt

            # *** Stage 3-2 : Set coefficients of butterworth filter *** !

            fs = self.days_in_year

            # long term
            cutlo = float(pl_cutlo)
            # a1, b1, c1 = bwfcoef(cutlo, pl_orderlo)
            bwf_sos1 = signal.butter(
                pl_orderlo, 12.0 / cutlo, btype="low", output="sos", fs=fs
            )

            # Short term
            cuthi = 12.0 / (float(nh / 2) + 1.0)
            # a2, b2, c2 = bwfcoef(cuthi, pl_orderhi)
            bwf_sos2 = signal.butter(
                pl_orderhi, 12.0 / cuthi, btype="low", output="sos", fs=fs
            )

            filt_lo = FiltFilt(bwf_sos1)
            filt_hi = FiltFilt(bwf_sos2)

            logger.info("")
            logger.info(" Low path filter result")
            logger.info(" +Cutoff month        : %3d", int(cutlo))
            logger.info(" +Order               : %3d", pl_orderlo)
            logger.info(" High path filter result")
            logger.info(" +Cutoff month        : %3d", int(cuthi))
            logger.info(" +Order               : %3d", pl_orderhi)

            # *** Stage 3-3 : Apply butterworth filter *** !
            nh = nh + 2
            nterm = nh
            sgm_temp = sgm_hmsp

            x_q = np.zeros(xdata, dtype=np.float64)

            # first grid index with x_t > p - day_interval (for Q -> X mapping)
            pfirst = np.searchsorted(x_t, p[:pdata] - self.day_interval, side="right")

            # grid points filled by each segment (p[i-1], p[i]) of Q wave
            seg = np.full(xdata, -1, dtype=np.int64)
            searchi = 0
            for i in range(1, pdata):
                if searchi >= xdata:
                    break
                # fill up to the first grid point within day_interval
                # of p[i] (to the end of the grid if not found)
                j = max(int(pfirst[i]), searchi)
                if j >= xdata or not abs(x_t[j] - p[i]) < self.day_interval:
                    j = xdata - 1
                seg[searchi : j + 1] = i
                searchi = j + 1
            xfill = np.flatnonzero(seg >= 0)
            seg = seg[xfill]

            for ite_i in range(0, 2):
                if ite_i == 0:
                    logger.info("")
                    logger.info(" +Low path filter")

                if ite_i == 1:
                    logger.info("")
                    logger.info(" +High path filter")

                judge = 0
                ite_j = 0 - 1

                while TRUE == 1:
                    ite_j += 1
                    logger.info("  +iteration = %2d - %2d", ite_i + 1, ite_j + 1)

                    # Map Q wave to X wave (linear interpolation)
                    aa = (q[1:pdata] - q[: pdata - 1]) / (p[1:pdata] - p[: pdata - 1])
                    bb = q[1:pdata] - aa * p[1:pdata]
                    x_q[xfill] = aa[seg - 1] * x_t[xfill] + bb[seg - 1]

                    # Low path filter
                    if ite_i == 0:
                        x_lo = filt_lo(x_q)

                        t_lo = matching(ndata, t, xdata, x_t, x_lo, self.day_interval)
                        q_lo = matching(pdata, p, xdata, x_t, x_lo, self.day_interval)
                        de_splo = de_sp - t_lo

                        # Smooth seasonal cycle with high path filter
                        hm_coef = frcoef(ndata, t, de_splo, nh, nterm)

                        x_hm = harmonic(x_t, hm_coef, nh)

                        x_hm_hi = filt_hi(x_hm)

                        trim_x_t = np.zeros(xdata, dtype=np.float64)
                        trim_x_hm_hi = np.zeros(xdata, dtype=np.float64)
                        j = 0 - 1

                        ntrim = len(range(tstart, tend + 1, 3))
                        trim_x_t[:ntrim] = x_t[tstart : tend + 1 : 3]
                        trim_x_hm_hi[:ntrim] = x_hm_hi[tstart : tend + 1 : 3]

                        hm_coef = frcoef(ntrim, trim_x_t, trim_x_hm_hi, nh, nterm)

                        t_hm = harmonic(t, hm_coef, nh)

                        # Sum deviation (y -sp -hm(new) -longterm)
                        de_sphm = de_sp - t_hm
                        de_sphmlo = de_splo - t_hm
                        sgm_sphmlo = np.std(de_sphmlo)

                        logger.info("   +Sigma(Y -sp -hm -longterm) = %12.7E", sgm_sphmlo)

                        if sgm_temp != 0 and 1.0 - sgm_sphmlo / sgm_temp <= 0.0001:
                            judge = 1

                        # Replace Q wave by Real wave (new Y-sp-hm)
                        q_de_sphm = matching(pdata, p, ndata, t, de_sphm, self.day_interval)

                        m = r[:pdata] == 1
                        q[:pdata][m] = q_de_sphm[:pdata][m]

                        sgm_temp = sgm_sphmlo

                        if judge == 1 or ite_j == 9:
                            # Remove longterm trend from Q wave (Real+Ideal)
                            q[:pdata] -= q_lo[:pdata]
                            break

                        # Replace Q wave by Ideal longterm trend
                        m = r[:pdata] == 0
                        q[:pdata][m] = q_lo[:pdata][m]

                    # High path filter
                    if ite_i == 1:
                        x_hi = filt_hi(x_q)

                        t_hi = matching(ndata, t, xdata, x_t, x_hi, self.day_interval)
                        q_hi = matching(pdata, p, xdata, x_t, x_hi, self.day_interval)

                        # Sum deviation (y -sp -hm -longterm -shortterm)
                        de_sphmlohi = de_sphmlo - t_hi
                        sgm_sphmlohi = np.std(de_sphmlohi)

                        logger.info(
                            "  +Sigma(Y -sp -hm -long -shortterm) = %12.7E", sgm_sphmlohi
                        )

                        if sgm_temp != 0 and 1.0 - sgm_sphmlohi / sgm_temp <= 0.0001:
                            judge = 1

                        sgm_temp = sgm_sphmlohi

                        if judge == 1 or ite_j == 9:
                            break

                        # Replace Q wave by Ideal shortterm trend
                        m = r[:pdata] == 0
                        q[:pdata][m] = q_hi[:pdata][m]

                    # end if

            x_hm = harmonic(x_t, hm_coef, nh)

            logger.info("")
            logger.info("  +COEFFICIENTS A(K)=")
            for i in range(0, nh):
                logger.info("    %s %10.4E", label[i], hm_coef[i])

            # Result
            x_tr = x_sp + x_lo
            x_ss = x_hm + x_hi
            x_tt = x_tr + x_ss

            # *** Stage 3-4 : Judge the flags and re-allocate 2D data ***

            # if value is out of sigma*pl_sigma, set to 2
            sgm_allow = sgm_sphmlohi * pl_sigma
            sumcnt = 0

            if sw_iteration == 1:
                # Judge by using averaged
                ij, i = flatten_ijmap(ijmap, ijdata)
                sumcnt += reviseflags(flag, ij, de_sphmlohi[i], sgm_allow)

                logger.info("")
                logger.info("  =>Reject data (sigma*%2d): %5d", pl_sigma, sumcnt)

                reportrej(rawndata, rawymd, rawy, flag, 2)

                if sumcnt == 0:
                    break

                # Re-Allocate 2D Array (reject flag 1 and 2)
                ndata, dt, d, t, y, ysdev, ijdata, ijmap = ijmapping(
                    pl_avgday, rawt, rawd, rawy, flag, asarray=True
                )

                logger.info("")
                logger.info("  =====> BACK TO STAGE 2 =====> ")

        logger.info("")
        logger.info("* * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        logger.info("* * *                 END FITTING                 * * *")
        logger.info("* * * * * * * * * * * * * * * * * * * * * * * * * * * *")

        # *** Stage 3 : END ***

        # * * *      Stage 4 : Output Files       * * *  !

        t_tt = matching(rawndata, rawt, xdata, x_t, x_tt, self.day_interval)
        t_tr = matching(rawndata, rawt, xdata, x_t, x_tr, self.day_interval)
        t_ss = matching(rawndata, rawt, xdata, x_t, x_ss, self.day_interval)

        if sw_iteration == 0:
            dev       = rawy - t_tt
            sgm_all   = np.std(dev[flag == 0])
            sgm_allow = sgm_all * pl_sigma

            logger.info("")
            logger.info("  +Sigma(Y -total) = %12.7E", sgm_all)

            ij = np.arange(rawndata)
            sumcnt += reviseflags(flag, ij, dev[:rawndata], sgm_allow)

            logger.info("")
            logger.info("  =>Reject data (sigma*%2d): %5d", pl_sigma, sumcnt)

            reportrej(rawndata, rawymd, rawy, flag, 2)
            logger.info("")

        self.syear = syear
        self.trange = (tstart, tend + 1)
        self.nh = nh
        self.hm_coef = hm_coef

        # raw_data : equal to raw data (no average in same day)
        self.raw_data = pd.DataFrame(
            {
                "d": rawd,
                "t": rawt,
                "qf": flag,
                "value": rawy,
                "smooth": t_tt,
                "trend": t_tr,
                "seasonal": t_ss,
                "detrended": rawy - t_tr,
                "deseasonal": rawy - t_ss,
                "deviation": rawy - t_tt,
                "date": pd.to_datetime(data["date"]),
            }
        )

        # eff_data : averaged effective raw data (flag 0)
        s_index = np.array([ijmap[i][0] for i in range(ndata)], dtype=np.int64)
        e_index = np.array([ijmap[i][-1] for i in range(ndata)], dtype=np.int64)
        cday0 = np.datetime64("1899-12-30", "D")  # see cday2date
        self.eff_data = pd.DataFrame(
            {
                "d": d,
                "t": t,
                "value": y,
                "sd": ysdev,
                "detrended": y - t_tr[s_index],
                "deseasonal": y - t_ss[s_index],
                "deviation": y - t_tt[s_index],
                "sdate": cday0 + rawd[s_index],
                "edate": cday0 + rawd[e_index],
            }
        )

        # fit_result : fitting result
        def dtrdtt(tr, t):
            a = np.array([NaN] * len(tr))
            a[1:-1] = (tr[2:] - tr[:-2]) / (t[2:] - t[:-2])
            return a

        self.fit_result = pd.DataFrame(
            {
                "d": x_d,
                "t": x_t,
                "smooth": x_tt,
                "trend": x_tr,
                "seasonal": x_ss,
                "harmonics": x_hm,
                "shortterm": x_hi,
                "growthrate": dtrdtt(x_tr, x_t),
                "date": np.datetime64("1899-12-30", "D") + x_d,
            },
        )

        # input data for bootstrap(); the results of the last bootstrap
        # are for the previous data
        self.input_data = data
        self.boot_result = None
        self.boot_smooth = None
        self.boot_trend = None
        self.boot_growthrate = None
        self.boot_info = None

        return self.fit_result

    def bootstrap(
        self,
        nboot: int = 1000,
        exclude_frac: float = 0.5,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Uncertainty of the fitting result by bootstrap

        The data are thinned out at random and fitted again, nboot times.
        The unit of thinning is the "Range of day" group (the day with
        avgday=1) of the effective data. In each member, a fraction
        exclude_frac of the units (round(exclude_frac * number of units))
        are excluded (flag -1), and fit() is run with the same parameters,
        including the outlier rejection. The smooth curve, the trend and
        the growth rate of each member are taken in the range where fit()
        gives its result (the data period +- extrap_days).

        The standard deviation over the members at each day, multiplied
        by the delete-d jackknife factor sqrt((1-f)/f) (Shao and Wu, 1989),
        is the uncertainty (stdev) of the fitting result with all data.
        The factor makes it nearly independent of f, and is 1 at f=0.5.
        f is the actual fraction of the excluded units (e.g. 0.4996 for
        exclude_frac=0.5 and 1265 units).

        Run fit() first. The results are in boot_all() and boot_annual().

        Args:
            nboot (int): number of members (default: 1000)
            exclude_frac (float): fraction of the excluded units,
                0 < exclude_frac < 1 (default: 0.5)
            seed (int): random seed (default: 0). If None, 128-bit entropy
                is drawn from the OS. The seed is logged and kept in
                boot_info.

        Returns:
            pd.DataFrame: stdev at each day of the result range
                - d: day (see date2cday)
                - stdev_value, stdev_trend, stdev_growthrate: stdev of
                  smooth, trend and growthrate (corrected by the factor)
                - ensmean_trend: mean of the trend over the members
                - nboot: number of members covering the day

        The smooth curve, the trend and the growth rate of each member are
        kept in boot_smooth, boot_trend and boot_growthrate (nboot x days,
        NaN out of the range).
        """

        if self.fit_result is None:
            raise ValueError("No data. Run fit() first.")
        if nboot < 2:
            raise ValueError(f"nboot must be 2 or more: {nboot}")

        loglevel = logger.level
        start = time.perf_counter()

        # units: "Range of day" groups of the effective data, as ijmapping
        data = self.input_data
        pre = self.preprocess(data)
        qf = pre["qf"].to_numpy(copy=True)
        rawd = pre["d"].to_numpy()
        valid = np.flatnonzero(qf == 0)
        d_group = (rawd - rawd.min()) // self.avgday
        _, unit = np.unique(d_group[valid], return_inverse=True)
        unit = unit.ravel()
        nunit = int(unit.max()) + 1

        exclude_frac = check_exclude_frac(exclude_frac, nunit)
        rng, entropy = make_rng(seed)

        logger.info("")
        logger.info(" * * * * * Start Bootstrap * * * * *")
        logger.info(" Number of members           : %8d", nboot)
        logger.info(" Number of data units        : %8d", nunit)
        logger.info(" Fraction of excluded units  : %8s", exclude_frac)
        logger.info(" Random generator            : PCG64DXSM")
        logger.info(" Seed                        : %s", entropy)

        # result range of the fit with all data
        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        win_d = self.fit_result["d"].to_numpy()[tslice]
        nwin = len(win_d)

        a_sm = np.full((nboot, nwin), np.nan)
        a_tr = np.full((nboot, nwin), np.nan)
        a_gr = np.full((nboot, nwin), np.nan)
        a_frac = np.zeros(nboot)

        member_data = data.copy()
        excluded = np.zeros(nunit, dtype=bool)
        nreport = max(nboot // 10, 1)

        try:
            for k in range(nboot):
                excl = exclude_units(rng, nunit, exclude_frac)
                a_frac[k] = len(excl) / nunit

                excluded[:] = False
                excluded[excl] = True
                member_qf = qf.copy()
                member_qf[valid[excluded[unit]]] = -1
                member_data["qf"] = member_qf

                member = N97Fit(
                    avgday=self.avgday,
                    sigma=self.sigma,
                    nhm=self.nhm,
                    cutlo=self.cutlo,
                    orderlo=self.orderlo,
                    orderhi=self.orderhi,
                    sw_iteration=self.sw_iteration,
                    loglevel="WARNING",
                )
                member.fit(member_data)

                # result range of the member, on the days of win_d
                mslice = slice(
                    member.trange[0] - int(member.extrap_days),
                    member.trange[1] + int(member.extrap_days),
                )
                mfit = member.fit_result.iloc[mslice]
                idx = mfit["d"].to_numpy() - win_d[0]
                sel = (idx >= 0) & (idx < nwin)
                a_sm[k, idx[sel]] = mfit["smooth"].to_numpy()[sel]
                a_tr[k, idx[sel]] = mfit["trend"].to_numpy()[sel]
                a_gr[k, idx[sel]] = mfit["growthrate"].to_numpy()[sel]

                if (k + 1) % nreport == 0 or k + 1 == nboot:
                    logger.setLevel(loglevel)
                    logger.info(
                        " +Member %6d / %d  (%7.1f s)",
                        k + 1,
                        nboot,
                        time.perf_counter() - start,
                    )
        finally:
            logger.setLevel(loglevel)

        frac = float(a_frac.mean())
        scale = float(jackknife_scale(frac))

        # days covered by no member (or one) give NaN (or 0) silently
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stdev_sm = np.nanstd(a_sm, axis=0)
            stdev_tr = np.nanstd(a_tr, axis=0)
            stdev_gr = np.nanstd(a_gr, axis=0)
            ensmean_tr = np.nanmean(a_tr, axis=0)

        self.boot_smooth = a_sm
        self.boot_trend = a_tr
        self.boot_growthrate = a_gr
        self.boot_info = {
            "nboot": nboot,
            "exclude_frac": exclude_frac,
            "frac": frac,
            "scale": scale,
            "nunit": nunit,
            "seed": entropy,
        }
        self.boot_result = pd.DataFrame(
            {
                "d": win_d,
                "stdev_value": stdev_sm * scale,
                "stdev_trend": stdev_tr * scale,
                "stdev_growthrate": stdev_gr * scale,
                "ensmean_trend": ensmean_tr,
                "nboot": np.sum(~np.isnan(a_tr), axis=0),
            }
        )

        logger.info("")
        logger.info(" Mean fraction of excluded   : %8.4f", frac)
        logger.info(" Stdev factor sqrt((1-f)/f)  : %8.4f", scale)
        logger.info(" * * * * * End Bootstrap * * * * *")

        return self.boot_result

    def preprocess(
        self, data: pd.DataFrame, inplace: bool = False, asarray: bool = False
    ) -> pd.DataFrame:
        if not inplace:
            data = data.copy()

        if "date" not in data.columns:
            if "year" in data.columns and "month" in data.columns and "day" in data.columns:
                data["date"] = pd.to_datetime(data[["year", "month", "day"]])
            else:
                raise ValueError("date column is not found")

        syear = data.iloc[0]["date"].year
        sdate = date(syear, 1, 1)

        # convert date to YYYYMMDD
        dts = data["date"].dt
        data["ymd"] = (dts.year * 10000 + dts.month * 100 + dts.day).astype(int)

        # convert date to continuous day (serial day of MS Excel, see date2cday)
        data["d"] = (
            data["date"].values.astype("datetime64[D]") - np.datetime64("1899-12-30", "D")
        ).astype(np.int64)

        # convert date to continuous year
        data["t"] = (
            (data["d"] - date2cday(sdate)).astype(np.float64) + 0.5
        ) / float(self.days_in_year)

        # convert date to continuous year (exact calculation)
#         data["t"] = data["date"].apply(
#             lambda x: float( ( date(x.year,x.month,x.day) - date(x.year,1,1) ).days + 0.5 ) # assume 12:00
#                     / float( ( date(x.year+1,1,1)         - date(x.year,1,1) ).days )
#                     + float( x.year - syear )
#         )

        # stop if reverse in time is found
        dt = data["t"].diff()
        if (dt < 0).any():
            revserse = data["ymd"][dt < 0].values
            raise ValueError(f"reverse in time is found. {revserse}")

        # check flag and set -1 if value is invalid(<=-999)
        data.loc[data["value"] <= -999.0, "qf"] = -1

        logger.info(" +Number of data     : %5d", len(data))

        if asarray:
            return (
                len(data),
                data["t"].values,
                data["d"].values,
                data["ymd"].values,
                data["value"].values,
                data["qf"].to_numpy(copy=True),
                syear,
            )
        else:
            return data

    def data_all(self, use_excel_date: bool = False):
        """raw data (no average in same day)
        Args:
            use_excel_date (bool, optional): If True, date is converted to continuous day.
        Returns:
            pd.DataFrame: raw data
        """

        if self.raw_data is None:
            raise ValueError("No data. Run fit() first.")

        df = pd.DataFrame(
            {
                "year": self.raw_data["t"] + self.syear,
                "value": self.raw_data["value"],
                "flg": self.raw_data["qf"],
                "detrended": self.raw_data["detrended"],
                "deseasonal": self.raw_data["deseasonal"],
                "deviation": self.raw_data["deviation"],
                "date": self.raw_data["date"],
            }
        )

        if use_excel_date:
            df["date"] = df["date"].apply(lambda x: date2cday(x))

        # NaN for flg < 0
        df.loc[df["flg"] < 0, ["detrended", "deseasonal", "deviation"]] = NaN

        return df

    def data_eff_ave(self, use_excel_date: bool = False):
        """averaged effective raw data (flag 0)
        Args:
            use_excel_date (bool, optional): If True, date is converted to continuous day.
        Returns:
            pd.DataFrame: averaged effective raw data
        """
        if self.eff_data is None:
            raise ValueError("No data. Run fit() first.")

        df = pd.DataFrame(
            {
                "year": self.eff_data["t"] + self.syear,
                "value": self.eff_data["value"],
                "detrended": self.eff_data["detrended"],
                "deseasonal": self.eff_data["deseasonal"],
                "deviation": self.eff_data["deviation"],
                "date": self.eff_data["sdate"],
            }
        )

        if use_excel_date:
            df["date"] = df["date"].apply(lambda x: date2cday(x))

        return df

    def data_eq_ave(self, use_excel_date: bool = False):
        """averaged effective raw data (flag 0) with all of interval

        Args:
            use_excel_date (bool, optional): If True, date is converted to continuous day.
        Returns:
            pd.DataFrame: averaged effective raw data with all of interval
        """

        if self.eff_data is None or self.fit_result is None:
            raise ValueError("No data. Run fit() first.")

        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )

        fit_all = self.fit_result.iloc[tslice, :].copy()
        fit_all["year"] = fit_all["t"] + self.syear

        df = pd.merge(
            fit_all[["d", "year", "date"]],
            self.eff_data[["d", "value", "detrended", "deseasonal", "deviation"]],
            how="left",
            on="d",
        )

        if use_excel_date:
            df["date"] = df["date"].apply(lambda x: date2cday(x))

        df = df[["year", "value", "detrended", "deseasonal", "deviation", "date"]]
        return df

    def data_rej(self, use_excel_date: bool = False):
        """rejected data
        Args:
            use_excel_date (bool, optional): If True, date is converted to continuous day.
        Returns:
            pd.DataFrame: rejected data
        """

        if self.raw_data is None:
            raise ValueError("No data. Run fit() first.")

        # rejected data (flag > 0)
        rej_data = self.raw_data[self.raw_data["qf"] > 0].copy()

        df = pd.DataFrame(
            {
                "year": rej_data["t"] + self.syear,
                "value": rej_data["value"],
                "flg": rej_data["qf"],
                "detrended": rej_data["detrended"],
                "deseasonal": rej_data["deseasonal"],
                "deviation": rej_data["deviation"],
                "date": rej_data["date"],
            }
        )

        if use_excel_date:
            df["date"] = df["date"].apply(lambda x: date2cday(x))

        return df

    def all(self, interval: int = 1, use_excel_date: bool = False) -> pd.DataFrame:
        """All of interval

        Args:
            interval (int, optional): interval of data. Defaults to 1.
            use_excel_date (bool, optional): If True, date is converted to continuous day.
        Returns:
            pd.DataFrame: all of interval
        """

        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
            interval,
        )
        fit_all = self.fit_result.iloc[tslice, :]

        df = pd.DataFrame(
            {
                "year": fit_all["t"] + self.syear,
                "value": fit_all["smooth"],
                "trend": fit_all["trend"],
                "seasonal": fit_all["seasonal"],
                "harmonics": fit_all["harmonics"],
                "shortterm": fit_all["shortterm"],
                "date": fit_all["date"],
                "growthrate": fit_all["growthrate"],
            }
        )

        if use_excel_date:
            df["date"] = df["date"].apply(lambda x: date2cday(x))

        return df

    def annual(self) -> pd.DataFrame:
        """Annual average

        Returns:
            pd.DataFrame: annual average
        """

        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        fit_all = self.fit_result.iloc[tslice, :]

        yr_group = fit_all.groupby(fit_all["date"].dt.year)
        ys = np.array([*yr_group.groups.keys()])
        yr_mean = groupmean(fit_all, fit_all["date"].dt.year, ["smooth", "trend"])

        df = pd.DataFrame(
            {
                "year": ys,
                "ave_fit": yr_mean["smooth"],
                "stdev_fit": yr_group["smooth"].std(ddof=0),
                "ave_trend": yr_mean["trend"],
                "stdev_trend": yr_group["trend"].std(ddof=0),
            }
        )
        return df

    def monthly(self) -> pd.DataFrame:
        """Monthly average

        Returns:
            pd.DataFrame: monthly average
        """

        # first: get start and end from sliced data
        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        fit_all = self.fit_result.iloc[tslice, :]
        dstart = fit_all["d"].min()
        dend = fit_all["d"].max()
        ystart = fit_all["date"].min().year
        yend = fit_all["date"].max().year

        # second: get monthly average between start year and end year
        fit_all = self.fit_result.copy()
        fit_all["year"] = fit_all["date"].apply(lambda x: x.year)
        fit_all["mn"] = fit_all["date"].apply(lambda x: x.month)
        fit_all = fit_all[fit_all["year"].between(ystart, yend)]

        # set NaN for out of range
        fit_all.loc[fit_all["d"] < dstart, ["smooth", "seasonal"]] = NaN
        fit_all.loc[fit_all["d"] > dend, ["smooth", "seasonal"]] = NaN

        ym_group = fit_all.groupby(["year", "mn"])
        ym_mean = groupmean(fit_all, ["year", "mn"], ["smooth", "seasonal"])

        ys = np.array([y for (y, m) in ym_group.groups.keys()])
        mn = np.array([m for (y, m) in ym_group.groups.keys()])

        df = pd.DataFrame(
            {
                "year": ys,
                "mn": mn,
                "ave_fit": ym_mean["smooth"],
                "stdev_fit": ym_group["smooth"].std(ddof=0),
                "ave_seasonal": ym_mean["seasonal"],
                "stdev_seasonal": ym_group["seasonal"].std(ddof=0),
            },
        )

        return df

    def multimonth(self) -> pd.DataFrame:
        """Multi-month average

        Returns:
            pd.DataFrame: multi-month average
        """

        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        fit_all = self.fit_result.iloc[tslice, :].copy()

        fit_all["mn"] = fit_all["date"].apply(lambda x: x.month)

        mn_group = fit_all.groupby("mn")
        mn_mean = groupmean(fit_all, "mn", ["smooth", "seasonal"])
        eyear = fit_all["date"].max().year

        df = pd.DataFrame(
            {
                "period": f"{self.syear}#{eyear}",
                "mn": mn_group.groups.keys(),
                "ave_fit": mn_mean["smooth"],
                "stdev_fit": mn_group["smooth"].std(ddof=0),
                "ave_seasonal": mn_mean["seasonal"],
                "stdev_seasonal": mn_group["seasonal"].std(ddof=0),
            },
        )
        df["period"] = df["period"].convert_dtypes()

        return df

    def period(self) -> pd.DataFrame:
        """Period average

        Returns:
            pd.DataFrame: period average
        """

        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        fit_all = self.fit_result.iloc[tslice, :]

        eyear = fit_all["date"].max().year

        df = pd.DataFrame(
            {
                "period": f"{self.syear}#{eyear}",
                "ave_fit": fit_all["smooth"].mean(),
                "stdev_fit": fit_all["smooth"].std(ddof=0),
                "ave_trend": fit_all["trend"].mean(),
                "stdev_trend": fit_all["trend"].std(ddof=0),
            },
            index=[0],
        )
        df["period"] = df["period"].convert_dtypes()

        return df

    def onemonth(self) -> pd.DataFrame:
        """One-month harmonic average

        Returns:
            pd.DataFrame: one-month average
        """
        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        fit_all = self.fit_result.iloc[tslice, :]

        hm_range = range(int(self.days_in_year) + 1)

        hm_t = np.array([float(i) / self.days_in_year for i in hm_range])
        hm_hm = harmonic(hm_t, self.hm_coef, self.nh)

        hm_t = np.append(hm_t, 1.0)
        hm_hm = np.append(hm_hm, harmonic(0.0, self.hm_coef, self.nh))

        tt_avgall = fit_all["smooth"].mean()
        tr_avgall = fit_all["trend"].mean()

        df = pd.DataFrame(
            {
                "norm_time": hm_t,
                "harmonics": hm_hm,
                "hm+ave_fit": hm_hm + tt_avgall,
                "hm+ave_trend": hm_hm + tr_avgall,
            }
        )

        return df

    def boot_all(self, interval: int = 1, use_excel_date: bool = False) -> pd.DataFrame:
        """Fitting result with the uncertainty by bootstrap

        Same range and interval as all().

        Args:
            interval (int, optional): interval of data. Defaults to 1.
            use_excel_date (bool, optional): If True, date is converted to continuous day.
        Returns:
            pd.DataFrame: fitting result with the uncertainty
                - value, trend: smooth curve and trend with all data
                - stdev_value, stdev_trend: stdev of them by bootstrap
                - ensmean_trend: mean of the trend over the members
                - nboot: number of members covering the day
                - growthrate: growth rate with all data (same as all())
                - stdev_growthrate: stdev of the growth rate by bootstrap
                  (computed from the growth rate of each member)
        """

        if self.boot_result is None:
            raise ValueError("No bootstrap result. Run bootstrap() first.")

        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        fit_all = self.fit_result.iloc[tslice, :]
        boot = self.boot_result

        df = pd.DataFrame(
            {
                "year": fit_all["t"].to_numpy() + self.syear,
                "value": fit_all["smooth"].to_numpy(),
                "trend": fit_all["trend"].to_numpy(),
                "stdev_value": boot["stdev_value"].to_numpy(),
                "stdev_trend": boot["stdev_trend"].to_numpy(),
                "ensmean_trend": boot["ensmean_trend"].to_numpy(),
                "nboot": boot["nboot"].to_numpy(),
                "date": fit_all["date"].to_numpy(),
                "growthrate": fit_all["growthrate"].to_numpy(),
                "stdev_growthrate": boot["stdev_growthrate"].to_numpy(),
            }
        ).iloc[::interval]

        if use_excel_date:
            df["date"] = df["date"].apply(lambda x: date2cday(x))

        return df

    def boot_annual(self) -> pd.DataFrame:
        """Annual values of the trend with the uncertainty by bootstrap

        Only the years fully in the range of all() (from Jan 1 to Jan 1 of
        the next year) are given.
            - ave_trend: mean of the trend in the year (same as annual())
            - stdev_trend: mean of stdev_trend of boot_all() in the year
            - ave_gr: trend(Jan 1 of the next year) - trend(Jan 1)
            - stdev_gr: sqrt(stdev_trend(Jan 1)**2
                             + stdev_trend(Jan 1 of the next year)**2)

        Returns:
            pd.DataFrame: annual values
        """

        if self.boot_result is None:
            raise ValueError("No bootstrap result. Run bootstrap() first.")

        tslice = slice(
            self.trange[0] - int(self.extrap_days),
            self.trange[1] + int(self.extrap_days),
        )
        fit_all = self.fit_result.iloc[tslice, :]
        days = fit_all["date"].to_numpy().astype("datetime64[D]")
        trend = fit_all["trend"].to_numpy()
        stdev = self.boot_result["stdev_trend"].to_numpy()

        def index(y, md):
            return int((np.datetime64(f"{y:04d}-{md}") - days[0]).astype(np.int64))

        rows = []
        for y in range(days[0].astype(object).year, days[-1].astype(object).year):
            i0 = index(y, "01-01")
            i1 = index(y + 1, "01-01")
            if i0 < 0 or i1 >= len(days):
                continue
            ave = group_mean(trend[i0:i1], np.zeros(i1 - i0, dtype=np.int64), 1)[0]
            rows.append(
                [
                    y,
                    ave,
                    np.mean(stdev[i0:i1]),
                    trend[i1] - trend[i0],
                    np.hypot(stdev[i0], stdev[i1]),
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "year",
                "ave_trend",
                "stdev_trend",
                "ave_gr",
                "stdev_gr",
            ],
        )
        return df
