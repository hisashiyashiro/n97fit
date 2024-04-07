from typing import Union
from datetime import date
from math import nan as NaN

import numpy as np
import pandas as pd
from scipy import signal

from .date import ijmapping, date2cday, cday2date
from .io import logger, reportrej
from .misc import reviseflag, matching
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
    raw_data: pd.DataFrame
    eff_data: pd.DataFrame
    fit_result: pd.DataFrame

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
        sumcnt = 0

        for i in range(0, ndata):
            if ysdev[i] != 0:
                sgm_allow = ysdev[i] * pl_sigma
                for j in range(0, ijdata[i]):
                    ij = ijmap[i][j]

                    flag[ij], cnt = reviseflag(rawy[ij] - y[i], flag[ij], sgm_allow)
                    sumcnt += cnt

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
        for i in range(0, xdata):
            x_d[i] = i + xstart
            x_t[i] = float(x_d[i] - sdate) / self.days_in_year
            if abs(x_t[i] - t[0]) <= self.day_interval:
                tstart = i
            if abs(x_t[i] - t[ndata - 1]) <= self.day_interval:
                tend = i

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

        temp_cnt = None
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

                t_ln = np.zeros(ndata, dtype=np.float64)
                t_hm = np.zeros(ndata, dtype=np.float64)

                for i in range(0, ndata):
                    t_ln[i] = linea * t[i] + lineb
                    t_hm[i] = harmonic(t[i], hm_coef, nh)

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
                    t_hm = np.zeros(ndata, dtype=np.float64)

                    hm_coef = frcoef(ndata, t, de_sp, nh, nterm)
                    for i in range(0, ndata):
                        t_hm[i] = harmonic(t[i], hm_coef, nh)

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
                sumcnt = 0

                for i in range(0, ndata):
                    for j in range(0, ijdata[i]):
                        ij = ijmap[i][j]

                        flag[ij], cnt = reviseflag(de_hmsp[i], flag[ij], sgm_allow)
                        sumcnt += cnt

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
            pdata = -1

            # extrapolate left side
            lpad_pdata = int((x_t[tstart] - x_t[0]) / dt - spt) - 1
            # extrapolate right side
            rpad_pdata = int((x_t[xdata - 1] - x_t[tend]) / dt + spt) - 1

            p = np.zeros(ndata + lpad_pdata + rpad_pdata, dtype=np.float64)
            q = np.zeros_like(p)
            r = np.zeros_like(p, dtype=np.int64)

            for i in range(0, lpad_pdata):
                pdata += 1
                p[pdata] = x_t[0] + dt * float(i + 1)
                q[pdata] = 0.0
                r[pdata] = 0

            for j in range(0, ndata):
                pdata += 1
                p[pdata] = t[j]
                q[pdata] = de_hmsp[j]
                r[pdata] = 1

                # interpolate large interval ( >= half year)
                if j < ndata - 1 and t[j + 1] - t[j] >= (self.days_in_year / 2.0):
                    temp_pdata = int((t[j + 1] - t[j]) / dt) - 1

                    for i in range(0, temp_pdata):
                        pdata += 1
                        p[pdata] = t[j] + dt * float(i + 1)
                        q[pdata] = 0.0
                        r[pdata] = 0

            for i in range(0, rpad_pdata):
                pdata += 1
                p[pdata] = x_t[tend] + dt * float(i + 1) + dt * spt
                q[pdata] = 0.0
                r[pdata] = 0

            pdata += 1

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

                    # Map Q wave to X wave
                    searchi = 0

                    for i in range(1, pdata):
                        aa = (q[i] - q[i - 1]) / (p[i] - p[i - 1])
                        bb = q[i] - aa * p[i]

                        for j in range(searchi, xdata):
                            x_q[j] = aa * x_t[j] + bb
                            if abs(x_t[j] - p[i]) < self.day_interval:
                                break
                        searchi = j + 1

                    # Low path filter
                    if ite_i == 0:
                        x_lo = signal.sosfiltfilt(bwf_sos1, x_q, padtype=None)

                        t_lo = matching(ndata, t, xdata, x_t, x_lo, self.day_interval)
                        q_lo = matching(pdata, p, xdata, x_t, x_lo, self.day_interval)
                        de_splo = de_sp - t_lo

                        # Smooth seasonal cycle with high path filter
                        hm_coef = frcoef(ndata, t, de_splo, nh, nterm)

                        x_hm = np.zeros(xdata, dtype=np.float64)

                        for i in range(0, xdata):
                            x_hm[i] = harmonic(x_t[i], hm_coef, nh)

                        x_hm_hi = signal.sosfiltfilt(bwf_sos2, x_hm, padtype=None)

                        trim_x_t = np.zeros(xdata, dtype=np.float64)
                        trim_x_hm_hi = np.zeros(xdata, dtype=np.float64)
                        j = 0 - 1

                        for i in range(tstart, tend + 1, 3):
                            j += 1
                            trim_x_t[j] = x_t[i]
                            trim_x_hm_hi[j] = x_hm_hi[i]

                        hm_coef = frcoef(ndata, trim_x_t, trim_x_hm_hi, nh, nterm)

                        for i in range(0, ndata):
                            t_hm[i] = harmonic(t[i], hm_coef, nh)

                        # Sum deviation (y -sp -hm(new) -longterm)
                        de_sphm = de_sp - t_hm
                        de_sphmlo = de_splo - t_hm
                        sgm_sphmlo = np.std(de_sphmlo)

                        logger.info("   +Sigma(Y -sp -hm -longterm) = %12.7E", sgm_sphmlo)

                        if sgm_temp != 0 and 1.0 - sgm_sphmlo / sgm_temp <= 0.0001:
                            judge = 1

                        # Replace Q wave by Real wave (new Y-sp-hm)
                        q_de_sphm = matching(pdata, p, ndata, t, de_sphm, self.day_interval)

                        for i in range(0, pdata):
                            if r[i] == 1:
                                q[i] = q_de_sphm[i]

                        sgm_temp = sgm_sphmlo

                        if judge == 1 or ite_j == 9:
                            # Remove longterm trend from Q wave (Real+Ideal)
                            for i in range(0, pdata):
                                q[i] = q[i] - q_lo[i]
                            break

                        # Replace Q wave by Ideal longterm trend
                        for i in range(0, pdata):
                            if r[i] == 0:
                                q[i] = q_lo[i]

                    # High path filter
                    if ite_i == 1:
                        x_hi = signal.sosfiltfilt(bwf_sos2, x_q, padtype=None)

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
                        for i in range(0, pdata):
                            if r[i] == 0:
                                q[i] = q_hi[i]

                    # end if

            for i in range(0, xdata):
                x_hm[i] = harmonic(x_t[i], hm_coef, nh)

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
                for i in range(0, ndata):
                    for j in range(0, ijdata[i]):
                        ij = ijmap[i][j]

                        if flag[ij] >= 0:
                            dev = de_sphmlohi[i]  # Judge by using averaged
                            flag[ij], cnt = reviseflag(dev, flag[ij], sgm_allow)
                            sumcnt += cnt

                logger.info("")
                logger.info("  =>Reject data (sigma*%2d): %5d", pl_sigma, sumcnt)

                reportrej(rawndata, rawymd, rawy, flag, 2)

                if sumcnt == temp_cnt:
                    break

                # Re-Allocate 2D Array (reject flag 1 and 2)
                ndata, dt, d, t, y, ysdev, ijdata, ijmap = ijmapping(
                    pl_avgday, rawt, rawd, rawy, flag, asarray=True
                )

                logger.info("")
                logger.info("  =====> BACK TO STAGE 2 =====> ")
                temp_cnt = sumcnt

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
            sgm_all   = np.std(dev)
            sgm_allow = sgm_all * pl_sigma

            logger.info("")
            logger.info("  +Sigma(Y -total) = %12.7E", sgm_all)

            for ij in range(0, rawndata):
                if flag[ij] >= 0:
                    flag[ij], cnt = reviseflag(dev[ij], flag[ij], sgm_allow)
                    sumcnt += cnt

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
        s_index = np.array([ijmap[i][0] for i in range(ndata)])
        e_index = np.array([ijmap[i][-1] for i in range(ndata)])
        self.eff_data = pd.DataFrame(
            {
                "d": d,
                "t": t,
                "value": y,
                "sd": ysdev,
                "detrended": y - t_tr[s_index],
                "deseasonal": y - t_ss[s_index],
                "deviation": y - t_tt[s_index],
                "sdate": np.array(
                    [cday2date(rawd[j]) for j in s_index],
                    dtype="datetime64[D]",
                ),
                "edate": np.array(
                    [cday2date(rawd[j]) for j in e_index],
                    dtype="datetime64[D]",
                ),
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
                "date": np.array(
                    [cday2date(cd) for cd in x_d],
                    dtype="datetime64[D]",
                ),
            },
        )

        return self.fit_result

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
        data["ymd"] = (
            data["date"].apply(lambda x: x.year * 10000 + x.month * 100 + x.day).astype(int)
        )

        # convert date to continuous day
        data["d"] = data["date"].apply(
            lambda x: date2cday(x),
        )

        # convert date to continuous year
        data["t"] = data["d"].apply(
            lambda x: ( float( x - date2cday(sdate) ) + 0.5 ) / float( self.days_in_year )
        )

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
                data["qf"].values,
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

        yr_group = fit_all.groupby(fit_all["t"].astype(int))
        ys = np.array([*yr_group.groups.keys()])

        df = pd.DataFrame(
            {
                "year": ys + self.syear,
                "ave_fit": yr_group["smooth"].mean(),
                "stdev_fit": yr_group["smooth"].std(ddof=0),
                "ave_trend": yr_group["trend"].mean(),
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

        ys = np.array([y for (y, m) in ym_group.groups.keys()])
        mn = np.array([m for (y, m) in ym_group.groups.keys()])

        df = pd.DataFrame(
            {
                "year": ys,
                "mn": mn,
                "ave_fit": ym_group["smooth"].mean(),
                "stdev_fit": ym_group["smooth"].std(ddof=0),
                "ave_trend": ym_group["seasonal"].mean(),
                "stdev_trend": ym_group["seasonal"].std(ddof=0),
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

        mn_group = fit_all.groupby(["mn"])
        eyear = int(fit_all["t"].max())

        df = pd.DataFrame(
            {
                "period": f"{self.syear}#{self.syear + eyear}",
                "mn": mn_group.groups.keys(),
                "ave_fit": mn_group["smooth"].mean(),
                "stdev_fit": mn_group["smooth"].std(ddof=0),
                "ave_trend": mn_group["seasonal"].mean(),
                "stdev_trend": mn_group["seasonal"].std(ddof=0),
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

        eyear = int(fit_all["t"].max())

        df = pd.DataFrame(
            {
                "period": f"{self.syear}#{self.syear + eyear}",
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
        hm_hm = np.array([harmonic(hm_t[i], self.hm_coef, self.nh) for i in hm_range])

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
