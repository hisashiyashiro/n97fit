n97fit: Curve fitting tool for trace gas observations
======

As known as "Nakazawa-fitting"

## Reference Paper

Nakazawa, T., Ishizawa, M., Higuchi, K. And Trivett, N.B.A. (1997), Two Curve Fitting Methods Applied to CO2 Flask Data. Environmetrics, 8: 197-218. [URL]

[URL]: https://doi.org/10.1002/(SICI)1099-095X(199705)8:3<197::AID-ENV248>3.0.CO;2-C

## Authors

- Takakiyo Nakazawa - Original work

See the AUTHORS file for details

## License

This project is licensed under the BSD 3-clause License - see the LICENSE file for details

## Acknowledgments

We would like to thank Prof. Shinji Morimoto and the staff and students of the Center for Atmospheric and Oceanic Studies in Tohoku University, who contributed to the improvement of this tool.

# How to use

## Prerequisites

- python >= 3.8
- numpy >= 1.21, < 3
- scipy >= 1.10, < 2
- pandas >= 2.0, < 4

## Running as a module

    $> python -m n97fit [input_file] [config_file] [output_dir]

## Input data format

    YYYYMMDD obsID   Lat   Lon   Hgt    Value  Anl.Date QF
    19891011  2135    13   144     2   131.17  19891023 -1
    19891012  2137    15   152     2   167.82  19891023  0
    ...

    YYYY     : Observation year
    MM       : Observation month
    DD       : Observation day
    obsID    : Observation ID (e.g. Flask number)
    Lat      : Latitude
    Log      : Longtitude
    Hgt      : Altitude
    Value    : Time series of concentration, isotope ratio, etc.
    Anl.Date : Analyzing/extracting day
    QF       : Quality flag

## Outputs

- data\_all.txt       : All of the input data
- data\_eff\_ave.txt  : Input data, averaged good quality data (flag=0) by "Range of day"
- data\_eq\_ave.txt   : Input data, averaged and aligned to the output interval of fitting results
- data\_rej.txt       : Input data, outliers
- fit\_all.txt        : Fitting results
- fit\_annual.txt     : Fitting results, annual mean
- fit\_monthly.txt    : Fitting results, monthly mean
- fit\_multimonth.txt : Fitting results, multi-monthly mean
- fit\_onemonth.txt   : Fitting results, one harmonics
- fit\_period.txt     : Fitting results, mean of the whole period

## Uncertainty by bootstrap

The uncertainty of the fitting results is estimated by resampling: the
data are resampled and fitted again, many times. The standard method is
the wild bootstrap of the residuals; the delete-d jackknife can also be
selected.

    $> python -m n97fit [input_file] [config_file] [output_dir] --bootstrap

    --bootstrap       : Estimate the uncertainty by resampling
    --method M        : wild (wild bootstrap of the residuals, default)
                        or jackknife (delete-d jackknife)
    --nboot N         : Number of members (default: 1000)
    --exclude-frac F  : Jackknife only. Fraction of the data excluded in
                        each member, 0 < F < 1 (default: 0.5)
    --seed S          : Random seed (default: 0)

or in python:

    nf = n97fit.N97Fit()
    nf.fit(data)
    nf.bootstrap()      # method="wild", nboot=1000, seed=0
    nf.boot_all()       # fitting results with the uncertainty
    nf.boot_annual()    # annual values with the uncertainty

Outputs (with --bootstrap):

- boot\_all.txt    : Fitting results (smooth curve, trend and growth
                     rate) and their standard deviations (stdev)
- boot\_annual.txt : Annual mean and growth rate of the trend, and
                     their stdev

Method:

- The unit of resampling is the "Range of day" group (the day with
  avgday=1) of the data used for the fit with all data.
- Wild bootstrap of the residuals (--method wild, default):
    - The days of the data are kept, and the residuals about the smooth
      curve s are flipped at random,
      y\*\_j = s\_j + v\_i (y\_j - s\_j) / sqrt(1 - h\_i),
      for each data j of unit i. v\_i is +1 or -1 with probability 1/2,
      and h\_i is the leverage of the smooth curve at unit i.
    - The members are fitted with the same parameters, without the
      rejection of outliers. The fit is then linear in the data, and
      h\_i is the change of the smooth curve at unit i when the data of
      unit i are increased by 1. It needs as many fits as the units.
    - Dividing by sqrt(1 - h\_i) undoes the shrinkage of the residuals
      about the fitted curve (h\_i <= 0.99).
    - Each residual keeps its own size, so the seasonal change of the
      noise is kept.
    - It gives the uncertainty by the errors of the data (including the
      short-term variations) for the days of the data.
- Delete-d jackknife (--method jackknife):
    - In each member, a fraction F of the units are excluded at random,
      and the data are fitted with the same parameters, including the
      rejection of outliers.
    - The stdev over the members is multiplied by sqrt((1-f)/f) (Shao
      and Wu, 1989). The corrected stdev is nearly independent of f, and
      the factor is 1 at f=0.5. f is the actual fraction of the
      excluded units, round(F x units) / units.
    - It also includes the variation by the days of the data, and gives
      a larger stdev than the wild bootstrap (15% larger on average for
      the sample data).
    - The bootstrap() of version 1.2.0 is this method. Resampling the
      data with replacement is not a bootstrap with this fitting: the
      duplicated data are averaged in the same day, so their weight is
      lost, and it ends up excluding about 37% of the data at random.
- The smooth curve, the trend and the growth rate of each member are
  taken in the range of fit\_all.txt (the data period +- 30 days). The
  members that cover each day are counted in the column nboot (always
  the number of members for the wild bootstrap).
- boot\_all.txt:
    - value, trend, growthrate: same as fit\_all.txt
    - stdev\_value, stdev\_trend, stdev\_growthrate: stdev of them.
      stdev\_growthrate is from the growth rate of each member, because
      the errors of the trend on successive days are highly correlated.
    - ensmean\_trend: mean of the trend over the members
- boot\_annual.txt (the years fully in the range):
    - ave\_trend: mean of the trend in the year (same as fit\_annual.txt)
    - ave\_gr: trend(Jan 1 of the next year) - trend(Jan 1)
    - stdev\_trend, stdev\_gr: stdev of them, from the annual values of
      each member. The members that do not cover the whole year are
      not used.
- Notes:
    - Each unit is resampled independently, so the correlation of the
      data in time is not considered. The stdev can be underestimated
      for strongly correlated data.
    - The stdev is large near both ends of the period and where the
      data are sparse. ensmean\_trend differs from the trend with all
      data there.
    - It takes about 40 seconds for 1000 members with 1265 days of data
      (the leverage takes about half of it for the wild bootstrap).

Wu, C. F. J. (1986), Jackknife, bootstrap and other resampling methods in regression analysis. Ann. Statist., 14(4): 1261-1295.

Liu, R. Y. (1988), Bootstrap procedures under some non-i.i.d. models. Ann. Statist., 16(4): 1696-1708.

Shao, J. and Wu, C. F. J. (1989), A general theory for jackknife variance estimation. Ann. Statist., 17(3): 1176-1197.

Davison, A. C. and Hinkley, D. V. (1997), Bootstrap Methods and their Application. Cambridge University Press.

# Release Notes

## 2026-11-08

- Version 1.2.0
- New feature: uncertainty of the fitting results by bootstrap (see
  "Uncertainty by bootstrap" above).
    - N97Fit.bootstrap() thins out the data at random and fits them
      again (default: 1000 members, half of the data excluded, seed 0).
      The stdev over the members is corrected by the delete-d jackknife
      factor sqrt((1-f)/f).
    - boot\_all() and boot\_annual() give the smooth curve, the trend,
      the growth rate and the annual values with their stdev.
    - The command line writes boot\_all.txt and boot\_annual.txt with
      the option --bootstrap (--nboot, --exclude-frac and --seed).
- The results of fit() and the other output files do not change.

## 2026-11-07

- Version 1.1.1
- Support pandas 3. The supported versions are python 3.8-3.14,
  numpy 1.21-2.x, scipy 1.10-1.x and pandas 2.0-3.x.
- Speed up fit() further (about 3x faster than 1.1.0) and the writing
  of the output files (about 2x faster).
- Better precision (the results change by 1e-6 or less in the output
  files):
    - The smoothing spline is solved by the Reinsch algorithm with the
      mean subtracted. The rounding error is about 50 times smaller at
      the data points and 30 times smaller far outside the data.
    - The means of each day group are computed by Kahan summation in
      the order of the data, which pandas 3 does not use.
    - The results agree within 1e-8 among the supported versions of
      numpy, scipy and pandas. Before, the growth rate differed by up
      to 3e-3 near both ends of the 1-day grid.
- The output files have the same layout with all pandas versions.
  pandas 2.0.0-2.0.2 put an extra space before string columns.

## 2026-11-07

- Version 1.1.0
- Speed up fit() by vectorizing the main loops (about 5x faster).
  This part does not change the results.
- Bug fixes (the results change):
    - The sigma of the final outlier check (sw\_iteration=0) is computed
      from the data with flag 0 only. Invalid values (-999.99) and
      rejected data were included before.
    - The refit of the smoothed harmonics uses all of the 3-day trimmed
      points.
    - Data gaps longer than half a year are interpolated as intended.
    - The right-side extrapolation points stay within the 1-day grid.
    - annual(), multimonth() and period() use calendar years.
    - The outer iteration (sw\_iteration=1) stops when no new data are
      rejected.
    - The 1-day grid is placed at 12:00 like the data. The "year" column
      of the fitting results shifts by half a day.
    - Accessors called before fit() raise ValueError.
    - No SyntaxWarning on Python 3.12 or later.
- Renamed columns: ave\_trend/stdev\_trend of fit\_monthly.txt and
  fit\_multimonth.txt are renamed to ave\_seasonal/stdev\_seasonal.
  They are the mean and standard deviation of the seasonal component.

## 2024-04-01

- Version 1.0.1
- Initial release of python version

## Original Release Notes (FORTRAN version)


    Program : lsfbf4-9-2.f

          This is a program of data selection and curve fitting
       by a digital filtering technique including Reinsch-type cubic
       spline, Fourier harmonics and linear interpolation. It is to
       derive the fitted curve to the observed data ( CO2, d13C,..),
       as well as to separate the seasonal, short- and long-term
       variations from the data.

          The average seasonal cycle is represented by harmonics
       (NH=0-6), but if you give NH, the average cycle with NH+1 is
       first calculated and then smoothed by a Butterworth filter
       which attenuates signals with a period of NH+1 to 50 %.
          Linear interpolation is used to calculate equally spaced
       data from residuals which are obtained by extracting the
       average seasonal cycle from actual data.
          The long-term trend and irregular parts of the seasonal
       cycle are driven by smoothing equally spaced data ,obtained
       above, with the Butterworth filter.  The extrapolated data
       for the trend and overall fit are given by the fit to the
       actual data with a Reinsch spline (T5=1825days), which are
       altered by an iterative procedure.
          Prior to use of this program, the number of harmonics has
       to be determined by the program 'PERI'.  The digital filter
      form of Reinsch spline is shown explicitly by the programs
       'COX1' or 'COX' in which the transfer function is plotted.

          KD must be sufficiently larger than the value of ((910
       +2920)/(mean data interval expressed in day)+data number of
       the record). KO must be sufficiently larger than the value
       of (910+2920+length of the data record expressed in day).
       If you want to change KD, you must change KD of subroutine
       or function Fourier, Frc1, Reinsch, Setup, Chol1d, Smooth,
       Extrad1 and Judge1,too.  If you want to change KO, you must
       change N or KO in subroutines Bwf, Scycle, Yac and Judge1,
       too.

                                 By T. Nakazawa (Mar. 8, 1991)

    * Before digital filtering, data values with more than 3 standard
     errors of estimated fit are identified as outliers and rejected
     from further analysis. (dafault setting)

    * The order of Butterworth filter is set to be 26th for long-term
     and to be 16th for seasonal cycle. (dafault setting)

                                Last modified by M. Ishizawa
                                                 (Feb. 4, 2002)

    Fortran 90 version refactoring
                        Developed by Hisashi Yashio (2008)
