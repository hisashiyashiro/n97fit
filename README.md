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

# Release Notes

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
