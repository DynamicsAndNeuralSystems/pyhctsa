Trends and drift
================

Why it matters
--------------

Many features describe fluctuations about a stable level: autocorrelations and correlation times, the shape of the spectrum, forecasting errors, symbolic and entropy measures that split the series at its mean.
A trend leaks into all of them.
Three mechanisms do most of the work.
The z-score denominator includes the trend's variance, so the fluctuations are shrunk; features that set their own delay from the autocorrelation time read a trended series as slow and choose long delays; and features that split the series at its mean or into global halves see the two ends of a trended series as different regimes.
In a screen of the default feature set on synthetic series, a linear rise of three standard deviations over the record moved about half the features by more than the realisation-to-realisation spread; a rise of one standard deviation moved almost nothing.

Other features measure the trend on purpose: :func:`~pyhctsa.operations.stationarity.trend`, :func:`~pyhctsa.operations.stationarity.monotonic_trend` (the Mann-Kendall tau and the Theil-Sen slope), :func:`~pyhctsa.operations.stationarity.fit_polynomial`, the drifting-mean and stationarity tests, and every ``*_raw`` feature computed without z-scoring.
These report what they should, and detrending the data would make them uninformative.

What pyhctsa does internally
----------------------------

Nothing is detrended inside the feature library.
Every feature is computed on the series as given (z-scored where its configuration says so), so that its name says what it computes on every series alike.

A small family of ``*_lindt`` features gives the linear correlation structure of the linearly detrended series alongside the originals: the autocorrelation at every configured lag (``ac_<tau>_lindt``), the partial autocorrelation function (``partial_autocorr_20_burg_lindt``), the first zero and 1/e crossing times of the autocorrelation function (``first_crossing_ac_0_both_lindt``, ``first_crossing_ac_1/e_both_lindt``), the shape of its decay (``autocorr_shape_drown_lindt``, ``autocorr_shape_pos_drown_lindt``) and the histogram automutual information at lags 1 to 5 (``histogram_ami_*_lindt``).
Each is the same operation applied to ``zscore(detrend_linear(x_z))``: the least-squares line through the z-scored series is removed and the residual z-scored again (:func:`~pyhctsa.utils.detrend_linear`, the ``preprocess: detrend`` setting of a feature configuration).
The ``*_lindt`` form is exactly invariant to an added line and otherwise differs from the original only by what the fitted line absorbs.
A series that is a line up to round-off has no residual to work with, so its ``*_lindt`` features are NaN.

Keep two limits in mind.
Only a line is removed: a curved drift, a step or a slow oscillation is removed only in part (a step of two standard deviations in the middle of the record still moves the ``*_lindt`` features, by about a third of what it does to the originals).
And a line fitted to a long-memory series absorbs some of its real low-frequency power, so the ``*_lindt`` autocorrelations of such a series sit a little below the originals: for a fractionally integrated series with ``d = 0.4`` and 1000 samples, by about half the realisation-to-realisation spread at every lag.
The features that set a delay from the autocorrelation time still use the raw series' timescale.

Recommendation
--------------

When the series in a dataset are trended, decide once what to remove and apply it to every series, so that the features compare series on the same footing.
Never detrend a subset of the series: a feature then means different things for different rows.

Look at the trend before deciding.
The features that measure it (``trend``, ``monotonic_trend``, ``fit_polynomial``, the KPSS test) are themselves a summary: on a z-scored series ``trend.stdRatio`` is the fraction of the standard deviation that survives a linear detrend, ``fit_polynomial_1`` to ``fit_polynomial_3`` say how much a line, a quadratic and a cubic leave, and ``monotonic_trend.tau`` says how consistently the series rises or falls.
A trend that a line explains is removed by :func:`~pyhctsa.utils.detrend_linear`; a curved drift or a slow oscillation needs a method of your own (a polynomial, a high-pass filter), applied to every series alike; a random walk is made stationary by differencing (``numpy.diff``, or the ``*_diff1`` features), not by detrending.

.. code-block:: python

   from pyhctsa.calculator import FeatureCalculator
   from pyhctsa.utils import detrend_linear

   data_lindt = [detrend_linear(x) for x in data]   # the least-squares line removed from every series
   res = FeatureCalculator().extract(data_lindt)

After this the ``*_lindt`` features equal their originals, and the features that measure trend report nothing: drop both from the result, or keep the trend measurements from a run on the original data if they matter to your question.
When the trend is itself the signal of interest (a growth rate, a drift between conditions), do not detrend; the trend measurements and the ``*_raw`` features carry it.
