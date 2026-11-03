Sampling rate
=============

Why it matters
--------------

Many features look at a series one sample apart.
Lag-1 entropies, symbolic features and complexity measures describe structure at the sampling interval.
If a series is sampled much faster than its dynamics evolve (oversampled), consecutive samples are nearly redundant, and these features mostly measure how smooth the series is at that scale rather than its dynamics.
Other features encode a timescale in samples, such as autocorrelation lags and spectral frequencies.
These are meant to change with the sampling rate, because they report the timescale itself.

What pyhctsa does internally
----------------------------

Most adaptive features set their delay with :func:`~pyhctsa.utils.get_tau` using the ``'ac1e'`` rule, the 1/e autocorrelation time, so that they look at structure relative to each series' own timescale.
The ``*_dec`` features are computed on the series decimated to one sample per 1/e time (:func:`~pyhctsa.utils.decimate_ac1e`), which makes them nearly invariant to oversampling.
The :func:`~pyhctsa.operations.correlation.oversampling` operation measures oversampling directly (Toker et al., 2020).

This per-series normalisation has limits.
Delays are integers, so when the 1/e time is only 1 or 2 samples it is floored and cannot adapt.
White measurement noise pulls the lag-1 autocorrelation below 1/e, so the 1/e time reads as 1 sample even when the underlying signal is oversampled.
For nonstationary or trending series a single timescale is not well defined.
Normalising each series separately also removes real differences in timescale between series.

Recommendation
--------------

If you know the band of your signal, low-pass filter (anti-alias) and resample the whole dataset with one integer factor before extracting features.
For example, for EEG recorded at 1000 Hz, anti-alias near 80-100 Hz and resample to 200-250 Hz.
Keep one sampling rate across a dataset, so that features compare series on the same footing.

:func:`~pyhctsa.sampling.sampling_report` shows a series' delays (``tau_ac1e``, ``tau_ac``, ``tau_mi``), its lag-1 autocorrelation, the oversampling statistic ``eta`` and a suggested decimation factor (about three samples per 1/e time), with warnings for near-critical sampling, short results, disagreeing delays and a noise floor.
Run it on a few representative series and choose one factor.
:func:`~pyhctsa.sampling.resample_dataset` then applies that factor to every series with an anti-aliasing filter (:func:`scipy.signal.decimate`, whose filter cuts off at 0.8 of the new Nyquist frequency).

.. code-block:: python

   from pyhctsa.calculator import FeatureCalculator
   from pyhctsa.sampling import sampling_report, resample_dataset

   fs = 1000.0                      # sampling rate of your recordings (Hz)
   rep = sampling_report(data[0], fs=fs)
   print(rep['tau_ac1e'], rep['decimation_factor'], rep['fs_suggested'], rep['warnings'])

   # factor 4: 1000 Hz -> 250 Hz, anti-aliased at 0.8 * 125 Hz = 100 Hz
   data_ds = resample_dataset(data, 4)
   res = FeatureCalculator().extract(data_ds)

The suggested factor depends on the series, so prefer a factor set by the signal band when you know it, and use the report to check that it leaves enough samples (a few hundred or more) and that ``tau_ac1e`` stays above 2.
