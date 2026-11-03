"""Sampling-rate diagnostics and dataset-wide resampling.

Many time-series features (lag-1 entropies, symbolic and complexity measures) look at
structure at the sampling interval, so their values depend on how finely a process is
sampled relative to its own dynamics. pyhctsa compensates within the feature set (delays
set by the 1/e autocorrelation time, the ``*_dec`` features), but when all series in a
dataset share one sampling rate it is better to choose that rate once, before extraction.
This module helps with that choice:

- :func:`sampling_report` summarises one series' timescales and suggests an integer
  decimation factor;
- :func:`resample_dataset` applies one anti-aliased decimation factor to every series.

See the "Sampling rate" page of the usage guide.
"""
import math
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import decimate

from .utils import _acf_fourier, get_tau

__all__ = ['sampling_report', 'resample_dataset']

# Report thresholds (see sampling_report's Notes).
_MIN_DECIMATED_LENGTH = 500
_MAX_TAU_RATIO = 10
_NOISE_FLOOR_ACF1 = 0.2
# scipy.signal.decimate recommends at most this factor per IIR stage.
_MAX_IIR_STAGE = 13


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def sampling_report(y: ArrayLike, fs: Union[float, None] = None, target_tau: float = 3) -> dict:
    """
    Summarise how finely a time series is sampled relative to its own timescales.

    Computes the delays that pyhctsa's adaptive features use
    (:func:`~pyhctsa.utils.get_tau`), the lag-1 autocorrelation and Toker et al.'s
    oversampling statistic (:func:`~pyhctsa.operations.correlation.oversampling`), and
    suggests the integer decimation factor that would leave about `target_tau` samples per
    1/e autocorrelation time. To resample a whole dataset, run the report on a few
    representative series, choose one factor, and pass it to :func:`resample_dataset`.

    Parameters
    ----------
    y : array-like
        The time series (a list, 1-D array or pandas Series).
    fs : float, optional
        The sampling rate (e.g. in Hz). If given, the report also expresses the timescale
        and the suggested rate in these units.
    target_tau : float, optional
        The number of samples per 1/e autocorrelation time to aim for (default 3). Must be
        positive.

    Returns
    -------
    dict
        Dictionary containing:
            - 'n': The length of the series.
            - 'tau_ac1e': The 1/e autocorrelation time in samples (``get_tau(y, 'ac1e')``),
              the default delay of most adaptive features.
            - 'tau_ac': The first zero crossing of the ACF in samples (``get_tau(y, 'ac')``).
            - 'tau_mi': The AMI-based delay in samples (``get_tau(y, 'mi')``, capped at
              'tau_ac1e').
            - 'acf1': The lag-1 autocorrelation.
            - 'eta': Toker et al.'s oversampling statistic, range(y) / mean(abs(diff(y)));
              values above 10 indicate oversampling.
            - 'decimation_factor': ``max(1, floor(tau_ac1e / target_tau))``, the integer
              factor that would leave about `target_tau` samples per 1/e time (NaN if
              'tau_ac1e' is NaN).
            - 'decimated_length': ``n // decimation_factor`` (NaN if the factor is NaN).
            - 'fs', 'tau_ac1e_seconds', 'fs_suggested', 'nyquist_suggested' (only when `fs`
              is given): the sampling rate, 'tau_ac1e' in units of ``1/fs``, the rate after
              decimation (``fs / decimation_factor``) and its Nyquist frequency.
            - 'warnings': A list of short strings describing problems (empty if none).

    Notes
    -----
    Warnings are raised when:

    - a delay is NaN (constant or very short series, or an ACF that never falls to 1/e);
    - ``tau_ac1e <= 2``: the series is near-critically sampled, so the adaptive delays sit
      at 1 or 2 samples (integer flooring) and per-series normalisation has nothing to work
      with;
    - decimating by the suggested factor would leave fewer than 500 samples;
    - ``max(tau_ac, tau_ac1e, tau_mi) / min(...) > 10``: the delays disagree wildly, as
      for a nonstationary or trending series, whose timescale is not well defined;
    - ``tau_ac1e == 1`` while ``acf1 > 0.2``: the series is correlated at lag 1 but white
      measurement noise (a broadband noise floor) pulls the ACF below 1/e at lag 2, so the
      1/e rule understates how oversampled the underlying signal is. Low-pass filtering
      before choosing a factor (see :func:`resample_dataset`) helps here.

    The 'mi' delay uses a Kraskov AMI estimate, which takes a second or so for long,
    slowly decorrelating series.

    Examples
    --------
    >>> import numpy as np
    >>> from pyhctsa.sampling import sampling_report
    >>> rng = np.random.default_rng(0)
    >>> rep = sampling_report(rng.standard_normal(2000), fs=100.0)
    >>> rep['tau_ac1e'], rep['decimation_factor']
    (1, 1)
    """
    if isinstance(target_tau, bool) or not _finite(target_tau) or target_tau <= 0:
        raise ValueError(f"target_tau must be a positive number, got {target_tau!r}.")
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)

    taus = {rule: get_tau(y, rule) for rule in ('ac1e', 'ac', 'mi')}
    tau_ac1e = taus['ac1e']
    acf1 = float(_acf_fourier(y)[1]) if n >= 2 else np.nan

    from .operations.correlation import oversampling  # deferred, as in pyhctsa.utils
    eta = oversampling(y)['eta'] if n >= 2 else np.nan

    if _finite(tau_ac1e):
        factor = max(1, int(math.floor(tau_ac1e / target_tau)))
        decimated_length = n // factor
    else:
        factor = decimated_length = np.nan

    report = {
        'n': n,
        'tau_ac1e': tau_ac1e,
        'tau_ac': taus['ac'],
        'tau_mi': taus['mi'],
        'acf1': acf1,
        'eta': float(eta),
        'decimation_factor': factor,
        'decimated_length': decimated_length,
    }
    if fs is not None:
        fs = float(fs)
        fs_suggested = fs / factor if _finite(factor) else np.nan
        report.update({
            'fs': fs,
            'tau_ac1e_seconds': tau_ac1e / fs if _finite(tau_ac1e) else np.nan,
            'fs_suggested': fs_suggested,
            'nyquist_suggested': fs_suggested / 2,
        })

    warnings = []
    missing = [f"tau_{rule}" for rule, tau in taus.items() if not _finite(tau)]
    if missing:
        warnings.append(f"{', '.join(missing)} undefined (constant or short series, or an ACF "
                        "that never falls to 1/e); no timescale for these rules.")
    if _finite(tau_ac1e) and tau_ac1e <= 2:
        warnings.append(f"tau_ac1e = {tau_ac1e}: near-critically sampled; adaptive delays are "
                        "floored at 1-2 samples and per-series normalisation has nothing to work with.")
    if _finite(factor) and factor > 1 and decimated_length < _MIN_DECIMATED_LENGTH:
        warnings.append(f"decimating by {factor} would leave only {decimated_length} samples "
                        f"(< {_MIN_DECIMATED_LENGTH}).")
    finite_taus = [t for t in taus.values() if _finite(t)]
    if len(finite_taus) >= 2 and max(finite_taus) / min(finite_taus) > _MAX_TAU_RATIO:
        warnings.append(f"delays disagree (ac1e={taus['ac1e']}, ac={taus['ac']}, mi={taus['mi']}): "
                        "likely nonstationary or trending, so a timescale is not well defined.")
    if tau_ac1e == 1 and _finite(acf1) and acf1 > _NOISE_FLOOR_ACF1:
        warnings.append(f"tau_ac1e = 1 but acf1 = {acf1:.2f}: a broadband noise floor; the 1/e "
                        "rule understates the oversampling of the underlying signal.")
    report['warnings'] = warnings
    return report


def _series_list(data) -> list:
    """A list of 1-D float arrays, following FeatureCalculator.extract's input conventions."""
    from .calculator import _standardise_inputs
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    elif isinstance(data, (list, tuple)) and len(data) > 0 and any(isinstance(ts, pd.Series) for ts in data):
        data = [ts.to_numpy() if isinstance(ts, pd.Series) else ts for ts in data]
    return [np.asarray(ts, dtype=float).ravel() for ts in _standardise_inputs(data)]


def _stages(factor: int) -> list:
    """Split `factor` into decimation stages of at most 13 where its prime factors allow."""
    primes, k, p = [], factor, 2
    while p * p <= k:
        while k % p == 0:
            primes.append(p)
            k //= p
        p += 1
    if k > 1:
        primes.append(k)
    stages = []
    for p in sorted(primes, reverse=True):  # pack the largest primes first
        for i, s in enumerate(stages):
            if s * p <= _MAX_IIR_STAGE:
                stages[i] = s * p
                break
        else:
            stages.append(p)
    return stages


def resample_dataset(data, factor: int, zero_phase: bool = True) -> list:
    """
    Decimate every series in a dataset by one integer factor, with an anti-aliasing filter.

    Applying the same factor to every series keeps differences in timescale between series
    (which a per-series normalisation would remove) while bringing the sampling interval
    closer to the dynamics. Choose the factor from domain knowledge of the signal band, or
    from :func:`sampling_report` on representative series.

    Each series is filtered and downsampled with :func:`scipy.signal.decimate` using its
    default anti-aliasing filter: an order-8 Chebyshev type I low-pass filter (IIR) with
    cutoff at 0.8 of the new Nyquist frequency, i.e. ``0.8 * fs / (2 * factor)``. Factors
    above 13 are applied in stages of at most 13 (as scipy recommends); a stage that cannot
    be split, a prime factor above 13, uses scipy's FIR filter instead (a Hamming-window FIR
    of order ``20 * stage``). The output keeps samples ``0, factor, 2 * factor, ...``, so a
    series of length ``n`` becomes ``ceil(n / factor)`` long.

    Parameters
    ----------
    data : array-like or list of array-like
        A single series (a list, 1-D array or pandas Series), a list of series (which may
        differ in length), or a 2-D array of shape (n_series, n_samples), as accepted by
        :meth:`FeatureCalculator.extract <pyhctsa.calculator.FeatureCalculator.extract>`.
        Series must be finite: a NaN spreads through the filter.
    factor : int
        The decimation factor, an integer >= 1. With ``factor == 1`` every series is
        returned unchanged (as a float array).
    zero_phase : bool, optional
        Filter forwards and backwards (no phase shift; default True), as in
        :func:`scipy.signal.decimate`. With False the filter is causal and delays the
        signal.

    Returns
    -------
    list of numpy.ndarray
        The decimated series, in input order (a list even for a single input series).

    Raises
    ------
    ValueError
        If `factor` is not an integer >= 1, or a series is too short to filter (the IIR
        filter needs more than 27 samples per stage).

    Examples
    --------
    >>> import numpy as np
    >>> from pyhctsa.sampling import resample_dataset
    >>> rng = np.random.default_rng(0)
    >>> out = resample_dataset([rng.standard_normal(1000), rng.standard_normal(600)], 4)
    >>> [len(x) for x in out]
    [250, 150]
    """
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer, float, np.floating)) \
            or not np.isfinite(factor) or float(factor) != int(factor) or factor < 1:
        raise ValueError(f"factor must be an integer >= 1, got {factor!r}.")
    factor = int(factor)
    series = _series_list(data)
    if factor == 1:
        return series
    stages = _stages(factor)
    out = []
    for i, x in enumerate(series):
        try:
            for q in stages:
                ftype = 'iir' if q <= _MAX_IIR_STAGE else 'fir'
                x = decimate(x, q, ftype=ftype, zero_phase=zero_phase)
        except ValueError as err:
            raise ValueError(f"Series {i} (length {len(series[i])}) could not be decimated "
                             f"by {factor}: {err}") from err
        out.append(np.asarray(x, dtype=float))
    return out
