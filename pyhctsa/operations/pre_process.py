from typing import Union
import re
import numba
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import make_lsq_spline
from scipy.signal import lfilter, resample_poly
from scipy.stats import boxcox
import logging
logger = logging.getLogger('pyhctsa')

from ..operations.correlation import autocorr
from ..operations.distribution import compare_ks_fit, outlier_test, simple_fit
from ..operations.nonlinearity import zero_one_test
from ..operations.stationarity import sliding_window, stat_av
from ..robust import bf_fit_sinusoids, bf_random, bf_random_seed
from ..utils import _round_half_away, _zscore_matlab, z_score

def _med_filt_1d(x: ArrayLike, k: int) -> ArrayLike:
    """Apply a length-k median filter to a 1D array x, as MATLAB's ``medfilt1``.

    The ends are padded with zeros (``medfilt1``'s default ``'zeropad'``), so the
    first and last samples are medians that include zeros, not the edge values.
    For odd k, y(i) is the median of x[i-(k-1)//2 : i+(k-1)//2+1]
    For even k, y(i) is the median of x[i-k//2 : i+k//2]

    Based on: https://gist.github.com/bhawkins/3535131.
    """
    assert k > 0, "Median filter length must be positive."
    assert x.ndim == 1, "Input must be one-dimensional."

    if k % 2 == 1:
        left_pad = right_pad = (k - 1) // 2
    else:
        left_pad = k // 2
        right_pad = k // 2 - 1

    n = len(x)
    xp = np.concatenate([np.zeros(left_pad, dtype=float), np.asarray(x, dtype=float),
                         np.zeros(right_pad, dtype=float)])
    y = np.empty((n, k))
    for i in range(k):
        y[:, i] = xp[i:i + n]
    return np.median(y, axis=1)

def _norm_diff(proc: float, orig: float) -> float:
    """Normalized difference (proc - orig) / (proc + orig) of two positive numbers.

    Lies in [-1, 1], is 0 when nothing changes (and, by convention, when both are 0),
    and equals tanh(log(proc / orig) / 2): a bounded version of the log-ratio that
    stays finite when the original value is near 0, where a plain ratio is unstable.
    """
    if proc + orig == 0:
        return 0.0
    return (proc - orig) / (proc + orig)


def _diff(proc: float, orig: float) -> float:
    """Difference, processed minus original (for statistics that can be negative or zero)."""
    return proc - orig


def _spline_detrend(y: np.ndarray, npieces: int, order: int) -> np.ndarray:
    """Remove a least-squares spline, as MATLAB's ``spap2(npieces, order, 1:N, y)``.

    The spline has `npieces` polynomial pieces and the given order (order 4 is cubic).
    As in ``spap2`` with a scalar first argument, the knots come from ``aptknt`` applied to
    ``npieces - 1 + order`` data sites spread evenly over the series: the interior knots
    are averages of ``order - 1`` consecutive sites.
    """
    N = len(y)
    x = np.arange(1, N + 1, dtype=float)
    k = min(order, N)
    maxpieces = N - k + 1
    if npieces < 1 or npieces > maxpieces:
        logger.warning(f"spline: the number of pieces must be between 1 and {maxpieces}; "
                       f"using {max(1, min(maxpieces, npieces))}.")
        npieces = max(1, min(maxpieces, npieces))
    if npieces == 1 and k == 1:
        knots = np.array([x[0], x[-1]])
    else:
        idx = np.array([_round_half_away(v) for v in np.linspace(1, N, npieces - 1 + k)], dtype=int)
        tau = x[idx - 1]
        n = len(tau)
        if k == 1:  # aptknt: midpoints between the sites
            knots = np.concatenate([[tau[0]], tau[:-1] + np.diff(tau) / 2, [tau[-1]]])
        else:  # aptknt: augknt([tau(1), aveknt(tau, k), tau(end)], k)
            interior = np.array([np.mean(tau[i + 1:i + k]) for i in range(n - k)])
            knots = np.concatenate([np.full(k, tau[0]), interior, np.full(k, tau[-1])])
    spl = make_lsq_spline(x, y, knots, k=k - 1)
    return y - spl(x)


def preproc_compare(y: ArrayLike, detrend_meth: str = 'medianf3') -> dict:
    """
    How time-series properties change after a preprocessing step.

    Applies a given preprocessing transformation (detrending, differencing,
    filtering or resampling) to the time series, z-scores the original and the
    processed series, and returns the change in each of a set of statistics from its
    value for the original to its value for the processed series. The statistics compare
    stationarity measures (StatAv, and the variation of the local mean and of the
    local standard deviation across windows), distributional fits (a Gaussian fit
    to the kernel-smoothed distribution, and the discrepancy from a fitted normal
    distribution) and the effect of trimming outliers.

    The change is the difference (processed minus original) for statistics that can be
    negative or zero, and the normalized difference (processed - original) /
    (processed + original) for positive statistics. The latter is between -1 and 1, is 0
    when nothing changes, and equals tanh(log(processed / original) / 2), a bounded
    version of the log-ratio that stays finite when the original value is near 0, where
    a plain ratio is unstable.

    Parameters
    ----------
    y : array-like
        Input time series.
    detrend_meth : str, optional
        The preprocessing to apply:

        - ``"poly<n>"``  : remove a polynomial of order n = 1-9, e.g., ``"poly1"``
          is a linear detrending
        - ``"sin<n>"``   : remove a sum of n = 1-8 sinusoids a1*sin(2*pi*f1*t + c1) + ... fitted
          by least squares to the mean-subtracted series, with frequencies searched
          deterministically between 1/(2N) and 1/2 - 1/(2N) cycles per sample
          (:func:`pyhctsa.robust.bf_fit_sinusoids`), e.g., ``"sin1"``
        - ``"spline<npieces><order>"`` : remove a least-squares spline with the given
          number of polynomial pieces and spline order, e.g., ``"spline24"`` is a
          cubic spline with 2 pieces
        - ``"diff<n>"``  : n successive differences, e.g., ``"diff1"``
        - ``"medianf<n>"``: running median filter of length n (zero-padded at the
          ends, like MATLAB's ``medfilt1``), e.g., ``"medianf3"``
        - ``"rav<n>"``   : running mean filter of length n, e.g., ``"rav5"``
        - ``"resample_<p>_<q>"`` : resample by the ratio p/q; e.g., ``"resample_1_2"``
          halves the length and ``"resample_10_1"`` multiplies it by 10
        - ``"logr"``     : log returns (positive data only; otherwise NaN)
        - ``"boxcox"``   : a Box-Cox transformation (positive data only; otherwise NaN)

        Default is ``"medianf3"``.

    Returns
    -------
    dict
        The change, from the original to the processed series, of each of these
        statistics (all of the series are z-scored first).

        Normalized differences (positive statistics):

        - ``statav2``: StatAv with 2 segments (``stat_av``)
        - ``swms2_2``, ``swms5_1``, ``swms10_1``: the standard deviation of the window
          means across windows (``sliding_window`` 'mean'), with 2 windows overlapping
          by half, and 5 and 10 non-overlapping windows
        - ``swss2_1``, ``swss5_1``, ``swss10_1``: the same for the window standard
          deviations (``sliding_window`` 'std')
        - ``kscn_olapint``: the overlap integral of the kernel-smoothed distribution
          with the best-fitting normal (``compare_ks_fit``)
        - ``olbt_s5``: the standard deviation after trimming the 5% most extreme values
          at each end, relative to that of the full series (``outlier_test``)

        Differences (statistics that can be negative or zero):

        - ``gauss1_kd_r2``, ``gauss1_kd_resAC1``, ``gauss1_kd_resrunsz``: the R^2, the
          lag-1 autocorrelation of the residuals, and the runs-test z-statistic of the
          residuals of a Gaussian fit to the kernel-smoothed distribution
        - ``kscn_peaksepy``, ``kscn_peaksepx``, ``kscn_relent``: the peak separation in
          height and in position, and the relative entropy, of the kernel-smoothed
          distribution against the best-fitting normal (``compare_ks_fit``)
        - ``olbt_m2``, ``olbt_m5``: the mean after trimming the 2% and 5% most extreme
          values at each end (``outlier_test``)

        A scalar NaN is returned if the processed series is identically zero (or, for
        ``'logr'`` and ``'boxcox'``, if the data are not all positive).
    """
    y = np.asarray(y, dtype=float)
    N = len(y)

    # ------------------------------------------------------------------
    # Apply preprocessing: y (raw) -> y_d (detrended/transformed)
    # ------------------------------------------------------------------
    if (m := re.fullmatch(r'poly([1-9])', detrend_meth)):
        # 1) Polynomial detrend
        order = int(m.group(1))
        r = np.arange(1, N + 1, dtype=float)
        y_d = y - np.polynomial.Polynomial.fit(r, y, order)(r)

    elif (m := re.fullmatch(r'sin([1-8])', detrend_meth)):
        # 2) Seasonal detrend: sum of sinusoids
        # (the mean is removed first: the sinusoids have no offset, and the frequencies are
        # bounded away from zero, so they could not otherwise absorb a non-zero mean)
        num_sin = int(m.group(1))
        if N <= 3 * num_sin:
            return np.nan  # too short to fit this many sinusoids
        y_c = y - np.mean(y)
        y_d = y_c - bf_fit_sinusoids(y_c, num_sin)[0]

    elif (m := re.fullmatch(r'spline(\d)(\d)', detrend_meth)):
        # 3) Spline detrend
        y_d = _spline_detrend(y, int(m.group(1)), int(m.group(2)))

    elif (m := re.fullmatch(r'diff(\d)', detrend_meth)):
        # 4) Differencing
        y_d = np.diff(y, n=int(m.group(1)))

    elif (m := re.fullmatch(r'medianf(\d+)', detrend_meth)):
        # 5) Median filter
        y_d = _med_filt_1d(y, int(m.group(1)))

    elif (m := re.fullmatch(r'rav(\d+)', detrend_meth)):
        # 6) Running average
        n = int(m.group(1))
        y_d = lfilter(np.ones(n) / n, [1], y)

    elif (m := re.fullmatch(r'resample_(\d+)_(\d+)', detrend_meth)):
        # 7) Resample
        y_d = resample_poly(y, int(m.group(1)), int(m.group(2)))

    elif detrend_meth == 'logr':
        # 8) Log returns
        if not np.all(y > 0):
            return np.nan
        y_d = np.diff(np.log(y))

    elif detrend_meth == 'boxcox':
        # 9) Box-Cox transformation
        if not np.all(y > 0):
            return np.nan
        y_d = boxcox(y)[0]

    else:
        raise ValueError(f"Invalid detrending method '{detrend_meth}'")

    # ------------------------------------------------------------------
    # Quick check that outputs are meaningful
    # ------------------------------------------------------------------
    if np.all(y_d == 0):
        return np.nan

    # ------------------------------------------------------------------
    # Statistical tests on original and processed series (z-score both)
    # ------------------------------------------------------------------
    y = z_score(y)
    y_d = z_score(y_d)

    out = {}

    # 1) Stationarity
    # (a) StatAv
    out['statav2'] = _norm_diff(stat_av(y_d, 'seg', 2), stat_av(y, 'seg', 2))

    # (b) Sliding window mean
    for win, step in [(2, 2), (5, 1), (10, 1)]:
        out[f'swms{win}_{step}'] = _norm_diff(sliding_window(y_d, 'mean', 'std', win, step),
                                              sliding_window(y, 'mean', 'std', win, step))

    # (c) Sliding window std
    for win, step in [(2, 1), (5, 1), (10, 1)]:
        out[f'swss{win}_{step}'] = _norm_diff(sliding_window(y_d, 'std', 'std', win, step),
                                              sliding_window(y, 'std', 'std', win, step))

    # 2) Gaussianity
    # (a) Gaussian fit to the kernel density estimate
    me1 = simple_fit(y_d, 'gauss1', 0)
    me2 = simple_fit(y, 'gauss1', 0)
    if not isinstance(me1, dict) or not isinstance(me2, dict):
        # fitting the Gaussian failed
        for key in ['r2', 'resAC1', 'resrunsz']:
            out[f'gauss1_kd_{key}'] = np.nan
    else:
        for key in ['r2', 'resAC1', 'resrunsz']:
            out[f'gauss1_kd_{key}'] = _diff(me1[key], me2[key])

    # (b) Compare the distribution to a fitted normal distribution
    me1 = compare_ks_fit(y_d, 'norm')
    me2 = compare_ks_fit(y, 'norm')
    if not isinstance(me1, dict) or not isinstance(me2, dict):
        for key in ['peaksepy', 'peaksepx', 'olapint', 'relent']:
            out[f'kscn_{key}'] = np.nan
    else:
        out['kscn_peaksepy'] = _diff(me1['peaksepy'], me2['peaksepy'])
        out['kscn_peaksepx'] = _diff(me1['peaksepx'], me2['peaksepx'])
        out['kscn_olapint'] = _norm_diff(me1['olapint'], me2['olapint'])
        out['kscn_relent'] = _diff(me1['relent'], me2['relent'])

    # 3) Outliers
    out['olbt_m2'] = _diff(outlier_test(y_d, 2, 'mean'), outlier_test(y, 2, 'mean'))
    out['olbt_m5'] = _diff(outlier_test(y_d, 5, 'mean'), outlier_test(y, 5, 'mean'))
    out['olbt_s5'] = _norm_diff(outlier_test(y_d, 5, 'std'), outlier_test(y, 5, 'std'))

    return out


def _iterate_stats(y: np.ndarray, y_d: np.ndarray) -> np.ndarray:
    """The ten statistics of ``preproc_iterate`` for one processed series ``y_d`` (and original ``y``)."""
    y = z_score(y)
    y_d = z_score(y_d)
    f = np.full(10, np.nan)

    # 1) Stationarity: StatAv, sliding-window mean and standard deviation
    f[0] = stat_av(y_d, 'seg', 5)
    f[1] = sliding_window(y_d, 'mean', 'std', 5, 2)
    f[2] = sliding_window(y_d, 'std', 'std', 5, 2) / sliding_window(y, 'std', 'std', 5, 2)

    # 2) Gaussianity: Gaussian fits to the kernel density and to a histogram, and a normal fit
    me = simple_fit(y_d, 'gauss1', 0)
    f[3] = me['rmse'] if isinstance(me, dict) else np.nan
    me = simple_fit(y_d, 'gauss1', 'sqrt')
    f[4] = me['rmse'] if isinstance(me, dict) else np.nan
    me = compare_ks_fit(y_d, 'norm')
    f[5] = me['adiff'] if isinstance(me, dict) else np.nan

    # 3) Outliers
    f[6] = outlier_test(y_d, 5, 'mean')

    # Cross-correlation with, and distance to, the original signal
    if len(y) == len(y_d):
        norm = np.sqrt(np.sum(y ** 2) * np.sum(y_d ** 2))
        f[7] = np.dot(y[:-1], y_d[1:]) / norm  # lag -1 of xcorr(y, y_d, 1, 'coeff')
        f[8] = np.dot(y[1:], y_d[:-1]) / norm  # lag +1
        f[9] = np.linalg.norm(y - y_d) / len(y)
    return f


def _profile_trend_jump(f: np.ndarray) -> tuple:
    """Trend and jump of a profile of a statistic across processing strengths.

    The profile is z-scored. The trend is the sum of its successive differences (last minus first).
    The jump is the largest t-statistic for a step change in the mean, over all split points: the
    split with the greatest absolute difference between the means before and after is chosen,
    and its difference is divided by the combined standard error of the two means.
    (NaN, NaN if any value is not finite.)
    """
    if not np.all(np.isfinite(f)):
        return np.nan, np.nan
    f = _zscore_matlab(f)
    n = len(f)
    trend = float(np.sum(np.diff(f)))

    def sd(v):  # MATLAB std: the standard deviation of a single value is 0
        return np.std(v, ddof=1) if len(v) > 1 else 0.0

    m1 = np.array([np.mean(f[:j + 1]) for j in range(n)])
    m2 = np.array([np.mean(f[j:]) for j in range(n)])
    se1 = np.array([sd(f[:j + 1]) / np.sqrt(j + 1) for j in range(n)])
    se2 = np.array([sd(f[j:]) / np.sqrt(n - j) for j in range(n)])
    i = int(np.argmax(np.abs(m1 - m2)))
    with np.errstate(divide='ignore', invalid='ignore'):
        jump = float(np.abs((m1[i] - m2[i]) / np.sqrt(se1[i] ** 2 + se2[i] ** 2)))
    return trend, jump


def preproc_iterate(y: ArrayLike, dt_meth: str = 'diff') -> dict:
    """
    How time-series properties change as a preprocessing step is applied more and more strongly.

    A preprocessing transformation is applied to the time series with increasing strength
    (for the number of times, or the window size, given by the method), and a set of
    statistics is computed on the (z-scored) processed series at each strength. Each
    statistic's profile across the strengths is then z-scored and summarized by a trend
    (the sum of its successive differences, i.e., the last minus the first value) and a jump
    (the largest t-statistic for a step change in the mean, over all split points).

    Parameters
    ----------
    y : array-like
        The input time series.
    dt_meth : str, optional
        The preprocessing to apply:

        - ``"spline"``: remove a least-squares cubic spline with 1, ..., 20 pieces
        - ``"diff"``: take incremental differences, 1, ..., 5 times (the method hctsa uses)
        - ``"medianf"``: a median filter with 25 window lengths from 1 to N/25
        - ``"rav"``: a running mean filter with 25 window lengths from 1 to N/25
        - ``"resampleup"``: progressively upsample the series, by factors 1, ..., 20
        - ``"resampledown"``: progressively downsample the series, by factors 1, ..., 20

        Default is ``"diff"``.

    Returns
    -------
    dict
        For each of the following statistics, measured on the processed series at each
        strength, a trend (key ending ``_trend``) and a jump (ending ``_jump``) of its profile
        across the strengths:

        - ``statav5``: StatAv with 5 segments (``stat_av``)
        - ``swms5_2``: the standard deviation of the window means in 5 windows overlapping by half
        - ``swss5_2``: the standard deviation of the window standard deviations in 5 windows
          overlapping by half, relative to that of the original series
        - ``gauss1_kd``: the root-mean-square error of a Gaussian fit to the kernel-smoothed
          distribution of values
        - ``gauss1_hsqrt``: the root-mean-square error of a Gaussian fit to a histogram of the
          values (square-root rule for the number of bins)
        - ``norm_kscomp``: the area between the kernel-smoothed distribution of the values and the
          best-fitting normal distribution (``compare_ks_fit``)
        - ``ol``: the mean after trimming the 5% highest and 5% lowest values (``outlier_test``)
        - ``xcn1``, ``xc1``: the cross-correlation between the original and processed series
          at lags -1 and +1
        - ``normdiff``: the distance between the original and processed series,
          norm(y - y_processed) / N

        The last three statistics need the processed series to be as long as the original, so
        they are NaN for ``'diff'`` and the resampling methods.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)

    # The number of times (or the strength with which) the processing is performed
    if dt_meth in ('spline', 'resampleup', 'resampledown'):
        n_range = np.arange(1, 21)
    elif dt_meth == 'diff':
        n_range = np.arange(1, 6)
    elif dt_meth in ('medianf', 'rav'):
        n_range = np.array([_round_half_away(v) for v in np.linspace(1, N / 25, 25)], dtype=int)
    else:
        raise ValueError(f"Unknown detrending method '{dt_meth}'")

    # Progressive processing with a running statistical evaluation
    outmat = np.full((len(n_range), 10), np.nan)
    for q, n in enumerate(n_range):
        n = int(n)
        if dt_meth == 'spline':
            y_d = _spline_detrend(y, n, 4)  # n pieces, cubic
        elif dt_meth == 'diff':
            y_d = np.diff(y, n=n)
        elif dt_meth == 'medianf':
            y_d = _med_filt_1d(y, n)
        elif dt_meth == 'rav':
            y_d = lfilter(np.ones(n) / n, [1], y)
        elif dt_meth == 'resampleup':
            y_d = resample_poly(y, n, 1)
        else:  # resampledown
            y_d = resample_poly(y, 1, n)
        outmat[q] = _iterate_stats(y, y_d)

    names = ['statav5', 'swms5_2', 'swss5_2', 'gauss1_kd', 'gauss1_hsqrt', 'norm_kscomp',
             'ol', 'xcn1', 'xc1', 'normdiff']
    out = {}
    for t, name in enumerate(names):
        out[f'{name}_trend'], out[f'{name}_jump'] = _profile_trend_jump(outmat[:, t])
    return out


def _piecewise_poly_residual(y: np.ndarray, order: int, num_bits: int) -> np.ndarray:
    """Remove a polynomial of the given order from each of ``num_bits`` equal pieces.

    The pieces come from hctsa's ``PP_PreProcess`` (``SUB_rempt``): boundaries at
    ``round(linspace(0, N, num_bits + 1))``, with the fit made against 1, ..., length of piece.
    """
    n = len(y)
    bits = np.array([_round_half_away(v) for v in np.linspace(0, n, num_bits + 1)], dtype=int)
    out = np.zeros(n)
    for k in range(num_bits):
        seg = y[bits[k]:bits[k + 1]]
        x = np.arange(1, len(seg) + 1, dtype=float)
        out[bits[k]:bits[k + 1]] = seg - np.polynomial.Polynomial.fit(x, seg, order)(x)
    return out


def _piecewise_poly_detrend(y: np.ndarray, order: int, num_bits: int) -> np.ndarray:
    """Remove a polynomial of the given order from each of ``num_bits`` equal pieces (z-scored result);
    see :func:`_piecewise_poly_residual`."""
    return z_score(_piecewise_poly_residual(y, order, num_bits))


def _rank_map_gaussian(y: np.ndarray, random_seed=None, draws: np.ndarray = None) -> np.ndarray:
    """Replace the values of y by Gaussian values of the same rank (hctsa's ``rmgd``).

    N Gaussian values are drawn and sorted, and the k-th smallest is given to the k-th smallest
    value of y. ``random_seed`` is as in ``remove_points``: an integer, ``None``/``'default'``
    for seed 0, or ``'none'`` for a seed drawn from NumPy's global random state. The draws come
    from the portable generator :func:`pyhctsa.robust.bf_random` (normal), so they are hctsa's
    for the same seed. The sorted draws can instead be supplied as ``draws``.
    """
    n = len(y)
    if draws is None:
        draws = bf_random(n, bf_random_seed(random_seed), 'normal')
    out = np.zeros(n)
    out[np.argsort(y, kind='stable')] = np.sort(draws)
    return out


def _ar_rms_error(data: np.ndarray, order: int) -> float:
    """In-sample RMS one-step prediction error of an AR model: ``sqrt(mean(pe(ar(data, order), data).^2))``.

    The model is MATLAB's default ``ar`` fit, forward-backward least squares on the samples with a
    full set of lagged values ('fb/now'). As ``pe`` does, the prediction errors of the first
    ``order`` samples are 0 (they are counted in the mean).
    """
    n = len(data)
    p = order
    # Forward and backward prediction regressions, solved jointly
    A = np.vstack([np.column_stack([data[p - j - 1:n - j - 1] for j in range(p)]),
                   np.column_stack([data[j + 1:n - p + j + 1] for j in range(p)])])
    b = np.concatenate([data[p:], data[:n - p]])
    theta = np.linalg.lstsq(A, b, rcond=None)[0]
    e = np.zeros(n)
    e[p:] = data[p:] - np.column_stack([data[p - j - 1:n - j - 1] for j in range(p)]) @ theta
    return float(np.sqrt(np.mean(e ** 2)))


def preproc_model_fit(y: ArrayLike, model: str = 'ar', order: int = 2,
                      random_seed: Union[int, str, None] = None) -> dict:
    """
    How the error of an AR model changes after preprocessing the series.

    Fits an autoregressive (AR) model to the time series and to a set of preprocessed versions of
    it, and returns the in-sample root-mean-square (RMS) one-step prediction error for each
    preprocessed version as a ratio of the RMS prediction error for the original series. Every
    version is z-scored before the model is fitted. The AR model is MATLAB's default
    ``ar`` fit (forward-backward least squares).

    Only one representative of each family of preprocessings (from hctsa's ``PP_PreProcess``)
    is fitted, as the other candidates were found to correlate at r >= 0.95 with one of these:

    - ``d1``: incremental differencing (first differences)
    - ``d2``: second differences
    - ``p1_20``: a straight line removed in each of 20 equal segments (piece-wise linear detrending)
    - ``p2_5``: a quadratic removed in each of 5 equal segments (piece-wise quadratic detrending)
    - ``rmgd``: the values replaced by Gaussian values of the same rank (rank mapping to a
      Gaussian distribution)

    Parameters
    ----------
    y : array-like
        The input time series.
    model : str, optional
        The time-series model to fit to the transformed series (currently ``'ar'`` is the only
        option).
    order : int, optional
        The order of the AR model to fit. Default is 2.
    random_seed : int, 'default', 'none' or None, optional
        The seed of the random draws used by ``rmgd`` (hctsa's ``BF_RandomSeed``): an integer,
        ``'default'`` or ``None`` for seed 0, or ``'none'`` for a seed drawn from NumPy's global
        random state. The draws come from the portable generator
        :func:`pyhctsa.robust.bf_random`, so they are hctsa's for the same seed.

    Returns
    -------
    dict
        The ratios of the RMS prediction error of the AR model for the preprocessed series to
        that for the original series: ``stderat_d1``, ``stderat_d2``, ``stderat_p1_20``,
        ``stderat_p2_5``, ``stderat_rmgd``. A ratio above 1 means the preprocessing left the
        series harder to predict, as when it removes a trend or slow dynamics that the model
        had been exploiting.
    """
    if model != 'ar':
        raise ValueError(f"Unknown model '{model}'")
    y = np.asarray(y, dtype=float).ravel()
    versions = {
        'nothing': y,
        'd1': np.diff(y, 1),
        'd2': np.diff(y, 2),
        'p1_20': _piecewise_poly_detrend(y, 1, 20),
        'p2_5': _piecewise_poly_detrend(y, 2, 5),
        'rmgd': _rank_map_gaussian(y, random_seed),
    }
    rms = {k: _ar_rms_error(z_score(v), order) for k, v in versions.items()}
    return {f'stderat_{k}': rms[k] / rms['nothing'] for k in ('d1', 'd2', 'p1_20', 'p2_5', 'rmgd')}


_NRLAZY_BOX = 512  # side of the hash grid used by TISEAN's nrlazy to find neighbors


@numba.njit(cache=True)
def _nrlazy_numba(x, m, d, num_iter, eps):
    """Schreiber's simple nonlinear noise reduction, a port of TISEAN's ``nrlazy`` (one component).

    ``x`` is the series rescaled to [0, 1] and ``eps`` the neighborhood radius in these units
    (maximum norm). Each iteration corrects every embedding vector to the mean of its neighbors
    (all vectors within ``eps`` in every coordinate, including itself), found as in the C code by
    hashing the first and last coordinates into a ``_NRLAZY_BOX``-squared grid. Each sample
    becomes the average of its corrections from the (up to m) vectors that contain it.
    Returns the new series and, for the last iteration, the number of neighbors of each vector
    (1 for the first (m-1)*d samples, which start no vector).
    """
    n = len(x)
    back = (m - 1) * d
    ibox = _NRLAZY_BOX - 1
    epsinv = 1.0 / eps
    nmf = np.ones(n, dtype=np.int64)
    for _ in range(num_iter):
        box = -np.ones((_NRLAZY_BOX, _NRLAZY_BOX), dtype=np.int64)
        nxt = np.zeros(n, dtype=np.int64)
        for i in range(back, n):
            bx = int(x[i] / eps) & ibox
            by = int(x[i - back] / eps) & ibox
            nxt[i] = box[bx, by]
            box[bx, by] = i
        corr = np.zeros(n)
        nf = np.zeros(n, dtype=np.int64)
        nmf[:] = 1
        hcor = np.zeros(m)
        for k in range(back, n):
            for q in range(m):
                hcor[q] = 0.0
            i = int(x[k] * epsinv) & ibox
            j = int(x[k - back] * epsinv) & ibox
            nfound = 0
            for i1 in range(i - 1, i + 2):
                i2 = i1 & ibox
                for j1 in range(j - 1, j + 2):
                    element = box[i2, j1 & ibox]
                    while element != -1:
                        q = 0
                        while q < m:
                            if abs(x[k - q * d] - x[element - q * d]) > eps:
                                break
                            q += 1
                        if q == m:
                            nfound += 1
                            for q in range(m):
                                hcor[q] += x[element - q * d]
                        element = nxt[element]
            for q in range(m):
                corr[k - q * d] += hcor[q] / nfound
                nf[k - q * d] += 1
            nmf[k] = nfound
        for k in range(n):
            if nf[k] > 0:
                x[k] = corr[k] / nf[k]
    return x, nmf


def _nrlazy(y: np.ndarray, m: int, d: int, num_iter: int, neighborhood_std: float):
    """Run ``_nrlazy_numba`` as TISEAN's ``nrlazy -m1,m -d -i -v`` does on the series ``y``.

    Returns ``(y_denoised, num_neighbors)``, or ``None`` where TISEAN would exit with an error
    (a constant series).
    """
    y = np.asarray(y, dtype=float)
    lo = y.min()
    interval = y.max() - lo
    if not interval > 0:
        return None
    x = (y - lo) / interval
    dvar = np.sqrt(abs(np.mean(x ** 2) - np.mean(x) ** 2))  # TISEAN 'variance': the standard deviation
    if not dvar > 0:
        return None
    x, nmf = _nrlazy_numba(x.copy(), int(m), int(d), int(num_iter), neighborhood_std * dvar)
    return x * interval + lo, nmf


def preproc_schreiber_denoise(y: ArrayLike, m: int = 5, d: int = 1, num_iter: int = 1,
                              neighborhood_std: float = 0.5) -> dict:
    """
    Nonlinear noise reduction, and how it changes the series.

    Applies Schreiber's simple nonlinear noise-reduction method, replacing each embedded point by
    the average of its state-space neighbors, as TISEAN's ``nrlazy`` (a C reimplementation that
    corrects the whole embedding vector, of the original single-component-correcting Fortran
    ``lazy``; it tends to do better than ``lazy`` on flow-like data). This is a Python (numba)
    port of ``nrlazy``, for a scalar time series. The method is that of

    Schreiber, T. "Extremely simple nonlinear noise reduction method", Phys. Rev. E 47, 2401
    (1993).

    It is the first stage of the pipeline of Toker et al., used for diagnosing oversampling and
    for classifying chaos with the 0-1 test.

    Denoising is not itself a scalar feature, so this function reports how several properties
    of the series change as a result of applying it.

    Parameters
    ----------
    y : array-like
        The input time series.
    m : int, optional
        The embedding dimension (nrlazy ``-m``). Default is 5.
    d : int, optional
        The embedding delay (nrlazy ``-d``). Default is 1.
    num_iter : int, optional
        The number of correction passes (nrlazy ``-i``). More iterations denoise more
        aggressively but risk distorting real dynamics; TISEAN's own default, and most published
        use, is 1.
    neighborhood_std : float, optional
        The neighborhood radius in units of the standard deviation of the data (nrlazy ``-v``;
        scale-invariant, unlike the raw ``-r`` option, which is a fixed fraction of the data
        interval). Default is 0.5.

    Returns
    -------
    dict
        - ``rmsCorrection``: the root-mean-square size of the correction made to the series
        - ``fracVarRemoved``: 1 - var(denoised) / var(original), the fraction of the variance
          attributed to noise and removed
        - ``corrOrigDenoised``: the correlation coefficient between the original and denoised series
        - ``ac1Change``: the lag-1 autocorrelation of the denoised series minus that of the
          original (denoising should smooth the series, increasing it)
        - ``meanNeighbors``: the mean number of neighbors per point found by nrlazy, counting
          the point itself
        - ``fracNoCorrection``: the fraction of points with a neighbor count of 1, that is, with no
          neighbors besides themselves, so that no correction was possible there (diagnoses whether
          ``neighborhood_std`` was too small for these data)
        - ``KDenoised``: the statistic K of the 0-1 test for chaos (``zero_one_test``) computed on
          the denoised series, since measurement noise inflates the apparent diffusion of the
          test's (p, q) trajectory
        - ``KChange``: ``KDenoised`` minus K of the original series: does removing noise change
          the chaos verdict for this series?

        All values are NaN if the series is too short for the embedding, or if nrlazy cannot
        process it (a constant series). The 0-1 test needs at least 200 samples, so ``KDenoised``
        and ``KChange`` are NaN for shorter series.
    """
    keys = ['rmsCorrection', 'fracVarRemoved', 'corrOrigDenoised', 'ac1Change', 'meanNeighbors',
            'fracNoCorrection', 'KDenoised', 'KChange']
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    nan_out = {k: np.nan for k in keys}

    # Need enough points for an embedding vector, with margin for the local statistics
    if N < max(50, 10 * ((m - 1) * d + 1)):
        logger.warning(f"Time series too short to denoise at m={m}, d={d}")
        return nan_out

    res = _nrlazy(y, m, d, num_iter, neighborhood_std)
    if res is None or not np.all(np.isfinite(res[0])):
        logger.warning("nrlazy cannot denoise this series")
        return nan_out
    y_den, num_neighbors = res

    out = {}
    out['rmsCorrection'] = float(np.sqrt(np.mean((y - y_den) ** 2)))
    out['fracVarRemoved'] = float(1 - np.var(y_den, ddof=1) / np.var(y, ddof=1))
    out['corrOrigDenoised'] = float(np.corrcoef(y, y_den)[0, 1])
    out['ac1Change'] = autocorr(y_den, 1, 'Fourier') - autocorr(y, 1, 'Fourier')
    out['meanNeighbors'] = float(np.mean(num_neighbors))
    out['fracNoCorrection'] = float(np.mean(num_neighbors == 1))

    # Does denoising change the 0-1-test chaos verdict?
    k_orig = zero_one_test(y, 20)['K']
    k_den = zero_one_test(y_den, 20)['K']
    out['KDenoised'] = float(k_den)
    out['KChange'] = float(k_den - k_orig)
    return out
