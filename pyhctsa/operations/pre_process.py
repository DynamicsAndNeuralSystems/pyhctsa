import re
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import make_lsq_spline
from scipy.optimize import least_squares
from scipy.signal import lfilter, resample_poly
from scipy.stats import boxcox
import logging
logger = logging.getLogger('pyhctsa')

from ..operations.distribution import outlier_test
from ..operations.stationarity import sliding_window, stat_av
from ..utils import _round_half_away, z_score

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


def _sin_start(y: np.ndarray, n: int) -> np.ndarray:
    """Start frequencies of a sum of n sinusoids, as MATLAB's ``sinnstart``.

    One frequency at a time: the peak of the FFT magnitude of the residuals of the fit so
    far, ignoring the peaks already used.
    """
    N = len(y)
    t = np.arange(1, N + 1, dtype=float)
    freqs, used, res = [], [], y.copy()
    for j in range(n):
        fy = np.abs(np.fft.fft(res))
        fy[used] = 0
        m = int(np.argmax(fy[:N // 2]))  # 0-based; MATLAB's maxloc is m + 1
        used.append(m)
        freqs.append(2 * np.pi * max(0.5, m) / (N - 1))
        X = np.column_stack([f(w * t) for w in freqs for f in (np.sin, np.cos)])
        ab = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ ab
    return np.array(freqs)


def _sin_detrend(y: np.ndarray, n: int) -> np.ndarray:
    """Remove a sum of n sinusoids a1*sin(b1*t + c1) + ... fitted to y against t = 1:N.

    Mirrors MATLAB's ``fit(t, y, 'sin<n>')``: the amplitude and phase of each sinusoid
    are linear coefficients (the sine and cosine weights), solved by least squares, and
    only the frequencies are optimized, starting from the FFT peaks of the series
    (``sinnstart``). The result is therefore the local least-squares fit nearest to that
    start, like MATLAB's, not necessarily the global one. Returns NaN if the fit fails.
    """
    N = len(y)
    t = np.arange(1, N + 1, dtype=float)

    def design(b):
        return np.column_stack([f(w * t) for w in b for f in (np.sin, np.cos)])

    def resid(b):
        X = design(b)
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    try:
        b0 = _sin_start(y, n)
        sol = least_squares(resid, b0, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14)
        r = resid(sol.x)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    if not np.all(np.isfinite(r)):
        return np.nan
    return r


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
        - ``"sin<n>"``   : remove a sum of n = 1-8 sinusoids a1*sin(b1*t + c1) + ...,
          fitted against the time index, e.g., ``"sin1"``
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
        - ``olbt_s5``: the standard deviation after trimming the 5% most extreme values
          at each end, relative to that of the full series (``outlier_test``)

        Differences (statistics that can be negative or zero):

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
        y_d = _sin_detrend(y, int(m.group(1)))
        if np.ndim(y_d) == 0:  # the fit failed
            return np.nan

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

    # 3) Outliers
    out['olbt_m2'] = _diff(outlier_test(y_d, 2, 'mean'), outlier_test(y, 2, 'mean'))
    out['olbt_m5'] = _diff(outlier_test(y_d, 5, 'mean'), outlier_test(y, 5, 'mean'))
    out['olbt_s5'] = _norm_diff(outlier_test(y_d, 5, 'std'), outlier_test(y, 5, 'std'))

    return out
