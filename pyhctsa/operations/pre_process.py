import re
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import make_lsq_spline
from scipy.optimize import least_squares
from scipy.signal import lfilter, resample_poly
from scipy.special import gammaln
from scipy.stats import boxcox, norm
import logging
logger = logging.getLogger('pyhctsa')

from ..operations.correlation import autocorr
from ..operations.distribution import compare_ks_fit, outlier_test
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


def _runstest_p(x: np.ndarray) -> float:
    """Two-sided p-value of MATLAB's ``runstest(x)`` (exact distribution, mean cutoff).

    Counts runs of values above and below the mean; values equal to the mean are dropped.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    v = np.mean(x) if x.size else np.nan
    x = x[x != v]
    N = x.size
    b = x > v
    n1 = int(b.sum())
    n0 = N - n1
    if N == 0:
        return 1.0
    nruns = 1 + int(np.sum(b[:-1] != b[1:]))
    if n1 == 0 or n0 == 0:
        plist = np.array([1.0])  # exactly one run is possible
    else:
        def lnck(a, k):
            a = np.asarray(a, dtype=float)
            k = np.asarray(k, dtype=float)
            with np.errstate(invalid='ignore'):
                out = gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
            return np.where((k < 0) | (k > a), -np.inf, out)
        maxruns = 2 * min(n1, n0) + 1
        R = np.arange(1, maxruns + 1)
        plist = np.zeros(maxruns)
        ev = R % 2 == 0
        k = R[ev] // 2
        plist[ev] = 2 * np.exp(lnck(n1 - 1, k - 1) + lnck(n0 - 1, k - 1) - lnck(N, n0))
        k = R[~ev] // 2
        plist[~ev] = (np.exp(lnck(n1 - 1, k - 1) + lnck(n0 - 1, k) - lnck(N, n0))
                      + np.exp(lnck(n1 - 1, k) + lnck(n0 - 1, k - 1) - lnck(N, n0)))
    pexact = plist[nruns - 1]
    plo = plist[:nruns - 1].sum()
    phi = plist[nruns:].sum()
    return float(min(1.0, 2 * (pexact + min(plo, phi))))


def _ksdensity(x: np.ndarray, m: int = 100) -> tuple:
    """Normal-kernel density estimate of x on m points, like MATLAB's ``[f, xi] = ksdensity(x)``.

    The bandwidth comes from the median absolute deviation (Silverman's rule) and the grid
    covers the data range extended by 3 bandwidths, as in ``ksdensity``. (MATLAB truncates
    the kernel at 4 bandwidths for large samples; this sums the full kernel.)
    """
    n = len(x)
    sig = np.median(np.abs(x - np.median(x))) / 0.6745
    if sig <= 0:
        sig = np.ptp(x)
    bw = sig * (4 / (3 * n)) ** (1 / 5)
    xi = np.linspace(np.min(x) - 3 * bw, np.max(x) + 3 * bw, m)
    f = np.mean(norm.pdf((xi[:, None] - x[None, :]) / bw), axis=1) / bw
    return xi, f


def _gauss1_kd_fit(x: np.ndarray) -> dict:
    """Fit a Gaussian to the kernel-smoothed distribution of x: hctsa's ``DN_SimpleFit(x, 'gauss1', 0)``.

    The distribution is a normal-kernel density estimate on 100 points (MATLAB's
    ``ksdensity`` defaults, see ``_ksdensity``); the model is a1*exp(-((u - b1)/c1)^2),
    fitted by least squares from the start point of MATLAB's ``gaussnstart``. Returns the R^2 (``r2``), the lag-1
    autocorrelation of the residuals (``resAC1``) and the p-value of a runs test on the
    residuals (``resruns``), or NaN if the fit fails.
    """
    xi, f = _ksdensity(x)

    # Start point (gaussnstart, one peak)
    k = np.nonzero(f == f.max())[0][-1]
    a0, b0 = f[k], xi[k]
    ok = (f > 0) & (f < a0)
    if not ok.any():
        return np.nan
    c0 = np.mean(np.abs(xi[ok] - b0) / np.sqrt(np.log(a0 / f[ok]))) / 2

    def basis(p):
        b, c = p
        return np.ones_like(xi) if c == 0 else np.exp(-((xi - b) / c) ** 2)

    def resid(p):  # separable least squares: the amplitude is linear
        A = basis(p)
        return f - A * (A @ f) / (A @ A)

    try:
        sol = least_squares(resid, [b0, c0], method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14)
        res = resid(sol.x)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.nan
    if not np.all(np.isfinite(res)):
        return np.nan
    sst = np.sum((f - np.mean(f)) ** 2)
    return {'r2': 1 - np.sum(res ** 2) / sst,
            'resAC1': float(np.ravel(autocorr(res, 1, 'Fourier'))[0]),
            'resruns': _runstest_p(res)}


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
        - ``kscn_olapint``: the overlap integral of the kernel-smoothed distribution
          with the best-fitting normal (``compare_ks_fit``)
        - ``olbt_s5``: the standard deviation after trimming the 5% most extreme values
          at each end, relative to that of the full series (``outlier_test``)

        Differences (statistics that can be negative or zero):

        - ``gauss1_kd_r2``, ``gauss1_kd_resAC1``, ``gauss1_kd_resruns``: the R^2, the
          lag-1 autocorrelation of the residuals, and the runs-test p-value of the
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

    # 2) Gaussianity
    # (a) Gaussian fit to the kernel density estimate
    me1 = _gauss1_kd_fit(y_d)
    me2 = _gauss1_kd_fit(y)
    if not isinstance(me1, dict) or not isinstance(me2, dict):
        # fitting the Gaussian failed
        for key in ['r2', 'resAC1', 'resruns']:
            out[f'gauss1_kd_{key}'] = np.nan
    else:
        for key in ['r2', 'resAC1', 'resruns']:
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
