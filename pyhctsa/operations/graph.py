from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import scipy
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.stats import expon, gumbel_l, norm
from ts2vg import NaturalVG
import logging
logger = logging.getLogger('pyhctsa')

from pyhctsa.operations.correlation import autocorr, first_crossing
from pyhctsa.utils import bin_picker
from pyhctsa.toolboxes.distribution_fits.distfits import evfit


def _hvg_links(x: np.ndarray) -> tuple:
    """
    Nearest neighbour at least as tall to the left and to the right of every node.

    Returns
    -------
    prev, nxt : ndarray of intp, shape (N,)
        ``prev[i]`` is the largest ``j < i`` with ``x[j] >= x[i]``, or -1 if none.
        ``nxt[i]`` is the smallest ``j > i`` with ``x[j] >= x[i]``, or -1 if none.

    Notes
    -----
    Two monotonic-stack passes, O(N) total.
    Ties must terminate the search (``>=``, not ``>``), otherwise equal-valued
    points are never linked and pairs separated by an equal-valued intermediate
    are linked wrongly; with ``>=`` the links reproduce the horizontal
    visibility graph (``x[k] < min(x[i], x[j])`` for all intermediate k).
    NaN nodes neither block visibility nor count as taller neighbours, matching
    the all-False semantics of ``slice >= nan``.
    """
    N = x.shape[0]
    prev = np.full(N, -1, dtype=np.intp)
    nxt = np.full(N, -1, dtype=np.intp)
    if N == 0:
        return prev, nxt

    # list access is markedly faster than repeated numpy scalar indexing
    xl = x.tolist()
    stack = []

    for i in range(N):
        v = xl[i]
        if v != v:  # NaN: never taller, never occluding
            continue
        while stack and not (xl[stack[-1]] >= v):
            stack.pop()
        if stack:
            prev[i] = stack[-1]
        stack.append(i)

    stack.clear()
    for i in range(N - 1, -1, -1):
        v = xl[i]
        if v != v:
            continue
        while stack and not (xl[stack[-1]] >= v):
            stack.pop()
        if stack:
            nxt[i] = stack[-1]
        stack.append(i)

    return prev, nxt


def _horiz_vgraph_degrees(ts_data: ArrayLike) -> np.ndarray:
    """
    Degree sequence of the horizontal visibility graph, without materialising
    the N x N adjacency matrix.

    The forward and backward link sets overlap only for pairs of equal value
    (``nxt[i] == j`` requires ``x[j] >= x[i]`` while ``prev[j] == i`` requires
    ``x[i] >= x[j]``), and those are found by both passes, so the backward
    copy is dropped. Then no edge is double counted and a bincount over edge
    endpoints gives the degrees exactly.
    """
    x = np.asarray(ts_data)
    N = x.shape[0]
    if N < 2:
        return np.zeros(N, dtype=np.int64)

    prev, nxt = _hvg_links(x)
    pm = prev >= 0
    nm = nxt >= 0
    # equal-valued pairs are found by both passes: keep only the forward copy
    eq = np.zeros(N, dtype=bool)
    eq[pm] = x[prev[pm]] == x[pm]
    pm = pm & ~eq
    endpoints = np.concatenate((
        np.flatnonzero(pm), prev[pm],
        np.flatnonzero(nm), nxt[nm],
    ))
    return np.bincount(endpoints, minlength=N)


def _runstest_pvalue(x: np.ndarray) -> float:
    """
    Exact two-sided p-value of the runs test about the mean, as MATLAB's ``runstest(x)``.

    Values equal to the mean are omitted. The exact distribution of the number
    of runs above/below the mean is used (MATLAB's default for this test).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 1.0
    v = np.mean(x)
    x = x[x != v]
    N = x.size
    above = x > v
    n1 = int(np.sum(above))
    n0 = N - n1
    if n1 == 0 or n0 == 0:
        return 1.0  # exactly one run
    nruns = 1 + int(np.sum(above[:-1] != above[1:]))

    def log_choose(n, k):
        if k < 0 or k > n:
            return -np.inf
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def prob(r):
        denom = log_choose(N, n0)
        if r % 2 == 0:
            k = r // 2
            return 2 * np.exp(log_choose(n1 - 1, k - 1) + log_choose(n0 - 1, k - 1) - denom)
        k = r // 2
        return (np.exp(log_choose(n1 - 1, k - 1) + log_choose(n0 - 1, k) - denom)
                + np.exp(log_choose(n1 - 1, k) + log_choose(n0 - 1, k - 1) - denom))

    plist = np.array([prob(r) for r in range(1, 2 * min(n1, n0) + 2)])
    pexact = plist[nruns - 1]
    plo = np.sum(plist[:nruns - 1])
    phi = np.sum(plist[nruns:])
    return float(min(1.0, 2 * (pexact + min(plo, phi))))


_SIMPLE_FIT_NAN = {'r2': np.nan, 'adjr2': np.nan, 'rmse': np.nan,
                   'resAC1': np.nan, 'resAC2': np.nan, 'resruns': np.nan}


def _simple_fit(x: np.ndarray, dmodel: str, num_bins: int) -> dict:
    """
    Fit a simple curve to the histogram of the values of x (hctsa's ``DN_SimpleFit``).

    The histogram of x with ``num_bins`` equal-width bins is normalized to a
    probability density, and a single Gaussian (``'gauss1'``, ``a*exp(-((x-b)/c)^2)``),
    exponential (``'exp1'``, ``a*exp(b*x)``) or power law (``'power1'``, ``a*x^b``)
    is fitted by nonlinear least squares. Returns the goodness-of-fit outputs
    (r2, adjr2, rmse, resAC1, resAC2, resruns), all NaN if the model cannot
    be fitted.

    The rmse is in units of probability density of the standardized series
    (rmse of the fit multiplied by the standard deviation of x).
    """
    nan_out = dict(_SIMPLE_FIT_NAN)
    num_bins = int(num_bins)
    if num_bins < 1 or x.size < 2:
        return nan_out

    counts, edges = np.histogram(x, bins=num_bins)
    dnx = (edges[:-1] + edges[1:]) / 2
    dny = counts / (np.sum(counts) * np.mean(np.diff(edges)))

    if dmodel == 'gauss1':
        def model(t, a, b, c):
            return a * np.exp(-((t - b) / c) ** 2)
        width = max(dnx[-1] - dnx[0], np.finfo(float).eps)
        i0 = int(np.argmax(dny))
        wmean = np.sum(dnx * dny) / np.sum(dny) if np.sum(dny) > 0 else dnx[i0]
        starts = [(dny[i0], b, c) for b in (dnx[i0], wmean) for c in (width / 8, width / 4, width / 2, width)]
    elif dmodel == 'exp1':
        def model(t, a, b):
            return a * np.exp(b * t)
        pos = dny > 0
        starts = []
        if np.sum(pos) >= 2:
            slope, icpt = np.polyfit(dnx[pos], np.log(dny[pos]), 1)
            starts.append((np.exp(icpt), slope))
        starts += [(np.max(dny), 0.0), (np.max(dny), -1.0 / max(np.ptp(dnx), 1e-12)),
                   (np.max(dny), 1.0 / max(np.ptp(dnx), 1e-12))]
    elif dmodel == 'power1':
        if np.any(dnx <= 0):
            return nan_out  # power functions cannot be fit to non-positive x
        def model(t, a, b):
            return a * t ** b
        pos = dny > 0
        starts = []
        if np.sum(pos) >= 2:
            slope, icpt = np.polyfit(np.log(dnx[pos]), np.log(dny[pos]), 1)
            starts.append((np.exp(icpt), slope))
        starts += [(np.max(dny), -1.0), (np.max(dny), 1.0), (np.mean(dny), 0.0)]
    else:
        raise ValueError(f"Invalid distribution model '{dmodel}' specified")

    nparams = len(starts[0])
    if len(dnx) < nparams:
        return nan_out  # fewer data points than coefficients
    # Bound the Gaussian width away from zero: its sign is irrelevant and c -> 0 is degenerate
    lower = [-np.inf, -np.inf, 1e-12] if dmodel == 'gauss1' else [-np.inf] * nparams
    best_sse, best_p = np.inf, None
    for p0 in starts:
        try:
            with np.errstate(all='ignore'):
                sol = least_squares(lambda p: model(dnx, *p) - dny, p0, bounds=(lower, np.inf),
                                    method='trf', x_scale='jac', xtol=1e-12, ftol=1e-12,
                                    gtol=1e-12, max_nfev=2000)
                sse = np.sum((dny - model(dnx, *sol.x)) ** 2)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        if np.isfinite(sse) and sse < best_sse:
            best_sse, best_p = sse, sol.x
    if best_p is None:
        return nan_out

    res = dny - model(dnx, *best_p)
    n = len(dnx)
    dfe = n - nparams
    sst = np.sum((dny - np.mean(dny)) ** 2)
    out = dict(nan_out)
    out['r2'] = 1 - best_sse / sst if sst > 0 else np.nan
    out['adjr2'] = 1 - (1 - out['r2']) * (n - 1) / dfe if dfe > 0 else np.nan
    out['rmse'] = np.sqrt(best_sse / dfe) * np.std(x, ddof=1) if dfe > 0 else np.nan
    out['resAC1'] = autocorr(res, 1, 'Fourier')[0]
    out['resAC2'] = autocorr(res, 2, 'Fourier')[0]
    out['resruns'] = _runstest_pvalue(res)
    return out


def _degree_entropy(k: np.ndarray) -> float:
    """
    Entropy of the histogram of k with MATLAB's ``'sqrt'`` binning rule (hctsa's
    ``EN_DistributionEntropy(k, 'hist', 'sqrt')``), in nats, with the Miller-Madow correction.

    MATLAB's ``histcounts(..., 'BinMethod', 'sqrt')`` picks ``ceil(sqrt(N))`` bins' worth
    of width, then rounds the width and the edges to "nice" values (the ``binpicker``
    rule), which for integer-valued degrees differs from NumPy's ``'sqrt'`` rule.
    """
    n = len(k)
    nbins = max(int(np.ceil(np.sqrt(n))), 1)
    xmin, xmax = np.float64(np.min(k)), np.float64(np.max(k))
    edges = bin_picker(xmin, xmax, None, (xmax - xmin) / nbins)
    counts, _ = np.histogram(k, bins=edges)
    px = counts / np.sum(counts)
    bin_widths = np.diff(edges)
    pos = px > 0
    out = -np.sum(px[pos] * np.log(px[pos] / bin_widths[pos]))
    return out + (np.count_nonzero(pos) - 1) / (2 * n)


def visibility_graph(y: ArrayLike, meth: str = 'horiz', max_l: Union[int, str] = 20000) -> dict:
    """
    Visibility graph analysis of a time series.

    Constructs a visibility graph of the time series, with one node per sample,
    and returns statistics on the distribution of the number of links per node
    (the degree). In the natural visibility graph (``'norm'``), two samples are
    linked if the straight line between them passes above every sample in between.
    In the horizontal visibility graph (``'horiz'``), they are linked if a
    horizontal line between them passes above every sample in between.
    The outputs summarize the degrees (mode, mean, spread, extremes, heaviness of the
    upper tail), the entropy of their histogram, fits of Gaussian, exponential and
    power-law curves to that histogram and of an extreme-value distribution to the
    degrees, and the autocorrelation of the sequence of degrees taken in time order.
    cf. [1] and [2].

    References
    ----------
    .. [1] "From time series to complex networks: The visibility graph"
            Lacasa, Lucas and Luque, Bartolo and Ballesteros, Fernando and Luque, Jordi
            and Nuno, Juan Carlos P. Natl. Acad. Sci. USA. 105(13) 4972 (2008)
    .. [2] "Horizontal visibility graphs: Exact results for random time series"
            Luque, B. and Lacasa, L. and Ballesteros, F. and Luque, J.
            Phys. Rev. E. 80(4) 046103 (2009)
    
    Parameters
    ----------
    y : array-like
        Input time series
    meth : str, optional
        Method for constructing the visibility graph:

        - 'horiz': Uses horizontal visibility (only horizontal lines link nodes)
        - 'norm': Uses natural visibility (standard visibility definition)

        Default is ``'horiz'``.

    max_l : int or str, optional
        Maximum number of samples to analyze. Longer time series are truncated
        to first max_l points. Set to ``'full'`` to analyze the entire time series
        with no cropping (a warning is logged if the series exceeds 50000 samples
        and ``meth`` is ``'norm'``, since the natural visibility graph may be slow).
        Only the degrees are computed, with no adjacency matrix stored, so memory is
        not a concern. Default is 20000.

    Returns
    -------
    dict
        Statistics on the degree distribution:

        - ``modek``, ``propmode``: the most common degree, and the proportion of nodes
          that have it
        - ``meank``, ``mediank``, ``stdk``: mean, median and standard deviation of the degrees
        - ``maxk``, ``mink``, ``rangek``, ``iqrk``: maximum, minimum, range and
          interquartile range of the degrees
        - ``skewnessk``: skewness of the degrees
        - ``maxonmedian``: maximum degree divided by the median degree
        - ``ol90``: mean of the degrees between the 5th and 95th percentiles,
          divided by the mean of all degrees
        - ``olu90``: how far the mean of the top 5% of degrees lies above the overall mean,
          in standard deviations of the degrees
        - ``dgaussk_*``, ``dexpk_*``, ``dpowerk_*`` (``r2``, ``adjr2``, ``rmse``,
          ``resAC1``, ``resAC2``, ``resruns``): goodness of fit and residual tests for a
          single Gaussian, exponential and power law fitted to the histogram of degrees
          (with as many bins as the range of the degrees; the rmse is in units of
          probability density of the degrees divided by their standard deviation)
        - ``gaussnlogL``, ``expnlogL``: mean negative log-likelihood per node of a
          Gaussian and of an exponential distribution fitted to the degrees
        - ``evparam1``, ``evparam2``, ``evnlogL``: location and scale of an extreme-value
          distribution fitted to the degrees, and its mean negative log-likelihood per node
        - ``entropy``: entropy of the histogram of degrees (square-root binning), in nats
        - ``kac1``, ``kac2``, ``kac3``: autocorrelation of the degree sequence (in time
          order) at lags 1, 2 and 3
        - ``ktau``: lag at which the autocorrelation of the degree sequence first
          crosses zero (interpolated)
    """
    y = np.asarray(y)
    N = len(y)
    if isinstance(max_l, str):
        if max_l != 'full':
            raise ValueError(f"Unknown max_l '{max_l}'; use an integer or 'full'.")
        # no cropping, but flag potentially slow computations for very long series
        if N > 50000 and meth == 'norm':
            logger.warning(f"Time series ({N} samples) exceeds 50000 with max_l='full'; "
                           "visibility graph computation may be slow")
    elif N > max_l:
        logger.info(f"Time series ({N} > {max_l}) is too long for visibility graph."
              f"Analyzing the first {max_l} samples.")
        y = y[:max_l]
    y = y - np.min(y) # adjust so that the minimum of y is at zero

    # Compute the visibility graph (degrees only; no adjacency matrix is stored):
    if meth == 'horiz':
        # O(N) time and memory
        k = _horiz_vgraph_degrees(y)
    elif meth == 'norm':
        vg = NaturalVG()
        vg.build(y, only_degrees=True)
        k = vg._degrees
    else:
        raise ValueError(f"Unknown visibility graph method '{meth}'")

    # statistics of k are reused throughout; compute each exactly once
    meank = np.mean(k)
    stdk = np.std(k, ddof=1)
    mediank = np.median(k)
    maxk = np.max(k)
    q05, q25, q75, q95 = np.quantile(k, [0.05, 0.25, 0.75, 0.95], method='hazen')

    out = {}
    # Degree distribution: basic statistics
    out['modek'] = scipy.stats.mode(k).mode
    out['propmode'] = np.sum(k == out['modek'])/len(k)
    out['meank'] = meank # mean number of links per node
    out['mediank'] = mediank
    out['stdk'] = stdk
    out['maxk'] = maxk
    out['mink'] = np.min(k)
    out['rangek'] = np.ptp(k)
    out['iqrk'] = q75 - q25
    out['skewnessk'] = scipy.stats.skew(k)
    out['maxonmedian'] = maxk/mediank # max on median (indicator of outlier)
    out['ol90'] = np.mean(k[(k >= q05) & (k <= q95)])/meank
    out['olu90'] = np.mean(k[k >= q95] - meank)/stdk

    # Fit distributions to the degree distribution (histogram with range(k) bins)
    kf = k.astype(float)
    for prefix, dmodel in (('dgaussk', 'gauss1'), ('dexpk', 'exp1'), ('dpowerk', 'power1')):
        fit = _simple_fit(kf, dmodel, int(np.ptp(k)))
        for name in ('r2', 'adjr2', 'rmse', 'resAC1', 'resAC2', 'resruns'):
            out[f'{prefix}_{name}'] = fit[name]

    # Likelihood: mean negative log-likelihood per node (the summed value is proportional
    # to the number of nodes)
    out['gaussnlogL'] = -np.mean(norm.logpdf(k, loc=meank, scale=stdk))
    out['expnlogL'] = -np.mean(expon.logpdf(k, scale=meank))

    # Extreme value distribution (type I, minima: as MATLAB's evfit)
    mu, sigma = evfit(kf)
    out['evparam1'] = mu
    out['evparam2'] = sigma
    out['evnlogL'] = -np.mean(gumbel_l.logpdf(kf, loc=mu, scale=sigma))

    # Entropy of the degree distribution
    out['entropy'] = _degree_entropy(kf)

    # Autocorr
    out['kac1'] = autocorr(k, 1, 'Fourier')[0]
    out['kac2'] = autocorr(k, 2, 'Fourier')[0]
    out['kac3'] = autocorr(k, 3, 'Fourier')[0]
    out['ktau'] = first_crossing(k, 'ac', 0, 'continuous')

    return out
