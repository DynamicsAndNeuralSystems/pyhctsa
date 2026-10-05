from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import scipy
from numba import njit
from scipy.stats import expon, gumbel_l, norm
import logging
from math import factorial

logger = logging.getLogger('pyhctsa')

from pyhctsa.operations.correlation import autocorr, first_crossing
from pyhctsa.robust import bf_fit_density_curve, bf_residual_stats
from pyhctsa.utils import get_tau, time_delay_embed
from pyhctsa.operations.entropy import _ordinal_pattern_rank, distribution_entropy
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


@njit(cache=True)
def _natural_vg_degrees(y: np.ndarray) -> np.ndarray:
    """
    Degree sequence of the natural visibility graph, without an adjacency matrix.

    A forward sweep from each node i keeps the largest slope seen so far: node j is
    visible from i iff its slope ``(y[j] - y[i])/(j - i)`` exceeds that of every node in
    between. Slopes that agree to within rounding error (collinear nodes, as for tied or
    quantized values) are treated as equal, which blocks the view, so the graph does not
    depend on how a slope was rounded: a slope must exceed the running maximum by a
    relative 1e-12. Once the running maximum slope ``m`` is positive, nodes beyond
    distance ``(max(y) - y[i])/m`` would have to lie above ``max(y)``, so the scan stops
    early.
    """
    n = y.shape[0]
    k = np.zeros(n, dtype=np.int64)
    ymax = np.max(y)
    vis_tol = 1e-12  # relative tolerance of the visibility test
    for i in range(n - 1):
        yi = y[i]
        m = y[i + 1] - yi  # largest slope from i seen so far: the neighbor is always visible
        k[i + 1] += 1
        k[i] += 1
        jlim = n - 1  # last node that can still be visible
        if m > 0:
            reach = np.floor((ymax - yi) / m)
            if reach < n:
                jlim = min(n - 1, i + int(reach) + 1)
        j = i + 2
        while j <= jlim:
            sj = (y[j] - yi) / (j - i)
            if sj > m + vis_tol * abs(m):
                m = sj
                k[j] += 1
                k[i] += 1
                if m > 0:
                    reach = np.floor((ymax - yi) / m)
                    if reach < n:
                        jlim = min(n - 1, i + int(reach) + 1)
            j += 1
    return k


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
          ``resAC1``, ``resAC2``, ``resrunsz``): goodness of fit, autocorrelation of the
          residuals at lags 1 and 2, and the signed z-statistic of a runs test on the
          residuals (:func:`pyhctsa.robust.bf_runs_z`), for a single Gaussian, exponential
          and power law fitted by deterministic least squares
          (:func:`pyhctsa.robust.bf_fit_density_curve`) to the distribution of degrees (the
          proportion of nodes at each integer degree, from the minimum to the maximum
          degree; the rmse is in units of probability density of the degrees divided by
          their standard deviation, so it does not depend on the number of nodes). Each is
          NaN if the degrees take no more distinct values than the fit has parameters
          (3 for the Gaussian, 2 for the others), or if the fit is exact
        - ``gaussnlogL``, ``expnlogL``: mean negative log-likelihood per node of a
          Gaussian and of an exponential distribution fitted to the degrees
        - ``evparam1``, ``evparam2``, ``evnlogL``: location and scale of an extreme-value
          distribution fitted to the degrees, and its mean negative log-likelihood per node
        - ``entropy``: entropy of the histogram of degrees (square-root bin rule with explicit
          edges), in nats
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
        k = _natural_vg_degrees(np.asarray(y, dtype=np.float64))
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

    # Distribution of the degrees: the proportion of nodes at each integer degree from the
    # minimum to the maximum (bins of width 1, so proportions are probability densities)
    kf = k.astype(float)
    k_vals = np.arange(np.min(k), np.max(k) + 1, dtype=float)
    k_prob = np.bincount(k - np.min(k), minlength=len(k_vals)) / len(k)

    # Least-squares fits of a Gaussian, an exponential and a power law to the distribution
    # (deterministic: pyhctsa.robust.bf_fit_density_curve)
    for prefix, curve, num_params in (('dgaussk', 'gauss', 3), ('dexpk', 'exp', 2), ('dpowerk', 'power', 2)):
        if np.sum(k_prob > 0) <= num_params:  # too few distinct degrees to fit this model meaningfully
            r2 = adjr2 = rmse = res_ac1 = res_ac2 = res_runsz = np.nan
        else:
            k_fit = bf_fit_density_curve(k_vals, k_prob, curve)
            res = k_prob - k_fit  # residuals, in order of increasing degree
            sse = np.sum(res ** 2)
            sstot = np.sum((k_prob - np.mean(k_prob)) ** 2)
            dfe = len(k_vals) - num_params  # degrees of freedom of the error
            with np.errstate(all='ignore'):
                r2 = 1 - sse / sstot
                adjr2 = 1 - (1 - r2) * (len(k_vals) - 1) / dfe
            rmse = np.sqrt(sse / dfe) * stdk  # in density units of the standardized degrees
            res_ac1, res_ac2, res_runsz = bf_residual_stats(res, sstot)
        out[f'{prefix}_r2'] = r2
        out[f'{prefix}_adjr2'] = adjr2
        out[f'{prefix}_rmse'] = rmse
        out[f'{prefix}_resAC1'] = res_ac1
        out[f'{prefix}_resAC2'] = res_ac2
        out[f'{prefix}_resrunsz'] = res_runsz

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
    out['entropy'] = distribution_entropy(kf, 'hist', 'sqrt')  # NaN for a constant degree sequence

    # Autocorr
    out['kac1'] = autocorr(k, 1, 'Fourier')
    out['kac2'] = autocorr(k, 2, 'Fourier')
    out['kac3'] = autocorr(k, 3, 'Fourier')
    out['ktau'] = first_crossing(k, 'ac', 0, 'continuous')

    return out


def ordinal_partition_network(y: ArrayLike, d: int = 3, tau: Union[int, str] = 1) -> dict:
    """
    Ordinal partition transition network measures.

    Symbolizes the time series into ordinal patterns (Bandt-Pompe) and builds a
    directed transition network in which nodes are the ordinal patterns actually
    observed and edges connect a pattern to whichever pattern immediately follows it
    in time. Network-topological measures of this ordinal partition transition
    network are then computed. The network is unweighted: an edge exists if the
    transition occurs at least once (self-transitions included).

    Entropy is not recomputed here, to avoid duplicating
    :func:`pyhctsa.operations.entropy.permutation_entropy`, which computes
    entropy-based measures on the same symbolization; Kulp et al. found entropy to
    be a weaker discriminator than the network measures computed below.

    References
    ----------
    .. [1] C.W. Kulp, J.M. Chobot, H.R. Freitas and G.D. Sprechini, "Using ordinal
        partition transition networks to analyze ECG data", Chaos 26(7), 073114 (2016).
    .. [2] M. McCullough, M. Small, T. Stemler and H.H.-C. Iu, "Time lagged ordinal
        partition networks for capturing dynamics of continuous dynamical systems",
        Chaos 25(5), 053101 (2015). The original (weighted) ordinal partition
        transition network, of which the unweighted version of Kulp et al. (used
        here) is a variant.
    .. [3] C. Bandt and B. Pompe, "Permutation entropy: a natural complexity measure
        for time series", Phys. Rev. Lett. 88(17), 174102 (2002). The underlying
        ordinal-pattern symbolization.

    Parameters
    ----------
    y : array-like
        The input time series.
    d : int, optional
        The ordinal pattern (embedding) dimension: windows of ``d`` consecutive
        (delay-``tau``-spaced) points are each mapped to their rank permutation, one
        of ``d!`` possible ordinal patterns. Default is 3.
    tau : int or str, optional
        The time delay: an integer number of samples, or a rule understood by
        :func:`pyhctsa.utils.get_tau` (``'ac'``, the first zero-crossing of the
        autocorrelation function; ``'ac1e'``, the floor of its first 1/e crossing;
        or ``'mi'``, the smaller of the first minimum of the Kraskov automutual
        information and the ``'ac1e'`` delay), as in the time-lagged networks of
        McCullough et al. Default is 1, as used throughout Kulp et al.

    Returns
    -------
    dict or float
        NaN if the embedding fails (e.g. the delay cannot be set) or there are fewer
        than 30 embedded points. Otherwise a dictionary with:

        - ``meanDegree``: the mean degree (average number of unique out-edges per
          visited node, ``m / n`` for ``m`` unique edges and ``n`` nodes; equal to the
          mean in-degree), the paper's central discriminating measure
        - ``NFP``: the number of forbidden (non-occurring) ordinal patterns, ``d!``
          minus the number of nodes
        - ``maxOutDegree``, ``maxInDegree``: the largest out-degree and in-degree
        - ``stdOutDegree``, ``stdInDegree``: the standard deviation of the out- and
          in-degrees
        - ``reciprocity``: the fraction of unique edges whose reverse edge also exists
        - ``maxEdgeWeight``: the proportion of all observed transitions taken up by the
          most frequent single transition

        (The maximum and spread of the degrees, reciprocity and edge-weight measures
        are not reported in the paper but are cheaply available from the same
        transition-pair computation.)
    """
    y = np.asarray(y, dtype=float).ravel()
    d = int(d)

    tau = get_tau(y, tau)
    if np.isnan(tau):
        logger.warning('Embedding failed (could not set the time delay)')
        return np.nan
    try:
        X = time_delay_embed(y, d, int(tau))
    except ValueError:
        logger.warning('Embedding failed')
        return np.nan
    nx = X.shape[0]
    if nx < 30:
        logger.warning(f'Time series too short for a meaningful ordinal partition network (Nx = {nx})')
        return np.nan

    # Ordinal-pattern ID of each window, compacted to the patterns actually observed
    # (node labels carry no meaning beyond identity)
    _, s = np.unique(_ordinal_pattern_rank(X), return_inverse=True)
    s = s.ravel()
    n = int(s.max()) + 1  # number of nodes

    # Directed transition network from consecutive symbols: unique directed edges
    # and how many times each repeats
    pair_code, weights = np.unique(s[:-1] * n + s[1:], return_counts=True)
    from_nodes, to_nodes = pair_code // n, pair_code % n
    m = pair_code.size  # number of unique directed edges

    out = {}
    out['meanDegree'] = m / n  # mean in-degree = mean out-degree = m/n
    out['NFP'] = float(factorial(d) - n)

    out_deg = np.bincount(from_nodes, minlength=n)
    in_deg = np.bincount(to_nodes, minlength=n)
    out['maxOutDegree'] = float(out_deg.max())
    out['maxInDegree'] = float(in_deg.max())
    out['stdOutDegree'] = float(np.std(out_deg, ddof=1)) if n > 1 else 0.0
    out['stdInDegree'] = float(np.std(in_deg, ddof=1)) if n > 1 else 0.0

    # Reciprocity: fraction of edges (i,j) for which the reverse edge (j,i) also occurs
    reverse_code = to_nodes * n + from_nodes
    out['reciprocity'] = np.isin(reverse_code, pair_code).sum() / m

    # Probability of the most frequent single transition
    out['maxEdgeWeight'] = weights.max() / weights.sum()

    return out
