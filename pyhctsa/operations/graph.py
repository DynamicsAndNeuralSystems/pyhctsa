import numpy as np
from numpy.typing import ArrayLike
import scipy
from scipy.stats import expon, norm
from ts2vg import NaturalVG
import logging
logger = logging.getLogger('pyhctsa')

from pyhctsa.operations.correlation import autocorr, first_crossing


def _hvg_links(x: np.ndarray) -> tuple:
    """
    Nearest strictly-taller neighbour to the left and to the right of every node.

    Returns
    -------
    prev, nxt : ndarray of intp, shape (N,)
        ``prev[i]`` is the largest ``j < i`` with ``x[j] > x[i]``, or -1 if none.
        ``nxt[i]`` is the smallest ``j > i`` with ``x[j] > x[i]``, or -1 if none.

    Notes
    -----
    Two monotonic-stack passes, O(N) total.
    NaN nodes neither block visibility nor count as taller neighbours, matching
    the all-False semantics of ``slice > nan``.
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
        while stack and not (xl[stack[-1]] > v):
            stack.pop()
        if stack:
            prev[i] = stack[-1]
        stack.append(i)

    stack.clear()
    for i in range(N - 1, -1, -1):
        v = xl[i]
        if v != v:
            continue
        while stack and not (xl[stack[-1]] > v):
            stack.pop()
        if stack:
            nxt[i] = stack[-1]
        stack.append(i)

    return prev, nxt


def _horiz_vgraph_degrees(ts_data: ArrayLike) -> np.ndarray:
    """
    Degree sequence of the horizontal visibility graph, without materialising
    the N x N adjacency matrix.

    The forward and backward link sets are disjoint as unordered pairs --
    ``nxt[i] == j`` requires ``x[j] > x[i]`` while ``prev[j] == i`` requires
    ``x[i] > x[j]`` -- so no edge is double counted and a bincount over edge
    endpoints gives the degrees exactly.
    """
    x = np.asarray(ts_data)
    N = x.shape[0]
    if N < 2:
        return np.zeros(N, dtype=np.int64)

    prev, nxt = _hvg_links(x)
    pm = prev >= 0
    nm = nxt >= 0
    endpoints = np.concatenate((
        np.flatnonzero(pm), prev[pm],
        np.flatnonzero(nm), nxt[nm],
    ))
    return np.bincount(endpoints, minlength=N)


def visibility_graph(y: ArrayLike, meth: str = 'horiz', max_l: int = 5000) -> dict:
    """
    Visibility graph analysis of a time series.

    Constructs a visibility graph of the time series and returns various statistics 
    on the properties of the resulting network.
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

    max_l : int, optional
        Maximum number of samples to analyze. Longer time series are truncated
        to first max_l points. Default is 5000.

    Returns
    -------
    dict
        Statistics on the degree distribution.
    """
    y = np.asarray(y)
    N = len(y)
    if N > max_l:
        # too long to store in memory
        logger.info(f"Time series ({N} > {max_l}) is too long for visibility graph."
              f"Analyzing the first {max_l} samples.")
        y = y[:max_l]
    y = y - np.min(y) # adjust so that the minimum of y is at zero

    # Compute the visibility graph:
    k = np.zeros(1)
    if meth == 'horiz':
        # degrees directly: O(N) time and memory instead of the O(N^2) matrix
        k = _horiz_vgraph_degrees(y)

    elif meth == 'norm':
        vg = NaturalVG()
        vg.build(y, only_degrees=True)
        k = vg._degrees

    # statistics of k are reused throughout; compute each exactly once
    meank = np.mean(k)
    stdk = np.std(k, ddof=1)
    mediank = np.median(k)
    maxk = np.max(k)
    q05, q25, q75, q95 = np.quantile(k, [0.05, 0.25, 0.75, 0.95], method='hazen')

    out = {}
    # Degree distribution: basic statistics
    out['mode'] = scipy.stats.mode(k).mode
    out['propmode'] = np.sum(k == out['mode'])/len(k)
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

    #Using likelihood now:
    out['gaussnlogL'] = -np.sum(norm.logpdf(k, loc=meank, scale=stdk))
    out['expnlogL'] = -np.sum(expon.logpdf(k, scale=meank))

    # Autocorr
    out['kac1'] = autocorr(k, 1, 'Fourier')[0]
    out['kac2'] = autocorr(k, 2, 'Fourier')[0]
    out['kac3'] = autocorr(k, 3, 'Fourier')[0]
    out['ktau'] = first_crossing(k, 'ac', 0, 'continuous')

    return out