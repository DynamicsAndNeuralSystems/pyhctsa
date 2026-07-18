import numpy as np
from typing import Union
from numpy.typing import ArrayLike
import logging
logger = logging.getLogger('pyhctsa')

from scipy.stats import mstats
from scipy.signal import resample_poly

from ..operations.correlation import first_crossing
from ..utils import binarize, sign_change

def surprise(y: ArrayLike, what_prior: str = 'dist', memory: float = 0.2, num_groups: int = 3,
             coarse_grain_method: str = 'quantile', num_iters: int = 500,
             random_seed: int = 0) -> dict:
    """
    Quantifies how surprised you would be of the next data point given recent memory.

    Coarse-grains the time series, turning it into a sequence of symbols of a 
    given alphabet size (`num_groups`), and quantifies measures of surprise of 
    a process with local memory of the past `memory` values of the symbolic string.
    For each sample, the 'information gained' (log(1/p)) is estimated using expectations 
    calculated from the previous `memory` samples.

    Parameters
    ----------
    y : array-like
        The input time series.
    what_prior : {'dist', 'T1', 'T2'}, optional
        The type of information to store in memory:

        - 'dist': the values of the time series in the previous memory samples (default),
        - 'T1': the one-point transition probabilities in the previous memory samples,
        - 'T2': the two-point transition probabilities in the previous memory samples.

        Default is ``'dist'``.

    memory : float, optional
        The memory length (either number of samples, or a proportion of the time-series length 
        if between 0 and 1). Default is 0.2.
    num_groups : int, optional
        The number of groups to coarse-grain the time series into. Default is 3.
    coarse_grain_method : {'quantile', 'updown', 'embed2quadrants'}, optional
        The coarse-graining or symbolization method:

        - 'quantile': equiprobable alphabet by value of each time-series datapoint (default),
        - 'updown': equiprobable alphabet by incremental changes in the time-series values,
        - 'embed2quadrants': 4-letter alphabet of the quadrant each data point resides in a 
            2D embedding space.
        
        Default is ``'quantile'``.

    num_iters : int, optional
        The number of iterations to repeat the procedure for. Default is 500.
    random_seed : int, optional
        Whether (and how) to reset the random seed. Default is 0.

    Returns
    -------
    dict
        Summaries of the series of information gains.
    """

    if (memory > 0) and (memory < 1):
        memory = int(np.round(memory * len(y)))

    if isinstance(num_groups, (int, float)):
        num_groups = int(num_groups)
    yth = coarse_grain(y, coarse_grain_method, num_groups)
    N = int(len(yth))
    num_iters = int(num_iters)
    memory = int(memory)

    # rs and the RNG are left exactly as-is so sampling is byte-for-byte identical.
    if random_seed is not None:
        np.random.seed(random_seed)
    rs = np.random.permutation(int(N - memory)) + memory
    rs = np.sort(rs[0:min(num_iters, len(rs))])
    rs = np.array([rs])

    targets = rs[0]
    n = targets.size

    store = np.zeros((num_iters, 1))

    if what_prior == 'dist':
        # p[t] = (#{ yth[t-memory:t] == yth[t] }) / memory
        # Computed via per-symbol cumulative counts: O(S*N), no big window matrix.
        symbols, inv = np.unique(yth, return_inverse=True)
        inv = np.asarray(inv).ravel()
        S = symbols.shape[0]
        cum = np.zeros((S, N + 1), dtype=np.int64)
        cum[:, 1:] = np.cumsum(inv[None, :] == np.arange(S)[:, None], axis=1)
        tv = inv[targets]                                   # symbol index of yth[t]
        counts = cum[tv, targets] - cum[tv, targets - memory]
        store[:n, 0] = counts / memory

    elif what_prior == 'T1':
        # p[t] = mean( md[j+1]==yth[t]  over j where md[j]==yth[t-1] )
        offs = np.arange(memory)
        W = yth[(targets - memory)[:, None] + offs[None, :]]   # (n, memory) windows
        tvals = yth[targets]
        prev1 = W[:, -1]                                        # == yth[t-1]
        mp = W[:, :-1] == prev1[:, None]
        mn = W[:, 1:]  == tvals[:, None]
        den = mp.sum(axis=1)
        num = (mp & mn).sum(axis=1)
        p_vec = np.zeros(n)
        nz = den != 0
        p_vec[nz] = num[nz] / den[nz]
        store[:n, 0] = p_vec

    elif what_prior == 'T2':
        # Left as the original per-target loop (indexing semantics are fragile).
        for i in range(n):
            t = targets[i]
            memory_data = yth[t - memory:t]
            inmem1 = np.where(memory_data[1:-1] == yth[t - 1])[0]
            inmem2 = np.where(memory_data[inmem1] == yth[t - 2])[0]
            if len(inmem2) == 0:
                p = 0
            else:
                p = np.sum(memory_data[inmem2 + 2] == yth[t]) / len(inmem2)
            store[i] = p
    else:
        raise ValueError(f"Unknown method: {what_prior}")

    # log(1/p); zeros (and any unfilled trailing rows) map to log(1)=0.
    store[store == 0] = 1
    store = -(np.log(store))

    out = {}
    if np.any(store > 0):
        out['min'] = min(store[store > 0])
    else:
        out['min'] = np.nan
    out['max'] = np.max(store)
    out['mean'] = np.mean(store)
    out['sum'] = np.sum(store)
    out['median'] = np.median(store)
    lq = mstats.mquantiles(store, 0.25, alphap=0.5, betap=0.5)
    out['lq'] = lq[0]
    uq = mstats.mquantiles(store, 0.75, alphap=0.5, betap=0.5)
    out['uq'] = uq[0]
    out['std'] = np.std(store, ddof=1)

    return out

def motif_two(y: ArrayLike, binarize_how: str = 'diff') -> dict:
    """
    Compute local motifs in a binary symbolization of the input time series.

    This function coarse-grains the input time series into a binary sequence
    using the specified binarization method, and computes the probabilities
    of binary words of lengths 1 through 4, along with their entropies.

    Parameters
    ----------
    y : array-like
        The input time series.

    binarize_how : str, optional
        The method used for binary transformation. One of:

        - 'diff': Encode increases in the time series as 1, and decreases as 0.
        - 'mean': Encode values above the mean as 1, and below as 0.
        - 'median': Encode values above the median as 1, and below as 0.

        Default is ``'diff'``.

    Returns
    -------
    dict
        A dictionary containing:

        - 'prob_len_1', 'prob_len_2', ..., 'prob_len_4': 
            Lists of probabilities for each binary word of lengths 1 to 4.
        - 'entropy_len_1', 'entropy_len_2', ..., 'entropy_len_4': 
            Entropy values associated with the word distributions of lengths 1 to 4.

    """
    # Generate a binarized version of the input time series
    y = np.asarray(y)
    y_bin = binarize(y, binarize_how)

    N = len(y_bin)
    if N < 5:
        logger.warning("Time series too short!")
        return np.nan

    b = y_bin.astype(np.intp)  # 0/1 symbols

    # First symbol = most-significant bit, so bincount index order matches the original
    # key order exactly: 0->dd,1->du,2->ud,3->uu (and analogously for lengths 3, 4).
    w2 = (b[:-1] << 1) | b[1:]
    w3 = (b[:-2] << 2) | (b[1:-1] << 1) | b[2:]
    w4 = (b[:-3] << 3) | (b[1:-2] << 2) | (b[2:-1] << 1) | b[3:]

    c1 = np.bincount(b,  minlength=2)
    c2 = np.bincount(w2, minlength=4)
    c3 = np.bincount(w3, minlength=8)
    c4 = np.bincount(w4, minlength=16)

    # Denominators = the original's shrinking window counts: N, N-1, N-2, N-3.
    p1 = c1 / N          # [d, u]
    p2 = c2 / (N - 1)    # [dd, du, ud, uu]
    p3 = c3 / (N - 2)
    p4 = c4 / (N - 3)

    out = {}
    out['d'], out['u'] = p1[0], p1[1]
    out['h'] = _f_entropy(p1)

    out['dd'], out['du'], out['ud'], out['uu'] = p2
    out['hh'] = _f_entropy(p2)

    (out['ddd'], out['ddu'], out['dud'], out['duu'],
     out['udd'], out['udu'], out['uud'], out['uuu']) = p3
    out['hhh'] = _f_entropy(p3)

    (out['dddd'], out['dddu'], out['ddud'], out['dduu'],
     out['dudd'], out['dudu'], out['duud'], out['duuu'],
     out['uddd'], out['uddu'], out['udud'], out['uduu'],
     out['uudd'], out['uudu'], out['uuud'], out['uuuu']) = p4
    out['hhhh'] = _f_entropy(p4)

    return out

def motif_three(y: ArrayLike, cg_how: str = 'quantile') -> dict:
    """
    Motifs in a coarse-graining of a time series to a 3-letter alphabet.

    Parameters
    ----------
    y : array-like
        Time series to analyze.
    cg_how : {'quantile', 'diffquant'}, optional
        The coarse-graining method to use:

        - 'quantile': equiprobable alphabet by time-series value
        - 'diffquant': equiprobably alphabet by time-series increments

        Default is ``'quantile'``.

    Returns
    -------
    dict
        Statistics on words of length 1, 2, 3, and 4.
    """

    # Coarse-grain the data y -> yt
    y = np.asarray(y)
    num_letters = 3
    if cg_how == 'quantile':
        yt = coarse_grain(y, 'quantile', num_letters)
    elif cg_how == 'diffquant':
        yt = coarse_grain(np.diff(y), 'quantile', num_letters)
    else:
        raise ValueError(f"Unknown coarse-graining method {cg_how}")

    N = len(yt)

    # Symbols in {0,1,2}. Encode each length-k window in base 3 with the FIRST symbol
    # as the most-significant digit, so bincount index == i*3^(k-1)+... matches the
    # C-order layout of the original out2/out3/out4 arrays (and thus the dict keys).
    s = np.asarray(yt).astype(np.intp) - 1

    # Slicing to full windows (s[:-1], s[:-2], s[:-3]) reproduces the original's
    # trailing-index trimming exactly: only start positions with a complete window remain.
    w2 = s[:-1] * 3 + s[1:]
    w3 = s[:-2] * 9 + s[1:-1] * 3 + s[2:]
    w4 = s[:-3] * 27 + s[1:-2] * 9 + s[2:-1] * 3 + s[3:]

    c1 = np.bincount(s,  minlength=3)
    c2 = np.bincount(w2, minlength=9)
    c3 = np.bincount(w3, minlength=27)
    c4 = np.bincount(w4, minlength=81)

    # Same denominators as the original: N, N-1, N-2, N-3.
    out1 = c1 / N
    out2 = (c2 / (N - 1)).reshape(3, 3)
    out3 = (c3 / (N - 2)).reshape(3, 3, 3)
    out4 = (c4 / (N - 3)).reshape(3, 3, 3, 3)

    out = {'a': out1[0], 'b': out1[1], 'c': out1[2], 'h': _f_entropy(out1)}

    out.update({
        'aa': out2[0, 0], 'ab': out2[0, 1], 'ac': out2[0, 2],
        'ba': out2[1, 0], 'bb': out2[1, 1], 'bc': out2[1, 2],
        'ca': out2[2, 0], 'cb': out2[2, 1], 'cc': out2[2, 2],
        'hh': _f_entropy(out2),
    })

    out.update({f'{chr(97+i)}{chr(97+j)}{chr(97+k)}': out3[i, j, k]
                for i in range(3) for j in range(3) for k in range(3)})
    out['hhh'] = _f_entropy(out3)

    out.update({f'{chr(97+i)}{chr(97+j)}{chr(97+k)}{chr(97+l)}': out4[i, j, k, l]
                for i in range(3) for j in range(3) for k in range(3) for l in range(3)})
    out['hhhh'] = _f_entropy(out4)

    return out

def _f_entropy(x):
    """Entropy of a set of counts, log(0) = 0"""
    xpos = x[x > 0]
    return -np.sum(xpos * np.log(xpos))

def binary_stretch(x: ArrayLike, stretch_what: str = 'lseq1') -> float:
    """
    Characterize stretches of 0s or 1s in a binarized time series.

    This function binarizes the input time series based on its mean:
    values above the mean are converted to 1, and values below to 0.
    It then computes a statistic related to the lengths of consecutive
    0s or 1s in the resulting binary sequence, depending on the `stretch_what`
    argument.

    **Note**: Due to an implementation error in the original version, this
    function does not correctly compute the *longest* stretch of 0s or 1s,
    but still produces a potentially interesting statistic.

    Parameters
    ----------
    x : array-like
        The input time series.

    stretch_what : str, optional
        Specifies which binary symbol's stretch length to analyze:

        - 'lseq1': Analyze stretches related to consecutive 1s.
        - 'lseq0': Analyze stretches related to consecutive 0s.

        Default is ``'lseq1'``.

    Returns
    -------
    float
        A statistic related to the stretch length of consecutive 0s or 1s,
        normalized by the time-series length.
    """
    x = np.asarray(x)
    N = len(x) # time series length
    x = np.where(x > 0, 1, 0)

    if stretch_what == 'lseq1':
        # longest stretch of 1s [this code doesn't actualy measure this!]
        indices = np.where(x == 1)[0]
        diffs = np.diff(indices) - 1.5
        sign_changes = sign_change(diffs, 1)
        if sign_changes.size > 1:
            out = np.max(np.diff(sign_changes)) / N
        else:
            out = None
    elif stretch_what == 'lseq0':
        # longest stretch of 0s [this code doesn't actualy measure this!]
        indices = np.where(x == 0)[0]
        diffs = np.diff(indices) - 1.5
        sign_changes = sign_change(diffs, 1)
        if sign_changes.size > 1:
            out = np.max(np.diff(sign_changes)) / N
        else:
            out = None
    else:
        raise ValueError(f"Unknown input {stretch_what}")
    
    return out if out is not None else 0

def binary_stats(y: ArrayLike, binary_method: str = 'diff') -> dict:
    """
    Compute statistics on a binary symbolisation of the input time series.

    The time series is first symbolized as a binary string of 0s and 1s 
    using a specified coarse-graining (symbolisation) method. Then, various 
    statistics are computed to characterize the structure of the resulting 
    binary sequence.

    Parameters
    ----------
    y : array-like
        The input time series.

    binary_method : str, optional
        The binary symbolisation rule. One of:

        - 'diff': Encode as 1 if the time-series difference is positive, and 0 otherwise.
        - 'mean': Encode as 1 if the value is above the mean, 0 otherwise.

        Default is ``'diff'``.

    Returns
    -------
    dict
        Statistics computed on the binary symbolisation.
    """
    
    # Binarize the time series
    y = np.asarray(y)
    y_bin = binarize(y, binarize_how=binary_method)
    N = len(y_bin)

    out = {}
    out['pupstat2'] = np.count_nonzero(y_bin[N//2:] == 1) / np.count_nonzero(y_bin[:N//2] == 1)

    # Stretch of 0s: gaps between consecutive 1-positions (with 1-sentinels), minus 1.
    diff_y = np.diff(np.flatnonzero(np.concatenate(([1], y_bin, [1]))))
    stretch0 = diff_y[diff_y != 1] - 1

    # Stretch of 1s: gaps between consecutive 0-positions (with 0-sentinels), minus 1.
    diff_y = np.diff(np.flatnonzero(np.concatenate(([0], y_bin, [0])) == 0))
    stretch1 = diff_y[diff_y != 1] - 1

    out['pstretch1'] = len(stretch1) / N

    # --- stretch0: compute each reduction once, reuse for the /N variants ---
    if len(stretch0) == 0:
        out['longstretch0'] = 0
        out['longstretch0norm'] = 0
        out['meanstretch0'] = 0
        out['meanstretch0norm'] = 0
        out['stdstretch0'] = np.nan
        out['stdstretch0norm'] = np.nan
    else:
        max0 = np.max(stretch0)
        mean0 = np.mean(stretch0)
        std0 = np.std(stretch0, ddof=1)
        out['longstretch0'] = max0
        out['longstretch0norm'] = max0 / N
        out['meanstretch0'] = mean0
        out['meanstretch0norm'] = mean0 / N
        out['stdstretch0'] = std0
        out['stdstretch0norm'] = std0 / N

    # --- stretch1 ---
    if len(stretch1) == 0:
        out['longstretch1'] = 0
        out['longstretch1norm'] = 0
        out['meanstretch1'] = 0
        out['meanstretch1norm'] = 0
        out['stdstretch1'] = np.nan
    else:
        max1 = np.max(stretch1)
        mean1 = np.mean(stretch1)
        std1 = np.std(stretch1, ddof=1)
        out['longstretch1'] = max1
        out['longstretch1norm'] = max1 / N
        out['meanstretch1'] = mean1
        out['meanstretch1norm'] = mean1 / N
        out['stdstretch1'] = std1
        out['stdstretch1norm'] = std1 / N

    out['meanstretchdiff'] = (out['meanstretch1'] - out['meanstretch0']) / N
    out['stdstretchdiff'] = (out['stdstretch1'] - out['stdstretch0']) / N

    out['diff21stretch1'] = np.mean(stretch1 == 2) - np.mean(stretch1 == 1)
    out['diff21stretch0'] = np.mean(stretch0 == 2) - np.mean(stretch0 == 1)

    return out

def transition_matrix(y: ArrayLike, how_to_cg: str = 'quantile',
                      num_groups: int = 2, tau: Union[int, str] = 1) -> dict:
    """
    Transition probabilities between time-series states. 
    The time series is coarse-grained according to a given method.

    The input time series is transformed into a symbolic string using an
    equiprobable alphabet of num_groups letters. The transition probabilities are
    calculated at a lag tau.

    Related to the idea of quantile graphs from time series, cf. [1]

    References
    ----------
    .. [1] Andriana et al. (2011). Duality between Time Series and Networks. PLoS ONE.
        https://doi.org/10.1371/journal.pone.0023378

    Parameters
    -----------
    y : array-like
        Input time series.
    how_to_cg : str, optional
        The method of discretization. Default is ``'quantile'``.
    num_groups : int, optional
        number of groups in the course-graining. Default is 2.
    tau : int or str, optional
        analyze transition matrices corresponding to this lag. We
        could either downsample the time series at this lag and then do the
        discretization as normal, or do the discretization and then just
        look at this dicrete lag. Here we do the former. Can also set tau to 'ac'
        to set tau to the first zero-crossing of the autocorrelation function.
        Default is 1.

    Returns
    -------
    dict 
        A dictionary including the transition probabilities themselves, as well as the trace
        of the transition matrix, measures of asymmetry, and eigenvalues of the
        transition matrix.
    """
    y = np.asarray(y)
    if num_groups < 2:
        logger.warning("Too few groups for coarse-graining")
        return np.nan
    if tau == 'ac':
        tau = first_crossing(y, 'ac', 0, 'discrete')
        if np.isnan(tau):
            logger.warning("Time series too short to estimate tau")
            return np.nan
    if tau > 1:
        y = resample_poly(y, 1, int(tau))

    N = len(y)

    yth = coarse_grain(y, how_to_cg, num_groups)
    yth = np.ravel(yth)

    a = yth[:-1] - 1
    b = yth[1:] - 1
    idx = a * num_groups + b
    T = np.bincount(idx, minlength=num_groups * num_groups).astype(float)
    T = T.reshape(num_groups, num_groups)

    out = {}
    T = T / (N - 1)

    if num_groups == 2:
        for i in range(4):
            out[f'T{i+1}'] = T.transpose().flatten()[i]

    elif num_groups == 3:
        for i in range(9):
            out[f'T{i+1}'] = T.transpose().flatten()[i]

    elif num_groups > 3:
        for i in range(num_groups):
            out[f'TD{i+1}'] = T.transpose()[i, i]

    out['ondiag'] = np.trace(T)
    out['stddiag'] = np.std(np.diag(T), ddof=1)

    out['symdiff'] = np.sum(np.abs(T - T.T))
    out['symsumdiff'] = np.sum(np.tril(T, -1)) - np.sum(np.triu(T, 1))

    eig_t = np.linalg.eigvals(T)
    out['stdeig'] = np.std(eig_t, ddof=1)
    out['maxeig'] = np.max(np.real(eig_t))
    out['mineig'] = np.min(np.real(eig_t))
    out['maximeig'] = np.max(np.imag(eig_t))

    cov_t = np.cov(T.transpose())
    out['sumdiagcov'] = np.trace(cov_t)

    eig_cov_t = np.linalg.eigvals(cov_t)
    out['stdeigcov'] = np.std(eig_cov_t, ddof=1)
    out['maxeigcov'] = np.max(np.real(eig_cov_t))
    out['mineigcov'] = np.min(np.real(eig_cov_t))

    return out

def coarse_grain(y: list, how_to_cg: str, num_groups: int) -> np.ndarray:
    """
    Coarse-grains a continuous time series to a discrete alphabet.

    Parameters
    -----------
    y : array-like
        The input time series.
    how_to_cg : str
        The method of coarse-graining.
        Options: 

        - 'updown'
        - 'quantile'
        - 'embed2quadrants'
        - 'embed2octants'
        
    num_groups : int
        Specifies the size of the alphabet for 'quantile' and 'updown',
        or sets the time delay for the embedding subroutines.

    Returns
    --------
    yth : array-like
        The coarse-grained time series.
    """
    y = np.asarray(y)
    N = len(y)

    if how_to_cg not in ['updown', 'quantile', 'embed2quadrants', 'embed2octants']:
        raise ValueError(f"Unknown coarse-graining method '{how_to_cg}'")

    # Some coarse-graining/symbolization methods require initial processing:
    if how_to_cg == 'updown':
        y = np.diff(y)
        N = N - 1 # the time series is one value shorter than the input because of differencing
        how_to_cg = 'quantile' # successive differences and then quantiles

    elif how_to_cg in ['embed2quadrants', 'embed2octants']:
        # Construct the embedding
        if num_groups == 'tau':
            # First zero-crossing of the ACF
            tau = first_crossing(y, 'ac', 0, 'discrete')
        else:
            tau = num_groups
        
        if tau > N/25:
            tau = N // 25

        m1 = y[:-tau]
        m2 = y[tau:]

        # Look at which points are in which angular 'quadrant'
        upr = m2 >= 0 # points above the axis
        downr = m2 < 0 # points below the axis

        q1r = np.logical_and(upr, m1 >= 0) # points in quadrant 1
        q2r = np.logical_and(upr, m1 < 0) # points in quadrant 2
        q3r = np.logical_and(downr, m1 < 0) # points in quadrant 3
        q4r = np.logical_and(downr, m1 >= 0) # points in quadrant 4
    
    # Do the coarse graining
    yth = None  # Ensure yth is always defined
    if how_to_cg == 'quantile':
        th = np.quantile(y, np.linspace(0, 1, num_groups + 1), method='linear') # thresholds for dividing the time-series values
        yth = np.zeros(N, dtype=int)
        # turn the time series into a set of numbers from 1:num_groups
        for i in range(num_groups):
            if i == num_groups - 1:
                # Right-inclusive logic for the final boundary to catch the absolute max
                yth[(y >= th[i]) & (y <= th[i+1])] = i + 1
            else:
                # Left-inclusive logic [>=, <) for all other boundaries
                yth[(y >= th[i]) & (y < th[i+1])] = i + 1
        return yth

    elif how_to_cg == 'embed2quadrants': # divides based on quadrants in a 2-D embedding space
        # create alphabet in quadrants -- {1,2,3,4}
        yth = np.zeros(len(m1), dtype=int)
        yth[q1r] = 1
        yth[q2r] = 2
        yth[q3r] = 3
        yth[q4r] = 4
        
    elif how_to_cg == 'embed2octants': # divide based on octants in 2-D embedding space
        o1r = np.logical_and(q1r, m2 < m1) # points in octant 1
        o2r = np.logical_and(q1r, m2 >= m1) # points in octant 2
        o3r = np.logical_and(q2r, m2 >= -m1) # points in octant 3
        o4r = np.logical_and(q2r, m2 < -m1) # points in octant 4
        o5r = np.logical_and(q3r, m2 >= m1) # points in octant 5
        o6r = np.logical_and(q3r, m2 < m1) # points in octant 6
        o7r = np.logical_and(q4r, m2 < -m1) # points in octant 7
        o8r = np.logical_and(q4r, m2 >= -m1) # points in octant 8

        # create alphabet in octants -- {1,2,3,4,5,6,7,8}
        yth = np.zeros(len(m1), dtype=int)
        yth[o1r] = 1
        yth[o2r] = 2
        yth[o3r] = 3
        yth[o4r] = 4
        yth[o5r] = 5
        yth[o6r] = 6
        yth[o7r] = 7
        yth[o8r] = 8

    if yth is None:
        raise ValueError('Coarse-graining method did not assign yth.')

    if np.any(yth == 0):
        raise ValueError('All values in the sequence were not assigned to a group')

    return yth
