import numpy as np
from typing import Optional, Union
from numpy.typing import ArrayLike
import logging
logger = logging.getLogger('pyhctsa')

from scipy.stats import mstats
from scipy.signal import resample_poly

from ..operations.correlation import first_crossing
from ..toolboxes.matlab.matlab_fit import fit_exp1, fit_poly1, goodness_of_fit
from ..utils import binarize, matlab_quantile, sign_change

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

    if (memory > 0) and (memory < 1): #specify memory as a proportion of the time series length
        memory = int(np.round(memory*len(y)))

    # COURSE GRAIN
    # a coarse-grained time series using the numbers 1:num_groups
    if isinstance(num_groups, (int, float)):
        num_groups = int(num_groups)
    yth = coarse_grain(y, coarse_grain_method, num_groups)
    N = int(len(yth))
    num_iters = int(num_iters)
    memory = int(memory)

    # Use random sampling (original behavior)
    if random_seed is not None:
        np.random.seed(random_seed)
    rs = np.random.permutation(int(N - memory)) + memory
    rs = np.sort(rs[0:min(num_iters, len(rs))])
    rs = np.array([rs])

    # COMPUTE EMPIRICAL PROBABILITIES FROM TIME SERIES
    store = np.zeros([num_iters, 1])
    for i in range(0, rs.size): # rs.size
        if what_prior == 'dist':
            # uses the distribution up to memory to inform the next point
            # had to be careful with indexing, arange() works like matlab's : operator
            p = np.sum(yth[rs[0, i]-memory:rs[0, i]] == yth[rs[0, i]])/memory
            store[i] = p
        elif what_prior == 'T1':
            # uses one-point correlations in memory to inform the next point
            # estimate transition probabilities from data in memory
            # find where in memory this has been observbed before, and preceded it
            memory_data = yth[rs[0, i] - memory:rs[0, i]]
            inmem = np.where(memory_data[:-1] == yth[rs[0, i] - 1])[0]
            if len(inmem) == 0:
                p = 0
            else:
                p = np.mean(memory_data[inmem + 1] == yth[rs[0, i]])
            store[i] = p

        elif what_prior == 'T2':
            # Uses two-point correlations in memory to inform the next point
            memory_data = yth[rs[0, i] - memory:rs[0, i]]
            # Previous value observed in memory here
            inmem1 = np.where(memory_data[1:-1] == yth[rs[0, i] - 1])[0]
            inmem2 = np.where(memory_data[inmem1] == yth[rs[0, i] - 2])[0]
            if len(inmem2) == 0:
                p = 0
            else:
                p = np.sum(memory_data[inmem2 + 2] == yth[rs[0, i]]) / len(inmem2)
            store[i] = p
        else:
            raise ValueError(f"Unknown method: {what_prior}")
    # INFORMATION GAINED FROM NEXT OBSERVATION IS log(1/p) = -log(p)
    store[store == 0] = 1 # so that we set log[0] == 0

    out = {} # dictionary for outputs
    for i in range(0, len(store)):
        if store[i] == 0:
            store[i] = 1

    store = -(np.log(store))
    #minimum amount of information you can gain in this way
    if np.any(store > 0):
        out['min'] = min(store[store > 0]) # find the minimum value in the array, excluding zero
    else:
        out['min'] = np.nan
        
    # Calculate statistics
    out['max'] = np.max(store) # maximum amount of information you can gain in this way
    out['mean'] = np.mean(store)
    out['sum'] = np.sum(store)
    out['median'] = np.median(store)
    lq = mstats.mquantiles(store, 0.25, alphap=0.5, betap=0.5) # outputs an array of size one
    out['lq'] = lq[0] #convert array to int
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

    # Define the length of the new, symbolized sequence, N
    N = len(y_bin)

    if N < 5:
        logger.warning("Time series too short!")
        return np.nan
    # Binary sequences of length 1
    r1 = (y_bin == 1) # 1
    r0 = (y_bin == 0) # 0

    # ------ Record these -------
    # (Will be dependent outputs since signal is binary, sum to 1)
    out = {}
    out['u'] = np.mean(r1) # proportion 1 (corresponds to a movement up for 'diff')
    out['d'] = np.mean(r0) # proportion 0 (corresponds to a movement down for 'diff')
    pp = np.array([out['d'], out['u']])
    out['h'] = _f_entropy(pp)

    # Binary sequences of length 2:
    r1 = r1[:-1]
    r0 = r0[:-1]

    r00 = np.logical_and(r0, y_bin[1:] == 0)
    r01 = np.logical_and(r0, y_bin[1:] == 1)
    r10 = np.logical_and(r1, y_bin[1:] == 0)
    r11 = np.logical_and(r1, y_bin[1:] == 1)

    out['dd'] = np.mean(r00)  # down, down
    out['du'] = np.mean(r01)  # down, up
    out['ud'] = np.mean(r10)  # up, down
    out['uu'] = np.mean(r11)  # up, up

    pp = np.array([out['dd'], out['du'], out['ud'], out['uu']])
    out['hh'] = _f_entropy(pp)

    # -----------------------------
    # Binary sequences of length 3:
    # -----------------------------
    # Make sure ranges are valid for looking at the next one
    r00 = r00[:-1]
    r01 = r01[:-1]
    r10 = r10[:-1]
    r11 = r11[:-1]

    # 000
    r000 = np.logical_and(r00, y_bin[2:] == 0)
    # 001 
    r001 = np.logical_and(r00, y_bin[2:] == 1)
    r010 = np.logical_and(r01, y_bin[2:] == 0)
    r011 = np.logical_and(r01, y_bin[2:] == 1)
    r100 = np.logical_and(r10, y_bin[2:] == 0)
    r101 = np.logical_and(r10, y_bin[2:] == 1)
    r110 = np.logical_and(r11, y_bin[2:] == 0)
    r111 = np.logical_and(r11, y_bin[2:] == 1)

    # ----- Record these -----
    out['ddd'] = np.mean(r000)
    out['ddu'] = np.mean(r001)
    out['dud'] = np.mean(r010)
    out['duu'] = np.mean(r011)
    out['udd'] = np.mean(r100)
    out['udu'] = np.mean(r101)
    out['uud'] = np.mean(r110)
    out['uuu'] = np.mean(r111)

    ppp = np.array([out['ddd'], out['ddu'], out['dud'], 
                    out['duu'], out['udd'], out['udu'], 
                    out['uud'], out['uuu']])
    out['hhh'] = _f_entropy(ppp)

    # -------------------
    # 4
    # -------------------
    # Make sure ranges are valid for looking at the next one

    r000 = r000[:-1]
    r001 = r001[:-1]
    r010 = r010[:-1]
    r011 = r011[:-1]
    r100 = r100[:-1]
    r101 = r101[:-1]
    r110 = r110[:-1]
    r111 = r111[:-1]

    r0000 = np.logical_and(r000, y_bin[3:] == 0)
    r0001 = np.logical_and(r000, y_bin[3:] == 1)
    r0010 = np.logical_and(r001, y_bin[3:] == 0)
    r0011 = np.logical_and(r001, y_bin[3:] == 1)
    r0100 = np.logical_and(r010, y_bin[3:] == 0)
    r0101 = np.logical_and(r010, y_bin[3:] == 1)
    r0110 = np.logical_and(r011, y_bin[3:] == 0)
    r0111 = np.logical_and(r011, y_bin[3:] == 1)
    r1000 = np.logical_and(r100, y_bin[3:] == 0)
    r1001 = np.logical_and(r100, y_bin[3:] == 1)
    r1010 = np.logical_and(r101, y_bin[3:] == 0)
    r1011 = np.logical_and(r101, y_bin[3:] == 1)
    r1100 = np.logical_and(r110, y_bin[3:] == 0)
    r1101 = np.logical_and(r110, y_bin[3:] == 1)
    r1110 = np.logical_and(r111, y_bin[3:] == 0)
    r1111 = np.logical_and(r111, y_bin[3:] == 1)

    # ----- Record these -----
    out['dddd'] = np.mean(r0000)
    out['dddu'] = np.mean(r0001)
    out['ddud'] = np.mean(r0010)
    out['dduu'] = np.mean(r0011)
    out['dudd'] = np.mean(r0100)
    out['dudu'] = np.mean(r0101)
    out['duud'] = np.mean(r0110)
    out['duuu'] = np.mean(r0111)
    out['uddd'] = np.mean(r1000)
    out['uddu'] = np.mean(r1001)
    out['udud'] = np.mean(r1010)
    out['uduu'] = np.mean(r1011)
    out['uudd'] = np.mean(r1100)
    out['uudu'] = np.mean(r1101)
    out['uuud'] = np.mean(r1110)
    out['uuuu'] = np.mean(r1111)

    pppp = np.array([out['dddd'], out['dddu'], out['ddud'], 
                     out['dduu'], out['dudd'], out['dudu'], 
                     out['duud'], out['duuu'], out['uddd'], 
                     out['uddu'], out['udud'], out['uduu'], 
                     out['uudd'], out['uudu'], out['uuud'], 
                     out['uuuu']])
    out['hhhh'] = _f_entropy(pppp)

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

    # So we have a vectory yt with entries in {1, 2, 3}
    N = len(yt) # length of the symbolized sequence derived from the time series

    # ------------------------------------------------------------------------------
    # Words of length 1
    # ------------------------------------------------------------------------------
    out1 = np.zeros(3)
    r1 = [np.where(yt == i + 1)[0] for i in range(3)]
    for i in range(3):
        out1[i] = len(r1[i]) / N

    out = {
        'a': out1[0], 'b': out1[1], 'c': out1[2],
        'h': _f_entropy(out1)
    }

    # ------------------------------------------------------------------------------
    # Words of length 2
    # ------------------------------------------------------------------------------

    r1 = [r[:-1] if len(r) > 0 and r[-1] == N - 1 else r for r in r1]
    out2 = np.zeros((3, 3))
    r2 = [[r1[i][yt[r1[i] + 1] == j + 1] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            out2[i, j] = len(r2[i][j]) / (N - 1)

    out.update({
        'aa': out2[0, 0], 'ab': out2[0, 1], 'ac': out2[0, 2],
        'ba': out2[1, 0], 'bb': out2[1, 1], 'bc': out2[1, 2],
        'ca': out2[2, 0], 'cb': out2[2, 1], 'cc': out2[2, 2],
        'hh': _f_entropy(out2)
    })

    # ------------------------------------------------------------------------------
    # Words of length 3
    # ------------------------------------------------------------------------------

    r2 = [[r[:-1] if len(r) > 0 and r[-1] == N - 2 else r for r in row] for row in r2]
    out3 = np.zeros((3, 3, 3))
    r3 = [[[r2[i][j][yt[r2[i][j] + 2] == k + 1] for k in range(3)] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                out3[i, j, k] = len(r3[i][j][k]) / (N - 2)

    out.update({f'{chr(97+i)}{chr(97+j)}{chr(97+k)}': out3[i, j, k] 
                for i in range(3) for j in range(3) for k in range(3)})
    out['hhh'] = _f_entropy(out3)

    # ------------------------------------------------------------------------------
    # Words of length 4
    # ------------------------------------------------------------------------------

    r3 = [[[r[:-1] if len(r) > 0 and r[-1] == N - 3 else r for r in plane] for plane in cube] for cube in r3]
    out4 = np.zeros((3, 3, 3, 3))
    r4 = [[[[r3[i][j][k][yt[r3[i][j][k] + 3] == l + 1] for l in range(3)] for k in range(3)] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    out4[i, j, k, l] = len(r4[i][j][k][l]) / (N - 3)

    out.update({f'{chr(97+i)}{chr(97+j)}{chr(97+k)}{chr(97+l)}': out4[i, j, k, l] 
                for i in range(3) for j in range(3) for k in range(3) for l in range(3)})
    out['hhhh'] = _f_entropy(out4)

    return out

def _f_entropy(x):
    """Entropy of a set of counts, log(0) = 0"""
    return -np.sum(x[x > 0] * np.log(x[x > 0]))


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

    # Stationarity of binarised time series
    out = {}
    out['pupstat2'] = np.sum(y_bin[N//2:] == 1) / np.sum(y_bin[:N//2] == 1)

    # Consecutive strings of ones/zeros (normalized by length)
    diff_y = np.diff(np.where(np.concatenate(([1], y_bin, [1])))[0])
    stretch0 = diff_y[diff_y != 1] - 1

    diff_y = np.diff(np.where(np.concatenate(([0], y_bin, [0])) == 0)[0])
    stretch1 = diff_y[diff_y != 1] - 1

    # pstretches
    # Number of different stretches as proportion of the time-series length
    out['pstretch1'] = len(stretch1) / N

    if len(stretch0) == 0:
        out['longstretch0'] = 0
        out['longstretch0norm'] = 0
        out['meanstretch0'] = 0
        out['meanstretch0norm'] = 0
        out['stdstretch0'] = np.nan
        out['stdstretch0norm'] = np.nan
    else:
        out['longstretch0'] = np.max(stretch0)
        out['longstretch0norm'] = np.max(stretch0) / N
        out['meanstretch0'] = np.mean(stretch0)
        out['meanstretch0norm'] = np.mean(stretch0) / N
        out['stdstretch0'] = np.std(stretch0, ddof=1)
        out['stdstretch0norm'] = np.std(stretch0, ddof=1) / N

    if len(stretch1) == 0:
        out['longstretch1'] = 0
        out['longstretch1norm'] = 0
        out['meanstretch1'] = 0
        out['meanstretch1norm'] = 0
        out['stdstretch1'] = np.nan
    else:
        out['longstretch1'] = np.max(stretch1)
        out['longstretch1norm'] = np.max(stretch1) / N
        out['meanstretch1'] = np.mean(stretch1)
        out['meanstretch1norm'] = np.mean(stretch1) / N
        out['stdstretch1'] = np.std(stretch1, ddof=1)
        out['stdstretch1norm'] = np.std(stretch1, ddof=1) / N
    
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
    # check inputs
    y = np.asarray(y)
    if num_groups < 2:
        logger.warning("Too few groups for coarse-graining")
        return np.nan
    if tau == 'ac':
        # determine the tau from first zero of the ACF
        tau = first_crossing(y, 'ac', 0, 'discrete')
        if np.isnan(tau):
            logger.warning("Time series too short to estimate tau")
            return np.nan
    if tau > 1:
        y = resample_poly(y, 1, int(tau))
    
    N = len(y)

    yth = coarse_grain(y, how_to_cg, num_groups)
    # At this point we should have:
    # (*) yth: a thresholded y containing integers from 1 to num_groups
    yth = np.ravel(yth)

    T = np.zeros((num_groups,num_groups))
    for i in range(num_groups):
        ri = (yth == i + 1)
        if sum(ri) == 0:
            T[i,:] = 0
        else:
            ri_next = np.r_[False, ri[:-1]]
            for j in range(num_groups):
                T[i, j] = np.sum(yth[ri_next] == j + 1)

    out = {}
    # Normalize from counts to probabilities:
    T = T/(N - 1) # N-1 is appropriate because it's a 1-time transition matrix

    if num_groups == 2:
        for i in range(4):
            out[f'T{i+1}'] = T.transpose().flatten()[i]

    elif num_groups == 3:
        for i in range(9):
            out[f'T{i+1}'] = T.transpose().flatten()[i]

    elif num_groups > 3:
        for i in range(num_groups):
            out[f'TD{i+1}'] = T.transpose()[i, i]

    # (ii) Measures on the diagonal
    out['ondiag'] = np.trace(T) # trace
    out['stddiag'] = np.std(np.diag(T), ddof=1) # std of diagonal elements

    # (iii) Measures of symmetry:
    out['symdiff'] = np.sum(np.abs(T - T.T)) # sum of differences of individual elements
    # difference in sums of upper and lower triangular parts of T
    out['symsumdiff'] = np.sum(np.tril(T, -1)) - np.sum(np.triu(T, 1))

    # Measures from eigenvalues of T
    eig_t = np.linalg.eigvals(T)
    out['stdeig'] = np.std(eig_t, ddof=1)
    out['maxeig'] = np.max(np.real(eig_t))
    out['mineig'] = np.min(np.real(eig_t))
    out['maximeig'] = np.max(np.imag(eig_t))

    # Measures from covariance matrix
    cov_t = np.cov(T.transpose())
    out['sumdiagcov'] = np.trace(cov_t)

    # Eigenvalues of covariance matrix
    eig_cov_t = np.linalg.eigvals(cov_t)
    out['stdeigcov'] = np.std(eig_cov_t, ddof=1)
    out['maxeigcov'] = np.max(np.real(eig_cov_t))
    out['mineigcov'] = np.min(np.real(eig_cov_t))

    return out


def _seq_sum(x: ArrayLike) -> complex:
    """
    Sequential summation, as MATLAB's `sum` performs it over a short vector.
    NumPy sums pairwise instead, which can differ in the last bit -- enough to flip a
    downstream `>` comparison against a threshold derived from these same sums.
    """
    total = 0.0
    for v in np.asarray(x).ravel():
        total = total + v
    return total


def _seq_mean(x: ArrayLike) -> complex:
    """MATLAB's `mean`, i.e. its sequential `sum` divided by the count."""
    x = np.asarray(x)
    return _seq_sum(x) / x.size if x.size else np.nan


def _seq_std(x: ArrayLike) -> float:
    """
    MATLAB's sample standard deviation: sqrt(sum(abs(x - mean(x)).^2)/(n-1)), where the
    std of a scalar is 0 and the std of an empty vector is NaN. Accepts complex input,
    for which it returns the real spread about the complex mean.
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    xc = x - _seq_mean(x)
    return float(np.sqrt(np.real(_seq_sum(xc * np.conj(xc))) / (x.size - 1)))


def _exp_fit_gof(x: np.ndarray, y: np.ndarray, start_point) -> dict:
    """
    Fit ``a*exp(b*x)`` by nonlinear least squares and return the coefficients along
    with the goodness-of-fit statistics reported by MATLAB's Curve Fitting Toolbox.
    """
    a, b = fit_exp1(x, y, start_point)
    gof = goodness_of_fit(y, a * np.exp(b * x), num_coeffs=2)
    return {'a': a, 'b': b, 'r2': gof['rsquare'],
            'adjr2': gof['adjrsquare'], 'rmse': gof['rmse']}


def _discretize_quantile(y: np.ndarray, num_groups: int) -> np.ndarray:
    """
    Discretize a time series into `num_groups` equiprobable groups by quantile separation.

    This mirrors the (right-inclusive) binning of SB_TransitionpAlphabet's SUB_discretize,
    which differs from the (left-inclusive) binning used by `coarse_grain`.
    """
    # thresholds for dividing the time-series values
    th = matlab_quantile(y, np.linspace(0, 1, num_groups + 1))
    th[0] = th[0] - 1  # ensures the first point is included

    # turn the time series into a set of numbers from 1:num_groups
    yth = np.zeros(len(y), dtype=int)
    for li in range(num_groups):
        yth[(y > th[li]) & (y <= th[li + 1])] = li + 1

    if np.any(yth == 0):
        raise ValueError('Some values were not assigned to a group')

    return yth


def _transition_measures(yth: np.ndarray, num_groups: int) -> np.ndarray:
    """A set of metrics on the one-time transition matrix of a symbolized time series."""
    N = len(yth)

    # 1) Calculate the one-time transition matrix
    T = np.zeros((num_groups, num_groups))
    for j in range(num_groups):
        ri = (yth == j + 1)
        if not np.any(ri):
            T[j, :] = 0  # yth is never j
        else:
            # looking at the next element, so shift the indices forward by one
            ri_next = np.r_[False, ri[:-1]]
            for k in range(num_groups):
                T[j, k] = np.sum(yth[ri_next] == k + 1)
    T = T / (N - 1)  # N-1 is appropriate because it's a 1-time transition matrix

    # 2) return some quantities on the transition matrix, T
    out = np.zeros(8)
    #   (i) diagonal elements
    diag_t = np.diag(T)
    out[0] = _seq_sum(diag_t) / num_groups  # mean
    out[1] = np.max(diag_t)
    out[2] = _seq_sum(diag_t)  # trace

    #  (ii) measures of symmetry:
    # sum of differences of individual elements; MATLAB's sum(sum(M)) reduces down
    # the columns first, then across the resulting row vector
    asym = np.abs(T - T.T)
    out[3] = _seq_sum([_seq_sum(asym[:, j]) for j in range(num_groups)])

    # (iii) measures from covariance matrix:
    xc = T - np.array([_seq_sum(T[:, j]) for j in range(num_groups)]) / num_groups
    out[4] = _seq_sum(np.diag(xc.T @ xc) / (num_groups - 1))

    # (iv) measures from eigenvalues of T
    eig_t = np.linalg.eigvals(T)
    out[5] = _seq_std(eig_t)
    out[6] = np.max(np.real(eig_t))
    out[7] = np.min(np.real(eig_t))

    return out


def transition_p_alphabet(y: ArrayLike, num_groups: Optional[ArrayLike] = None,
                          tau: Union[int, str] = 1) -> dict:
    """
    How transition probabilities change with alphabet size.

    The time series is discretized by quantile separation into alphabets of a range
    of sizes, and the one-time transition matrix is computed for each. Statistics of
    those transition matrices are then tracked as a function of the alphabet size.

    Parameters
    ----------
    y : array-like
        The input time series.
    num_groups : array-like, optional
        The range of alphabet sizes to compare across. Must contain more than one
        value, each at least 2. Default is ``range(2, 11)``.
    tau : int or str, optional
        The time-delay. The time series is downsampled at this lag before being
        discretized. Can also be set to ``'ac'`` to use the first zero-crossing of
        the autocorrelation function. Default is 1.

    Returns
    -------
    dict
        The decay rate of the sum, mean, and maximum of the diagonal elements of the
        transition matrices, changes in symmetry, and statistics of their eigenvalues.
    """
    y = np.asarray(y, dtype=float)
    N = len(y)  # time-series length

    if num_groups is None:
        num_groups = np.arange(2, 11)  # compare across alphabet sizes from 2 to 10
    num_groups = np.atleast_1d(np.asarray(num_groups, dtype=int))

    if isinstance(tau, str):
        if tau != 'ac':
            raise ValueError(f"Unknown tau '{tau}'")
        # determine tau from first zero of autocorrelation
        tau = first_crossing(y, 'ac', 0, 'discrete')
        if np.isnan(tau):
            raise ValueError('Time series too short to estimate tau')
        if tau > N / 50:  # for highly-correlated signals
            tau = np.floor(N / 50)

    if np.size(tau) > 1 or num_groups.size == 1:
        raise NotImplementedError('Only a scalar tau with a range of alphabet sizes is '
                                  'supported.')
    tau = int(tau)

    if np.min(num_groups) < 2:
        raise ValueError('Need more than 2 groups')

    num_groups_range = num_groups
    if tau > 1:
        y = resample_poly(y, 1, tau)  # resample

    nfeat = 8  # the number of features calculated at each point
    store = np.zeros((len(num_groups_range), nfeat))
    for i, ng in enumerate(num_groups_range):
        yth = _discretize_quantile(y, int(ng))  # thresholded data: yth
        store[i, :] = _transition_measures(yth, int(ng))

    x = num_groups_range.astype(float)
    n = len(x)
    out = {}

    # 1) mean of diagonal elements of the transition matrix: shows an exponential
    # decay to zero
    fit = _exp_fit_gof(x, store[:, 0], [1, -0.2])
    out['meandiagfexp_a'] = fit['a']
    out['meandiagfexp_b'] = fit['b']
    out['meandiagfexp_r2'] = fit['r2']
    out['meandiagfexp_adjr2'] = fit['adjr2']
    out['meandiagfexp_rmse'] = fit['rmse']

    # 2) maximum of diagonal elements of the transition matrix: shows an exponential
    # decay to zero
    fit = _exp_fit_gof(x, store[:, 1], [1, -0.2])
    out['maxdiagfexp_a'] = fit['a']
    out['maxdiagfexp_b'] = fit['b']
    out['maxdiagfexp_r2'] = fit['r2']
    out['maxdiagfexp_adjr2'] = fit['adjr2']
    out['maxdiagfexp_rmse'] = fit['rmse']

    # 3) trace of T -- fit exponential
    fit = _exp_fit_gof(x, store[:, 2], [1, -0.2])
    out['trfexp_a'] = fit['a']
    out['trfexp_b'] = fit['b']
    out['trfexp_r2'] = fit['r2']
    out['trfexp_adjr2'] = fit['adjr2']
    out['trfexp_rmse'] = fit['rmse']

    # Also fit linear from the start to a fifth, a tenth of the starting value
    for thresh, name in ((5, 'trflin5_adjr2'), (10, 'trflin10adjr2')):
        r = np.flatnonzero(store[:, 2] > store[0, 2] / thresh)
        if len(r) > 2:
            a, b = fit_poly1(x[r], store[r, 2], [-0.05, 1])
            out[name] = goodness_of_fit(store[r, 2], a * x[r] + b, num_coeffs=2)['adjrsquare']
        else:
            out[name] = np.nan

    # 4) Symmetry; differences in diagonal elements -- return the slope
    out['symd_a'] = fit_poly1(x, store[:, 3], [0.1, 0])[0]

    # return approximately when starts to rise; where means before and
    # after a moving dividing point are most different
    if np.all(store[:, 3] == store[0, 3]):  # all the same
        out['symd_risept'] = np.nan
    else:
        mba = np.zeros((n, 2))  # means before and after
        sba = np.zeros((n, 2))  # standard deviation before and after
        for i in range(2, n):
            mba[i, 0] = _seq_mean(store[:i, 3])
            sba[i, 0] = _seq_std(store[:i, 3]) / np.sqrt(i)
            after = store[i + 1:, 3]
            mba[i, 1] = _seq_mean(after)
            sba[i, 1] = _seq_std(after) / np.sqrt(n - i)
        with np.errstate(invalid='ignore', divide='ignore'):
            tstats = np.abs((mba[:, 0] - mba[:, 1]) / np.sqrt(sba[:, 0]**2 + sba[:, 1]**2))
        if np.all(np.isnan(tstats)):
            out['symd_risept'] = np.nan
        else:
            # MATLAB's max ignores NaNs; report the 1-based index of the first maximum
            out['symd_risept'] = float(np.nanargmax(tstats) + 1)

    # 5) trace of covariance matrix -- check jump:
    out['trcov_jump'] = store[1, 4] - store[0, 4]
    r1 = np.arange(1, n) if store[1, 4] > store[0, 4] else np.arange(n)
    # fit exponential decay to range without possible first jump
    fit = _exp_fit_gof(x[r1], store[r1, 4], [1, -0.5])
    out['trcovfexp_a'] = fit['a']
    out['trcovfexp_b'] = fit['b']
    out['trcovfexp_r2'] = fit['r2']
    out['trcovfexp_adjr2'] = fit['adjr2']
    out['trcovfexp_rmse'] = fit['rmse']

    # 6) Standard deviation of eigenvalues of T -- fit an exponential decay
    fit = _exp_fit_gof(x, store[:, 5], [1, -0.2])
    out['stdeigfexp_a'] = fit['a']
    out['stdeigfexp_b'] = fit['b']
    out['stdeigfexp_r2'] = fit['r2']
    out['stdeigfexp_adjr2'] = fit['adjr2']
    out['stdeigfexp_rmse'] = fit['rmse']

    # 7) maximum (real) eigenvalue of T -- fit an exponential decay
    fit = _exp_fit_gof(x, store[:, 6], [1, -0.2])
    out['maxeig_fexpa'] = fit['a']
    out['maxeig_fexpb'] = fit['b']
    out['maxeig_fexpr2'] = fit['r2']
    out['maxeig_fexpadjr2'] = fit['adjr2']
    out['maxeig_fexprmse'] = fit['rmse']

    # 8) minimum (real) eigenvalue of T -- fit an exponential decay
    fit = _exp_fit_gof(x, store[:, 7], [1, -0.2])
    out['mineigfexp_a'] = fit['a']
    out['mineigfexp_b'] = fit['b']
    out['mineigfexp_r2'] = fit['r2']
    out['mineigfexp_adjr2'] = fit['adjr2']
    out['mineigfexp_rmse'] = fit['rmse']

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
