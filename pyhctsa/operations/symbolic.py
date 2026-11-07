from itertools import product

import numpy as np
from typing import Optional, Union
from numpy.typing import ArrayLike
import logging
logger = logging.getLogger('pyhctsa')

from ..operations.correlation import autocorr
from ..robust import bf_exp_fit
from ..utils import dict_output, _sample_std, binarize, sign_change, get_tau, time_delay_embed

def surprise(y: ArrayLike, what_prior: str = 'dist', memory: float = 0.2, num_groups: int = 3,
             coarse_grain_method: str = 'quantile', num_iters: int = 500,
             random_seed: int = 0) -> dict:
    """
    Quantifies how surprised you would be of the next data point given recent memory.

    Coarse-grains the time series, turning it into a sequence of symbols of a 
    given alphabet size (`num_groups`), and quantifies measures of surprise of 
    a process with local memory of the past `memory` values of the symbolic string.
    For each test point, the 'information gained' (log(1/p)) is estimated using expectations
    calculated from the previous `memory` samples. The test points are every point with a full
    memory before it, or, for long series, a fixed subsample of ``num_iters`` of them.

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
    num_groups : int or str, optional
        The number of groups to coarse-grain the time series into (2 for 'updown'), or, for
        'embed2quadrants'/'embed2octants', the time delay of the embedding: a number of
        samples, ``'ac1e'`` (the floor of the first 1/e crossing of the autocorrelation
        function), ``'mi'`` (the smaller of the first minimum of the Kraskov automutual
        information and the 'ac1e' delay; see :func:`pyhctsa.utils.get_tau`), or ``'tau'``
        (the first zero-crossing of the autocorrelation function, kept for backward
        compatibility). Default is 3.
    coarse_grain_method : {'quantile', 'diff', 'updown', 'embed2quadrants', 'embed2octants'}, optional
        The coarse-graining or symbolization method (see :func:`coarse_grain`):

        - 'quantile': equiprobable alphabet by value of each time-series datapoint (default),
        - 'diff': equiprobable alphabet by the value of the incremental changes in the
          time series (not a literal sign split; called 'updown' before it was renamed in hctsa),
        - 'updown': a binary split by the sign of each increment (requires ``num_groups=2``;
          not equiprobable),
        - 'embed2quadrants': 4-letter alphabet of the quadrant each data point resides in a
          2D embedding space,
        - 'embed2octants': 8-letter alphabet of the octant each data point resides in.

        Default is ``'quantile'``.

    num_iters : int, optional
        The number of test points to use. If there are at most ``2 * num_iters`` points with a
        full memory before them, all of them are used; otherwise ``num_iters`` of them, spread
        evenly by a golden-ratio sequence (deterministic, and free of aliasing with periodic
        series). Default is 500.
    random_seed : int, optional
        Ignored: the test points are deterministic. Kept so that existing calls still run.

    Returns
    -------
    dict
        Summaries of the series of information gains, with keys:

        - 'min', 'max', 'median', 'mean', 'sum', 'std', 'lq', 'uq': the minimum (of the
          nonzero values), maximum, median, mean, sum, standard deviation, and lower and
          upper quartiles of the information gain over the test points,
        - 'propUnseen': the proportion of test points whose antecedent pattern (the
          current symbol itself for 'dist'; the preceding 1 or 2 symbols for 'T1'/'T2')
          was never observed in the memory window (always 0 for 'dist'),
        - 'effectSize': ``|mean - 1| / std``, the standardized distance of the mean
          information gain from 1 nat,
        - 'tstat': ``effectSize * sqrt(number of test points)``.

        All NaN if there is no test point (the series is no longer than ``memory``) or if the
        coarse-graining is undefined (the embedding delay for
        'embed2quadrants'/'embed2octants' cannot be determined: a constant series, or
        ``'ac1e'`` for a series whose autocorrelation function never falls to 1/e). ``effectSize``
        and ``tstat`` are NaN if the information gain does not vary (its standard deviation at
        rounding level relative to its mean, e.g. for a perfectly periodic series).
    """

    if (memory > 0) and (memory < 1): #specify memory as a proportion of the time series length
        memory = int(np.round(memory*len(y)))

    # COURSE GRAIN
    # a coarse-grained time series using the numbers 1:num_groups
    if isinstance(num_groups, (int, float)):
        num_groups = int(num_groups)
    yth = coarse_grain(y, coarse_grain_method, num_groups)
    if np.isscalar(yth) and np.isnan(yth):
        # No coarse-graining exists (the embedding delay is undefined): every output is NaN
        return {k: np.nan for k in ('min', 'max', 'median', 'mean', 'sum', 'std', 'lq', 'uq',
                                    'propUnseen', 'effectSize', 'tstat')}
    N = int(len(yth))
    num_iters = int(num_iters)
    memory = int(memory)

    # Select the test points (0-based; can't test the beginning of the time series, up to memory)
    num_available = N - memory
    if num_available <= 2 * num_iters:
        rs = np.arange(memory, N)  # every point with a full memory before it
    else:
        # num_iters points spread over the available range by a golden-ratio sequence
        rs = np.unique(memory + np.floor(num_available * np.mod(np.arange(1, num_iters + 1) * 0.6180339887498949, 1)).astype(int))
    if rs.size == 0:  # the series is no longer than the memory: no test points
        return {k: np.nan for k in ('min', 'max', 'median', 'mean', 'sum', 'std', 'lq', 'uq',
                                    'propUnseen', 'effectSize', 'tstat')}
    rs = np.array([rs])

    # The alphabet size for the Krichevsky-Trofimov smoothing below. For
    # 'embed2quadrants'/'embed2octants', num_groups is the embedding delay, not an
    # alphabet size, so the alphabet size is fixed by the number of quadrants/octants.
    if coarse_grain_method == 'embed2quadrants':
        num_symbols = 4
    elif coarse_grain_method == 'embed2octants':
        num_symbols = 8
    else:
        num_symbols = num_groups

    # COMPUTE EMPIRICAL PROBABILITIES FROM TIME SERIES
    # Sized to the number of test points actually available, min(num_iters, N-memory)
    num_test = rs.size
    store = np.zeros(num_test)
    n_antecedent_all = np.zeros(num_test)  # how many times the antecedent pattern was seen in memory
    for i in range(0, num_test):
        if what_prior == 'dist':
            # uses the distribution up to memory to inform the next point
            # the memory window is the `memory` samples before the test point
            num_matches = np.sum(yth[rs[0, i]-memory:rs[0, i]] == yth[rs[0, i]])
            n_antecedent = memory
        elif what_prior == 'T1':
            # uses one-point correlations in memory to inform the next point
            # estimate transition probabilities from data in memory
            # find where in memory this has been observbed before, and preceded it
            memory_data = yth[rs[0, i] - memory:rs[0, i]]
            inmem = np.where(memory_data[:-1] == yth[rs[0, i] - 1])[0]
            n_antecedent = len(inmem)
            if n_antecedent == 0:
                num_matches = 0
            else:
                num_matches = np.sum(memory_data[inmem + 1] == yth[rs[0, i]])

        elif what_prior == 'T2':
            # Uses two-point correlations in memory to inform the next point
            memory_data = yth[rs[0, i] - memory:rs[0, i]]
            # Previous value observed in memory here
            inmem1 = np.where(memory_data[1:-1] == yth[rs[0, i] - 1])[0]
            inmem2 = np.where(memory_data[inmem1] == yth[rs[0, i] - 2])[0]
            n_antecedent = len(inmem2)
            if n_antecedent == 0:
                num_matches = 0
            else:
                # inmem2 indexes into inmem1, not directly into memory_data
                num_matches = np.sum(memory_data[inmem1[inmem2] + 2] == yth[rs[0, i]])
        else:
            raise ValueError(f"Unknown method: {what_prior}")
        # Krichevsky-Trofimov-style smoothed probability estimate: always in (0, 1),
        # the uniform prior 1/num_symbols when the antecedent was never observed
        store[i] = (num_matches + 0.5) / (n_antecedent + 0.5 * num_symbols)
        n_antecedent_all[i] = n_antecedent

    # INFORMATION GAINED FROM NEXT OBSERVATION IS log(1/p) = -log(p)
    out = {} # dictionary for outputs

    # proportion of test points whose antecedent pattern was never observed in memory
    # (always 0 for 'dist')
    prop_unseen = np.mean(n_antecedent_all == 0)

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
    out['lq'] = np.quantile(store, 0.25, method='hazen')
    out['uq'] = np.quantile(store, 0.75, method='hazen')
    out['std'] = np.std(store, ddof=1)
    out['propUnseen'] = prop_unseen

    # Standardized distance of the mean information gain from 1 (the length-stable form),
    # and the corresponding t-statistic, which grows with the number of test points. When the
    # surprise does not vary (std at rounding level relative to the mean, e.g. for a perfectly
    # periodic series) the ratio is rounding noise, so both are NaN
    if not out['std'] >= 1e-8 * out['mean']:
        out['effectSize'] = np.nan
        out['tstat'] = np.nan
    else:
        out['effectSize'] = np.abs(out['mean'] - 1) / out['std']
        out['tstat'] = out['effectSize'] * np.sqrt(num_test)

    return out

_SYMBOLIC_TAU_RULES = ('ac', 'ac1e', 'mi', 'ac-rank', 'ac1e-rank', 'mi-rank')


def _resolve_tau(y: np.ndarray, tau: Union[int, float, str]) -> Union[int, float]:
    """
    Resolve the lag of hctsa's SB_MotifTwo/Three, SB_TransitionMatrix and SB_TransitionPAlphabet.

    `tau` is a number of samples (at least 1), or a string that sets it from the series:
    ``'ac'`` (first zero crossing of the autocorrelation function), ``'ac1e'`` (floor of its
    first 1/e crossing) or ``'mi'`` (the smaller of the first minimum of the Kraskov automutual
    information and the 'ac1e' delay), or one of these computed from the ranks of the series
    (``'ac-rank'``, ``'ac1e-rank'``, ``'mi-rank'``); see :func:`pyhctsa.utils.get_tau`. Returns
    NaN if the delay cannot be determined (e.g., an undefined ACF of a constant series, or an ACF
    that never falls to 1/e for 'ac1e').
    """
    if isinstance(tau, str):
        if tau not in _SYMBOLIC_TAU_RULES:
            raise ValueError(f"Unknown tau '{tau}': use an integer or one of {_SYMBOLIC_TAU_RULES}")
        tau = get_tau(y, tau)
    if np.isnan(tau):
        return np.nan
    if tau < 1:
        raise ValueError(f"The lag tau must be at least 1, got {tau}")
    return int(tau)


def _lagged_word_probs(s: np.ndarray, length: int, tau: int, alphabet_size: int) -> tuple:
    """
    Probabilities of the lagged words of a symbol sequence, and the number of words.

    The words of `length` symbols at lag `tau` are ``(s[t], s[t + tau], ..., s[t + (length-1)*tau])``
    for every ``t = 0, ..., n - 1 - (length-1)*tau`` (the rows of
    :func:`~pyhctsa.utils.time_delay_embed`), for symbols ``s`` in ``0, ..., alphabet_size - 1``.
    Returns an array of shape ``(alphabet_size,) * length`` whose element ``[i, j, ...]`` is the
    fraction of words ``(i, j, ...)`` (all NaN if there is no word), and the number of words.
    """
    shape = (alphabet_size,) * length
    n_words = len(s) - (length - 1) * tau
    if n_words <= 0:
        return np.full(shape, np.nan), 0
    words = time_delay_embed(s, length, tau).astype(np.int64)
    code = words @ alphabet_size ** np.arange(length - 1, -1, -1)  # first symbol most significant
    return np.bincount(code, minlength=alphabet_size ** length).reshape(shape) / n_words, n_words


@dict_output
def motif_two(y: ArrayLike, binarize_how: str = 'diff', tau: Union[int, str] = 1) -> dict:
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

        - 'diff': Encode increases in the time series as 1, and decreases or no change as 0.
        - 'mean': Encode values above the mean as 1, and at or below the mean as 0.
        - 'median': Encode values above the median as 1, and at or below the median as 0.

        Default is ``'diff'``.

    tau : int or str, optional
        The lag of the words. The whole series is binarized to symbols ``s``, and the words of
        length ``L`` are ``(s[t], s[t + tau], ..., s[t + (L-1)*tau])`` for every ``t``, as in a
        time-delay embedding; for 'diff' the symbols are the signs of the increments at lag
        tau, ``y[t + tau] - y[t]``. An integer, or a rule that sets it from the series:
        ``'ac'`` (the first zero-crossing of the autocorrelation function), ``'ac1e'`` (the
        floor of its first 1/e crossing) or ``'mi'`` (the smaller of the first minimum of the
        Kraskov automutual information and the 'ac1e' delay), or one of these computed from the
        ranks of the series (``'ac-rank'``, ``'ac1e-rank'``, ``'mi-rank'``); see
        :func:`pyhctsa.utils.get_tau`.
        With the order-based symbols ('diff', 'median') and a rank-based rule, the result is
        unchanged by a strictly increasing transformation of the series. Default is 1. NaN is
        returned if tau cannot be determined, or leaves fewer than two words of length 4.

    Returns
    -------
    dict
        The probability of each binary word of length 1 to 4, keyed by the word in the letters
        'u' (1) and 'd' (0) ('u', 'd', 'dd', 'du', ..., 'uuuu'), and the Miller-Madow-corrected
        entropy of the words of each length ('h', 'hh', 'hhh', 'hhhh'). NaN if the series is
        too short for two words of length 4.

    """
    tau = _resolve_tau(np.asarray(y), tau)
    if np.isnan(tau):
        return np.nan

    # Binarize the whole series (for 'diff', the increments at lag tau)
    y_bin = binarize(y, binarize_how, lag=tau).astype(int)

    # A median split fixes the marginal symbol frequencies (used for the
    # Miller-Madow degrees of freedom in _f_entropy)
    fixed_marginals = (binarize_how == 'median')

    if len(y_bin) - 3 * tau < 2:  # fewer than two words of length 4
        logger.warning("Time series too short!")
        return np.nan

    # Probabilities of the words of 1 to 4 symbols at lag tau, and their Miller-Madow-corrected
    # entropies
    out = {}
    for length in range(1, 5):
        p, n_words = _lagged_word_probs(y_bin, length, tau, 2)
        if length == 1:
            out['u'], out['d'] = p[1], p[0]  # 'u': an increase for 'diff'
        else:
            out.update(zip(map(''.join, product('du', repeat=length)), p.ravel()))
        out['h' * length] = _f_entropy(p, n_words, length, 2, fixed_marginals)

    return out

@dict_output
def motif_three(y: ArrayLike, cg_how: str = 'quantile', tau: Union[int, str] = 1) -> dict:
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

    tau : int or str, optional
        The lag of the words. The whole series is coarse-grained to letters ``s``, and the words
        of length ``L`` are ``(s[t], s[t + tau], ..., s[t + (L-1)*tau])`` for every ``t``, as in
        a time-delay embedding; for 'diffquant' the letters code the increments at lag tau,
        ``y[t + tau] - y[t]``. An integer, or a rule that sets it from the series:
        ``'ac'`` (the first zero-crossing of the autocorrelation function), ``'ac1e'`` (the
        floor of its first 1/e crossing) or ``'mi'`` (the smaller of the first minimum of the
        Kraskov automutual information and the 'ac1e' delay), or one of these computed from the
        ranks of the series (``'ac-rank'``, ``'ac1e-rank'``, ``'mi-rank'``); see
        :func:`pyhctsa.utils.get_tau`.
        With 'quantile' and a rank-based rule, the result is unchanged by a strictly increasing
        transformation of the series. Default is 1. NaN is returned if tau cannot be
        determined or leaves no symbol, and each output is NaN for a word length with no word.

    Returns
    -------
    dict
        Statistics on words of length 1, 2, 3, and 4.
    """

    y = np.asarray(y)
    tau = _resolve_tau(y, tau)
    if np.isnan(tau):
        return np.nan

    if cg_how == 'diffquant' and tau >= len(y):  # no increment at this lag
        return np.nan

    # Coarse-grain the whole series (for 'diffquant', its increments at lag tau) to {1, 2, 3}
    if cg_how == 'quantile':
        yt = coarse_grain(y, 'quantile', 3)
    elif cg_how == 'diffquant':
        yt = coarse_grain(y, 'diff', 3, lag=tau)
    else:
        raise ValueError(f"Unknown coarse-graining method {cg_how}")

    # Probabilities of the words of 1 to 4 letters at lag tau, and their Miller-Madow-corrected
    # entropies (NaN if there is no word)
    out = {}
    for length in range(1, 5):
        p, n_words = _lagged_word_probs(yt - 1, length, tau, 3)
        out.update(zip(map(''.join, product('abc', repeat=length)), p.ravel()))
        out['h' * length] = _f_entropy(p, n_words, length, 3, True) if n_words > 0 else np.nan

    return out

def _f_entropy(p, num_samples=None, word_length=1, alphabet_size=2, fixed_marginals=False):
    """
    Miller-Madow-corrected entropy of a probability array, in nats (log(0) = 0).

    The plug-in entropy is biased downwards by df/(2 num_samples), with
    df = (number of occupied words) - 1. When the coarse-graining fixes the
    marginal symbol frequencies (`fixed_marginals`), df is reduced by
    word_length*(alphabet_size - 1), so that for words of length 1 df = 0.
    """
    p = np.asarray(p, dtype=float).ravel()
    r = p > 0
    h = -np.sum(p[r] * np.log(p[r]))
    if num_samples is not None and num_samples > 0:
        df = int(np.sum(r)) - 1
        if fixed_marginals:
            df -= word_length * (alphabet_size - 1)
        if df > 0:
            h += df / (2 * num_samples)
    return h


def binary_stretch(x: ArrayLike, stretch_what: str = 'gaps1') -> float:
    """
    Homogeneity of the gaps between like symbols in a binarized time series.

    This is hctsa's SB_BinaryGapHomogeneity (formerly SB_BinaryStretch). The input is
    binarized at zero (values above zero become 1, the rest 0; the time series is
    typically z-scored first, so this is a split about the mean). The gaps between
    successive 1s (or 0s) are then characterized by the longest block of gaps of one type
    (shorter or longer than one sample) between like symbols, as a proportion of the
    time-series length.

    **Note**: Despite its former name, this does not measure the *longest run* of 0s
    or 1s (an implementation quirk of the original that is retained), but it is a
    potentially interesting statistic.

    Parameters
    ----------
    x : array-like
        The input time series.

    stretch_what : {'gaps1', 'gaps0'}, optional
        Which binary symbol's gaps to analyze (formerly ``'lseq1'`` and ``'lseq0'``):

        - 'gaps1': Analyze gaps between consecutive 1s.
        - 'gaps0': Analyze gaps between consecutive 0s.

        Default is ``'gaps1'``.

    Returns
    -------
    float
        The statistic, normalized by the time-series length (0 if the symbol does not
        occur often enough to define it).
    """
    x = np.asarray(x)
    N = len(x) # time series length
    x = np.where(x > 0, 1, 0)

    if stretch_what == 'gaps1':
        # longest stretch of 1s [this code doesn't actually measure this!]
        indices = np.where(x == 1)[0]
    elif stretch_what == 'gaps0':
        # longest stretch of 0s [this code doesn't actually measure this!]
        indices = np.where(x == 0)[0]
    else:
        raise ValueError(f"Unknown input '{stretch_what}' (expected 'gaps1' or 'gaps0')")

    diffs = np.diff(indices) - 1.5
    sign_changes = sign_change(diffs, 1)
    if sign_changes.size > 1:
        out = np.max(np.diff(sign_changes)) / N
    else:
        out = None

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
        - 'median': Encode as 1 if the value is above the median, 0 otherwise.
        - 'iqr': Encode as 1 if the value is within the interquartile range (above the 25th
          percentile and below the 75th), 0 otherwise.

        Default is ``'diff'``.

    Returns
    -------
    dict
        Statistics computed on the binary symbolisation. The standard deviations of the
        stretch lengths are NaN if there are no stretches of that symbol, and 0 if there
        is a single stretch.
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
        out['stdstretch0'] = _sample_std(stretch0)
        out['stdstretch0norm'] = _sample_std(stretch0) / N

    if len(stretch1) == 0:
        out['longstretch1'] = 0
        out['longstretch1norm'] = 0
        out['meanstretch1'] = 0
        out['meanstretch1norm'] = 0
        out['stdstretch1'] = np.nan
        out['stdstretch1norm'] = np.nan
    else:
        out['longstretch1'] = np.max(stretch1)
        out['longstretch1norm'] = np.max(stretch1) / N
        out['meanstretch1'] = np.mean(stretch1)
        out['meanstretch1norm'] = np.mean(stretch1) / N
        out['stdstretch1'] = _sample_std(stretch1)
        out['stdstretch1norm'] = _sample_std(stretch1) / N
    
    out['meanstretchdiff'] = (out['meanstretch1'] - out['meanstretch0']) / N
    out['stdstretchdiff'] = (out['stdstretch1'] - out['stdstretch0']) / N

    out['diff21stretch1'] = np.mean(stretch1 == 2) - np.mean(stretch1 == 1)
    out['diff21stretch0'] = np.mean(stretch0 == 2) - np.mean(stretch0 == 1)

    return out

def binary_stats_ar1(y: ArrayLike, binary_method: str = 'mean') -> dict:
    """
    Binary run-length statistics normalized against an AR(1) null.

    Binarizes the time series (as :func:`binary_stats`) and compares the resulting
    run-length statistics to their analytic expectation under a Gaussian AR(1) null
    process with the same lag-1 autocorrelation as the series. Ratios near 1 indicate that
    the binary run structure is what linear autocorrelation alone would give.

    For a stationary Gaussian process u (``u = y`` for 'mean'; ``u = diff(y)`` for 'diff',
    i.e., whichever series is actually thresholded at zero), the probability that
    consecutive samples lie on the same side of the mean follows the arcsine law,
    ``p = 1/2 + arcsin(rho)/pi``, with ``rho`` the lag-1 autocorrelation of u. Treating the
    binary sign sequence as a two-state Markov chain with persistence probability p gives
    geometrically distributed run lengths, so the expected mean run length is ``1/(1-p)`` and
    the expected proportion of runs of 1s (per sample) is ``(1-p)/2``. Only the statistics
    whose theory holds up empirically (the mean run lengths and the number of runs) get an
    AR(1)-normalized counterpart; for the fuller set of empirical run-length statistics see
    :func:`binary_stats`.

    Parameters
    ----------
    y : array-like
        The input time series.
    binary_method : {'mean', 'diff'}, optional
        The binary symbolization rule: 'mean' (1 above the mean, 0 below) or 'diff' (1 for an
        increase, 0 otherwise). Unlike :func:`binary_stats`, 'median' and 'iqr' are not
        supported: the theory is specific to a sign threshold at the mean of a (possibly
        transformed) Gaussian series. Default is ``'mean'``.

    Returns
    -------
    dict
        - 'pstretch1': the number of runs of 1s divided by the length of the binary string,
        - 'meanstretch0', 'meanstretch1': the mean run length of 0s, and of 1s (NaN if there
          are no such runs),
        - 'ar1_p': the AR(1)-implied persistence probability p,
        - 'meanstretch_ar1exp': the expected mean run length, ``1/(1-p)``,
        - 'pstretch1_ar1exp': the expected value of 'pstretch1', ``(1-p)/2``,
        - 'meanstretch0_ar1rat', 'meanstretch1_ar1rat': 'meanstretch0' and 'meanstretch1'
          divided by 'meanstretch_ar1exp',
        - 'pstretch1_ar1rat': 'pstretch1' divided by 'pstretch1_ar1exp'.

        In the degenerate limit of a lag-1 autocorrelation of 1 the ratios are NaN,
        'meanstretch_ar1exp' is Inf and 'pstretch1_ar1exp' is 0.
    """
    if binary_method not in ('mean', 'diff'):
        raise ValueError(f"binary_stats_ar1 supports binary_method 'mean' or 'diff' only "
                         f"(not '{binary_method}')")
    y = np.asarray(y, dtype=float)

    # The series that is actually sign-thresholded at its mean
    u = y if binary_method == 'mean' else np.diff(y)

    # Binarize (the case for which the arcsine law below is exact)
    y_bin = binarize(y, binarize_how=binary_method)
    N = len(y_bin)  # note: N = len(y) - 1 for 'diff'

    # Empirical run-length statistics (cf. binary_stats)
    diff_y = np.diff(np.where(np.concatenate(([1], y_bin, [1])))[0])
    stretch0 = diff_y[diff_y != 1] - 1
    diff_y = np.diff(np.where(np.concatenate(([0], y_bin, [0])) == 0)[0])
    stretch1 = diff_y[diff_y != 1] - 1

    out = {}
    out['pstretch1'] = len(stretch1) / N
    out['meanstretch0'] = np.mean(stretch0) if len(stretch0) else np.nan  # all 1s: no runs of 0s
    out['meanstretch1'] = np.mean(stretch1) if len(stretch1) else np.nan  # all 0s: no runs of 1s

    # AR(1)-null persistence probability, via the arcsine law
    with np.errstate(all='ignore'):
        rho = autocorr(u, 1, 'Fourier')
    # guard against tiny numerical overshoot outside [-1, 1]; a NaN (constant series) is
    # taken as 1
    rho = 1.0 if np.isnan(rho) else max(min(rho, 1.0), -1.0)
    p = 0.5 + np.arcsin(rho) / np.pi
    out['ar1_p'] = p

    if p >= 1 - 1e-8:  # degenerate limit (rho -> 1): the expected run length diverges
        out['meanstretch_ar1exp'] = np.inf
        out['meanstretch0_ar1rat'] = np.nan
        out['meanstretch1_ar1rat'] = np.nan
        out['pstretch1_ar1exp'] = 0.0
        out['pstretch1_ar1rat'] = np.nan
        return out

    exp_mean_stretch = 1 / (1 - p)
    exp_pstretch1 = (1 - p) / 2

    # The null expectation for meanstretch0 and meanstretch1 is the same value (symmetric
    # about the mean by construction)
    out['meanstretch_ar1exp'] = exp_mean_stretch
    out['meanstretch0_ar1rat'] = out['meanstretch0'] / exp_mean_stretch
    out['meanstretch1_ar1rat'] = out['meanstretch1'] / exp_mean_stretch
    out['pstretch1_ar1exp'] = exp_pstretch1
    out['pstretch1_ar1rat'] = out['pstretch1'] / exp_pstretch1

    return out


@dict_output
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
    .. [1] Campanharo, A. S. L. O., Sirer, M. I., Malmgren, R. D., Ramos, F. M. & Amaral, L. A. N. (2011).
        Duality between Time Series and Networks. PLoS ONE 6(8), e23378.
        https://doi.org/10.1371/journal.pone.0023378

    Parameters
    -----------
    y : array-like
        Input time series.
    how_to_cg : str, optional
        The method of discretization: ``'quantile'`` (equiprobable, the default) or
        ``'updown'`` (a true binary up/down split by the sign of each increment: NOT
        equiprobable, and requires ``num_groups=2``; see :func:`coarse_grain`).
        ``'diff'`` (equiprobable by increment) is also accepted.
    num_groups : int, optional
        number of groups in the course-graining. Default is 2.
    tau : int or str, optional
        The lag of the transitions. The whole series is discretized to states ``s``, and the
        transitions are the pairs ``(s[t], s[t + tau])`` for every ``t``; for 'diff' and
        'updown' the states code the increments at lag tau, ``y[t + tau] - y[t]``. An integer,
        or a rule that sets it from the series:
        ``'ac'`` (the first zero-crossing of the autocorrelation function), ``'ac1e'`` (the
        floor of its first 1/e crossing) or ``'mi'`` (the smaller of the first minimum of the
        Kraskov automutual information and the 'ac1e' delay), or one of these computed from the
        ranks of the series (``'ac-rank'``, ``'ac1e-rank'``, ``'mi-rank'``); see
        :func:`pyhctsa.utils.get_tau`.
        With 'quantile' or 'updown' and a rank-based rule, the result is unchanged by a strictly
        increasing transformation of the series. Default is 1. NaN is returned if tau cannot be
        determined or leaves no transition.

    Returns
    -------
    dict 
        A dictionary including the transition probabilities themselves, as well as the trace
        of the transition matrix, measures of asymmetry (``symdiff``, ``symsumdiff`` and the
        Kullback-Leibler divergence ``transKLdiv`` between the matrix and its transpose),
        eigenvalues of the transition matrix (including ``secondeig``, ``specgap`` and
        ``lam2mod``, the modulus of the second-largest-modulus eigenvalue of the
        row-normalized matrix), and the Miller-Madow-corrected conditional entropy of the
        next state ``transEntropy``. NaN is returned if tau cannot be determined or leaves no
        transition.
        Note that the matrix is normalized by the number of transitions, so it holds joint
        (not row-normalized) probabilities.
    """
    # check inputs
    y = np.asarray(y, dtype=float)
    if num_groups < 2:
        raise ValueError('Too few groups for coarse-graining')
    tau = _resolve_tau(y, tau)
    if np.isnan(tau):  # undefined delay (e.g., constant series, or an ACF that never falls to 1/e)
        return np.nan
    if len(y) - tau * (2 if how_to_cg in ('diff', 'updown') else 1) < 1:  # no transition at this lag
        return np.nan

    # (((1))) Discretize the whole time series (for 'diff' and 'updown', its increments at lag
    # tau) to a symbolic string, containing integers from 1 to num_groups
    yth = coarse_grain(y, how_to_cg, num_groups, lag=tau)

    # (((2))) Compute the tau-step transition matrix (Markov for tau = 1)
    T = _transition_matrix(yth, num_groups, tau)

    # (((3))) Output measures from the transition matrix
    out = {}

    # (i) Raw values of the transition matrix; only for num_groups = 2, 3 are all
    # elements returned (in column-major order, T1 = T[0, 0], T2 = T[1, 0], ...), otherwise just the diagonal
    if num_groups in (2, 3):
        for i, v in enumerate(T.flatten(order='F')):
            out[f'T{i+1}'] = v
    else:
        for i in range(num_groups):
            out[f'TD{i+1}'] = T[i, i]

    # (ii) Measures on the diagonal
    diag_t = np.diag(T)
    out['ondiag'] = np.sum(diag_t)  # trace
    out['stddiag'] = _sample_std(diag_t)  # std of diagonal elements

    # (iii) Measures of symmetry:
    out['symdiff'] = np.sum(np.abs(T - T.T))  # sum of differences of individual elements
    # difference in sums of upper and lower triangular parts of T
    out['symsumdiff'] = np.sum(np.tril(T, -1)) - np.sum(np.triu(T, 1))

    # Kullback-Leibler divergence between T and its transpose, over the pairs where both
    # T(i,j) and T(j,i) are nonzero (a reversal-asymmetry measure, 0 iff T is symmetric)
    t_f = T.flatten(order='F')
    tt_f = T.T.flatten(order='F')
    kl_mask = (t_f > 0) & (tt_f > 0)
    out['transKLdiv'] = np.sum(t_f[kl_mask] * np.log(t_f[kl_mask] / tt_f[kl_mask]))

    # (iv) Measures from eigenvalues of T
    eig_t = np.linalg.eigvals(T)
    out['stdeig'] = _sample_std(eig_t)  # std of eigenvalues
    out['maxeig'] = np.max(np.real(eig_t))  # maximum eigenvalue
    out['mineig'] = np.min(np.real(eig_t))  # minimum eigenvalue
    # mean eigenvalue is equivalent to the trace
    out['maximeig'] = np.max(np.imag(eig_t))  # maximum imaginary part of eigenvalues

    # Second-largest (real) eigenvalue and the spectral gap (num_groups >= 2, so a second
    # eigenvalue always exists)
    real_eig = np.sort(np.real(eig_t))[::-1]
    out['secondeig'] = real_eig[1]
    out['specgap'] = out['maxeig'] - out['secondeig']

    # Modulus of the second-largest-modulus eigenvalue of the row-normalized transition
    # matrix P(i,j) = T(i,j)/sum_j T(i,j); NaN if some state never occurs as a source
    src_prob = T.sum(axis=1)
    if np.any(src_prob == 0):
        out['lam2mod'] = np.nan
    else:
        abs_eig_p = np.sort(np.abs(np.linalg.eigvals(T / src_prob[:, None])))[::-1]
        out['lam2mod'] = abs_eig_p[1]

    # Transition (conditional) entropy, H(X_{t+1}|X_t) = H(joint) - H(marginal), with a
    # Miller-Madow correction (M_joint - M_marginal)/(2 (N - 1)) for the number of
    # occupied bins
    row_sums = src_prob
    p_joint = t_f[t_f > 0]
    p_marg = row_sums[row_sums > 0]
    h_joint = -np.sum(p_joint * np.log(p_joint))
    h_marginal = -np.sum(p_marg * np.log(p_marg))
    out['transEntropy'] = (h_joint - h_marginal
                           + (p_joint.size - p_marg.size) / (2 * (len(yth) - tau)))

    # (v) Measures from the covariance matrix:
    cov_t = np.cov(T, rowvar=False)
    out['sumdiagcov'] = np.trace(cov_t)  # trace of covariance matrix

    # (vi) Eigenvalues of the covariance matrix. It is symmetric, so its
    # eigenvalues are real, as `eigvalsh` returns
    # (these measures don't make much sense in the case of 2 groups):
    eig_cov_t = np.linalg.eigvalsh(cov_t)
    out['stdeigcov'] = _sample_std(eig_cov_t)  # std of eigenvalues of covariance matrix
    out['maxeigcov'] = np.max(eig_cov_t)  # max eigenvalue of covariance matrix
    out['mineigcov'] = np.min(eig_cov_t)  # min eigenvalue of covariance matrix

    return out


def _linear_adjr2(x: np.ndarray, y: np.ndarray) -> float:
    """Adjusted R^2 of an ordinary least-squares line fitted to y against x."""
    n = len(y)
    p = np.polyfit(x, y, 1)
    rsq = 1 - np.sum((y - np.polyval(p, x)) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return 1 - (1 - rsq) * (n - 1) / (n - 2)


def _transition_matrix(yth: np.ndarray, num_groups: int, tau: int = 1) -> np.ndarray:
    """
    The tau-step transition matrix of a symbolized time series: the fraction of the pairs
    ``(yth[t], yth[t + tau])`` that go from state i to state j, for states 1 to `num_groups`.
    """
    return _lagged_word_probs(np.asarray(yth) - 1, 2, tau, num_groups)[0]


def _transition_measures(yth: np.ndarray, num_groups: int, tau: int = 1) -> np.ndarray:
    """A set of metrics on the tau-step transition matrix of a symbolized time series."""
    T = _transition_matrix(yth, num_groups, tau)

    out = np.zeros(6)
    #   (i) diagonal elements
    diag_t = np.diag(T)
    out[0] = np.mean(diag_t)
    out[1] = np.max(diag_t)
    out[2] = np.sum(diag_t)  # trace

    #  (ii) measures of symmetry:
    out[3] = np.sum(np.abs(T - T.T))  # sum of differences of individual elements

    # (iii) measures from covariance matrix:
    out[4] = np.trace(np.cov(T, rowvar=False))  # trace of the covariance matrix

    # (iv) measures from eigenvalues of T
    eig_t = np.linalg.eigvals(T)
    out[5] = _sample_std(eig_t)

    return out


@dict_output
def transition_p_alphabet(y: ArrayLike, num_groups: Optional[ArrayLike] = None,
                          tau: Union[int, str] = 1) -> dict:
    """
    How transition probabilities change with alphabet size.

    The time series is discretized by quantile separation into alphabets of a range
    of sizes, and the tau-step transition matrix is computed for each. Statistics of
    those transition matrices are then tracked as a function of the alphabet size.
    The exponential fits are global least-squares fits (:func:`~pyhctsa.robust.bf_exp_fit`,
    with no offset), so R^2 lies between 0 and 1; the linear fits are ordinary least squares.

    Parameters
    ----------
    y : array-like
        The input time series.
    num_groups : array-like, optional
        The range of alphabet sizes to compare across. Must contain more than one
        value, each at least 2. Default is ``range(2, 11)``.
    tau : int or str, optional
        The lag of the transitions: the pairs ``(s[t], s[t + tau])`` of the discretized series,
        for every ``t``. An integer, or a rule that sets it from the series:
        ``'ac'`` (the first zero-crossing of the autocorrelation function), ``'ac1e'`` (the
        floor of its first 1/e crossing) or ``'mi'`` (the smaller of the first minimum of the
        Kraskov automutual information and the 'ac1e' delay), or one of these computed from the
        ranks of the series (``'ac-rank'``, ``'ac1e-rank'``, ``'mi-rank'``); see
        :func:`pyhctsa.utils.get_tau`.
        With a rank-based rule, the result is unchanged by a strictly increasing transformation
        of the series. NaN is returned if the delay is undefined or leaves no transition.
        Default is 1.

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

    if np.size(tau) > 1 or num_groups.size == 1:
        # (hctsa does not support varying tau either: "This setting kind of doesn't work yet")
        raise NotImplementedError('Only a scalar tau with a range of alphabet sizes is '
                                  'supported.')
    tau = _resolve_tau(y, tau if isinstance(tau, str) else np.ravel(tau)[0])
    if np.isnan(tau) or tau >= len(y):  # undefined delay (e.g., constant series), or no transition
        return np.nan

    if np.min(num_groups) < 2:
        raise ValueError('Need more than 2 groups')

    num_groups_range = num_groups

    nfeat = 6  # the number of features calculated at each point
    store = np.zeros((len(num_groups_range), nfeat))
    for i, ng in enumerate(num_groups_range):
        yth = coarse_grain(y, 'quantile', int(ng))  # thresholded data: yth
        store[i, :] = _transition_measures(yth, int(ng), tau)

    x = num_groups_range.astype(float)
    n = len(x)
    out = {}

    # 1) mean of diagonal elements of the transition matrix: shows an exponential
    # decay to zero
    fit = bf_exp_fit(x, store[:, 0], False)
    out['meandiagfexp_a'] = fit['a']
    out['meandiagfexp_b'] = fit['b']
    out['meandiagfexp_r2'] = fit['r2']
    out['meandiagfexp_adjr2'] = fit['adjr2']
    out['meandiagfexp_rmse'] = fit['rmse']

    # 2) maximum of diagonal elements of the transition matrix: shows an exponential
    # decay to zero
    fit = bf_exp_fit(x, store[:, 1], False)
    out['maxdiagfexp_a'] = fit['a']
    out['maxdiagfexp_b'] = fit['b']
    out['maxdiagfexp_r2'] = fit['r2']
    out['maxdiagfexp_adjr2'] = fit['adjr2']
    out['maxdiagfexp_rmse'] = fit['rmse']

    # 3) trace of T -- fit exponential
    fit = bf_exp_fit(x, store[:, 2], False)
    out['trfexp_a'] = fit['a']
    out['trfexp_b'] = fit['b']
    out['trfexp_r2'] = fit['r2']
    out['trfexp_adjr2'] = fit['adjr2']
    out['trfexp_rmse'] = fit['rmse']

    # Also fit linear from the start to a fifth, a tenth of the starting value
    for thresh, name in ((5, 'trflin5_adjr2'), (10, 'trflin10adjr2')):
        r = np.flatnonzero(store[:, 2] > store[0, 2] / thresh)
        if len(r) > 2:
            out[name] = _linear_adjr2(x[r], store[r, 2])
        else:
            out[name] = np.nan

    # 4) Symmetry; differences in diagonal elements -- return the slope
    out['symd_a'] = np.polyfit(x, store[:, 3], 1)[0]

    # return approximately when starts to rise; where means before and
    # after a moving dividing point are most different
    if np.all(store[:, 3] == store[0, 3]):  # all the same
        out['symd_risept'] = np.nan
    else:
        mba = np.zeros((n, 2))  # means before and after
        sba = np.zeros((n, 2))  # standard deviation before and after
        for i in range(2, n - 2):
            mba[i, 0] = np.mean(store[:i, 3])
            sba[i, 0] = _sample_std(store[:i, 3]) / np.sqrt(i)
            after = store[i + 1:, 3]
            mba[i, 1] = np.mean(after)
            sba[i, 1] = _sample_std(after) / np.sqrt(n - i)
        with np.errstate(invalid='ignore', divide='ignore'):
            tstats = np.abs((mba[:, 0] - mba[:, 1]) / np.sqrt(sba[:, 0]**2 + sba[:, 1]**2))
        if np.all(np.isnan(tstats)):
            out['symd_risept'] = np.nan
        else:
            # the 1-based index of the first maximum, ignoring NaNs
            out['symd_risept'] = float(np.nanargmax(tstats) + 1)

    # 5) trace of covariance matrix -- check jump:
    out['trcov_jump'] = store[1, 4] - store[0, 4]
    r1 = np.arange(1, n) if store[1, 4] > store[0, 4] else np.arange(n)
    # fit exponential decay to range without possible first jump
    fit = bf_exp_fit(x[r1], store[r1, 4], False)
    out['trcovfexp_a'] = fit['a']
    out['trcovfexp_b'] = fit['b']
    out['trcovfexp_r2'] = fit['r2']
    out['trcovfexp_adjr2'] = fit['adjr2']
    out['trcovfexp_rmse'] = fit['rmse']

    # 6) Standard deviation of eigenvalues of T -- fit an exponential decay
    fit = bf_exp_fit(x, store[:, 5], False)
    out['stdeigfexp_a'] = fit['a']
    out['stdeigfexp_b'] = fit['b']
    out['stdeigfexp_r2'] = fit['r2']
    out['stdeigfexp_adjr2'] = fit['adjr2']
    out['stdeigfexp_rmse'] = fit['rmse']

    return out


def coarse_grain(y: list, how_to_cg: str, num_groups: Union[int, str], lag: int = 1) -> np.ndarray:
    """
    Coarse-grains a continuous time series to a discrete alphabet.

    Parameters
    -----------
    y : array-like
        The input time series.
    how_to_cg : str
        The method of coarse-graining.
        Options: 

        - 'quantile': an equiprobable alphabet by the value of each point,
        - 'diff': as 'quantile', but applied to the increments ``diff(y)``: an equiprobable
          alphabet by the *value* of each increment (not a literal sign split; this was called
          'updown' before hctsa renamed it),
        - 'updown': a true binary up/down split by the raw sign of each increment
          (``diff(y) > 0`` gives symbol 2, otherwise 1); ``num_groups`` must be 2. Unlike the
          other methods this is NOT equiprobable: the two states can be arbitrarily
          imbalanced for a drifting series,
        - 'embed2quadrants', 'embed2octants': the alphabet is the quadrant (4 symbols) or
          octant (8 symbols) of each point of a 2-D time-delay embedding.

    num_groups : int or str
        The size of the alphabet for 'quantile' and 'diff' (must be 2 for 'updown'), or the
        time delay for the embedding methods: a number of samples, or a string that sets it
        from the series: ``'ac1e'`` (the floor of the first 1/e crossing of the
        autocorrelation function), ``'mi'`` (the smaller of the first minimum of the Kraskov
        automutual information and the 'ac1e' delay; see :func:`pyhctsa.utils.get_tau`), or
        ``'tau'`` (the first zero-crossing of the autocorrelation function, kept for
        backward compatibility). A delay is capped at floor(N/25).
    lag : int, optional
        For 'diff' and 'updown', the lag of the increments, ``y[t + lag] - y[t]``. Default is 1.

    Returns
    --------
    yth : array-like
        The coarse-grained time series. NaN (a scalar) if the embedding delay for
        'embed2quadrants'/'embed2octants' cannot be determined (``num_groups='tau'`` or
        ``'ac1e'`` for a constant series, or ``'ac1e'`` for a series whose autocorrelation
        function never falls to 1/e).
    """
    y = np.asarray(y)
    N = len(y)

    if how_to_cg not in ['updown', 'diff', 'quantile', 'embed2quadrants', 'embed2octants']:
        raise ValueError(f"Unknown coarse-graining method '{how_to_cg}'")

    if how_to_cg == 'updown' and num_groups != 2:
        raise ValueError(f"'updown' is a true binary up/down split: num_groups must be 2 "
                         f"(got {num_groups}). Use 'diff' for a multi-level equiprobable "
                         "alphabet of the increments.")

    # Some coarse-graining/symbolization methods require initial processing:
    yth = None  # Ensure yth is always defined
    if how_to_cg == 'diff':
        y = y[lag:] - y[:-lag]
        N = N - lag  # the increments are lag values fewer than the input
        how_to_cg = 'quantile' # successive differences and then quantiles

    elif how_to_cg == 'updown':
        # True binary up/down split: 2 if the increment is positive, 1 otherwise
        y = y[lag:] - y[:-lag]
        N = N - lag
        yth = 1 + (y > 0).astype(int)

    elif how_to_cg in ['embed2quadrants', 'embed2octants']:
        # Construct the embedding
        if isinstance(num_groups, str):
            if num_groups in ('ac1e', 'mi'):
                # adaptive delay (NaN if undefined, or if the ACF never crosses 1/e)
                tau = get_tau(y, num_groups)
            elif num_groups == 'tau':
                # first zero-crossing of the ACF
                tau = get_tau(y, 'ac')
            else:
                raise ValueError(f"Unknown embedding delay '{num_groups}': use a number of "
                                 "samples, 'ac1e', 'mi' or 'tau'")
            if np.isnan(tau):  # undefined delay: no coarse-graining
                return np.nan
        else:
            tau = num_groups

        if tau > N/25:
            tau = max(1, N // 25)  # (hctsa's N // 25 is a degenerate delay of 0 for N < 25)
        tau = int(tau)

        m1 = y[:N - tau]
        m2 = y[tau:]

        # Look at which points are in which angular 'quadrant'
        upr = m2 >= 0 # points above the axis
        downr = m2 < 0 # points below the axis

        q1r = np.logical_and(upr, m1 >= 0) # points in quadrant 1
        q2r = np.logical_and(upr, m1 < 0) # points in quadrant 2
        q3r = np.logical_and(downr, m1 < 0) # points in quadrant 3
        q4r = np.logical_and(downr, m1 >= 0) # points in quadrant 4
    
    # Do the coarse graining
    if how_to_cg == 'quantile':
        th = np.quantile(y, np.linspace(0, 1, num_groups + 1), method='hazen') # thresholds for dividing the time-series values
        th[0] = th[0] - 1 # this ensures the first point is included
        yth = np.zeros(N, dtype=int)
        # turn the time series into a set of numbers from 1:num_groups
        for i in range(num_groups):
            yth[(y > th[i]) & (y <= th[i+1])] = i + 1

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
