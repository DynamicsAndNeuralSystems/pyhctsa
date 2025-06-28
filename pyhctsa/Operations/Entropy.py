import numpy as np
from typing import Union
from numba import njit
from numpy.typing import ArrayLike
from sklearn.neighbors import KDTree
from ..Utilities.utils import ZScore
from ..Toolboxes.Max_Little.rpde_wrapper import close_returns_analysis

def RPDE(y: ArrayLike, m: int = 2, tau: int = 1, epsilon: float = 0.12, TMax : int = -1) -> dict:
    """
    Recurrence period density entropy (RPDE).

    Fast RPDE analysis on an input signal to obtain an estimate of the normalized entropy (H_norm)
    and other related statistics. Based on Max Little's original rpde code.

    Parameters
    ----------
    y : array-like
        Input signal (must be a 1D array or list).
    m : int, optional
        Embedding dimension (default: 2).
    tau : int, optional
        Embedding time delay (default: 1).
    epsilon : float, optional
        Recurrence neighbourhood radius (default: 0.12).
    TMax : int, optional
        Maximum recurrence time. If not specified (default: -1), all recurrence times are returned.

    Returns
    -------
    dict
        Dictionary containing:
            - 'H_norm': Estimated normalized RPDE value.
            - 'H': Unnormalized entropy.
            - 'rpd': Recurrence period density (probability distribution).
            - 'propNonZero': Proportion of non-zero entries in rpd.
            - 'meanNonZero': Mean value of non-zero rpd entries (rescaled by N).
            - 'maxRPD': Maximum value of rpd (rescaled by N).

    References
    ----------
    M. Little, P. McSharry, S. Roberts, D. Costello, I. Moroz (2007),
    "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection",
    BioMedical Engineering OnLine 2007, 6:23.
    """

    y = np.asarray(y)
    rpd = np.array(close_returns_analysis(y, m, tau, epsilon))
    if TMax > -1:
        rpd = rpd[:TMax]
    rpd = np.divide(rpd, np.sum(rpd))
    N = len(rpd)
    ip = rpd > 0
    H = -np.sum(rpd[ip] * np.log(rpd[ip]))
    H_norm = np.divide(H, np.log(N))
    out = {}
    out['H'] = H
    out["H_norm"] = H_norm

    # prop of non-zero entries
    out['propNonZero'] = np.mean(rpd > 0) # proportion of rpds that are non-zero
    out['meanNonZero'] = np.mean(rpd[ip]) * N # mean value when rpd is non-zero (rescale by N)
    out['maxRPD'] = np.max(rpd) * N # maximum value of rpd (rescale by N)

    return out 

def ApproximateEntropy(x : ArrayLike, mnom : int = 1, rth : float = 0.2) -> float:
    """
    Approximate Entropy of a time series

    ApEn(m,r).

    Parameters
    -----------
    y : array-like
        The input time series
    mnom : int, optional
        The embedding dimension (default is 1)
    rth : float, optional
        The threshold for judging closeness/similarity (default is 0.2)

    Returns
    --------
    float
        The Approximate Entropy value

    References:
    -----------
    S. M. Pincus, "Approximate entropy as a measure of system complexity",
    P. Natl. Acad. Sci. USA, 88(6) 2297 (1991)

    For more information, cf. http://physionet.org/physiotools/ApEn/
    """
    x = np.asarray(x)
    r = rth * np.std(x, ddof=1) # threshold of similarity
    phi = _app_samp_entropy(x, order=mnom, r=r, metric="chebyshev", approximate=True)
    return np.subtract(phi[0], phi[1])

def _embed(x, order, delay=1):
    """Safe embedding that supports order=1."""
    x = np.asarray(x)
    if order < 1:
        raise ValueError("Order must be at least 1.")
    N = x.shape[0]
    if N - (order - 1) * delay <= 0:
        raise ValueError("Time series is too short for the given order and delay.")
    return np.array([x[i:i + order * delay:delay] for i in range(N - (order - 1) * delay)])

def _app_samp_entropy(x, order, r, metric="chebyshev", approximate=True):
    """Modified version of _app_samp_entropy that supports order=1."""
    phi = np.zeros(2)
    emb_data1 = _embed(x, order, 1)
    if approximate:
        pass
    else:
        emb_data1 = emb_data1[:-1]

    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64)

    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi

def ComplexityInvariantDistance(y : ArrayLike) -> dict:
    """
    Complexity-invariant distance

    Computes two estimates of the 'complexity' of a time series based on the 
    stretched-out length of the lines in its line graph. These features are 
    based on the method described by Batista et al. (2014), designed for use 
    in complexity-invariant distance calculations.

    Parameters
    ----------
    y : array-like
        One-dimensional time series input.

    Returns
    -------
    dict
        A dictionary containing the following features:
        
        - 'CE1' : float
            Root mean square of successive differences.
        - 'CE2' : float
            Mean length of line segments between consecutive points using 
            Euclidean distance (Pythagorean theorem).
        - 'minCE1' : float
            Minimum CE1 value computed from sorted time series.
        - 'minCE2' : float
            Minimum CE2 value computed from sorted time series.
        - 'CE1_norm' : float
            Normalized CE1: CE1 / minCE1.
        - 'CE2_norm' : float
            Normalized CE2: CE2 / minCE2.

    References
    ----------
    Batista, G. E. A. P. A., Keogh, E. J., Tataw, O. M., & de Souza, V. M. A. 
    (2014). CID: an efficient complexity-invariant distance for time series. 
    Data Mining and Knowledge Discovery, 28(3), 634–669. 
    https://doi.org/10.1007/s10618-013-0312-3
    """
    y = np.asarray(y)
    #% Original definition (in Table 2 of paper cited above)
    # % sum -> mean to deal with non-equal time-series lengths
    # % (now scales properly with length)

    f_CE1 = lambda y: np.sqrt(np.mean(np.power(np.diff(y), 2)))
    #% Definition corresponding to the line segment example in Fig. 9 of the paper
    #% cited above (using Pythagoras's theorum):
    f_CE2 = lambda y: np.mean(np.sqrt(1 + np.power(np.diff(y), 2)))

    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    # % Defined as a proportion of the minimum such value possible for this time series,
    # % this would be attained from putting close values close; i.e., sorting the time
    # % series
    y_sorted = np.sort(y)
    minCE1 = f_CE1(y_sorted)
    minCE2 = f_CE2(y_sorted)

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':       CE1,
           'CE2':       CE2,
           'minCE1':    minCE1,
           'minCE2':    minCE2,
           'CE1_norm':  CE1_norm,
           'CE2_norm':  CE2_norm}

    return out

def LZComplexity(x: ArrayLike, nbits: int = 2, preProc: Union[str, list] = [], rng : int = 0) -> float:
    """
    Compute the normalized Lempel-Ziv (LZ) complexity of an n-bit encoding of a time series.

    This function measures the complexity of a time series by counting the number of distinct
    symbol sequences (phrases) in its n-bit symbolic encoding, normalized by the expected
    number for a random (noise) sequence. Optionally, a preprocessing step can be applied
    before symbolization.

    Parameters
    ----------
    x : array-like
        Input time series (1-D array or list).
    nbits : int, optional
        Number of bits (alphabet size) to encode the data into (default: 2).
    preProc : str or list, optional
        Preprocessing method to apply before symbolization. Currently supported:
            - 'diff': Use z-scored first differences of the time series.
    rng : int, optional
        Random seed for reproducibility (default: 0). Used for adding small random
        noise to break ties during symbolization.

    Returns
    -------
    float
        Normalized Lempel-Ziv complexity: the number of distinct symbol sequences
        divided by the expected number for a noise sequence.
    """
    rng = np.random.RandomState(rng) # fix the seed for reproducibility
    x = np.asarray(x, dtype=np.float64).ravel()
    if preProc == "diff":
        x = ZScore(np.diff(x))

    if x.size == 0 or nbits < 2:
        return 0.0

    symbols = _symbolise_lz(x, nbits, rng)
    c = _lz_complexity(symbols)

    return (c * np.log(x.size)) / (x.size * np.log(nbits))

@njit(cache=True, fastmath=True)
def _lz_complexity(symbols: np.ndarray) -> int:
    """
    Count phrases exactly as in the original Python function.
    Input must be a 1-D int32/64 NumPy array whose values start at 1.
    """
    n = symbols.size
    if n == 0:
        return 0

    c  = 1 # phrase counter
    ns = 1 # phrase start
    nq = 1  # phrase length
    k  = 2 # overall scan pointer

    while k < n:
        is_substring = False
        max_i = ns - nq
        # brute-force search
        for i in range(max_i + 1):
            match = True
            for j in range(nq):
                if symbols[i + j] != symbols[ns + j]:
                    match = False
                    break
            if match:
                is_substring = True
                break

        if is_substring:
            nq += 1
        else:
            c  += 1
            ns += nq
            nq  = 1
        k += 1

    return c

def _symbolise_lz(x: np.ndarray, n_bins: int, rng) -> np.ndarray:
    nx = x.size
    noisy = x + np.finfo(np.float64).eps * rng.randn(nx)
    order = np.argsort(noisy, kind="mergesort") 
    ranks = np.arange(1, nx + 1)
    symbols = np.floor(ranks * (n_bins / (nx + 1)))

    out = np.empty_like(symbols)
    out[order] = symbols + 1                          
    return out
