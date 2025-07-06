import numpy as np
from typing import Union
from ..Operations.Correlation import FirstCrossing
from ..Utilities.utils import binarize, signChange
from scipy.signal import resample as ssre
from numpy.typing import ArrayLike
from loguru import logger

def MotifTwo(y : ArrayLike, binarizeHow : str = 'diff') -> dict:
    """
    Compute local motifs in a binary symbolization of the input time series.

    This function coarse-grains the input time series into a binary sequence
    using the specified binarization method, and computes the probabilities
    of binary words of lengths 1 through 4, along with their entropies.

    Parameters
    ----------
    y : array-like
        The input time series.

    binarizeHow : str, optional, default='diff'
        The method used for binary transformation. One of:
            - 'diff': Encode increases in the time series as 1, and decreases as 0.
            - 'mean': Encode values above the mean as 1, and below as 0.
            - 'median': Encode values above the median as 1, and below as 0.

    Returns
    -------
    result : dict
        A dictionary containing:
            - 'prob_len_1', 'prob_len_2', ..., 'prob_len_4': 
              Lists of probabilities for each binary word of lengths 1 to 4.
            - 'entropy_len_1', 'entropy_len_2', ..., 'entropy_len_4': 
              Entropy values associated with the word distributions of lengths 1 to 4.

    """
    # Generate a binarized version of the input time series
    y = np.asarray(y)
    yBin = binarize(y, binarizeHow)

    # Define the length of the new, symbolized sequence, N
    N = len(yBin)

    if N < 5:
        logger.warning("Time series too short!")
        return np.nan
    
    # Binary sequences of length 1
    r1 = (yBin == 1) # 1
    r0 = (yBin == 0) # 0

    # ------ Record these -------
    # (Will be dependent outputs since signal is binary, sum to 1)
    # (Default hctsa library measures just the u output: up)
    out = {}
    out['u'] = np.mean(r1) # proportion 1 (corresponds to a movement up for 'diff')
    out['d'] = np.mean(r0) # proportion 0 (corresponds to a movement down for 'diff')
    pp = np.array([out['d'], out['u']])
    out['h'] = _f_entropy(pp)

    # Binary sequences of length 2:
    r1 = r1[:-1]
    r0 = r0[:-1]

    r00 = np.logical_and(r0, yBin[1:] == 0)
    r01 = np.logical_and(r0, yBin[1:] == 1)
    r10 = np.logical_and(r1, yBin[1:] == 0)
    r11 = np.logical_and(r1, yBin[1:] == 1)

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
    r000 = np.logical_and(r00, yBin[2:] == 0)
    # 001 
    r001 = np.logical_and(r00, yBin[2:] == 1)
    r010 = np.logical_and(r01, yBin[2:] == 0)
    r011 = np.logical_and(r01, yBin[2:] == 1)
    r100 = np.logical_and(r10, yBin[2:] == 0)
    r101 = np.logical_and(r10, yBin[2:] == 1)
    r110 = np.logical_and(r11, yBin[2:] == 0)
    r111 = np.logical_and(r11, yBin[2:] == 1)

    # ----- Record these -----
    out['ddd'] = np.mean(r000)
    out['ddu'] = np.mean(r001)
    out['dud'] = np.mean(r010)
    out['duu'] = np.mean(r011)
    out['udd'] = np.mean(r100)
    out['udu'] = np.mean(r101)
    out['uud'] = np.mean(r110)
    out['uuu'] = np.mean(r111)

    ppp = np.array([out['ddd'], out['ddu'], out['dud'], out['duu'], out['udd'], out['udu'], out['uud'], out['uuu']])
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

    r0000 = np.logical_and(r000, yBin[3:] == 0)
    r0001 = np.logical_and(r000, yBin[3:] == 1)
    r0010 = np.logical_and(r001, yBin[3:] == 0)
    r0011 = np.logical_and(r001, yBin[3:] == 1)
    r0100 = np.logical_and(r010, yBin[3:] == 0)
    r0101 = np.logical_and(r010, yBin[3:] == 1)
    r0110 = np.logical_and(r011, yBin[3:] == 0)
    r0111 = np.logical_and(r011, yBin[3:] == 1)
    r1000 = np.logical_and(r100, yBin[3:] == 0)
    r1001 = np.logical_and(r100, yBin[3:] == 1)
    r1010 = np.logical_and(r101, yBin[3:] == 0)
    r1011 = np.logical_and(r101, yBin[3:] == 1)
    r1100 = np.logical_and(r110, yBin[3:] == 0)
    r1101 = np.logical_and(r110, yBin[3:] == 1)
    r1110 = np.logical_and(r111, yBin[3:] == 0)
    r1111 = np.logical_and(r111, yBin[3:] == 1)

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

    pppp = np.array([out['dddd'], out['dddu'], out['ddud'], out['dduu'], out['dudd'], out['dudu'], out['duud'], out['duuu'],
            out['uddd'], out['uddu'], out['udud'], out['uduu'], out['uudd'], out['uudu'], out['uuud'], out['uuuu']])
    out['hhhh'] = _f_entropy(pppp)

    return out

def MotifThree(y : ArrayLike, cgHow : str = 'quantile') -> dict:
    """
    Motifs in a coarse-graining of a time series to a 3-letter alphabet.

    Parameters:
    -----------
    y : np.ndarray
        Time series to analyze.
    cgHow : {'quantile', 'diffquant'}, optional
        The coarse-graining method to use:
        - 'quantile': equiprobable alphabet by time-series value
        - 'diffquant': equiprobably alphabet by time-series increments
        Default is 'quantile'.

    Returns:
    --------
    Dict[str, float]
        Statistics on words of length 1, 2, 3, and 4.
    """

    # Coarse-grain the data y -> yt
    y = np.asarray(y)
    numLetters = 3
    if cgHow == 'quantile':
        yt = CoarseGrain(y, 'quantile', numLetters)
    elif cgHow == 'diffquant':
        yt = CoarseGrain(np.diff(y), 'quantile', numLetters)
    else:
        raise ValueError(f"Unknown coarse-graining method {cgHow}")

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

# helper function for MotifThree
def _f_entropy(x):
    """Entropy of a set of counts, log(0) = 0"""
    return -np.sum(x[x > 0] * np.log(x[x > 0]))


def BinaryStretch(x : ArrayLike, stretchWhat : str = 'lseq1') -> float:
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

    stretchWhat : str, optional, default='lseq1'
        Specifies which binary symbol's stretch length to analyze:
            - 'lseq1': Analyze stretches related to consecutive 1s.
            - 'lseq0': Analyze stretches related to consecutive 0s.

    Returns
    -------
    float
        A statistic related to the stretch length of consecutive 0s or 1s,
        normalized by the time-series length.
    """
    x = np.asarray(x)
    N = len(x) # time series length
    x = np.where(x > 0, 1, 0)

    if stretchWhat == 'lseq1':
        # longest stretch of 1s [this code doesn't actualy measure this!]
        indices = np.where(x == 1)[0]
        diffs = np.diff(indices) - 1.5
        sign_changes = signChange(diffs, 1)
        if sign_changes.size > 1:
            out = np.max(np.diff(sign_changes)) / N
        else:
            out = None
    elif stretchWhat == 'lseq0':
        # longest stretch of 0s [this code doesn't actualy measure this!]
        indices = np.where(x == 0)[0]
        diffs = np.diff(indices) - 1.5
        sign_changes = signChange(diffs, 1)
        if sign_changes.size > 1:
            out = np.max(np.diff(sign_changes)) / N
        else:
            out = None
    else:
        raise ValueError(f"Unknown input {stretchWhat}")
    
    return out if out is not None else 0

def BinaryStats(y : ArrayLike, binaryMethod : str = 'diff') -> dict:
    """
    Compute statistics on a binary symbolization of the input time series.

    The time series is first symbolized as a binary string of 0s and 1s 
    using a specified coarse-graining (symbolization) method. Then, various 
    statistics are computed to characterize the structure of the resulting 
    binary sequence.

    Parameters
    ----------
    y : array-like
        The input time series.

    binaryMethod : str, optional, default='diff'
        The binary symbolization rule. One of:
            - 'diff' : Encode as 1 if the time-series difference is positive,
                       and 0 otherwise.
            - 'mean' : Encode as 1 if the value is above the mean, 0 otherwise.

    Returns
    -------
    dict
        Statistics computed on the binary symbolization.
    """
    
    # Binarize the time series
    y = np.asarray(y)
    yBin = binarize(y, binarizeHow=binaryMethod)
    N = len(yBin)

    # Stationarity of binarised time series
    out = {}
    out['pupstat2'] = np.sum(yBin[N//2:] == 1) / np.sum(yBin[:N//2] == 1)

    # Consecutive strings of ones/zeros (normalized by length)
    diff_y = np.diff(np.where(np.concatenate(([1], yBin, [1])))[0])
    stretch0 = diff_y[diff_y != 1] - 1

    diff_y = np.diff(np.where(np.concatenate(([0], yBin, [0])) == 0)[0])
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

def TransitionMatrix(y : ArrayLike, howtocg : str = 'quantile', numGroups : int = 2, tau : Union[int, str] = 1) -> dict:
    """
    Transition probabilities between time-series states. 
    The time series is coarse-grained according to a given method.

    The input time series is transformed into a symbolic string using an
    equiprobable alphabet of numGroups letters. The transition probabilities are
    calculated at a lag tau.

    Related to the idea of quantile graphs from time series.
    cf. Andriana et al. (2011). Duality between Time Series and Networks. PLoS ONE.
    https://doi.org/10.1371/journal.pone.0023378

    Parameters:
    -----------
    y : array_like
        Input time series (column vector)
    howtocg : str, optional
        The method of discretization (currently 'quantile' is the only
        option)
    numGroups : int, optional
        number of groups in the course-graining
    tau : int or str, optional
        analyze transition matricies corresponding to this lag. We
        could either downsample the time series at this lag and then do the
        discretization as normal, or do the discretization and then just
        look at this dicrete lag. Here we do the former. Can also set tau to 'ac'
        to set tau to the first zero-crossing of the autocorrelation function.

    Returns:
    --------
    dict 
        A dictionary including the transition probabilities themselves, as well as the trace
        of the transition matrix, measures of asymmetry, and eigenvalues of the
        transition matrix.
    """
    # check inputs
    y = np.asarray(y)
    if numGroups < 2:
        raise ValueError("Too few groups for coarse-graining")
    if tau == 'ac':
        # determine the tau from first zero of the ACF
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
        if np.isnan(tau):
            raise ValueError("Time series too short to estimate tau")
    if tau > 1: # calculate transition matrix at a non-unit lag
        # downsample at rate 1:tau
        y = ssre(y, int(np.ceil(len(y) / tau)))
    
    N = len(y)

    yth = CoarseGrain(y, howtocg, numGroups)
    # At this point we should have:
    # (*) yth: a thresholded y containing integers from 1 to numGroups
    yth = np.ravel(yth)

    T = np.zeros((numGroups,numGroups))
    for i in range(numGroups):
        ri = (yth == i + 1)
        if sum(ri) == 0:
            T[i,:] = 0
        else:
            ri_next = np.r_[False, ri[:-1]]
            for j in range(numGroups):
                T[i, j] = np.sum(yth[ri_next] == j + 1)

    out = {}
    # Normalize from counts to probabilities:
    T = T/(N - 1) # N-1 is appropriate because it's a 1-time transition matrix

    if numGroups == 2:
        for i in range(4):
            out[f'T{i+1}'] = T.transpose().flatten()[i] # transpose to match MATLAB column major

    elif numGroups == 3:
        for i in range(9):
            out[f'T{i+1}'] = T.transpose().flatten()[i] # transpose to match MATLAB column major

    elif numGroups > 3:
        for i in range(numGroups):
            out[f'TD{i+1}'] = T.transpose()[i, i]

    # (ii) Measures on the diagonal
    out['ondiag'] = np.trace(T) # trace
    out['stddiag'] = np.std(np.diag(T), ddof=1) # std of diagonal elements

    # (iii) Measures of symmetry:
    out['symdiff'] = np.sum(np.abs(T - T.T)) # sum of differences of individual elements
    out['symsumdiff'] = np.sum(np.tril(T, -1)) - np.sum(np.triu(T, 1)) # difference in sums of upper and lower triangular parts of T

    # Measures from eigenvalues of T
    eig_T = np.linalg.eigvals(T)
    out['stdeig'] = np.std(eig_T, ddof=1)
    out['maxeig'] = np.max(np.real(eig_T))
    out['mineig'] = np.min(np.real(eig_T))
    out['maximeig'] = np.max(np.imag(eig_T))

    # Measures from covariance matrix
    cov_T = np.cov(T.transpose()) # need to transpose T to get same output as MATLAB's cov func. 
    out['sumdiagcov'] = np.trace(cov_T)

    # Eigenvalues of covariance matrix
    eig_cov_T = np.linalg.eigvals(cov_T)
    out['stdeigcov'] = np.std(eig_cov_T, ddof=1)
    out['maxeigcov'] = np.max(eig_cov_T)
    #out['mineigcov'] = np.min(eig_cov_T)

    return out

def CoarseGrain(y : list, howtocg : str, numGroups : int) -> np.ndarray:
    """
    Coarse-grains a continuous time series to a discrete alphabet.

    Parameters:
    -----------
    y : array-like
        The input time series.
    howtocg : str
        The method of coarse-graining.
        Options: 'updown', 'quantile', 'embed2quadrants', 'embed2octants'
    numGroups : int
        Specifies the size of the alphabet for 'quantile' and 'updown',
        or sets the time delay for the embedding subroutines.

    Returns:
    --------
    yth : array-like
        The coarse-grained time series.
    """
    y = np.asarray(y)
    N = len(y)

    if howtocg not in ['updown', 'quantile', 'embed2quadrants', 'embed2octants']:
        raise ValueError(f"Unknown coarse-graining method '{howtocg}'")

    # Some coarse-graining/symbolization methods require initial processing:
    if howtocg == 'updown':
        y = np.diff(y)
        N = N - 1 # the time series is one value shorter than the input because of differencing
        howtocg = 'quantile' # successive differences and then quantiles

    elif howtocg in ['embed2quadrants', 'embed2octants']:
        # Construct the embedding
        if numGroups == 'tau':
            # First zero-crossing of the ACF
            tau = FirstCrossing(y, 'ac', 0, 'discrete')
        else:
            tau = numGroups
        
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
    if howtocg == 'quantile':
        th = np.quantile(y, np.linspace(0, 1, numGroups + 1), method='hazen') # thresholds for dividing the time-series values
        th[0] -= 1  # Ensure the first point is included
        # turn the time series into a set of numbers from 1:numGroups
        yth = np.zeros(N, dtype=int)
        for i in range(numGroups):
            yth[(y > th[i]) & (y <= th[i+1])] = i + 1

    elif howtocg == 'embed2quadrants': # divides based on quadrants in a 2-D embedding space
        # create alphabet in quadrants -- {1,2,3,4}
        yth = np.zeros(len(m1), dtype=int)
        yth[q1r] = 1
        yth[q2r] = 2
        yth[q3r] = 3
        yth[q4r] = 4
        
    elif howtocg == 'embed2octants': # divide based on octants in 2-D embedding space
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
