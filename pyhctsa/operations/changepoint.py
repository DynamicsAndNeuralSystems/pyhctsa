import numpy as np

from ..toolboxes.Max_Little.ML_l1pwc import l1pwc, l1pwc_lmax

def stepdetect(y: np.ndarray, method: str = 'l1pwc', params: float | int = 10) -> dict:
    """
    Analysis of discrete steps in a time series.

    Gives information about discrete steps in the signal, using the function
    `l1pwc` from Max A. Little's step detection toolkit.

    References
    ----------
    .. [1] S.J. Kim, K. Koh, S. Boyd and D. Gorinevsky (2007) "An Interior-Point Method for Large-Scale 
        ℓ1 -Regularized Least Squares," in IEEE Journal of Selected Topics in Signal Processing, 
        vol. 1, no. 4, pp. 606-617, Dec. 2007, doi: 10.1109/JSTSP.2007.910971. 
    .. [2] M.A. Little, Nick S. Jones (2010) "Sparse Bayesian Step-Filtering for
        High-Throughput Analysis of Molecular Machine Dynamics", in 2010 IEEE
        International Conference on Acoustics, Speech and Signal Processing,
        ICASSP 2010 Proceedings.

    Parameters
    ----------
    y : array-like
        The input time series.
    method : str
        The step-detection method. Currently only `l1pwc` is supported.
    params : float or str
        The parameters for the given method used. For `l1pwc` this corresponds to lambda. Default is 10.

    Returns
    -------
    dict
        Statistics on the output of the step-detection method, including the intervals
        between change points, the proportion of constant segments, the reduction in
        variance from removing the piece-wise constants, and stationarity in the
        occurrence of change points.
    """

    y = np.asarray(y).ravel()
    N = len(y)

    out = {}
    if method == 'l1pwc':
        lamb = params
        if lamb < 1:  # specify as a proportion of lambdamax
            lamb = l1pwc_lmax(y) * lamb
        steppedy, E, s, lambda_max = l1pwc(y, lamb, False)  # defaults for stoptol/maxiter

        # l1pwc returns x as N x L; collapse to 1-D so diff/subtraction behave
        steppedy = np.asarray(steppedy).ravel()
        E = np.atleast_1d(E)
        s = np.atleast_1d(s)

        # Round to remove numerical fluctuations of order less than 1e-4
        steppedy = np.round(steppedy * 1e4) / 1e4
        out = {
            "E": E[0],
            "s": s[0],          # collapses to 1 for some parameter values
            "lambdamax": lambda_max,
        }
        # Get step indices from steppedy; these give the index of the start of
        # each run. Kept 1-based to match MATLAB, since every statistic below
        # was written against 1-based indices.
        whch = np.flatnonzero(np.diff(steppedy) != 0)
        if whch.size:
            chpts = np.concatenate(([1], whch + 2))
        else:
            chpts = np.array([1])   # no changes
    elif method == 'kv':
        raise NotImplementedError('step detection method not yet implemented.')
    else:
        raise ValueError(f'unknown step detection method: {method}')

    num_change_points = len(chpts)

    # Intervals -- of change
    chints = np.diff(np.concatenate([chpts, [N]]))

    # Number of constant segments per sample
    out["nsegments"] = num_change_points / N  # will be 1 if there are no changes

    # How much reduces variance
    out["rmsoff"] = np.std(y, ddof=1) - np.std(y - steppedy, ddof=1)

    # Reduces variance per step
    out["rmsoffpstep"] = out["rmsoff"] / num_change_points

    # Ratio of number of steps in first half of time series to second half
    sum1 = np.sum(chpts < N / 2) - 1  # (exclude the chpt that's always sitting at 1)
    sum2 = np.sum(chpts >= N / 2)
    if (sum2 > 0) and (sum1 > 0):
        if sum2 > sum1:
            out["ratn12"] = sum1 / sum2
        else:
            out["ratn12"] = sum2 / sum1
    else:
        out["ratn12"] = 0

    # Difference between number of change points in first and second half of data
    # as a proportion of total number of change points
    # Maximal (1) when all change points are in one half of data
    # Minimal (0) when same number of change points in both halves
    out["diffn12"] = abs(sum1 - sum2) / num_change_points

    # Proportion of really short steps:
    out["pshort_3"] = np.sum(chints <= 3) / N

    # Mean interval between steps:
    out["meanstepint"] = np.mean(chints) / N

    # Mean interval greater than 3 samples, per sample:
    long_ints = chints[chints > 3]
    out["meanstepintgt3"] = np.mean(long_ints) / N if long_ints.size else np.nan

    # Mean error on step interval distribution:
    out["meanerrstepint"] = np.std(chints, ddof=1) / np.sqrt(len(chints))

    # Maximum step interval:
    out["maxstepint"] = np.max(chints) / N

    # Minimum step interval:
    out["minstepint"] = np.min(chints) / N

    # Median step interval:
    out["medianstepint"] = np.median(chints) / N

    return out

def l1pwc_sweep_lambda(y: np.ndarray, lambdar: np.ndarray) -> dict:
    """
    Dependence of step detection on regularization parameter.
 
    Gives information about discrete steps in the signal across a range of
    regularization parameters lambda, using the function l1pwc from Max Little's
    step detection toolkit.
 
    cf.,
    "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
 
    Parameters
    ----------
    y : array_like
        The input time series.
    lambdar : array_like
        A vector specifying the lambda parameters to use.
 
    Returns
    -------
    dict
        At each iteration, the CP_ML_StepDetect code was run with a given
        lambda, and the number of segments, and reduction in root mean square
        error from removing the piecewise constants was recorded. Outputs
        summarize how these quantities vary with lambda.
    """
    y = np.asarray(y, dtype=float)
    lambdar = np.atleast_1d(np.asarray(lambdar, dtype=float))
 
    Llambdar = len(lambdar)
    nsegs = np.zeros(Llambdar)
    rmserrs = np.zeros(Llambdar)
    rmserrpsegs = np.zeros(Llambdar)
 
    for i in range(Llambdar):
        lam = lambdar[i]
        # Run the (stochastic) step detection algorithm:
        outi = stepdetect(y, method='l1pwc', params=lam)
        nsegs[i] = outi['nsegments']
        rmserrs[i] = outi['rmsoff']
        rmserrpsegs[i] = outi['rmsoffpstep']
 
    # ------------------------------------------------------------------------------
    # Define the FirstUnder function:
    # ------------------------------------------------------------------------------
    # Often finding the first time a certain vector (x) drops under a given
    # threshold (y). This function does it:
 
    def FirstUnder(x, thresh):
        idx = np.flatnonzero(x < thresh)
        return idx[0] if idx.size > 0 else None
 
    def NaNIfEmpty(idx):
        # Returns a NaN if idx is empty, otherwise returns lambdar at idx.
        return np.nan if idx is None else lambdar[idx]
 
    out = {}
 
    # ------------------------------------------------------------------------------
    # (*) Use rmsunderx subfunction to analyze when RMS errors drop under a set of
    # thresholds, *x*, for the first time:
    # ------------------------------------------------------------------------------
    out['rmserrsu05'] = NaNIfEmpty(FirstUnder(rmserrs, 0.5))
    out['rmserrsu02'] = NaNIfEmpty(FirstUnder(rmserrs, 0.2))
    out['rmserrsu01'] = NaNIfEmpty(FirstUnder(rmserrs, 0.1))
 
    # ------------------------------------------------------------------------------
    # (*) Use the nsegsunderx subfunction to analyze when nseg drops under a set of
    # thresholds, *x*, for the first time:
    # nsegunderx = @(x) find(nsegs < x, 1, 'first');
    # ------------------------------------------------------------------------------
    out['nsegsu005'] = NaNIfEmpty(FirstUnder(nsegs, 0.05))
    out['nsegsu001'] = NaNIfEmpty(FirstUnder(nsegs, 0.01))
 
    # Calculate the correlation between the number of segments and rmserrs
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.corrcoef(nsegs, rmserrs)
    out['corrsegerr'] = R[1, 0]
 
    # Maximum rmserrpsegment
    indbest = int(np.argmax(rmserrpsegs))  # where the maximum occurs
    out['bestrmserrpseg'] = rmserrpsegs[indbest]
    out['bestlambda'] = lambdar[indbest]
 
    return out