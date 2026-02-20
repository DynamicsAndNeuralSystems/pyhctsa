from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from arch.unitroot import VarianceRatio
from scipy.stats import jarque_bera, norm, wilcoxon
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.diagnostic import acorr_ljungbox

def variance_ratio_test(y : ArrayLike, periods : Union[int, list[int], float] = 2,
                        iids : Union[int, list[int]] = 0) -> dict:
    """
    Variance ratio test for random walk.

    Implements the variance ratio test using the VarianceRatio function from arch.unitroot.

    The test assesses the null hypothesis of a random walk in the time series,
    which is rejected for some critical p-value.

    Parameters
    ----------
    y : array-like
        The input time series.
    periods : int or list of int, optional
        A scalar or vector of period(s) to use for the test.
    iids : int or list of int, optional
        A scalar or vector of boolean values (0 or 1) indicating whether to assume
        independent and identically distributed (IID) innovations for each period.

    Returns
    -------
    dict
        Dictionary of test results.
    """
    y = np.asarray(y)
    out = {}
    def logical_check(lst):
        return all(x in (0, 1) for x in lst)
    if isinstance(periods, list):
        # Return statistics on multiple outputs for multiple periods/IIDs
        # check that IIDS is also a list
        if isinstance(iids, list):
            # some checks on the data types...
            if len(iids) != len(periods):
                raise ValueError(f"Length of IIDs list ({len(iids)}) does not match "
                    "the list of periods ({len(periods)}).")
            if not logical_check(iids):
                raise ValueError("List of IIDs must only be logicals (0 or 1).")

            vrs = []
            for (i, p) in enumerate(periods):
                robust = True if iids[i] == 0 else False 
                vr = VarianceRatio(y, lags=p, robust=robust)
                vrs.append(vr)
            all_pvals = np.array([p.pvalue for p in vrs])
            all_stats = np.array([s.stat for s in vrs])
            out['maxpValue'] = np.max(all_pvals)
            out['minpValue'] = np.min(all_pvals)
            out['meanpValue'] = np.mean(all_pvals)

            imaxp = np.argmax(all_pvals)
            iminp = np.argmin(all_pvals)
            out['periodmaxpValue'] = periods[imaxp]
            out['periodminpValue'] = periods[iminp]
            out['IIDperiodmaxpValue'] = iids[imaxp]
            out['IIDperiodminpValue'] = iids[iminp]

            out['meanstat'] = np.mean(all_stats)
            out['maxstat'] = np.max(all_stats)
            out['minstat'] = np.min(all_stats)
        else:
            raise ValueError(f"Expected iids to be a list of bools, since periods "
                             f"are also a list. Got data type: {type(iids)} instead.")
    elif isinstance(periods, (int, float, np.number)):
        periods = int(periods)
        robust = True if iids == 0 else False 
        vr = VarianceRatio(y, lags=periods, robust=robust)
        out = {}
        out['pValue'] = vr.pvalue
        out['stat'] = vr.stat
        out['ratio'] = vr.vr
    else:
        raise ValueError(f"Unknown data type for periods: {type(periods)}, "
                         "select either integer or list of integers.")

    return out

def hypothesis_test(x : ArrayLike, the_test: str = "signtest") -> float:
    """
    Perform statistical hypothesis testing on a time series.

    Applies a specified statistical test and returns its p-value. Tests are chosen
    to evaluate different null hypotheses about the time series properties.

    Parameters
    ----------
    x : ArrayLike
        Input time series
    the_test : str, optional
        Type of hypothesis test to perform:

        - 'signtest': Tests if median equals zero
        - 'runstest': Tests for randomness in sequence
        - 'ztest': Tests if mean equals zero (assumes unit variance)
        - 'signrank': Wilcoxon signed rank test for zero median
        - 'jbtest': Jarque-Bera test for normality
        - 'lbq': Ljung-Box Q-test for autocorrelation
        
        Default is 'signtest'.

    Returns
    -------
    float
        P-value from the statistical test. A small p-value (< 0.05) typically
        indicates rejection of the null hypothesis.
    """
    x = np.asarray(x)
    p = np.nan
    if the_test == "signtest":
        _, p = sign_test(x)
    elif the_test == "runstest":
        _, p = runstest_1samp(x, cutoff='mean', correction=True)
    elif the_test == "jbtest":
        s = jarque_bera(x)
        p = s.pvalue
    elif the_test == "ztest":
        x_mean = np.mean(x)
        n = len(x)
        sigma = 1
        zval = (x_mean - 0) / (sigma / np.sqrt(n))
        p = 2 * norm.cdf(-abs(zval))
    elif the_test == "signrank":
        _, p = wilcoxon(x)
    elif the_test == "lbq":
        # Ljung-Box Q-test for residual autocorrelation
        t = np.sum(~np.isnan(x)) # get the effective sample size
        n_lags = min(20, t-1)
        p = acorr_ljungbox(x, lags=[n_lags])['lb_pvalue'].to_numpy()[0]
    else:
        raise ValueError(f"Unknown test: {the_test}.")
    return p
