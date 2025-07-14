from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import jarque_bera, wilcoxon, norm
import numpy as np
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from numpy.typing import ArrayLike
from arch.unitroot import VarianceRatio
from typing import Union

def VarianceRatioTest(y : ArrayLike, periods : Union[int, list[int]] = 2, IIDs : Union[int, list[int]] = 0) -> dict:
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
    IIDs : int or list of int, optional
        A scalar or vector of boolean values (0 or 1) indicating whether to assume
        independent and identically distributed (IID) innovations for each period.

    Returns
    -------
    dict
        Dictionary of test results.
    """
    y = np.asarray(y)
    out = {}
    logical_check = lambda lst: all(x in (0, 1) for x in lst)
    if isinstance(periods, list):
        # Return statistics on multiple outputs for multiple periods/IIDs
        # check that IIDS is also a list
        if isinstance(IIDs, list):
            # some checks on the data types...
            if len(IIDs) != len(periods):
                raise ValueError(f"Length of IIDs list ({len(IIDs)}) does not match the list of periods ({len(periods)}).")
            if not logical_check(IIDs):
                raise ValueError("List of IIDs must only be logicals (0 or 1).")

            vrs = []
            for (i, p) in enumerate(periods):
                robust = True if IIDs[i] == 0 else False 
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
            out['IIDperiodmaxpValue'] = IIDs[imaxp]
            out['IIDperiodminpValue'] = IIDs[iminp]

            out['meanstat'] = np.mean(all_stats)
            out['maxstat'] = np.max(all_stats)
            out['minstat'] = np.min(all_stats)
        else:
            raise ValueError(f"Expected IIDs to be a list of bools, since periods are also a list. Got data type: {type(IIDs)} instead.")
    elif isinstance(periods, int):
        robust = True if IIDs == 0 else False 
        vr = VarianceRatio(y, lags=periods, robust=robust)
        out = {}
        out['pValue'] = vr.pvalue
        out['stat'] = vr.stat
        out['ratio'] = vr.vr
    else:
        raise ValueError(f"Unknown data type for periods: {type(periods)}, select either integer or list of integers.")

    return out

def HypothesisTest(x : ArrayLike, theTest: str = "signtest") -> float:
    """
    Statistical hypothesis test applied to a time series.

    Applies a specified statistical hypothesis test to the input time series and returns the p-value.

    Tests are implemented using functions from Python's statsmodels and scipy libraries.

    Parameters
    ----------
    x : array-like
        The input time series.
    theTest : {'signtest', 'runstest', 'vartest', 'ztest', 'signrank', 'jbtest', 'lbq'}, optional
        The hypothesis test to perform:
            - 'signtest': Sign test
            - 'runstest': Runs test
            - 'vartest': Variance test (not implemented)
            - 'ztest': Z-test
            - 'signrank': Wilcoxon signed rank test for zero median
            - 'jbtest': Jarque-Bera test of composite normality
            - 'lbq': Ljung-Box Q-test for residual autocorrelation
        Default is 'signtest'.

    Returns
    -------
    float
        p-value from the specified statistical test.
    """
    x = np.asarray(x)
    p = np.nan
    if theTest == "signtest":
        _, p = sign_test(x)
    elif theTest == "runstest":
        _, p = runstest_1samp(x, cutoff='mean', correction=True)
    elif theTest == "jbtest":
        s = jarque_bera(x)
        p = s.pvalue
    elif theTest == "ztest":
        xmean = np.mean(x)
        n = len(x)
        sigma = 1
        zval = (xmean - 0) / (sigma / np.sqrt(n))
        p = 2 * norm.cdf(-abs(zval))
    elif theTest == "signrank":
        _, p = wilcoxon(x)
    elif theTest == "lbq":
        # Ljung-Box Q-test for residual autocorrelation
        T = np.sum(~np.isnan(x)) # get the effective sample size
        nLags = min(20, T-1)
        p = acorr_ljungbox(x, lags=[nLags])['lb_pvalue'].to_numpy()[0]
    else:
        raise ValueError(f"Unknown test: {theTest}.")
    return p
