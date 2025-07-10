from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import jarque_bera, wilcoxon, norm
import numpy as np
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from numpy.typing import ArrayLike

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
    return p
