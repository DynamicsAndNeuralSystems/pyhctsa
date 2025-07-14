import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import lfilter
from pyhctsa.Operations.Correlation import AutoCorr

def ARCov(y : ArrayLike, p : int = 2) -> dict:
    """
    Fits an autoregressive (AR) model of a given order p.

    Uses the arcov approach (covariance method) to fit an AR model to the input time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    p : int, optional
        The AR model order. Default is 2.

    Returns
    -------
    dict
        Dictionary containing the parameters of the fitted model, the variance estimate
        of a white noise input to the AR model, the root-mean-square (RMS) error of a
        reconstructed time series, and the autocorrelation of residuals.
    """
    y = np.asarray(y)
    model = AutoReg(y, lags=p, trend='n')
    results = model.fit()
    phi = results.params
    a = np.concatenate(([1], -phi))
    e = results.sigma2
    out = {}
    out['e'] = e
    for i in range(len(a)):
        out[f'a{i+1}'] = a[i]
    # Residual analysis
    b_coeffs = np.concatenate(([0], -a[1:]))
    # Predict y from its past values
    y_est = lfilter(b_coeffs, [1], y)
    err = y - y_est
    out['res_mu'] = np.mean(err)
    out['res_std'] = np.std(err, ddof=1)
    out['res_AC1'] = AutoCorr(err, 1, 'Fourier')[0]
    out['res_AC2'] = AutoCorr(err, 2, 'Fourier')[0]
    return out
