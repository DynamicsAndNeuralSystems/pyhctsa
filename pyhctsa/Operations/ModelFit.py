import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import lfilter
from pyhctsa.Operations.Correlation import AutoCorr
from scipy.stats import ks_1samp, norm
import numba
from typing import Union
from pyhctsa.Utilities.utils import ZScore


def ExpSmoothing(x : ArrayLike, ntrain : Union[None, int, float] = None, alpha : Union[str, float] = 'best') -> dict:
    """
    Exponential smoothing time-series prediction model.

    Fits an exponential smoothing model to the time series using a training set to
    fit the optimal smoothing parameter, alpha, and then applies the result to
    predict the rest of the time series.

    Reference
    ---------
    "The Analysis of Time Series", C. Chatfield, CRC Press LLC (2004).
    Code adapted from Siddharth Arora (Siddharth.Arora@sbs.ox.ac.uk).

    Parameters
    ----------
    x : array-like
        The input time series.
    ntrain : int or float, optional
        The number of samples to use for training. Can be an integer or a proportion of the time-series length.
    alpha : str or float, optional
        The exponential smoothing parameter. If 'best', the function optimizes alpha on the training set.

    Returns
    -------
    dict
        Dictionary including the fitted alpha and statistics on the residuals from the prediction phase.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    out = {}

    # --- Check Inputs ---
    if ntrain is None:
        ntrain = min(100, N)
    
    if 0 < ntrain < 1:
        ntrain = int(np.floor(N * ntrain))
        
    min_train, max_train = 100, 1000
    
    if ntrain > max_train:
        print(f"Training set size reduced from {ntrain} to {max_train}.")
        ntrain = max_train
        
    if ntrain < min_train:
        print(f"Training set size increased from {ntrain} to {min_train}.")
        ntrain = min_train
        
    if N < ntrain:
        print("Time series is too short for the specified training size.")
        return np.nan
        
    # --- Find Optimal Alpha ---
    if alpha == 'best':
        xtrain = x[:ntrain]
        
        # (1) Initial coarse search
        alphar = np.linspace(0.1, 0.9, 5)
        # Note: Original MATLAB has a bug `rmses = zeros(4,1)`. We correct this.
        rmses = np.zeros_like(alphar)
        
        for i, a in enumerate(alphar):
            xf = +_fit_exp_smooth(xtrain, a)
            fore, orig = xf[2:], xtrain[2:]
            rmses[i] = np.sqrt(np.mean((fore - orig)**2)) if len(fore) > 0 else np.nan

        # Check for valid RMSEs before fitting
        valid_indices = ~np.isnan(rmses)
        if np.sum(valid_indices) < 3:
            print("Not enough valid points for quadratic fit; choosing best alpha from search.")
            alphamin = alphar[np.nanargmin(rmses)] if np.any(valid_indices) else 0.5
        else:
            # Fit quadratic to the 3 points with the lowest RMSE
            # np.argsort on `rmses[valid_indices]` finds the indices within that slice
            sorted_rmse_indices = np.argsort(rmses[valid_indices])
            # Get the indices of the original `alphar` and `rmses` arrays
            original_indices = np.where(valid_indices)[0][sorted_rmse_indices[:3]]

            alphar_fit = alphar[original_indices]
            rmses_fit = rmses[original_indices]
            
            p = np.polyfit(alphar_fit, rmses_fit, 2)
            out.update({'alphamin_1': -p[1] / (2 * p[0]), 'p1_1': abs(p[0]), 'cup_1': np.sign(p[0])})
            
            if p[0] < 0:  # Concave down (found a maximum), pick a boundary
                y_boundary = np.polyval(p, [0.01, 1.0])
                alphamin = [0.01, 1.0][np.argmin(y_boundary)]
            else:  # Concave up (found a minimum)
                alphamin = -p[1] / (2 * p[0])

                # (2) Refined search around the found minimum
                low_b, high_b = alphamin - 0.1, alphamin + 0.1
                if low_b <= 0: low_b, high_b = 0.01, max(alphamin, 0) + 0.1
                elif high_b >= 1: low_b, high_b = min(alphamin, 1) - 0.1, 1.0
                
                alphar_ref = np.linspace(low_b, high_b, 5)
                rmses_ref = np.zeros_like(alphar_ref)

                for i, a in enumerate(alphar_ref):
                    xf = _fit_exp_smooth(xtrain, a)
                    fore, orig = xf[2:], xtrain[2:]
                    rmses_ref[i] = np.sqrt(np.mean((fore - orig)**2)) if len(fore) > 0 else np.nan
                
                valid_ref = ~np.isnan(rmses_ref)
                if not np.any(valid_ref):
                    print("Could not compute RMSE in refined search; using previous alpha.")
                else:
                    p2 = np.polyfit(alphar_ref[valid_ref], rmses_ref[valid_ref], 2)
                    if p2[0] < 0: # Bad fit, fallback to best alpha in search
                        alphamin = alphar_ref[np.nanargmin(rmses_ref)]
                    else: # Minimum of the new quadratic fit
                        alphamin = -p2[1] / (2 * p2[0])
                        
        alpha = np.clip(alphamin, 0.01, 1.0)
        out['alphamin'] = alpha

    if np.isnan(alpha):
        raise ValueError("Alpha optimization failed, resulting in NaN.")

    # --- Final Fit and Residual Analysis ---
    y_fit = _fit_exp_smooth(x, alpha)
    yp, xp = y_fit[2:], x[2:]
    
    if len(yp) < 2:
        print("Not enough points to calculate residual statistics.")
        residout = {'mean': np.nan, 'std': np.nan, 'AC1': np.nan}
    else:
        residuals = yp - xp
        residout = ResidualAnalysis(residuals)
    
    out.update(residout)
    return out

@numba.jit(nopython=True, cache=True)
def _fit_exp_smooth(x: np.ndarray, a: float) -> np.ndarray:
    """
    An accelerated version of _fit_exp_smooth using Numba.
    """
    n = x.shape[0]
    xf = np.zeros(n)
    
    for i in range(1, n - 1):
        # Calculate s_0 = mean(x[0:i])
        s0 = np.mean(x[0:i])
        
        # Smooth up to the current point `i`
        s_prev = s0
        s_curr = 0.0 
        for j in range(1, i + 1):
            s_curr = a * x[j] + (1 - a) * s_prev
            s_prev = s_curr
            
        # The forecast for time `i+1` is the smoothed value at time `i`
        xf[i + 1] = s_curr
        
    return xf


def ResidualAnalysis(e):
    """
    Analysis of residuals from a model fit.

    Given an input residual time series `e`, this function returns a dictionary with fields corresponding to statistical tests on the residuals.
    These tests are motivated by the general expectation that model residuals should be uncorrelated.

    Parameters
    ----------
    e : array-like
        Raw residuals as prediction minus data (e = yp - y), provided as a column vector.

    Returns
    -------
    dict
        Dictionary of statistics on the residuals.
    """
    e = np.asarray(e)
    N = len(e)
    # basic stats on resids
    out = {}
    out['meane'] = np.mean(e)
    out['meanabs'] = np.mean(np.abs(e))
    out['rmse'] = np.sqrt(np.mean(e**2))
    std_e = np.std(e, ddof=1)
    out['stde'] = std_e
    out['mms'] = np.abs(np.mean(e)) + np.abs(np.std(e, ddof=1))
    out['maxonmean'] = np.max(e)/np.abs(np.mean(e))

    if std_e == 0:
        e = np.zeros(len(e))
    else:
        e = ZScore(e)
    
    # TODO: Identify any low-frequency trends in residuals
    # Analyze autocorrelation in residuals
    maxLag = 25
    autoCorrResid = AutoCorr(e, list(range(1, maxLag+1)), 'Fourier')
    sqrtN = np.sqrt(N)

    # Output first 3 ACs
    out['ac1'] = autoCorrResid[0]
    out['ac2'] = autoCorrResid[1]
    out['ac3'] = autoCorrResid[2]
    out['ac1n'] = np.abs(autoCorrResid[0]) * sqrtN
    out['ac2n'] = np.abs(autoCorrResid[1]) * sqrtN
    out['ac3n'] = np.abs(autoCorrResid[2]) * sqrtN

    #% Median normalized distance from zero
    out['acmnd0'] = np.median(np.abs(autoCorrResid)) * sqrtN
    out['acsnd0'] = np.std(np.abs(autoCorrResid), ddof=1) * sqrtN
    out['propbth'] = np.sum(np.abs(autoCorrResid) < 2.6/sqrtN)/maxLag

    # % First time to get below the significance threshold
    ftbth_indices = np.where(np.abs(autoCorrResid) < 2.6/sqrtN)[0]
    if ftbth_indices.size > 0:  # Alternative way to check if array is empty
        out['ftbth'] = ftbth_indices[0] + 1
    else:
        out['ftbth'] = maxLag + 1

    # Durbin-Watson test statistic (like AC1)
    out['dwts'] = np.sum((e[1:] - e[:-1])**2) / np.sum(e**2)

    # Distribution tests
    res = ks_1samp(e, norm.cdf)
    out['normksstat'] = res.statistic
    out['normp'] = res.pvalue

    return out


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
