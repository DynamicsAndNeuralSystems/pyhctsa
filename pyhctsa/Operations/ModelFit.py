import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.signal import lfilter
from pyhctsa.Operations.Correlation import AutoCorr
from scipy.stats import ks_1samp, norm, t
import numba
from typing import Union
from ..Utilities.utils import ZScore
from ..Operations.Stationarity import SlidingWindow
from ..Operations.Correlation import FirstCrossing, AutoCorr

def LocalSimple(y, forecastMeth = 'mean', trainLength = 3):
    y = np.asarray(y)
    N = len(y)
    # % Do the local prediction
    if trainLength == 'ac':
        lp = FirstCrossing(y, 'ac', 0, 'discrete')
    else:
        lp = trainLength # the length of the subsegment preceeding to use to predict the subsequent value
    evalr = np.arange(lp, N) #range over which to evaluate the forecast
    if np.size(evalr) == 0:
        print("This time series is too short for forecasting")
        return np.nan
    res = np.zeros(len(evalr))
    if forecastMeth == 'mean':
        for i in range(len(evalr)):
            res[i] = np.mean(y[evalr[i]-lp:evalr[i]]) - y[evalr[i]] # prediction - value
    elif forecastMeth == 'median':
        for i in range(len(evalr)):
            res[i] = np.median(y[evalr[i]-lp:evalr[i]]) - y[evalr[i]]  # prediction - value
    elif forecastMeth == 'lfit':
        for i in range(len(evalr)):
            # Fit linear
            p = np.polyfit(np.arange(1, lp+1), y[evalr[i]-lp:evalr[i]], 1)
            res[i] = np.polyval(p, lp+1) - y[evalr[i]]  # prediction - value
    else:
        raise ValueError(f"Unknown forecasting method: {forecastMeth}")
    
    #Output statistics on the residuals, res
    #% Mean residual (mean error/bias):
    out = {}
    out['meanerr'] = np.mean(res)
    #% Spread of residuals:
    out['stderr'] = np.std(res, ddof=1)
    out['meanabserr'] = np.mean(np.abs(res))
    #% Stationarity of residuals:
    out['sws'] = SlidingWindow(res, 'std', 'std', 5, 1) # across five non-overlapping segments
    out['swm'] = SlidingWindow(res, 'mean', 'std', 5, 1) # across five non-overlapping segments
    #% TODO Normality of residuals:
    #% Autocorrelation structure of the residuals:
    out['ac1'] = AutoCorr(res, 1, 'Fourier')[0]
    out['ac2'] = AutoCorr(res, 2, 'Fourier')[0]
    out['taures'] = FirstCrossing(res, 'ac', 0, 'continuous')
    out['tauresrat'] = FirstCrossing(res, 'ac', 0, 'continuous')/FirstCrossing(y, 'ac', 0, 'continuous')

    return out

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


def ResidualAnalysis(e : ArrayLike) -> dict:
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


def _arconf_from_arfit(fitted_ar, theConfInterval : float = 0.95):
    params = fitted_ar.params
    has_intercept = fitted_ar.model.trend == 'c'
    if has_intercept:
        w = params[0]
        A = params[1:]
    else:
        w = None
        A = params
    # degress of freedom
    dof = fitted_ar.df_resid
    t_crit = t.ppf(0.5 + theConfInterval / 2, df=dof) # quantiles of the t distrib
    Cest = fitted_ar.sigma2 # the noise covariance/variance
    uinv = fitted_ar.cov_params() / Cest
    all_errs = t_crit * np.sqrt(np.diag(uinv) * Cest)
    if has_intercept:
        w_err = all_errs[0]
        A_err = all_errs[1:]
        return {'w_err': w_err, 'A_err': A_err}
    else:
        A_err = all_errs
        return {'A_err': A_err}

def _get_criteria(sel, N, type = "aic"):
    # pop the first key
    keys = None
    se = None

    if type == "aic":
        se = sel.aic
        se.pop(0)
        keys = se.keys()
    elif type == "bic":
        se = sel.bic
        se.pop(0)
        keys = se.keys()
    else:
        return ValueError(f"Unknown crtieria: {type}!")
    
    orlist = np.array([i[-1] for i in list(keys)])
    ps_len = len(keys)
    orlist_sorted_idxs = np.argsort(orlist)
    criteria_vals = np.zeros(ps_len)
    
    for i in range(ps_len):
        key_i = list(keys)[orlist_sorted_idxs[i]] # both aic and bic ordered the same way
        val = se.get(key_i)/N # normalise by num observations
        criteria_vals[i] = val
    
    return criteria_vals

def ARFit(y : ArrayLike, pmin : int = 1, pmax : int = 10, selector : str = 'sbc') -> dict:
    """
    Statistics of a fitted AR model to a time series.

    Fits autoregressive (AR) models of orders p = pmin, pmin + 1, ..., pmax to the input time series,
    selects the optimal model order using Schwartz's Bayesian Criterion (SBC), and returns statistics
    on the fitted model, residuals, and confidence intervals.

    Reference
    ----------
    - "Estimation of parameters and eigenmodes of multivariate autoregressive models",
      A. Neumaier and T. Schneider, ACM Trans. Math. Softw. 27, 27 (2001)
    - "Algorithm 808: ARFIT---a Matlab package for the estimation of parameters and eigenmodes of multivariate autoregressive models",
      T. Schneider and A. Neumaier, ACM Trans. Math. Softw. 27, 58 (2001)

    Parameters
    ----------
    y : array-like
        The input time series.
    pmin : int, optional
        The minimum AR model order to fit. Default is 1.
    pmax : int, optional
        The maximum AR model order to fit. Default is 10.
    selector : str, optional
        Criterion to select optimal model order (e.g., 'sbc', cf. ARFIT package documentation). Default is 'sbc'.

    Returns
    -------
    dict
        Dictionary containing statistics of a fitted AR model to a time series.
    """
    y = np.asarray(y)
    N = len(y)
    if selector in ['bic', 'sbc']: # bic and sbc are the same metrics
        selector = 'bic'
    #(I) Fit AR model)
    sel = ar_select_order(y, maxlag=pmax, ic=selector, glob=False, trend='n') # bic is the same as sbc
    p_optimal = max(pmin, np.max(sel.ar_lags)) if sel.ar_lags is not None else pmin
    ps = np.arange(pmin, pmax+1)
    # fit the AR model using the optimal number of lags from above
    model = AutoReg(y, lags=p_optimal, trend='n')
    res = model.fit()
    popt = len(res.params)
    Aest = res.params
    #2) Coefficients Aest
    out = {}
    out['A1'] = Aest[0]
    for i in range(2, 7):
        if popt >= i:
            out[f'A{i}'] = Aest[i-1]
        else:
            out[f'A{i}'] = 0 # % set all the higher order coefficients are all zero
    # (ii) Summary statistics on the coefficients
    out['maxA'] = np.max(Aest)
    out['minA'] = np.min(Aest)
    out['meanA'] = np.mean(Aest)
    out['stdA'] = np.std(Aest, ddof=1) if len(Aest) > 1 else 0
    out['sumA'] = np.sum(Aest)
    out['rmsA'] = np.sqrt(sum(Aest**2))
    out['sumsqA'] = np.sum(Aest**2)

    #(3) Noise covariance matrix, Cest
    # In our case of a univariate time series, just a scalar for the noise magnitude.
    Cest = res.sigma2
    out['C'] = Cest


    # #(4) Schwartz's Bayesian Criterion, SBC (BIC)
    bics = _get_criteria(sel, N, "bic")
    for i in range(len(bics)):
        out[f'sbc_{ps[i]}'] = bics[i]

    # return minimum 
    out['minsbc'] = np.min(bics)
    popt_sbc = ps[np.argmin(bics)]
    out['popt_sbc'] = popt_sbc
    
    # Akiake Information Criteria (AIC) as a viable alternative to Akiake's FPE for final prediction error (FPE)
    aics = _get_criteria(sel, N, "aic")
    n = aics.size
    for i in range(len(aics)):
        out[f'fpe_{ps[i]}'] = aics[i]
    # return minimum 
    out['minfpe'] = np.min(aics)
    popt_fpe = ps[np.argmin(aics)]
    out['popt_fpe'] = popt_fpe

    #%% (II) Test Residuals
    ljung_box_results = acorr_ljungbox(res.resid, lags=[20], model_df=p_optimal, return_df=False) # test of auto correlation in the residuals (test up to lag 20)
    out['res_siglev'] = ljung_box_results.iloc[0]['lb_pvalue']

    # Correlation test of residuals
    resids = res.resid
    # out['res_ac1'] = AutoCorr(resids, 1, 'Fourier')[0]
    # out['res_ac1_norm'] = out['res_ac1']/np.sqrt(N)

    #Calculate correlations up to 20, return how many exceed significance threshold
    acf = AutoCorr(resids, list(range(1, 21)), 'Fourier')
    out['pcorr_res'] = np.sum(np.abs(acf) > 1.96/np.sqrt(N))/20

    # Confidence Intervals
    Aerr = _arconf_from_arfit(res, 0.95)['A_err']
    out['aerr_min'] = np.min(Aerr)
    out['aerr_max'] = np.max(Aerr)
    out['aerr_mean'] = np.mean(Aerr)

    # TODO: Add eigendecomposition analysis

    return out
