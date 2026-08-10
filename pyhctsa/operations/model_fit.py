from typing import Union

import numba
import numpy as np
from numpy.typing import ArrayLike
from numpy.lib.stride_tricks import sliding_window_view
from hmmlearn.hmm import GaussianHMM
from scipy.signal import lfilter
from scipy.stats import ks_1samp, norm, t
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from lmfit.models import SineModel
import logging
logger = logging.getLogger('pyhctsa')

from ..operations.correlation import autocorr, first_crossing
from ..operations.stationarity import sliding_window
from ..toolboxes.matlab.gpml.gpml import CovSEisoNoise, gp_predict, gp_train
from ..toolboxes.matlab.optimizers import minimize
from ..utils import z_score, ljung_box_pvalue

def hmm_fit(y: ArrayLike, train_p: float = 0.8, num_states: int = 3, random_seed: int = 0) -> dict:
    """
    Fits a Hidden Markov Model to sequential data.

    Parameters
    ----------
    y : array-like
        The input time series.
    train_p : float
        The proportion of data to train on, 0 < train_p < 1. Default is 0.8.
    num_states : int
        The number of states in the HMM. Default is 3.
    random_seed : int
        Random seed. Default is 0.

    Returns
    -------
    dict
        Dictionary of statistics based on the fitted HMM. 

    """
    #Actually highly stochastic, so for reproducible results helps to set the
    #random seed.
    y = np.asarray(y)
    n_samples = len(y)
    out = {}

    # 1. Split data into training and test sets
    n_train = int(np.floor(train_p * n_samples))
    n_test = n_samples-n_train
    if n_train <= 0 or n_train > n_samples:
        raise ValueError("Invalid training proportion 'train_p' results in an invalid training set size.")
    
    y_train = y[:n_train]
    y_test = y[n_train:]
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    num_states = int(num_states)

    model = GaussianHMM(n_components=num_states,
                    covariance_type='tied',
                    n_iter=30,
                    tol=0.0001,
                    params='stmc',
                    init_params='stmc',
                    random_state=random_seed)
    
    model.fit(y_train_reshaped)
    means_sorted = np.sort(model.means_.flatten())
    for i, mu in enumerate(means_sorted):
        out[f'Mu_{i+1}'] = mu
    out['meanMu'] = np.mean(means_sorted)
    out['rangeMu'] = np.ptp(means_sorted)
    out['maxMu'] = np.max(means_sorted)
    out['minMu'] = np.min(means_sorted)

    # Covariance Cov
    out['Cov'] = model.covars_.flatten()[0]

    #% Transition matrix
    p_matrix = model.transmat_

    out['Pmeandiag'] = np.mean(np.diag(p_matrix))
    out['std_mean_p'] = np.std(np.mean(p_matrix, axis=0), ddof=1)
    out['max_p'] = np.max(p_matrix)
    out['mean_p'] = np.mean(p_matrix)
    out['std_p'] = np.std(p_matrix, ddof=1)

    #% Within-sample log-likelihood
    train_ll = model.score(y_train_reshaped)
    out['LLtrainpersample'] = model.monitor_.history[-1] / n_train
    out['nit'] = model.monitor_.iter

    #Calculate log likelihood for the test data
    out['LLtestpersample'] = model.score(y_test_reshaped)/n_test
    out['LLdifference'] = out['LLtestpersample'] - out['LLtrainpersample']

    return out

def fit_subsegments(y: ArrayLike, model: str = 'ar', order: int = 2, subset_how: str = 'uniform',
                    sample_p: Union[list, tuple] = [20, 0.1]) -> dict:
    """
    Robustness of model parameters across different segments of a time series.

    The spread of parameters obtained (including in-sample goodness of fit statistics) 
    provides some indication of stationarity. Values of goodness of fit provide some 
    indication of model suitability.

    Parameters
    ----------
    y : array-like
        The input time series.
    model : str, optional
        The model to fit in each segment of the time series:

        - 'ar': fits an AR model of a specified order.
            Outputs are how the fitted AR parameters vary across the different 
            segments of time series.
        - 'arma': Not yet implemented.
        - 'ss': Not yet implemented.

        Default is ``'ar'``.

    order : int, optional
        The order of the model to fit (used for 'ar', 'ss', or 'arma' models). Default is 2.
    subset_how : str, optional
        How to choose segments from the time series, either:

        - 'uniform' (uniformly) 
        - 'rand' (at random) [not implemented].

        Default is ``'uniform'``.
         
    sample_p : list or tuple, optional
        A two-vector specifying how many segments to take and of what length.
        Of the form [n_samples, length], where length can be a proportion of the time-series length.
        For example, [20, 0.1] takes 20 segments of 10% the time-series length.
        Default is [20, 0.1].

    Returns
    -------
    dict
        Dictionary of statistics on the spread and mean of fitted model parameters 
        and goodness of fit across segments.
    """
    y = np.asarray(y)
    N = len(y)
    num_pred = sample_p[0]
    if subset_how == 'uniform':
        if len(sample_p) == 1:  # size will depend on number of unique subsegments
            # num_pred+1 boundaries = num_pred portions
            spts = np.round(np.linspace(0, N, num_pred + 1)).astype(int)
            r = np.zeros((num_pred, 2), dtype=int)
            r[:, 0] = spts[:num_pred] + 1  # +1 for 1-based indexing (if needed)
            r[:, 1] = spts[1:]
        else:
            if sample_p[1] < 1:  # specified a fraction of time series
                l = int(np.floor(N * sample_p[1]))
            else:  # specified an absolute interval
                l = int(sample_p[1])
            # num_pred boundaries
            spts = np.round(np.linspace(1, N - l + 1, num_pred)).astype(int)
            r = np.zeros((num_pred, 2), dtype=int)
            r[:, 0] = spts
            r[:, 1] = spts + l - 1
    elif subset_how == 'rand':
        raise NotImplementedError("Subset method not yet implemented.")
    else:
        raise ValueError(f"Unknown subset method: {subset_how}")
    # Fit the model to each training set
    if model == 'ar':
        avals = np.zeros((num_pred,order))
        for i in range(num_pred):
            dat = y[r[i, 0]:r[i, 1]]
            m = AutoReg(dat, lags=order, trend='n')
            results = m.fit()
            avals[i, :] = -results.params
        #% Statistics on fitted AR parameters
        out = {}
        for i in range(order):
            out[f'a_{i+1}_std'] = np.std(avals[:, i], ddof=1)
            out[f'a_{i+1}_mean'] = np.mean(avals[:, i])
            out[f'a_{i+1}_max'] = np.max(avals[:, i])
            out[f'a_{i+1}_min'] = np.min(avals[:, i])
    elif model in ['arsbc', 'ss', 'arma']:
        raise NotImplementedError("Model not yet implemented.")
    else:
        raise ValueError(f"Unknown model: {model}")
    return out

def loop_local_simple(y: ArrayLike, forecast_meth: str = 'mean') -> dict:
    """
    How simple local forecasting depends on window length.
    
    Analyzes the outputs of local_simple for a range of local window lengths, l.
    Loops over the length of the data to use for local_simple prediction.
    
    Parameters
    ----------
    y : array-like
        The input time series.
    forecast_meth : str, optional
        The prediction method:

        - 'mean': local mean prediction
        - 'median': local median prediction

        Default is ``'mean'``.
        
    Returns
    -------
    dict
        Dictionary containing statistics about how forecasting performance varies
        with window length.
    """
    y = np.asarray(y)
    if forecast_meth == 'mean':
        train_length_range = np.arange(1, 11)
    elif forecast_meth == 'median':
        train_length_range = np.arange(1, 19, 2)
    else:
        raise ValueError(f"Unknown prediction method: {forecast_meth}")
    stats_st = np.zeros((len(train_length_range), 5))
    for i in range(len(train_length_range)):
        outtmp = local_simple(y, forecast_meth, train_length_range[i])
        stats_st[i, 0] = outtmp['stderr']
        stats_st[i, 1] = outtmp['sws']
        stats_st[i, 2] = outtmp['swm']
        stats_st[i, 3] = outtmp['ac1']
        stats_st[i, 4] = outtmp['ac2']
    # Compute statistics from the shapes of the curves
    # (1) root mean square error
    out = {}
    std_err_chnn = np.mean(np.diff(stats_st[:, 0]))/(np.ptp(stats_st[:, 0]))
    out['stderr_chn'] = std_err_chnn
    out['stderr_meansgndiff'] = np.mean(np.sign(np.diff(stats_st[:, 0])))
    # (ii) Is there a peak?
    if std_err_chnn < 1: # on the whole decreasing, as expected
        wigv = np.max(stats_st[:, 0])
        wig = np.where(stats_st[:, 0] == wigv)[0][0]  # find first occurrence
        if wig != 0 and stats_st[wig - 1, 0] > wigv:
            wig = np.nan  # maximum is not a local maximum; previous value exceeds it
        elif wig != len(train_length_range) - 1 and stats_st[wig + 1, 0] > wigv:
            wig = np.nan  # maximum is not a local maximum; the next value exceeds it
    else:
        wigv = np.min(stats_st[:, 0])
        wig = np.where(stats_st[:, 0] == wigv)[0][0]  # find first occurrence
        if wig != 0 and stats_st[wig - 1, 0] < wigv:
            wig = np.nan  # minimum is not a local minimum; previous value is less
        elif wig != len(train_length_range) - 1 and stats_st[wig + 1, 0] < wigv:
            wig = np.nan  # minimum is not a local minimum; the next value is less
    if not np.isnan(wig):
        out['stderr_peakpos'] = wig
        out['stderr_peaksize'] = wigv / np.mean(stats_st[:, 0])
    else:  # put NaNs in all the outputs
        out['stderr_peakpos'] = np.nan
        out['stderr_peaksize'] = np.nan

    #% (2)-(5) Curve statistics for the remaining metrics:
    #%   sws (sliding window stationarity), swm (sliding window mean), ac1, ac2
    for name, col in (('sws', 1), ('swm', 2), ('ac1', 3), ('ac2', 4)):
        curve = stats_st[:, col]
        out[f'{name}_chn'] = np.mean(np.diff(curve)) / np.ptp(curve)
        out[f'{name}_meansgndiff'] = np.mean(np.sign(np.diff(curve)))
        out[f'{name}_stdn'] = np.std(curve, ddof=1) / np.ptp(curve)

    return out

def local_simple(y: ArrayLike, forecast_meth: str = 'mean',
                 train_length: Union[int, str] = 3) -> dict:
    """
    Simple local time-series forecasting.
    
    Simple predictors using the past trainLength values of the time series to
    predict its next value.
    
    Parameters
    ----------
    y : array-like
        The input time series.
    forecast_meth : str, optional
        The forecasting method:

        - 'mean': local mean prediction using the past trainLength time-series values
        - 'median': local median prediction using the past trainLength time-series values  
        - 'lfit': local linear prediction using the past trainLength time-series values

        Default is ``'mean'``.

    train_length : int or str, optional
        The number of time-series values to use to forecast the next value.
        If 'ac', uses first zero-crossing of autocorrelation function.
        Default is 1.
        
    Returns
    -------
    dict
        Dictionary containing output statistics on the residuals of the simple forecasting method. 

    """
    y = np.asarray(y)
    N = len(y)
    # % Do the local prediction
    if train_length == 'ac':
        lp = first_crossing(y, 'ac', 0, 'discrete')
    else:
        #the e length of the subsegment preceding to use to predict the subsequent value
        train_length = int(train_length)
        lp = train_length 
    evalr = np.arange(lp, N) #range over which to evaluate the forecast
    if np.size(evalr) == 0:
        logger.warning("This time series is too short for forecasting")
        return np.nan
    if forecast_meth in ('mean', 'median'):
        # All length-lp windows at once. W.mean(axis=1) reduces the same contiguous
        # elements in the same order as np.mean(window), so it is bit-identical.
        W = sliding_window_view(y, lp)[:len(evalr)]
        pred = W.mean(axis=1) if forecast_meth == 'mean' else np.median(W, axis=1)
        res = pred - y[evalr]  # prediction - value
    elif forecast_meth == 'lfit':
        res = np.zeros(len(evalr))
        for i in range(len(evalr)):
            # Fit linear
            p = np.polyfit(np.arange(1, lp+1), y[evalr[i]-lp:evalr[i]], 1)
            res[i] = np.polyval(p, lp+1) - y[evalr[i]]  # prediction - value
    else:
        raise ValueError(f"Unknown forecasting method: {forecast_meth}")
    
    #Output statistics on the residuals, res
    #% Mean residual (mean error/bias):
    out = {}
    out['meanerr'] = np.mean(res)
    #% Spread of residuals:
    out['stderr'] = np.std(res, ddof=1)
    out['meanabserr'] = np.mean(np.abs(res))
    #% Stationarity of residuals:
    out['sws'] = sliding_window(res, 'std', 'std', 5, 1) # across five non-overlapping segments
    out['swm'] = sliding_window(res, 'mean', 'std', 5, 1) # across five non-overlapping segments
    #% TODO Normality of residuals
    #% Autocorrelation structure of the residuals:
    out['ac1'] = autocorr(res, 1, 'Fourier')[0]
    out['ac2'] = autocorr(res, 2, 'Fourier')[0]
    taures = first_crossing(res, 'ac', 0, 'continuous')
    out['taures'] = taures
    out['tauresrat'] = taures / first_crossing(y, 'ac', 0, 'continuous')

    return out

def exp_smoothing(x: ArrayLike, n_train: Union[None, int, float] = None,
                  alpha: Union[str, float] = 'best') -> dict:
    """
    Exponential smoothing time-series prediction model.

    Fits an exponential smoothing model to the time series using a training set to
    fit the optimal smoothing parameter, alpha, and then applies the result to
    predict the rest of the time series.

    References
    ----------
    .. [1] "The Analysis of Time Series", C. Chatfield, CRC Press LLC (2004).
        Code adapted from Siddharth Arora (Siddharth.Arora@sbs.ox.ac.uk).

    Parameters
    ----------
    x : array-like
        The input time series.
    n_train : int or float, optional
        The number of samples to use for training. Can be an integer or a 
        proportion of the time-series length. Default is `None`.
    alpha : str or float, optional
        The exponential smoothing parameter. If ``'best'``, the function
        optimizes alpha on the training set. Default is ``'best'``.

    Returns
    -------
    dict
        Dictionary including the fitted alpha and statistics on the 
        residuals from the prediction phase.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    out = {}

    # --- Check Inputs ---
    if n_train is None:
        n_train = min(100, N)
    
    if 0 < n_train < 1:
        n_train = int(np.floor(N * n_train))
        
    min_train, max_train = 100, 1000
    
    if n_train > max_train:
        logger.info(f"Training set size reduced from {n_train} to {max_train}.")
        n_train = max_train
        
    if n_train < min_train:
        logger.info(f"Training set size increased from {n_train} to {min_train}.")
        n_train = min_train
        
    if N < n_train:
        logger.warning("Time series is too short for the specified training size.")
        return np.nan
        
    # --- Find Optimal Alpha ---
    if alpha == 'best':
        xtrain = x[:n_train]

        def _rmse_for_alpha(a):
            xf = _fit_exp_smooth(xtrain, a)
            fore, orig = xf[2:], xtrain[2:]
            return np.sqrt(np.mean((fore - orig)**2)) if len(fore) > 0 else np.nan

        # (1) Initial coarse search
        alphar = np.linspace(0.1, 0.9, 5)
        rmses = np.array([_rmse_for_alpha(a) for a in alphar])

        # Check for valid RMSEs before fitting
        valid_indices = ~np.isnan(rmses)
        if np.sum(valid_indices) < 3:
            logger.info("Not enough valid points for quadratic fit; choosing best alpha from search.")
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
                rmses_ref = np.array([_rmse_for_alpha(a) for a in alphar_ref])

                valid_ref = ~np.isnan(rmses_ref)
                if not np.any(valid_ref):
                    logger.info("Could not compute RMSE in refined search; using previous alpha.")
                else:
                    p2 = np.polyfit(alphar_ref[valid_ref], rmses_ref[valid_ref], 2)
                    if p2[0] < 0: # Bad fit, fallback to best alpha in search
                        alphamin = alphar_ref[np.nanargmin(rmses_ref)]
                    else: # Minimum of the new quadratic fit
                        alphamin = -p2[1] / (2 * p2[0])
                        
        alpha = np.clip(alphamin, 0.01, 1.0)
        out['alphamin'] = alpha

    if np.isnan(alpha):
        logger.warning("Alpha optimization failed, resulting in NaN.")
        return np.nan

    # --- Final Fit and Residual Analysis ---
    y_fit = _fit_exp_smooth(x, alpha)
    yp, xp = y_fit[2:], x[2:]
    
    if len(yp) < 2:
        logger.warning("Not enough points to calculate residual statistics.")
        residout = {'mean': np.nan, 'std': np.nan, 'AC1': np.nan}
    else:
        residuals = yp - xp
        residout = residual_analysis(residuals)
    
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

def residual_analysis(e: ArrayLike) -> dict:
    """
    Analysis of residuals from a model fit.

    Given an input residual time series `e`, this function returns a dictionary 
    with fields corresponding to statistical tests on the residuals. These tests 
    are motivated by the general expectation that model residuals should be uncorrelated.

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
    out['mms'] = np.abs(out['meane']) + np.abs(std_e)

    if std_e == 0:
        e = np.zeros(len(e))
    else:
        e = z_score(e)
    
    # Analyze autocorrelation in residuals
    max_lag = 25
    autocorr_resid = autocorr(e, list(range(1, max_lag+1)), 'Fourier')
    sqrt_n = np.sqrt(N)

    # Output first 3 ACs
    out['ac1'] = autocorr_resid[0]
    out['ac2'] = autocorr_resid[1]
    out['ac3'] = autocorr_resid[2]
    out['ac1n'] = np.abs(autocorr_resid[0]) * sqrt_n
    out['ac2n'] = np.abs(autocorr_resid[1]) * sqrt_n
    out['ac3n'] = np.abs(autocorr_resid[2]) * sqrt_n

    #% Median normalized distance from zero
    out['acmnd0'] = np.median(np.abs(autocorr_resid)) * sqrt_n
    out['acsnd0'] = np.std(np.abs(autocorr_resid), ddof=1) * sqrt_n
    out['propbth'] = np.sum(np.abs(autocorr_resid) < 2.6/sqrt_n)/max_lag

    # % First time to get below the significance threshold
    ftbth_indices = np.where(np.abs(autocorr_resid) < 2.6/sqrt_n)[0]
    if ftbth_indices.size > 0:  # Alternative way to check if array is empty
        out['ftbth'] = ftbth_indices[0] + 1
    else:
        out['ftbth'] = max_lag + 1

    # Durbin-Watson test statistic (like AC1)
    out['dwts'] = np.sum((e[1:] - e[:-1])**2) / np.sum(e**2)

    # Distribution tests
    res = ks_1samp(e, norm.cdf)
    out['normksstat'] = res.statistic
    out['normp'] = res.pvalue

    return out

def ar_cov(y: ArrayLike, p: int = 2) -> dict:
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
    out['res_AC1'] = autocorr(err, 1, 'Fourier')[0]
    out['res_AC2'] = autocorr(err, 2, 'Fourier')[0]

    return out

def _arconf_from_arfit(fitted_ar, the_conf_interval: float = 0.95) -> dict:
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
    t_crit = t.ppf(0.5 + the_conf_interval / 2, df=dof) # quantiles of the t distrib
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

def _get_criteria(sel, N, crit="aic"):
    if crit == "aic":
        se = sel.aic
    elif crit == "bic":
        se = sel.bic
    else:
        raise ValueError(f"Unknown criteria: {crit}!")

    se.pop(0)  # drop the zero-lag (intercept-only) model
    keys = list(se.keys())
    # Order by AR order (the last element of each lag-tuple key); aic and bic
    # are keyed identically, so this ordering is shared.
    order = np.argsort([k[-1] for k in keys])
    return np.array([se[keys[i]] / N for i in order])  # normalise by num observations

def ar_fit(y: ArrayLike, p_min: int = 1, p_max: int = 10, selector: str = 'sbc') -> dict:
    """
    Statistics of a fitted AR model to a time series.

    Fits autoregressive (AR) models of orders p = p_min, p_min + 1, ...,  to the input time series,
    selects the optimal model order using Schwartz's Bayesian Criterion (SBC), and returns statistics
    on the fitted model, residuals, and confidence intervals.

    References
    ---------
    .. [1] "Estimation of parameters and eigenmodes of multivariate autoregressive models",
        A. Neumaier and T. Schneider, ACM Trans. Math. Softw. 27, 27 (2001)
    .. [2] "Algorithm 808: ARFIT---a Matlab package for the estimation of parameters and eigenmodes of multivariate autoregressive models",
        T. Schneider and A. Neumaier, ACM Trans. Math. Softw. 27, 58 (2001)

    Parameters
    ----------
    y : array-like
        The input time series.
    p_min : int, optional
        The minimum AR model order to fit. Default is 1.
    p_max : int, optional
        The maximum AR model order to fit. Default is 10.
    selector : str, optional
        Criterion to select optimal model order (e.g., 'sbc', cf. ARFIT package documentation). Default is ``'sbc'``.

    Returns
    -------
    dict
        Dictionary containing statistics of a fitted AR model to a time series.
    """
    y = np.asarray(y)
    N = len(y)
    p_min = int(p_min)
    p_max = int(p_max)
    if selector in ['bic', 'sbc']: # bic and sbc are the same metrics
        selector = 'bic'
    #(I) Fit AR model)
    sel = ar_select_order(y, maxlag=p_max, ic=selector, glob=False, trend='n') # bic is the same as sbc
    p_optimal = max(p_min, np.max(sel.ar_lags)) if sel.ar_lags is not None else p_min
    ps = np.arange(p_min, p_max+1)
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
    #test of auto correlation in the residuals (test up to lag 20)
    out['res_siglev'] = ljung_box_pvalue(res.resid, n_lags=20, model_df=p_optimal)

    # Correlation test of residuals.
    # autocorr(...,'Fourier') computes the full FFT ACF internally, so compute
    # lags 1..20 once and reuse lag 1 (acf[0]) rather than running two FFTs.
    resids = res.resid
    acf = autocorr(resids, list(range(1, 21)), 'Fourier')
    out['res_ac1'] = acf[0]
    out['res_ac1_norm'] = out['res_ac1']/np.sqrt(N)

    #Calculate correlations up to 20, return how many exceed significance threshold
    out['pcorr_res'] = np.sum(np.abs(acf) > 1.96/np.sqrt(N))/20

    # Confidence Intervals
    a_err = _arconf_from_arfit(res, 0.95)['A_err']
    out['aerr_min'] = np.min(a_err)
    out['aerr_max'] = np.max(a_err)
    out['aerr_mean'] = np.mean(a_err)

    return out

def is_seasonal(y: ArrayLike) -> int:
    """
    Fits a 'sin1' (single frequency sinusoid) model to the time series. 
    The output is binary: 1 if the goodness of fit, R^2, exceeds 0.3 and
    the amplitude of the fitted periodic component exceeds 0.5, and 0 otherwise.

    Parameters
    ----------
    y : array-like
        The input time series.
    
    Returns
    -------
    bool
        Binary: 1 (= seasonal), 0 (= non-seasonal)
    """
    y = np.asarray(y).flatten()
    N = len(y)
    r = np.arange(1, N + 1)
    
    model = SineModel()
    params = model.guess(y, x=r) 
    
    result = model.fit(y, params, x=r)
    # extract the amplitude
    a1 = result.params['amplitude'].value
    r_squared = result.rsquared

    #% Condition 1: fit is ok
    th_fit = 0.3 # % r2 > th_fit
    #% Condition 2: amplitude is not too small
    th_ampl = 0.5 #% a1 > th_ampl

    out = 0 # test thinks the time series doesn't have any strong periodicities
    if r_squared > th_fit and abs(a1) > th_ampl:
        out = 1 # test thinks the time series has strong periodicities
    
    return out

def _gp_learn_hyperp(tt: np.ndarray, yt: np.ndarray, cov, nfevals: int = -50) -> np.ndarray:
    """
    learn GP hyperparameters for the time series ``(tt, yt)``.

    The GP is a mean-zero process with a Gaussian likelihood and Laplace
    inference; ``nfevals`` is negative, so it caps the number of function
    evaluations rather than the number of line searches.

    Returns the flattened hyperparameter vector ``[cov..., lik]`` -- gpml
    unwraps the hyperparameter struct with its fields alphabetised (cov, lik,
    mean), and the mean is empty for a mean-zero process.

    Raises ``numpy.linalg.LinAlgError`` if the covariance loses positive
    definiteness, the counterpart of gpml's ``MATLAB:posdef`` error.
    """
    nhps = cov.n_hyp
    # Initial values: the length parameter is in the ballpark of the difference
    # between time elements; the remaining log-hyperparameters start at zero and
    # the likelihood noise at log(0.1).
    hyp0 = np.concatenate([[np.log(np.mean(np.diff(tt)))],
                           np.zeros(nhps - 1), [np.log(0.1)]])

    def _nlz(theta):
        hyp = {'cov': theta[:nhps], 'lik': theta[nhps], 'mean': np.zeros(0)}
        nlZ, dnlZ = gp_train(hyp, cov, tt, yt)
        return nlZ, np.concatenate([dnlZ['cov'], dnlZ['lik'], dnlZ['mean']])

    theta, _, _ = minimize(hyp0, _nlz, nfevals)
    return theta


def gp_fit_across(y: ArrayLike, cov_func: str = 'covSEiso_covNoise',
                  npoints: int = 20) -> dict:
    """
    Gaussian Process time-series modeling for local prediction.

    Trains a Gaussian Process model on equally-spaced points throughout the time
    series and uses the model to predict its intermediate values.

    Parameters
    ----------
    y : array-like
        The input time series.
    cov_func : str
        The covariance function. Only ``'covSEiso_covNoise'``, the gpml
        ``covSum`` of a squared exponential and a noise term, is supported -- it
        is the only configuration hctsa instantiates. Default is
        ``'covSEiso_covNoise'``.
    npoints : int
        The number of points through the time series to fit the GP model to.
        Default is 20.

    Returns
    -------
    dict
        Dictionary summarising the error and the fitted hyperparameters.
    """
    if cov_func != 'covSEiso_covNoise':
        raise ValueError(
            "Only cov_func='covSEiso_covNoise' is supported "
            f"(got {cov_func!r}); it is the only variant used by hctsa.")

    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    npoints = int(npoints)

    cov = CovSEisoNoise
    nhps = cov.n_hyp

    tt = np.floor(_linspace(1, N, npoints))
    yt = y[tt.astype(int) - 1]

    try:
        theta = _gp_learn_hyperp(tt, yt, cov)
    except np.linalg.LinAlgError:
        logger.warning('Lack of positive definite matrix for this time series')
        return {k: np.nan for k in
                ('rmserr', 'meanstderr', 'stdmu', 'meanS', 'stdS', 'mlikelihood',
                 'logh1', 'logh2', 'logh3', 'h_lonN')}

    loghyper = theta[:nhps]
    hyp = {'cov': loghyper, 'lik': theta[nhps], 'mean': np.zeros(0)}

    # Evaluate over the whole space now, predicting at the test times ts
    if N <= 2000:
        ts = np.arange(1, N + 1, dtype=float)
    else:  # memory constraints force us to crudely resample
        ts = np.floor(_linspace(1, N, 2000) + 0.5)  # MATLAB round()
    y_ts = y[ts.astype(int) - 1]

    mu, S2, _, _ = gp_predict(hyp, cov, tt, yt, ts)

    # Output statistics
    S = np.sqrt(S2)  # standard deviation function, S
    out = {}
    # rms error from mean function, mu
    out['rmserr'] = np.mean(np.sqrt((y_ts - mu) ** 2))
    out['meanstderr'] = np.mean(np.abs(y_ts - mu) / S)
    out['stdmu'] = np.std(mu, ddof=1)
    out['meanS'] = np.mean(S)
    out['stdS'] = np.std(S, ddof=1)

    # Marginal likelihood
    try:
        out['mlikelihood'] = gp_train(hyp, cov, ts, y_ts, want_dnlZ=False)[0]
    except Exception:
        out['mlikelihood'] = np.nan

    # Log-hyperparameters
    for i in range(nhps):
        out[f'logh{i + 1}'] = loghyper[i]

    # Give extra output based on length parameter on length of time series
    out['h_lonN'] = np.exp(loghyper[0]) / N

    return out


def gp_local_prediction(y: ArrayLike, cov_func: str = 'covSEiso_covNoise',
                        num_train: int = 10, num_test: int = 3,
                        num_preds: int = 20, pmode: str = 'randomgap',
                        random_seed: int = 0) -> dict:
    """
    Gaussian Process time-series model for local prediction.

    Fits a Gaussian Process model to a section of the time series and uses it to
    predict the subsequent datapoints, repeated at equally-spaced positions
    through the time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    cov_func : str
        The covariance function. Only ``'covSEiso_covNoise'``, the gpml
        ``covSum`` of a squared exponential and a noise term, is supported -- it
        is the only configuration hctsa instantiates. Default is
        ``'covSEiso_covNoise'``.
    num_train : int
        The number of training samples (for each iteration). Default is 20.
    num_test : int
        The number of testing samples (for each iteration). Default is 5.
    num_preds : int
        The number of predictions to make. Default is 10.
    pmode : str
        The prediction mode, one of:

        - ``'frombefore'``: predicts the following values of the time series by
          training on preceding values,
        - ``'beforeafter'``: predicts the preceding time series values by
          training on the following values,
        - ``'randomgap'``: predicts random values within a segment of time
          series by training on the other values in that segment.

        Default is ``'frombefore'``.
    random_seed : int or None
        Seed for the Mersenne Twister, reset before each prediction (as
        ``BF_ResetSeed`` does), used by the ``'randomgap'`` mode. ``None``
        leaves the stream alone, matching ``BF_ResetSeed('none')``. Default
        is 0.

    Returns
    -------
    dict
        Summaries of the quality of the predictions made, the mean and spread of
        the obtained hyperparameter values, and the marginal likelihoods.
    """
    if cov_func != 'covSEiso_covNoise':
        raise ValueError(
            "Only cov_func='covSEiso_covNoise' is supported "
            f"(got {cov_func!r}); it is the only variant used by hctsa.")

    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    num_train, num_test, num_preds = int(num_train), int(num_test), int(num_preds)

    cov = CovSEisoNoise
    nhps = cov.n_hyp

    if pmode in ('frombefore', 'randomgap'):
        spns = np.floor(_linspace(1, N - (num_test + num_train), num_preds))
    elif pmode == 'beforeafter':
        spns = np.floor(_linspace(1, N - (num_test + num_train * 2), num_preds))
    else:
        raise ValueError(f"Unknown prediction mode {pmode!r}")
    spns = spns.astype(int)

    out_keys = (
        'maxstderr', 'maxabserr', 'minstderr', 'minabserr', 'meanstderr',
        'meanabserr', 'meanstderr_run', 'meanabserr_run', 'maxstderr_run',
        'maxabserr_run', 'minstderr_run', 'minabserr_run', 'maxerrbar',
        'meanerrbar', 'minerrbar',
        *(f'{s}logh{i + 1}' for i in range(nhps) for s in ('mean', 'std')),
        'maxmlik', 'minmlik', 'stdmlik',
    )

    mus = np.zeros((num_test, num_preds))        # predicted values
    stderrs = np.zeros((num_test, num_preds))    # standard errors on predictions
    yss = np.zeros((num_test, num_preds))        # test values
    mlikelihoods = np.zeros(num_preds)           # marginal likelihoods of model
    loghypers = np.zeros((nhps, num_preds))      # log-hyperparameters

    rng = np.random.RandomState() if random_seed is None else None

    for i in range(num_preds):
        # (0) Set up test and training sets
        sp = spns[i]
        if pmode == 'frombefore':
            tt = np.arange(1, num_train + 1, dtype=float)          # times (from 1)
            yt = y[sp - 1:sp - 1 + num_train]                      # training data
            ts = np.arange(num_train + 1, num_train + num_test + 1, dtype=float)
            ys = y[sp - 1 + num_train:sp - 1 + num_train + num_test]  # test data

        elif pmode == 'randomgap':
            if random_seed is not None:
                rng = _ml_rng(random_seed)
            n = num_train + num_test
            t = np.arange(1, n + 1, dtype=float)
            r = _ml_randperm(n, rng)
            yy = y[sp - 1:sp - 1 + n]

            rt = np.sort(r[:num_train])
            tt, yt = t[rt - 1], yy[rt - 1]

            rs = np.sort(r[num_train:])
            ts, ys = t[rs - 1], yy[rs - 1]

        else:  # 'beforeafter'
            n = 2 * num_train + num_test
            t = np.arange(1, n + 1, dtype=float)
            yy = y[sp - 1:sp - 1 + n]

            rt = np.concatenate([np.arange(1, num_train + 1),
                                 np.arange(num_train + num_test + 1, n + 1)])
            tt, yt = t[rt - 1], yy[rt - 1]

            rs = np.arange(num_train + 1, num_train + num_test + 1)
            ts, ys = t[rs - 1], yy[rs - 1]

        # Process to normalize scales (the same transformation for both sets)
        yt_mean, yt_std = np.mean(yt), np.std(yt, ddof=1)
        ys = (ys - yt_mean) / yt_std
        yt = (yt - yt_mean) / yt_std

        # (1) Learn hyperparameters from the training set
        try:
            theta = _gp_learn_hyperp(tt, yt, cov)
        except np.linalg.LinAlgError:
            logger.warning('Unable to learn hyperparameters for this time series')
            return {k: np.nan for k in out_keys}

        loghyper = theta[:nhps]
        loghypers[:, i] = loghyper
        hyp = {'cov': loghyper, 'lik': theta[nhps], 'mean': np.zeros(0)}

        # Marginal likelihood for this model, with hyperparameters optimized
        # over the training data
        mlikelihoods[i] = -gp_train(hyp, cov, tt, yt, want_dnlZ=False)[0]

        # (2) Evaluate at the test points, based on the training time/data
        mu, S2, _, _ = gp_predict(hyp, cov, tt, yt, ts)

        mus[:, i] = mu                     # ~predicted values for time-series points
        stderrs[:, i] = 2 * np.sqrt(S2)    # ~errors on those predictions
        yss[:, i] = ys

    # (1) Prediction error measures
    allabserrs = np.abs(mus - yss)                 # absolute errors
    allstderrs = allabserrs / stderrs   # in units of 95% confidence-interval bars

    out = {}
    # Largest/smallest/mean error across all runs:
    out['maxstderr'] = np.max(allstderrs)
    out['maxabserr'] = np.max(allabserrs)
    out['minstderr'] = np.min(allstderrs)
    out['minabserr'] = np.min(allabserrs)
    out['meanstderr'] = np.mean(allstderrs)
    out['meanabserr'] = np.mean(allabserrs)

    # Summary of how it did on each run:
    stderr_run = np.mean(allstderrs, axis=0)
    abserr_run = np.mean(allabserrs, axis=0)

    out['meanstderr_run'] = np.mean(stderr_run)
    out['meanabserr_run'] = np.mean(abserr_run)
    out['maxstderr_run'] = np.max(stderr_run)
    out['maxabserr_run'] = np.max(abserr_run)
    out['minstderr_run'] = np.min(stderr_run)
    out['minabserr_run'] = np.min(abserr_run)

    # Error bar stats:
    out['maxerrbar'] = np.max(stderrs)     # largest error bar
    out['meanerrbar'] = np.mean(stderrs)   # mean error bar length
    out['minerrbar'] = np.min(stderrs)     # minimum error bar length

    # (2) Hyperparameter measures: mean and std for each hyperparameter
    for i in range(nhps):
        out[f'meanlogh{i + 1}'] = np.mean(loghypers[i, :])
        out[f'stdlogh{i + 1}'] = np.std(loghypers[i, :], ddof=1)

    # (3) Marginal likelihood measures
    out['maxmlik'] = np.max(mlikelihoods)
    out['minmlik'] = np.min(mlikelihoods)
    out['stdmlik'] = np.std(mlikelihoods, ddof=1)

    return out


def _ml_rng(seed: int) -> np.random.RandomState:
    """
    ``rng(seed, 'twister')``, as a numpy ``RandomState``.
    """
    return np.random.RandomState(5489 if seed == 0 else seed)


def _ml_randperm(n: int, rng: np.random.RandomState) -> np.ndarray:
    """
    MATLAB's ``randperm(n)``: the 1-based ordering that sorts ``rand(1, n)``.
    """
    return np.argsort(rng.random_sample(n), kind='stable') + 1


def _linspace(d1: float, d2: float, n: int) -> np.ndarray:
    """
    Helper function for gp_fit_across
    """
    n1 = n - 1
    y = d1 + np.arange(n) * (d2 - d1) / n1
    y[0] = d1
    if n1 > 0:
        y[n - 1] = d2
    return y
