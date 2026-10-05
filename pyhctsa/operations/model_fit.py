import warnings
from typing import Union

import numba
import numpy as np
from numpy.typing import ArrayLike
from numpy.lib.stride_tricks import sliding_window_view
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import curve_fit
from scipy.signal import lfilter
from scipy.special import gammaincc
from scipy.stats import ks_1samp, norm, t
from statsmodels.tsa.ar_model import AutoReg
from lmfit.models import SineModel
import logging
logger = logging.getLogger('pyhctsa')

from ..operations.correlation import autocorr, first_crossing
from ..operations.physics import _ksdensity
from ..operations.stationarity import sliding_window
from ..toolboxes.matlab.gpml.gpml import CovSEisoNoise, gp_predict, gp_train
from ..toolboxes.matlab.optimizers import minimize
from ..utils import _linspace, _ml_randperm, _ml_rng, get_tau, matlab_quantile, z_score

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
        Random seed for the initial parameters of the fit. Default is 0.

    Returns
    -------
    dict
        Dictionary of statistics based on the fitted HMM: the sorted state means
        (``Mu_1``, ...) and their ``meanMu``, ``rangeMu``, ``maxMu``, ``minMu``;
        the tied covariance ``Cov``; the transition matrix summaries
        ``Pmeandiag``, ``stdmeanP``, ``maxP``, ``meanP``, ``stdP``; the training
        log-likelihood per sample ``LLtrainpersample`` and the number of EM
        iterations ``nit``; and the test log-likelihood per sample
        ``LLtestpersample`` and ``LLdifference``.

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

    # Initialize and iterate Baum-Welch as in Zoubin Ghahramani's ZG_hmm (used by hctsa):
    # random state means around the data mean (scaled by the data standard
    # deviation), random start probabilities and transition matrix, a tied
    # variance equal to the data variance; at most 30 cycles, stopping when the
    # proportional change in the log-likelihood falls below tol. (hmmlearn's
    # k-means initialization and absolute tolerance find different, generally
    # poorer, local optima: the fitted-model statistics then do not follow the
    # distribution of hctsa's.)
    rng = _ml_rng(0 if random_seed is None else int(random_seed))
    tol = 1e-4
    cov0 = np.var(y_train, ddof=1)
    mu0 = rng.randn(num_states, 1) * np.sqrt(cov0) + np.mean(y_train)
    pi0 = rng.random_sample(num_states)
    pi0 = pi0 / pi0.sum()
    p0 = rng.random_sample((num_states, num_states))
    p0 = p0 / p0.sum(axis=1, keepdims=True)

    model = GaussianHMM(n_components=num_states,
                        covariance_type='tied',
                        n_iter=1,  # one EM cycle per fit() call, so that we control the stopping rule
                        tol=0,
                        params='stmc',
                        init_params='')
    model.startprob_ = pi0
    model.transmat_ = p0
    model.means_ = mu0
    model.covars_ = np.array([[cov0]])

    LL = []  # log-likelihood of the training data at the start of each cycle
    lik_base = 0.0
    for cycle in range(1, 31):
        model.fit(y_train_reshaped)  # one E step and M step
        lik = model.monitor_.history[-1]
        old_lik = LL[-1] if LL else 0.0
        LL.append(lik)
        if cycle <= 2:
            lik_base = lik
        elif lik < old_lik:
            pass  # a decrease (numerical violation): keep going, as ZG_hmm does
        elif (lik - lik_base) < (1 + tol) * (old_lik - lik_base) or not np.isfinite(lik):
            break

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
    out['stdmeanP'] = np.std(np.mean(p_matrix, axis=0), ddof=1)
    out['maxP'] = np.max(p_matrix)
    out['meanP'] = np.mean(p_matrix)
    out['stdP'] = np.std(p_matrix, ddof=1)

    #% Within-sample log-likelihood
    out['LLtrainpersample'] = np.max(LL) / n_train
    out['nit'] = len(LL)

    #Calculate log likelihood for the test data
    out['LLtestpersample'] = model.score(y_test_reshaped)/n_test
    out['LLdifference'] = out['LLtestpersample'] - out['LLtrainpersample']

    return out

def _ar_fb(seg: np.ndarray, order: int) -> tuple:
    """
    AR model by forward-backward least squares (MATLAB's default ``ar`` estimator).

    Minimizes the sum of the squared forward and backward prediction errors over the
    segment (no windowing, no mean removal). Returns the coefficients of the polynomial
    ``1 + a_1 z^-1 + ... + a_p z^-p`` (the negative of the usual AR coefficients) and
    Akaike's final prediction error, ``FPE = (SSE/N) (1 + p/N) / (1 - p/N)``, where
    ``SSE`` is the sum of squared forward residuals.
    """
    n = len(seg)
    p = order
    fwd = sliding_window_view(seg, p + 1)            # rows: y[t-p], ..., y[t]
    lags_f, tgt_f = fwd[:, p - 1::-1], fwd[:, p]     # y[t-1], ..., y[t-p]; y[t]
    lags_b, tgt_b = fwd[:, 1:], fwd[:, 0]    # y[t+1], ..., y[t+p]; y[t]
    X = -np.vstack([lags_f, lags_b])
    b = np.concatenate([tgt_f, tgt_b])
    a = np.linalg.lstsq(X, b, rcond=None)[0]
    sse_f = np.sum((tgt_f + lags_f @ a) ** 2)
    fpe = sse_f / n * (1 + p / n) / (1 - p / n)
    return a, fpe

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

        - 'arsbc': fits an AR model of the best order (1 to 10) by the Schwarz
            Bayesian criterion (ARFIT algorithm, zero mean). Outputs are how the
            optimal order and the SBC vary across segments (``orders_*``, ``sbcs_*``).
            The ``order`` input is not used.
        - 'ar': fits an AR model of a specified order by forward-backward least
            squares (MATLAB's default ``ar`` estimator). Outputs are how Akaike's
            final prediction error (``fpe_*``) and the fitted AR parameters
            (``a_k_*``, as in the polynomial 1 + a_1 z^-1 + ..., the negative of the
            usual coefficients) vary across segments.
        - 'arcrosspred': splits the series into ``sample_p`` non-overlapping segments
            (requires ``subset_how='uniform'`` and a scalar-like ``sample_p``, a segment
            count), fits an AR model of the given order to each, and uses every
            segment's model to predict, one step ahead, every segment (including itself).
            The result is a matrix of cross-prediction root-mean-square errors (row:
            predicting model, column: predicted segment), of which the spread and
            off-diagonal statistics are returned. NaN if any segment is shorter than
            ``5 * (order + 1)`` or an AR fit fails.
        - 'arma': Not implemented (deregistered in hctsa).
        - 'ss': Not yet implemented.

        Default is ``'ar'``.

    order : int, optional
        The order of the model to fit (used for 'ar', 'ss', or 'arma' models). Default is 2.
    subset_how : str, optional
        How to choose segments from the time series, either:

        - 'uniform' (uniformly) 
        - 'rand' (at random) [not implemented].

        Default is ``'uniform'``.
         
    sample_p : list, tuple or int, optional
        A two-vector specifying how many segments to take and of what length.
        Of the form [n_samples, length], where length can be a proportion of the time-series length.
        For example, [20, 0.1] takes 20 segments of 10% the time-series length.
        For ``model='arcrosspred'``, an integer (or length-1 list): the number of
        non-overlapping segments to partition the series into.
        Default is [20, 0.1].

    Returns
    -------
    dict
        Dictionary of statistics on the spread and mean of fitted model parameters 
        and goodness of fit across segments. For ``'arcrosspred'``: ``std``, ``range``,
        ``iqr`` (over all entries of the cross-prediction error matrix), ``stdoffdiag``,
        ``rangeoffdiag``, ``iqroffdiag`` (over the positive off-diagonal entries),
        ``stdmean``, ``rangemean``, ``stdmedian``, ``rangemedian`` (across predicted
        segments, of the mean and of the median error), ``rangerange``, ``stdrange``,
        ``rangestd``, ``stdstd`` (across predicted segments, of the range or standard
        deviation of the errors) and ``mineig`` (smallest real part of the eigenvalues
        of the matrix).
    """
    y = np.asarray(y)
    N = len(y)
    if np.ndim(sample_p) == 0:
        sample_p = [sample_p]
    num_pred = int(sample_p[0])
    if model == 'arcrosspred' and (subset_how != 'uniform' or len(sample_p) != 1):
        raise ValueError("'arcrosspred' requires subset_how = 'uniform' and a scalar sample_p "
                         "(a non-overlapping segment count)")
    if subset_how == 'uniform':
        if len(sample_p) == 1:  # size will depend on number of unique subsegments
            # num_pred+1 boundaries = num_pred portions
            spts = np.floor(_linspace(0, N, num_pred + 1) + 0.5).astype(int)  # MATLAB round()
            r = np.zeros((num_pred, 2), dtype=int)
            r[:, 0] = spts[:num_pred] + 1  # +1 for 1-based indexing (if needed)
            r[:, 1] = spts[1:]
        else:
            if sample_p[1] < 1:  # specified a fraction of time series
                l = int(np.floor(N * sample_p[1]))
            else:  # specified an absolute interval
                l = int(sample_p[1])
            # num_pred boundaries
            spts = np.floor(_linspace(1, N - l + 1, num_pred) + 0.5).astype(int)  # MATLAB round()
            r = np.zeros((num_pred, 2), dtype=int)
            r[:, 0] = spts
            r[:, 1] = spts + l - 1
    elif subset_how == 'rand':
        raise NotImplementedError("Subset method not yet implemented.")
    else:
        raise ValueError(f"Unknown subset method: {subset_how}")
    # Fit the model to each training set (r is 1-based and inclusive, as in MATLAB)
    out = {}
    if model == 'arsbc':
        # AR model of the best order (1-10) by SBC, zero mean
        orders = np.zeros(num_pred)
        sbcs = np.zeros(num_pred)
        for i in range(num_pred):
            try:
                _, A_est, _, sbc, _, _ = _arfit(y[r[i, 0] - 1:r[i, 1]], 1, 10, 'sbc', zero=True)
            except ValueError as err:
                logger.warning(f'Time series segment is too short for ARFIT: {err}')
                return np.nan
            orders[i] = len(A_est)
            sbcs[i] = np.min(sbc)
        vals, counts = np.unique(orders, return_counts=True)
        out['orders_mode'] = vals[np.argmax(counts)]  # smallest value among ties, as MATLAB mode
        out['orders_mean'] = np.mean(orders)
        out['orders_std'] = np.std(orders, ddof=1)
        out['orders_max'] = np.max(orders)
        out['orders_min'] = np.min(orders)
        out['orders_range'] = np.ptp(orders)
        out['sbcs_mean'] = np.mean(sbcs)
        out['sbcs_std'] = np.std(sbcs, ddof=1)
        out['sbcs_range'] = np.ptp(sbcs)
        out['sbcs_min'] = np.min(sbcs)
        out['sbcs_max'] = np.max(sbcs)
    elif model == 'ar':
        # AR model of the specified order (forward-backward least squares)
        fpes = np.zeros(num_pred)
        avals = np.zeros((num_pred, order))
        for i in range(num_pred):
            avals[i, :], fpes[i] = _ar_fb(y[r[i, 0] - 1:r[i, 1]], order)
        # statistics on the FPE
        out['fpe_std'] = np.std(fpes, ddof=1)
        out['fpe_mean'] = np.mean(fpes)
        out['fpe_max'] = np.max(fpes)
        out['fpe_min'] = np.min(fpes)
        out['fpe_range'] = np.ptp(fpes)
        # statistics on the fitted AR parameters, as in the polynomial 1 + a_1 z^-1 + ...
        for i in range(order):
            out[f'a_{i+1}_std'] = np.std(avals[:, i], ddof=1)
            out[f'a_{i+1}_mean'] = np.mean(avals[:, i])
            out[f'a_{i+1}_max'] = np.max(avals[:, i])
            out[f'a_{i+1}_min'] = np.min(avals[:, i])
    elif model == 'arcrosspred':
        # AR model for each of num_pred non-overlapping segments; every model then
        # predicts, one step ahead, every segment (including its own)
        seg_len = r[:, 1] - r[:, 0] + 1
        if np.any(seg_len < 5 * (order + 1)):
            logger.warning(f'Segments too short to reliably cross-predict with an AR({order}) model')
            return np.nan
        segs = [y[r[i, 0] - 1:r[i, 1]] for i in range(num_pred)]
        try:
            coefs = [_ar_fb(seg, order)[0] for seg in segs]
        except np.linalg.LinAlgError:
            return np.nan
        xperr = np.zeros((num_pred, num_pred))
        for j, seg in enumerate(segs):
            # lags y[t-1], ..., y[t-p] for t = p, ..., n-1
            lags = sliding_window_view(seg, order + 1)[:, order - 1::-1]
            target = seg[order:]
            for i in range(num_pred):
                # MATLAB's predict estimates the initial conditions, which makes the
                # first `order` one-step errors zero: only t >= order contribute
                e = -lags @ coefs[i] - target
                xperr[i, j] = np.sqrt(np.sum(e ** 2) / len(seg))
        if not np.all(np.isfinite(xperr)):
            return np.nan
        iqr = lambda v: np.diff(matlab_quantile(v, [0.25, 0.75]))[0]
        out['std'] = np.std(xperr.ravel(), ddof=1)
        out['range'] = np.ptp(xperr)
        out['iqr'] = iqr(xperr.ravel())
        offdiag = np.concatenate([xperr[np.tril_indices(num_pred, -1)],
                                  xperr[np.triu_indices(num_pred, 1)]])
        offdiag = offdiag[offdiag > 0]
        if offdiag.size == 0:
            out['iqroffdiag'] = out['stdoffdiag'] = out['rangeoffdiag'] = np.nan
        else:
            out['iqroffdiag'] = iqr(offdiag)
            out['stdoffdiag'] = np.std(offdiag, ddof=1) if offdiag.size > 1 else 0.0
            out['rangeoffdiag'] = np.ptp(offdiag)
        # how differently each segment's model behaves as a predictor vs as a target
        col_mean, col_med = np.mean(xperr, axis=0), np.median(xperr, axis=0)
        col_range, col_std = np.ptp(xperr, axis=0), np.std(xperr, axis=0, ddof=1)
        out['stdmean'] = np.std(col_mean, ddof=1)
        out['rangemean'] = np.ptp(col_mean)
        out['stdmedian'] = np.std(col_med, ddof=1)
        out['rangemedian'] = np.ptp(col_med)
        out['rangerange'] = np.ptp(col_range)
        out['stdrange'] = np.std(col_range, ddof=1)
        out['rangestd'] = np.ptp(col_std)
        out['stdstd'] = np.std(col_std, ddof=1)
        out['mineig'] = np.min(np.linalg.eigvals(xperr).real)
    elif model in ['ss', 'arma']:
        raise NotImplementedError("Model not yet implemented.")
    else:
        raise ValueError(f"Unknown model: {model}")
    return out

def _fit_exp_curve(x: np.ndarray, y: np.ndarray, prefix: str) -> dict:
    """
    Fit y = a * exp(b * x) + c by nonlinear least squares.

    The starting point is [range(y), -0.5, min(y)], as in hctsa's
    FC_LoopLocalSimple. Returns the parameters (``a``, ``b``, ``c``) and the
    goodness of fit (``r2``, ``adjr2``, ``rmse``, with ``rmse`` using the
    degrees-of-freedom-adjusted residual variance), all NaN if the fit fails.
    """
    keys = [f'{prefix}_{k}' for k in ('a', 'b', 'c', 'r2', 'adjr2', 'rmse')]
    try:
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("non-finite data")
        popt, _ = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y,
                            p0=[np.ptp(y), -0.5, np.min(y)], maxfev=10000)
        res = y - (popt[0] * np.exp(popt[1] * x) + popt[2])
        sse = np.sum(res ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        n, dfe = len(y), len(y) - 3
        r2 = 1 - sse / sst
        vals = [popt[0], popt[1], popt[2], r2, 1 - (1 - r2) * (n - 1) / dfe, np.sqrt(sse / dfe)]
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        vals = [np.nan] * 6
    return dict(zip(keys, vals))

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

        - 'mean': local mean prediction, with window lengths 1, 2, ..., 10
        - 'median': local median prediction, with window lengths 1, 3, ..., 19

        Default is ``'mean'``.
        
    Returns
    -------
    dict
        Dictionary containing statistics about how forecasting performance varies
        with window length: for each of the residual standard deviation
        (``stde``), ``sws``, ``swm``, ``ac1`` and ``ac2``, the normalized mean
        change (``_chn``), the mean sign of the changes (``_meansgndiff``) and, for
        the last four, ``_stdn``; ``sws_fexp_a``, ``_b``, ``_c``, ``_r2``,
        ``_adjr2`` and ``_rmse`` (an exponential fit a*exp(b*l) + c to the
        ``sws`` curve; NaN if the fit fails); ``stde_peakpos`` (1-based position in the list
        of window lengths of the extreme value of the ``stde`` curve) and
        ``stde_peaksize``.
    """
    y = np.asarray(y)
    if forecast_meth == 'mean':
        train_length_range = np.arange(1, 11)
    elif forecast_meth == 'median':
        train_length_range = np.arange(1, 20, 2)  # 1:2:19, as in hctsa
    else:
        raise ValueError(f"Unknown prediction method: {forecast_meth}")
    stats_st = np.zeros((len(train_length_range), 5))
    for i in range(len(train_length_range)):
        outtmp = local_simple(y, forecast_meth, train_length_range[i])
        stats_st[i, 0] = outtmp['stde']
        stats_st[i, 1] = outtmp['sws']
        stats_st[i, 2] = outtmp['swm']
        stats_st[i, 3] = outtmp['ac1']
        stats_st[i, 4] = outtmp['ac2']
    # Compute statistics from the shapes of the curves
    # (1) root mean square error
    out = {}
    std_err_chnn = np.mean(np.diff(stats_st[:, 0]))/(np.ptp(stats_st[:, 0]))
    out['stde_chn'] = std_err_chnn
    out['stde_meansgndiff'] = np.mean(np.sign(np.diff(stats_st[:, 0])))
    # (ii) Is there a peak?
    if std_err_chnn < 0: # on the whole decreasing, as expected: look for a maximum
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
        out['stde_peakpos'] = wig + 1  # 1-based position, as MATLAB find
        out['stde_peaksize'] = wigv / np.mean(stats_st[:, 0])
    else:  # put NaNs in all the outputs
        out['stde_peakpos'] = np.nan
        out['stde_peaksize'] = np.nan

    #% (2)-(5) Curve statistics for the remaining metrics:
    #%   sws (sliding window stationarity), swm (sliding window mean), ac1, ac2
    for name, col in (('sws', 1), ('swm', 2), ('ac1', 3), ('ac2', 4)):
        curve = stats_st[:, col]
        out[f'{name}_chn'] = np.mean(np.diff(curve)) / np.ptp(curve)
        out[f'{name}_meansgndiff'] = np.mean(np.sign(np.diff(curve)))
        out[f'{name}_stdn'] = np.std(curve, ddof=1) / np.ptp(curve)
        if name == 'sws':
            # exponential fit f(l) = a exp(b l) + c to the sws curve
            out.update(_fit_exp_curve(train_length_range.astype(float), curve, 'sws_fexp'))

    return out

def _gauss1_r2(x: np.ndarray) -> float:
    """
    R-squared of a Gaussian fit to the kernel-density estimate of x.

    Equivalent to hctsa's ``DN_SimpleFit(x, 'gauss1', 0).r2``: the curve
    a * exp(-((t - b) / c) ** 2) is fitted by nonlinear least squares to MATLAB's
    default ``ksdensity`` estimate of x (100 points). NaN if the fit fails.
    """
    try:
        dny, dnx = _ksdensity(np.asarray(x, dtype=float))
        gauss1 = lambda t, a, b, c: a * np.exp(-((t - b) / c) ** 2)
        i0 = int(np.argmax(dny))
        popt, _ = curve_fit(gauss1, dnx, dny,
                            p0=[dny[i0], dnx[i0], (dnx[-1] - dnx[0]) / 4], maxfev=10000)
        sse = np.sum((dny - gauss1(dnx, *popt)) ** 2)
        return float(1 - sse / np.sum((dny - np.mean(dny)) ** 2))
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.nan

def local_simple(y: ArrayLike, forecast_meth: str = 'mean',
                 train_length: Union[int, str] = 3) -> dict:
    """
    Simple local time-series forecasting.
    
    Simple predictors using the past ``train_length`` values of the time series to
    predict its next value. The residuals (prediction minus data) are summarized with
    the ``'core'`` level of :func:`residual_analysis`, plus the Gaussianity of their
    distribution. The first ``train_length`` values are used for training only and are
    not forecast.

    Parameters
    ----------
    y : array-like
        The input time series.
    forecast_meth : str, optional
        The forecasting method:

        - 'mean': local mean prediction using the past ``train_length`` time-series values
        - 'median': local median prediction using the past ``train_length`` time-series values
        - 'lfit': local linear prediction using the past ``train_length`` time-series values

        Default is ``'mean'``.

    train_length : int or str, optional
        The number of time-series values to use to forecast the next value, or a
        string that sets it from the series: 'ac' (the first zero crossing of the
        autocorrelation function of ``y``, discrete), 'ac1e' (the floor of its first
        1/e crossing) or 'mi' (the smaller of the first minimum of the Kraskov
        automutual information and the 'ac1e' delay), as in
        :func:`pyhctsa.utils.get_tau`. For 'lfit' with 'ac', 'ac1e' or 'mi', the
        window is at least 2 (a straight line needs two points).
        Default is 3.

    Returns
    -------
    dict
        The 11 ``'core'`` residual statistics (``meane``, ``meanabs``, ``stde``,
        ``maxonstd``, ``ac1``, ``ac2``, ``ac3``, ``propbth``, ``taurat``, ``sws``,
        ``swm``) and ``normr2``, the R-squared of a Gaussian fit to the
        kernel-smoothed distribution of the residuals. NaN if the series is too short
        to forecast or the window cannot be set.

    """
    y = np.asarray(y)
    N = len(y)
    # % Do the local prediction
    if isinstance(train_length, str) and train_length in ('ac1e', 'mi'):
        # adaptive window (hctsa BF_GetTau): NaN if it cannot be set
        train_length = get_tau(y, train_length)
        if np.isnan(train_length):
            logger.warning("Could not set the training length from the series")
            return np.nan
        if forecast_meth == 'lfit':
            train_length = max(train_length, 2)  # a straight line needs at least two points
    if isinstance(train_length, str) and train_length == 'ac':
        lp = first_crossing(y, 'ac', 0, 'discrete')
        if np.isnan(lp):
            logger.warning("Could not set the training length from the autocorrelation function")
            return np.nan
        if forecast_meth == 'lfit':
            lp = max(lp, 2)  # a straight line needs at least two points
    else:
        # the length of the subsegment preceding to use to predict the subsequent value
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

    # Output statistics on the residuals, res, through the shared contract ('core' level)
    out = residual_analysis(res, y, 'core')
    #% Normality of residuals: r-squared of a Gaussian fit to their kernel-density estimate
    out['normr2'] = _gauss1_r2(res)

    return out

def exp_smoothing(x: ArrayLike, n_train: Union[None, int, float] = None,
                  alpha: Union[str, float] = 'best') -> dict:
    """
    Exponential smoothing time-series prediction model.

    Fits an exponential smoothing model to the time series using a training set to
    fit the optimal smoothing parameter, alpha, and then applies the result to
    predict the rest of the time series. The residual statistics are computed on
    the held-out samples only (those after the first ``n_train``), and ``nan`` is
    returned if fewer than 50 samples remain after the training set.

    The residuals (prediction minus data) are summarized with the ``'full'`` level of
    :func:`residual_analysis`.

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
        Dictionary including the fitted alpha (``alphamin``, with the quadratic-fit
        outputs ``alphamin_1``, ``p1_1`` and ``cup_1``) and the 15 ``'full'``
        statistics on the residuals from the prediction phase: ``meane``, ``meanabs``,
        ``stde``, ``maxonstd``, ``ac1``, ``ac2``, ``ac3``, ``propbth``, ``taurat``,
        ``sws``, ``swm``, ``ftbth``, ``normksstat``, ``popt`` and ``minsbc``.
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
        
    if N < n_train + 50:  # too few samples held out after the training set
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
                # (hctsa compares the fitted curve at alpha = 0 and alpha = 1)
                alphamin = 0.01 if np.polyval(p, 0.0) < np.polyval(p, 1.0) else 1.0
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
    # residuals on the held-out part only (after the n_train samples used to fit alpha)
    yp, xp = y_fit[n_train:], x[n_train:]
    
    if len(yp) < 2:
        logger.warning("Not enough points to calculate residual statistics.")
        residout = {}
    else:
        residuals = yp - xp  # prediction minus data
        residout = residual_analysis(residuals, xp, 'full')
    
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

def _zscore(x: np.ndarray) -> np.ndarray:
    # MATLAB's zscore: no guard against (near-)constant input, which gives NaN
    # for exactly constant data (the guarded utils.z_score raises instead)
    with np.errstate(all='ignore'):
        return (x - np.mean(x)) / np.std(x, ddof=1)

def residual_analysis(e: ArrayLike, y: Union[ArrayLike, None] = None,
                      level: str = 'full') -> dict:
    """
    Canonical summary of the residuals from a model fit.

    The shared residual-summary contract of the model-fitting and forecasting
    operations (hctsa's ``MF_ResidualAnalysis``): every operation that produces a
    residual series reports it through this function, so the same quantity carries
    the same name everywhere. Two levels are available: ``'core'`` is cheap (no test
    and no model fit), ``'full'`` adds diagnostics that need extra machinery.

    Parameters
    ----------
    e : array-like
        The residuals, as prediction minus data (``e = yp - y``).
    y : array-like, optional
        The original time series the model was fitted to. It is only used for
        ``taurat``, which compares the residual timescale to the data timescale;
        ``taurat`` is NaN if ``y`` is not supplied. Default is ``None``.
    level : {'full', 'core'}, optional
        The summary level. Default is ``'full'``.

    Returns
    -------
    dict
        ``'core'`` (11 fields):

        - ``meane``: mean residual
        - ``meanabs``: mean absolute residual
        - ``stde``: standard deviation of the residuals
        - ``maxonstd``: largest absolute residual, in units of the residual standard
          deviation (0 if the residuals are constant)
        - ``ac1``, ``ac2``, ``ac3``: residual autocorrelation at lags 1 to 3
        - ``propbth``: proportion of the first 25 autocorrelations within the
          significance band, ``2.6/sqrt(N)``
        - ``taurat``: residual decorrelation time (first zero crossing of the ACF)
          divided by that of the data (NaN without ``y``, or if the data timescale
          is 0 or undefined)
        - ``sws``, ``swm``: stationarity of the residual standard deviation and mean
          across 5 windows

        ``'full'`` adds 4 fields:

        - ``ftbth``: first lag at which the autocorrelation drops below significance
          (26 if it never does)
        - ``normksstat``: Kolmogorov-Smirnov statistic against a standard normal
        - ``popt``: order of the SBC-selected zero-mean AR model (orders 1 to 10, fitted
          with the ARFIT algorithm) of the z-scored residuals (NaN if the fit fails)
        - ``minsbc``: the corresponding Schwarz criterion (NaN if the fit fails)

    Notes
    -----
    Callers: ``local_simple`` (``'core'``), ``ar_cov`` (``'core'``), ``ar_fit``
    (``'core'``, on the negated ARFIT residuals), ``exp_smoothing`` (``'full'``) and
    ``nonlinearity.nlpe`` (``'full'``, hctsa's ``NL_nlpe`` passes the series as ``y``:
    ``residual_analysis(res, y, 'full')``; the one-argument call still works but
    leaves ``taurat`` NaN).
    """
    if level not in ('core', 'full'):
        raise ValueError(f"Unknown summary level '{level}' (expected 'core' or 'full')")
    e = np.asarray(e, dtype=float).ravel()
    N = len(e)
    if np.all(e > 0):
        logger.warning('Very weird that ALL model residuals are positive...')
    elif np.all(e < 0):
        logger.warning('Very weird that ALL model residuals are negative...')

    # Location, scale and shape
    out = {}
    out['meane'] = np.mean(e)
    out['meanabs'] = np.mean(np.abs(e))
    std_e = np.std(e, ddof=1)
    out['stde'] = std_e
    out['maxonstd'] = 0.0 if std_e == 0 else np.max(np.abs(e)) / std_e

    # z-score the residuals for everything that follows (all of it is scale-free)
    e_z = np.zeros(N) if std_e == 0 else _zscore(e)

    # Serial correlation
    max_lag = 25
    acf = np.asarray(autocorr(e_z, list(range(1, max_lag + 1)), 'Fourier'))
    sqrt_n = np.sqrt(N)
    out['ac1'] = acf[0]
    out['ac2'] = acf[1]
    out['ac3'] = acf[2]
    # proportion of the autocorrelation function within the significance band
    out['propbth'] = np.sum(np.abs(acf) < 2.6 / sqrt_n) / max_lag

    # Residual decorrelation time relative to that of the data
    if y is None:
        out['taurat'] = np.nan
    else:
        y = np.asarray(y, dtype=float).ravel()
        tau_y = first_crossing(_zscore(y), 'ac', 0, 'continuous')
        tau_e = first_crossing(e_z, 'ac', 0, 'continuous')
        if tau_y == 0 or not np.isfinite(tau_y):
            out['taurat'] = np.nan
        else:
            out['taurat'] = tau_e / tau_y

    # Stationarity of the residuals (on the raw, not z-scored, residuals)
    out['sws'] = sliding_window(e, 'std', 'std', 5, 1)
    out['swm'] = sliding_window(e, 'mean', 'std', 5, 1)

    if level == 'core':
        return out

    # (full only) Whiteness, normality and an AR fit to the residuals
    below = np.where(np.abs(acf) < 2.6 / sqrt_n)[0]
    out['ftbth'] = below[0] + 1 if below.size > 0 else max_lag + 1
    out['normksstat'] = ks_1samp(e_z, norm.cdf).statistic

    # does an AR model still find structure in the residuals?
    try:
        _, A_est, _, sbc, _, _ = _arfit(e_z, 1, 10, 'sbc', zero=True)
        out['popt'] = len(A_est)
        out['minsbc'] = np.min(sbc)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as err:
        logger.warning(f'Error fitting AR model to residuals using the ARFIT algorithm: {err}')
        out['popt'] = np.nan
        out['minsbc'] = np.nan
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
        Dictionary containing the variance estimate of the white noise input to
        the AR model (``noisevar``), the parameters of the fitted model
        (``a1``, ..., ``a{p+1}``, with ``a1 = 1``), and the 11 ``'core'`` statistics of
        the residuals of the reconstructed time series (see :func:`residual_analysis`).
        The residuals are prediction minus data.
    """
    y = np.asarray(y)
    model = AutoReg(y, lags=p, trend='n')
    results = model.fit()
    phi = results.params
    a = np.concatenate(([1], -phi))
    out = {}
    out['noisevar'] = results.sigma2
    for i in range(len(a)):
        out[f'a{i+1}'] = a[i]
    # Residual analysis
    b_coeffs = np.concatenate(([0], -a[1:]))
    # Predict y from its past values
    y_est = lfilter(b_coeffs, [1], y)
    err = y_est - y  # prediction minus data (the residual_analysis convention)
    out.update(residual_analysis(err, y, 'core'))

    return out

def _arfit(v: ArrayLike, pmin: int, pmax: int, selector: str = 'sbc',
           zero: bool = True) -> tuple:
    """
    Stepwise least-squares AR model for a univariate series (ARFIT algorithm).

    Port of ``ARFIT_arfit`` [1]_ for a single variable and a single realization. All
    orders ``pmin`` to ``pmax`` are compared on the same ``N - pmax`` equations (via one
    QR factorization), and the order is chosen by Schwarz's criterion (``'sbc'``) or
    the log of Akaike's final prediction error (``'fpe'``).

    References
    ----------
    .. [1] T. Schneider and A. Neumaier, "Algorithm 808: ARFIT---a Matlab package for
        the estimation of parameters and eigenmodes of multivariate autoregressive
        models", ACM Trans. Math. Softw. 27, 58 (2001)

    Parameters
    ----------
    v : array-like
        The time series.
    pmin, pmax : int
        The range of orders.
    selector : {'sbc', 'fpe'}
        The order-selection criterion.
    zero : bool
        If True fit a zero-mean model, ``v[k] = A1 v[k-1] + ... + Ap v[k-p] + noise``;
        otherwise also fit an intercept.

    Returns
    -------
    w, A, C, sbc, fpe, th
        The intercept (0 if ``zero``), the coefficients ``A1, ..., Ap`` at the selected
        order, the noise variance, the criteria for orders ``pmin`` to ``pmax``, and
        ``th = (dof, Uinv)`` for the confidence intervals and eigenmodes.

    Raises
    ------
    ValueError
        If the series is too short or the fit is degenerate (e.g. a constant series).
    """
    v = np.asarray(v, dtype=float).ravel()
    n = len(v)
    pmin, pmax = int(pmin), int(pmax)
    if pmin != pmax and pmax < pmin:
        raise ValueError('PMAX must be greater than or equal to PMIN.')
    if selector not in ('sbc', 'fpe'):
        raise ValueError(f"Unknown order selector '{selector}'.")
    mcor = 0 if zero else 1
    ne = n - pmax                # number of equations
    npmax = pmax + mcor          # maximum number of parameters
    if ne <= npmax:
        raise ValueError(f'Time series (N = {n}) too short.')

    # ARFIT_arqr: QR factorization of the data matrix for order pmax
    K = np.zeros((ne, npmax + 1))
    if mcor:
        K[:, 0] = 1.0
    for j in range(1, pmax + 1):
        K[:, mcor + j - 1] = v[pmax - j:n - j]
    K[:, npmax] = v[pmax:]
    q = npmax + 1
    delta = (q ** 2 + q + 1) * np.finfo(float).eps  # Higham's choice for a Cholesky factorization
    scale = np.sqrt(delta) * np.sqrt(np.sum(K ** 2, axis=0))
    R = np.triu(np.linalg.qr(np.vstack([K, np.diag(scale)]), mode='r'))

    # ARFIT_arord: order selection criteria for orders pmin:pmax
    imax = pmax - pmin + 1
    sbc = np.zeros(imax)
    fpe = np.zeros(imax)
    logdp = np.zeros(imax)
    with np.errstate(all='ignore'):
        R22 = R[npmax, npmax]
        Mp = (1.0 / R22) ** 2
        logdp[imax - 1] = 2.0 * np.log(abs(R22))
        i = imax - 1
        for p in range(pmax, pmin - 1, -1):
            np_i = p + mcor
            if p < pmax:
                Rp = R[np_i, npmax]
                L = np.sqrt(1.0 + Rp * Mp * Rp)
                Nn = Rp * Mp / L
                Mp = Mp - Nn * Nn
                logdp[i] = logdp[i + 1] + 2.0 * np.log(abs(L))
            sbc[i] = logdp[i] - np.log(ne) * (ne - np_i) / ne
            fpe[i] = logdp[i] - np.log(ne * (ne - np_i) / (ne + np_i))
            i -= 1
    if not (np.all(np.isfinite(sbc)) and np.all(np.isfinite(fpe))):
        raise ValueError('Degenerate AR fit (non-finite order-selection criteria).')

    # order of the model
    iopt = int(np.argmin(sbc if selector == 'sbc' else fpe))
    popt = pmin + iopt
    np_opt = popt + mcor
    R11 = R[:np_opt, :np_opt].copy()
    R12 = R[:np_opt, npmax]
    R22v = R[np_opt:npmax + 1, npmax]
    con = 1.0
    if mcor:
        con = np.max(scale[1:npmax + 1]) / scale[0]  # improve condition of R11
        R11[:, 0] *= con
    Aaug = np.linalg.solve(R11, R12)
    if mcor:
        w = Aaug[0] * con
        A = Aaug[1:]
    else:
        w = 0.0
        A = Aaug
    dof = ne - np_opt
    C = float(R22v @ R22v) / dof
    invR11 = np.linalg.inv(R11)
    if mcor:
        invR11[0, :] *= con
    Uinv = invR11 @ invR11.T
    return w, A, C, sbc, fpe, (dof, Uinv)

def _arfit_residuals(w: float, A: np.ndarray, v: np.ndarray, k: Union[int, None] = None) -> tuple:
    """
    Residuals of a fitted AR model and the Li-McLeod portmanteau test (ARFIT_arres).

    Returns ``(siglev, res)``: the significance level of the test for whiteness of the
    residuals (autocorrelations up to lag ``k``, default ``min(20, N - p - 1)``) and the
    residuals, *data minus fit*.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = len(v)
    p = len(A)
    nres = n - p
    if k is None:
        k = min(20, nres - 1)
    if k <= p:
        raise ValueError('Maximum lag of residual correlation matrices too small.')
    if k >= nres:
        raise ValueError('Maximum lag of residual correlation matrices too large.')
    res = v[p:] - w
    for j in range(1, p + 1):
        res = res - A[j - 1] * v[p - j:n - j]
    resc = res - np.mean(res)
    c0 = np.sum(resc ** 2)
    cl = np.array([np.sum(resc[:nres - l] * resc[l:]) / c0 for l in range(1, k + 1)])
    lmp = nres * np.sum(cl ** 2) + k * (k + 1) / 2 / nres
    dof_lmp = k - p
    return float(gammaincc(dof_lmp / 2, lmp / 2)), res

def _arfit_modes(A: np.ndarray, C: float, th: tuple, conf: float = 0.95) -> tuple:
    """
    Eigenmodes of a univariate AR model (the parts of ARFIT_armode that MF_arfit uses).

    Returns ``(per, tau, exctn, lam)``: the period and its confidence interval
    (``per`` has shape ``(2, p)``), the damping time and its confidence interval
    (``tau``, ``(2, p)``), the relative excitations and the eigenvalues of the
    companion matrix.
    """
    p = len(A)
    dof, Uinv = th
    t = _t_quantile(dof, 0.5 + conf / 2)
    A1 = np.zeros((p, p))
    A1[0, :] = A
    if p > 1:
        A1[1:, :-1] = np.eye(p - 1)
    lam, BigS = np.linalg.eig(A1)
    # the eigenvectors are only used in products invariant to their phase and norm
    BigS_inv = np.linalg.inv(BigS)
    Sigma_A = Uinv * C
    cov_dcpld = BigS_inv[:, 0] * C * np.conj(BigS_inv[:, 0])
    per = np.zeros((2, p))
    tau = np.zeros((2, p))
    exctn = np.zeros(p)
    with np.errstate(all='ignore'):
        for j in range(p):
            a, b = lam[j].real, lam[j].imag
            abs_lambda_sq = abs(lam[j]) ** 2
            tau[0, j] = -2.0 / np.log(abs_lambda_sq)
            exctn[j] = (cov_dcpld[j] / (1 - abs_lambda_sq)).real
            dot_lam = BigS_inv[j, 0] * BigS[:, j]
            dot_a, dot_b = dot_lam.real, dot_lam.imag
            phi = tau[0, j] ** 2 / abs_lambda_sq * (a * dot_a + b * dot_b)
            tau[1, j] = t * np.sqrt(phi @ Sigma_A @ phi)
            if b == 0 and a >= 0:    # purely real, nonnegative eigenvalue
                per[0, j] = np.inf
                per[1, j] = 0.0
            elif b == 0 and a < 0:   # purely real, negative eigenvalue
                per[0, j] = 2.0
                per[1, j] = 0.0
            else:                    # complex eigenvalue
                per[0, j] = 2 * np.pi / abs(np.arctan2(b, a))
                phi = per[0, j] ** 2 / (2 * np.pi * abs_lambda_sq) * (b * dot_a - a * dot_b)
                per[1, j] = t * np.sqrt(phi @ Sigma_A @ phi)
        exctn = exctn / np.sum(exctn)
    return per, tau, exctn, lam

def _t_quantile(dof: int, p: float) -> float:
    """Student-t quantile (ARFIT_tquant)."""
    return float(t.ppf(p, df=dof))

def ar_fit(y: ArrayLike, p_min: int = 1, p_max: int = 10, selector: str = 'sbc') -> dict:
    """
    Statistics of a fitted AR model to a time series.

    Fits zero-mean autoregressive (AR) models of orders p = p_min, ..., p_max to the
    input time series with the ARFIT algorithm [1]_ [2]_, selects the optimal order
    using Schwarz's Bayesian Criterion (SBC) or the final prediction error (FPE), and
    returns statistics on the fitted model, its residuals, confidence intervals and
    eigenmodes. As in ARFIT, all orders are fitted to the same ``N - p_max`` equations.

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
    selector : {'sbc', 'fpe'}, optional
        Criterion to select the optimal model order (cf. the ARFIT documentation;
        ``'bic'`` and ``'aic'`` are accepted as aliases). Default is ``'sbc'``.

    Returns
    -------
    dict
        - ``A1`` ... ``A6``: the first six AR coefficients (NaN beyond the selected order)
        - ``maxA``, ``minA``, ``meanA``, ``stdA``, ``sumA``, ``rmsA``: summaries of the
          coefficients
        - ``C``: the noise variance
        - ``sbc_k``, ``minsbc``, ``popt_sbc``, ``aroundmin_sbc``: Schwarz's criterion for
          each order, its minimum, the position (1-based, within ``p_min..p_max``) of
          the minimum, and its size relative to the neighboring values
        - ``fpe_k``, ``minfpe``, ``popt_fpe``, ``aroundmin_fpe``: the same for the
          logarithm of Akaike's final prediction error
        - ``res_siglev``: significance level of the Li-McLeod test for autocorrelation
          in the residuals (up to lag 20)
        - the 11 ``'core'`` statistics of the residuals (see :func:`residual_analysis`)
        - ``aerr_min``, ``aerr_max``, ``aerr_mean``: 95% confidence intervals on the
          coefficients
        - ``maxReLambda``, ``maxImLambda``, ``maxabsLambda``, ``stdabsLambda``: the eigenvalues
          of the AR model's companion matrix
        - ``hasInfper``, ``meanper``, ``stdper``, ``maxper``, ``minper``, ``meanpererr``,
          ``meantau``, ``maxtau``, ``mintau``, ``stdtau``, ``meantauerr``, ``maxexctn``,
          ``minexctn``, ``meanexctn``, ``stdexctn``: periods, damping times (with
          confidence intervals) and excitations of the eigenmodes

        NaN if the series is too short for ARFIT.
    """
    y = np.asarray(y, dtype=float).ravel()
    p_min = int(p_min)
    p_max = int(p_max)
    if selector in ('bic', 'sbc'):  # bic and sbc are the same metrics
        selector = 'sbc'
    elif selector in ('aic', 'fpe'):
        selector = 'fpe'
    else:
        raise ValueError(f"Unknown order selector '{selector}'.")

    # (I) Fit the AR model
    try:
        _, Aest, Cest, sbc, fpe, th = _arfit(y, p_min, p_max, selector, zero=True)
    except ValueError as err:
        logger.warning(f'Could not fit an AR model with the ARFIT algorithm: {err}')
        return np.nan
    ps = np.arange(p_min, p_max + 1)
    popt = len(Aest)

    # (i) Coefficients
    out = {}
    out['A1'] = Aest[0]
    for i in range(2, 7):
        if popt >= i:
            out[f'A{i}'] = Aest[i-1]
        else:
            out[f'A{i}'] = np.nan  # not estimated at the selected order (NaN, not 0)
    # (ii) Summary statistics on the coefficients
    out['maxA'] = np.max(Aest)
    out['minA'] = np.min(Aest)
    out['meanA'] = np.mean(Aest)
    out['stdA'] = np.std(Aest, ddof=1) if len(Aest) > 1 else 0.0
    out['sumA'] = np.sum(Aest)
    out['rmsA'] = np.sqrt(np.sum(Aest ** 2))

    # (iii) Noise covariance matrix: for a univariate series, a scalar noise variance
    out['C'] = Cest

    # (iv) Order-selection criteria: Schwarz's Bayesian Criterion and the log FPE
    for crit, vals in (('sbc', sbc), ('fpe', fpe)):
        for i in range(len(ps)):
            out[f'{crit}_{ps[i]}'] = vals[i]
        out[f'min{crit}'] = np.min(vals)
        pos = int(np.argmin(vals))  # first minimum
        out[f'popt_{crit}'] = pos + 1  # as hctsa: a position in p_min:p_max
        if len(vals) == 1:
            around = np.nan
        elif pos == 0:
            around = abs(vals[1])
        elif pos == len(vals) - 1:
            around = abs(vals[pos - 1])
        else:
            around = np.mean(np.abs([vals[pos - 1], vals[pos + 1]]))
        out[f'aroundmin_{crit}'] = abs(np.min(vals)) / around

    # (II) Test the residuals
    try:
        siglev, res = _arfit_residuals(0.0, Aest, y)
    except ValueError as err:
        logger.warning(f'Could not test the AR residuals: {err}')
        return np.nan
    out['res_siglev'] = siglev
    # ARFIT returns data minus fit; the contract is prediction minus data
    out.update(residual_analysis(-res, y, 'core'))

    # (III) Confidence intervals on the coefficients
    t_crit = _t_quantile(th[0], 0.5 + 0.95 / 2)
    a_err = t_crit * np.sqrt(np.diag(th[1]) * Cest)
    out['aerr_min'] = np.min(a_err)
    out['aerr_max'] = np.max(a_err)
    out['aerr_mean'] = np.mean(a_err)

    # (IV) Eigendecomposition
    per, tau, exctn, lam = _arfit_modes(Aest, Cest, th)
    out['maxReLambda'] = np.max(lam.real)
    out['maxImLambda'] = np.max(lam.imag)
    out['maxabsLambda'] = np.max(np.abs(lam))
    out['stdabsLambda'] = np.std(np.abs(lam), ddof=1) if len(lam) > 1 else 0.0

    per_special = ~np.isfinite(per[0])
    per_filtered = np.where(per_special, np.nan, per[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        out['hasInfper'] = int(np.sum(per_special))
        out['meanper'] = np.nanmean(per_filtered)
        out['stdper'] = np.nanstd(per_filtered, ddof=1) if np.sum(~np.isnan(per_filtered)) > 1 else (
            0.0 if np.sum(~np.isnan(per_filtered)) == 1 else np.nan)
        out['maxper'] = np.nanmax(per_filtered)
        out['minper'] = np.nanmin(per_filtered)
        out['meanpererr'] = np.nanmean(per[1])
    out['meantau'] = np.mean(tau[0])
    out['maxtau'] = np.max(tau[0])
    out['mintau'] = np.min(tau[0])
    out['stdtau'] = np.std(tau[0], ddof=1) if len(tau[0]) > 1 else 0.0
    out['meantauerr'] = np.mean(tau[1])
    out['maxexctn'] = np.max(exctn)
    out['minexctn'] = np.min(exctn)
    out['meanexctn'] = np.mean(exctn)
    out['stdexctn'] = np.std(exctn, ddof=1) if len(exctn) > 1 else 0.0

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
    # Initial values, set component by component as in MF_GP_LearnHyperp for
    # covSum{covSEiso, covNoise}: the SE length scale is in the ballpark of the
    # difference between time elements, its log-magnitude starts at zero, the noise
    # covariance at log(0.1), and so does the likelihood noise.
    hyp0 = np.array([np.log(np.mean(np.diff(tt))), 0.0, np.log(0.1), np.log(0.1)])
    assert nhps == 3

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
    out['rmserr'] = np.sqrt(np.mean((y_ts - mu) ** 2))
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
        Seed for the Mersenne Twister, reset once before the loop over windows
        (as ``BF_ResetSeed`` does), used by the ``'randomgap'`` mode. ``None``
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
    if pmode == 'randomgap' and random_seed is not None:
        # reset the seed once, before the loop over windows: successive windows
        # then draw different random splits (a reproducible sequence)
        rng = _ml_rng(random_seed)

    for i in range(num_preds):
        # (0) Set up test and training sets
        sp = spns[i]
        if pmode == 'frombefore':
            tt = np.arange(1, num_train + 1, dtype=float)          # times (from 1)
            yt = y[sp - 1:sp - 1 + num_train]                      # training data
            ts = np.arange(num_train + 1, num_train + num_test + 1, dtype=float)
            ys = y[sp - 1 + num_train:sp - 1 + num_train + num_test]  # test data

        elif pmode == 'randomgap':
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
