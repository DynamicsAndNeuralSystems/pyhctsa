import warnings
from typing import Union

import numba
import numpy as np
from numpy.typing import ArrayLike
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from scipy.signal import lfilter
from scipy.special import gammaincc
from scipy.stats import ks_1samp, norm, t
from lmfit.models import SineModel
import logging
logger = logging.getLogger('pyhctsa')

from ..operations.correlation import autocorr, first_crossing
from ..operations.physics import _ksdensity
from ..operations.stationarity import sliding_window
from ..toolboxes.matlab.gpml.gpml import CovSEisoNoise, gp_predict, gp_train
from ..robust import bf_exp_fit
from ..toolboxes.matlab.optimizers import minimize
from ..utils import _linspace, _ml_randperm, _ml_rng, _zscore_matlab, get_tau, matlab_quantile, z_score

@numba.njit(cache=True, error_model='numpy')
def _zg_hmm_em(x, mu, cov, P, pi, n_cycles, tol, cov_floor):
    """
    Baum-Welch EM for a Gaussian-emission HMM with a variance shared by all states, as
    Zoubin Ghahramani's ``ZG_hmm`` (hctsa's ``ZG_hmm``, with its covariance floor and
    log-domain emission scaling). Returns the fitted parameters and the log-likelihood
    at the start of each cycle (before that cycle's M step).
    """
    T = len(x)
    K = len(mu)
    mu = mu.copy()
    P = P.copy()
    pi = pi.copy()
    LL = np.zeros(n_cycles)
    n_done = 0
    lik = 0.0
    likbase = 0.0
    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))
    B = np.zeros((T, K))
    gamma = np.zeros((T, K))
    scale = np.zeros(T)
    shift = np.zeros(T)
    for cycle in range(1, n_cycles + 1):
        # --- E step (forward-backward with scaling)
        logk2 = np.log((2 * np.pi) ** (-0.5)) - 0.5 * np.log(cov)
        for t in range(T):
            m = -np.inf
            for l in range(K):
                d = x[t] - mu[l]
                lb = logk2 - 0.5 * d * d / cov
                B[t, l] = lb
                if lb > m:
                    m = lb
            shift[t] = m
            for l in range(K):
                B[t, l] = np.exp(B[t, l] - m)
        s = 0.0
        for l in range(K):
            alpha[0, l] = pi[l] * B[0, l]
            s += alpha[0, l]
        scale[0] = s
        for l in range(K):
            alpha[0, l] /= s
        for t in range(1, T):
            s = 0.0
            for l in range(K):
                a = 0.0
                for j in range(K):
                    a += alpha[t - 1, j] * P[j, l]
                alpha[t, l] = a * B[t, l]
                s += alpha[t, l]
            scale[t] = s
            for l in range(K):
                alpha[t, l] /= s
        for l in range(K):
            beta[T - 1, l] = 1.0 / scale[T - 1]
        for t in range(T - 2, -1, -1):
            for j in range(K):
                a = 0.0
                for l in range(K):
                    a += beta[t + 1, l] * B[t + 1, l] * P[j, l]
                beta[t, j] = a / scale[t]
        for t in range(T):
            s = 0.0
            for l in range(K):
                gamma[t, l] = alpha[t, l] * beta[t, l]
                s += gamma[t, l]
            for l in range(K):
                gamma[t, l] /= s
        sxi = np.zeros((K, K))
        for t in range(T - 1):
            s = 0.0
            for j in range(K):
                for l in range(K):
                    s += P[j, l] * alpha[t, j] * beta[t + 1, l] * B[t + 1, l]
            for j in range(K):
                for l in range(K):
                    sxi[j, l] += P[j, l] * alpha[t, j] * beta[t + 1, l] * B[t + 1, l] / s
        loglik = 0.0
        for t in range(T):
            loglik += np.log(scale[t]) + shift[t]
        # --- M step
        gsum = np.zeros(K)
        for l in range(K):
            num = 0.0
            for t in range(T):
                gsum[l] += gamma[t, l]
                num += gamma[t, l] * x[t]
            mu[l] = num / gsum[l]
        for j in range(K):
            rs = 0.0
            for l in range(K):
                rs += sxi[j, l]
            for l in range(K):
                P[j, l] = sxi[j, l] / rs
        for l in range(K):
            pi[l] = gamma[0, l]
        c = 0.0
        for l in range(K):
            for t in range(T):
                d = x[t] - mu[l]
                c += gamma[t, l] * d * d
        gtot = 0.0
        for l in range(K):
            gtot += gsum[l]
        cov = max(c / gtot, cov_floor)
        # --- convergence
        oldlik = lik
        lik = loglik
        LL[cycle - 1] = lik
        n_done = cycle
        if cycle <= 2:
            likbase = lik
        elif lik < oldlik:
            pass  # a decrease (numerical violation): keep going, as ZG_hmm does
        elif (lik - likbase) < (1 + tol) * (oldlik - likbase) or not np.isfinite(lik):
            break
    return mu, cov, P, pi, LL[:n_done]


def _zg_hmm_fit(y_train: np.ndarray, num_states: int, n_cycles: int = 30,
                rhos: tuple = (0.9, 0.5, 0.99), floor_frac: float = 0.01,
                tol: float = 1e-4) -> tuple:
    """
    Deterministic Gaussian HMM fit (hctsa's ``ZG_hmm_fit``, used by ``MF_hmm_Fit`` and
    ``MF_hmm_CompareNStates``; shared by :func:`hmm_fit` and :func:`hmm_compare_n_states`).

    Baum-Welch EM (Zoubin Ghahramani's ``ZG_hmm``) is run from six fixed starting points and
    the fit with the highest final training log-likelihood is kept. The starts have a shared
    variance equal to the variance of the data, equal initial-state probabilities, and a
    transition matrix with probability ``rho`` of staying in a state (and ``(1-rho)/(K-1)`` of
    moving to each other one) for each ``rho`` in ``rhos``, and two placements of the K state
    means: at the ``(k-1/2)/K`` quantiles of the data (sorted, element ``ceil(N(k-1/2)/K)``)
    and evenly spaced from ``mean - std`` to ``mean + std``. The shared variance is not
    allowed to fall below ``floor_frac`` times the data variance. At most ``n_cycles`` cycles,
    stopping when the proportional change in log-likelihood falls below ``tol``.

    Returns ``(mu, cov, P, pi, LL)``: state means, shared variance, transition matrix,
    initial-state probabilities and the log-likelihood at each cycle (NaN-filled parameters
    and ``LL = [nan]`` if no start gives a finite fit).
    """
    x = np.ascontiguousarray(y_train, dtype=float).ravel()
    K = int(num_states)
    N = len(x)
    v = np.var(x, ddof=1)
    xs = np.sort(x)
    idx = np.ceil(N * (np.arange(1, K + 1) - 0.5) / K).astype(int) - 1
    mean_sets = [xs[idx], np.mean(x) + np.std(x, ddof=1) * _linspace(-1, 1, K)]
    pi0 = np.ones(K) / K
    best_ll = -np.inf
    best = (np.full(K, np.nan), np.nan, np.full((K, K), np.nan), np.full(K, np.nan), np.array([np.nan]))
    for r in range(2 * len(rhos)):
        mu0 = mean_sets[0 if r < len(rhos) else 1]
        rho = rhos[r % len(rhos)]
        if K > 1:
            P0 = (1 - rho) / (K - 1) * np.ones((K, K)) + (rho - (1 - rho) / (K - 1)) * np.eye(K)
        else:
            P0 = np.ones((1, 1))
        mu, cov, P, pi, LL = _zg_hmm_em(x, mu0, v, P0, pi0, n_cycles, tol, floor_frac * v)
        ok = np.all(np.isfinite(mu)) and np.isfinite(cov) and np.all(np.isfinite(P)) \
            and np.all(np.isfinite(pi)) and np.isfinite(LL[-1])
        if ok and LL[-1] > best_ll:
            best_ll = LL[-1]
            best = (mu, cov, P, pi, LL)
    return best


@numba.njit(cache=True, error_model='numpy')
def _zg_hmm_loglik(x, mu, cov, P, pi):
    """Log-likelihood of ``x`` under a fitted Gaussian HMM (hctsa's ``ZG_hmm_cl``)."""
    T = len(x)
    K = len(mu)
    tiny = np.exp(-700.0)
    logk2 = np.log((2 * np.pi) ** (-0.5)) - 0.5 * np.log(cov)
    alpha = np.zeros(K)
    new = np.zeros(K)
    B = np.zeros(K)
    lik = 0.0
    for t in range(T):
        m = -np.inf
        for l in range(K):
            d = x[t] - mu[l]
            B[l] = logk2 - 0.5 * d * d / cov
            if B[l] > m:
                m = B[l]
        for l in range(K):
            B[l] = np.exp(B[l] - m)
        s = 0.0
        for l in range(K):
            if t == 0:
                new[l] = pi[l] * B[l]
            else:
                a = 0.0
                for j in range(K):
                    a += alpha[j] * P[j, l]
                new[l] = a * B[l]
            s += new[l]
        for l in range(K):
            alpha[l] = new[l] / (s + tiny)
        if s == 0:
            s = tiny
        lik += np.log(s) + m
    return lik


def hmm_fit(y: ArrayLike, train_p: float = 0.8, num_states: int = 3) -> dict:
    """
    A hidden Markov model fitted to the first part of the series, and how well it describes the rest.

    Fits a hidden Markov model (HMM) with Gaussian emissions to the first ``train_p``
    proportion of the time series (hctsa's ``MF_hmm_Fit``, using Zoubin Ghahramani's ``ZG_hmm``
    EM). The emissions of all states share one variance (a tied covariance). The model is
    trained with at most 30 cycles of EM (Baum-Welch), or until convergence.

    The fit is deterministic. EM is run from six fixed starting points (state means at the
    quantiles ``(k-1/2)/num_states`` of the training data, or evenly spaced within one standard
    deviation of its mean; a variance equal to that of the training data; and probabilities
    0.5, 0.9 and 0.99 of staying in a state); the fit with the highest training log-likelihood is
    kept (see :func:`_zg_hmm_fit`). The shared variance cannot fall below 1% of the variance of
    the training data, so that states placed on a few repeated values do not give unbounded
    likelihoods.

    Parameters
    ----------
    y : array-like
        The input time series.
    train_p : float
        The proportion of data to train on, 0 < train_p < 1. Default is 0.8.
    num_states : int
        The number of states in the HMM. Default is 3.

    Returns
    -------
    dict
        Dictionary of statistics based on the fitted HMM: the sorted state means
        (``Mu_1``, ...) and their ``meanMu``, ``rangeMu``, ``maxMu``, ``minMu``;
        the tied covariance ``Cov``; the transition matrix summaries
        ``Pmeandiag``, ``stdmeanP`` (standard deviation across states of the mean probability
        of moving into each state), ``maxP`` and ``stdP``; the training
        log-likelihood per sample ``LLtrainpersample`` (the highest reached);
        and the test log-likelihood per sample ``LLtestpersample`` and
        ``LLdifference`` (test minus training).

    """
    y = np.asarray(y, dtype=float).ravel()
    n_samples = len(y)
    out = {}

    # 1. Split data into training and test sets
    n_train = int(np.floor(train_p * n_samples))
    n_test = n_samples - n_train
    if n_train <= 0 or n_train > n_samples:
        raise ValueError("Invalid training proportion 'train_p' results in an invalid training set size.")
    if n_test == 0:
        raise ValueError('No data for test set for HMM fitting')

    y_train = y[:n_train]
    y_test = y[n_train:]
    num_states = int(num_states)

    # 2. Train the HMM (deterministic EM from fixed starts, see _zg_hmm_fit)
    mu, cov, p_matrix, pi, LL = _zg_hmm_fit(y_train, num_states)

    means_sorted = np.sort(mu)
    for i, m in enumerate(means_sorted):
        out[f'Mu_{i+1}'] = m
    out['meanMu'] = np.mean(means_sorted)
    out['rangeMu'] = np.ptp(means_sorted)
    out['maxMu'] = np.max(means_sorted)
    out['minMu'] = np.min(means_sorted)

    # Covariance Cov
    out['Cov'] = cov

    # Transition matrix
    out['Pmeandiag'] = np.mean(np.diag(p_matrix))
    out['stdmeanP'] = np.std(np.mean(p_matrix, axis=0), ddof=1)
    out['maxP'] = np.max(p_matrix)
    out['stdP'] = np.std(p_matrix, ddof=1)

    # Within-sample log-likelihood
    out['LLtrainpersample'] = np.max(LL) / n_train

    # Log-likelihood of the test data
    out['LLtestpersample'] = _zg_hmm_loglik(y_test, mu, cov, p_matrix, pi) / n_test
    out['LLdifference'] = out['LLtestpersample'] - out['LLtrainpersample']

    return out


def _armax_fit(y: np.ndarray, p: int, q: int) -> tuple:
    """
    Prediction-error (least-squares) fit of an ARMA(p, q) model in the System Identification
    Toolbox convention ``A(q) y(t) = C(q) e(t)`` with ``A = 1 + a_1 q^-1 + ... + a_p q^-p`` and
    ``C = 1 + c_1 q^-1 + ... + c_q q^-q``, minimizing the sum of squared one-step prediction
    errors with zero initial conditions (as ``armax``).

    Returns ``(a, c, loss, cov)``: the coefficient vectors (including the leading 1), the loss
    function (mean squared prediction error), and the covariance matrix of the parameter
    estimates ``[a_1, ..., a_p, c_1, ..., c_q]``.
    """
    from scipy.optimize import least_squares
    n = len(y)
    n_par = p + q

    def resid(th):
        return lfilter(np.r_[1.0, th[:p]], np.r_[1.0, th[p:]], y)

    if n_par > 0:
        # Two starting points: an AR fit with no MA part, and a Hannan-Rissanen estimate (regress
        # y on its lags and on the residuals of a long AR fit, standing in for the innovations)
        m = max(p, q)
        starts = []
        if p > 0:
            X = np.column_stack([-y[p - i - 1:n - i - 1] for i in range(p)])
            starts.append(np.r_[np.linalg.lstsq(X, y[p:], rcond=None)[0], np.zeros(q)])
        else:
            starts.append(np.zeros(n_par))
        L = min(max(10, 2 * n_par), n // 4)
        X = np.column_stack([y[L - i - 1:n - i - 1] for i in range(L)])
        e_hat = np.r_[np.zeros(L), y[L:] - X @ np.linalg.lstsq(X, y[L:], rcond=None)[0]]
        cols = [-y[m - i - 1:n - i - 1] for i in range(p)] + [e_hat[m - i - 1:n - i - 1] for i in range(q)]
        th_hr = np.linalg.lstsq(np.column_stack(cols), y[m:], rcond=None)[0]
        if q > 0:  # an invertible start: reflect roots of C outside the unit circle
            roots = np.roots(np.r_[1.0, th_hr[p:]])
            roots = np.where(np.abs(roots) > 1, 1 / np.conj(roots), roots)
            th_hr[p:] = np.real(np.poly(roots))[1:]
        starts.append(th_hr)
        sol = None
        for th0 in starts:
            try:
                cand = least_squares(resid, th0, method='lm')
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(cand.cost) and (sol is None or cand.cost < sol.cost):
                sol = cand
        if sol is None:
            raise ValueError('ARMA model could not be fitted')
        th = sol.x
        jac = sol.jac
        loss = np.sum(sol.fun ** 2) / n
        # covariance of the estimates: noise variance times the inverse of J'J
        cov = loss * n / (n - n_par) * np.linalg.pinv(jac.T @ jac)
    else:
        th = np.zeros(0)
        loss = np.mean(y ** 2)
        cov = np.zeros((0, 0))
    return np.r_[1.0, th[:p]], np.r_[1.0, th[p:]], loss, cov


def _armax_residuals(a: np.ndarray, c: np.ndarray, y: np.ndarray, steps: int) -> np.ndarray:
    """
    Prediction errors of ``yp - y`` for a ``steps``-ahead predictor of the ARMA model
    ``a(q) y = c(q) e`` (as ``predict(m, data, steps, 'init', 'e')``): the initial state of the
    predictor is the one that minimizes the squared prediction error.
    """
    # y = H e with H = C/A; the k-step predictor error is e_k = (H_k A / C) y, where H_k is
    # the first k terms of the impulse response of H
    h = lfilter(c, a, np.r_[1.0, np.zeros(steps - 1)])
    b = np.convolve(h, a)
    state_len = max(len(b), len(c)) - 1
    e0 = lfilter(b, c, y)
    if state_len == 0:
        return -e0
    # the output is linear in the initial filter state: estimate it by least squares
    basis = np.zeros((len(y), state_len))
    for j in range(state_len):
        zi = np.zeros(state_len)
        zi[j] = 1.0
        basis[:, j] = lfilter(b, c, y, zi=zi)[0] - e0
    zi_hat = np.linalg.lstsq(basis, -e0, rcond=None)[0]
    return -(e0 + basis @ zi_hat)


def armax(y: ArrayLike, orders: Union[list, tuple] = (3, 3), p_train: float = 0.8,
          num_steps: int = 1) -> dict:
    """
    The coefficients of a fitted ARMA model, and how well it predicts the later part of the series.

    Fits an autoregressive moving-average (ARMA) model with orders ``[p, q]`` to the whole
    time series by minimizing the one-step prediction error (hctsa's ``MF_armax``, which uses
    ``armax`` from MATLAB's System Identification Toolbox). The coefficients, their
    uncertainties, and the goodness of fit are from this fit. The model is then fitted again to
    the first ``p_train`` proportion of the time series and used to predict the remainder
    ``num_steps`` samples ahead; the prediction residuals (prediction minus data) are
    summarized with :func:`residual_analysis`.

    Parameters
    ----------
    y : array-like
        The input time series.
    orders : sequence of two ints, optional
        ``[p, q]``, the AR and MA orders of the model. Default is ``(3, 3)``.
    p_train : float, optional
        The proportion of the data to train the model on (the remainder is used for testing).
        Default is 0.8.
    num_steps : int, optional
        The number of steps ahead to predict when testing the model. Default is 1.

    Returns
    -------
    dict
        From the model fitted to the entire series, in the MATLAB convention
        ``y(t) + a_1 y(t-1) + ... + a_p y(t-p) = e(t) + c_1 e(t-1) + ... + c_q e(t-q)``:

        - ``AR_1``, ..., ``AR_p``: the AR coefficients ``a_1, ..., a_p`` (the negatives of the
          usual AR coefficients)
        - ``MA_1``, ..., ``MA_q``: the MA coefficients ``c_1, ..., c_q``
        - ``maxda``, ``maxdc``: the largest estimated standard deviation of the AR and MA
          coefficients (from the covariance of the parameter estimates)
        - ``noisevar``, ``lossfn``, ``fpe``: the noise variance, the loss function (mean squared
          prediction error), and Akaike's final prediction error of the fit

        From the residuals of the predictions of the held-out portion (see
        :func:`residual_analysis`, ``'full'``): ``meane``, ``meanabs``, ``stde``, ``maxonstd``,
        ``ac1``, ``ac2``, ``ac3``, ``propbth``, ``ftbth``, ``taurat``, ``sws``, ``swm``,
        ``normksstat``, ``popt``, ``minsbc``.

    Notes
    -----
    The model is fitted by Levenberg-Marquardt least squares on the one-step prediction errors,
    from the better of an AR start and a Hannan-Rissanen start, until convergence, with zero
    initial conditions. MATLAB's ``armax`` instead stops at a loose tolerance (so its coefficients
    are partly the starting values) and, with its default initial condition ``'auto'``, sometimes
    estimates the initial conditions by backcasting (for highly predictable series). So the
    coefficients and everything that depends on them agree with MATLAB's only roughly, and
    poorly where an ARMA model is poorly identified (nearly cancelling AR and MA polynomials, as
    for near-white series, where only the loss, ``noisevar`` and ``fpe`` are well-determined).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    p, q = int(orders[0]), int(orders[1])

    # Fit to the whole time series
    a, c, loss, cov = _armax_fit(y, p, q)
    n_par = p + q
    out = {}
    for i in range(1, p + 1):
        out[f'AR_{i}'] = a[i]
    for i in range(1, q + 1):
        out[f'MA_{i}'] = c[i]
    sd = np.sqrt(np.diag(cov))
    out['maxda'] = np.max(sd[:p]) if p > 0 else 0.0
    out['maxdc'] = np.max(sd[p:]) if q > 0 else 0.0
    out['noisevar'] = loss * n / (n - n_par)
    out['lossfn'] = loss
    out['fpe'] = loss * (1 + n_par / n) / (1 - n_par / n)

    # Fit to the training portion, predict the test portion (overlapping by one sample)
    n_cut = int(np.floor(p_train * n))
    y_train = y[:n_cut]
    y_test = y[n_cut - 1:]
    a_tr, c_tr, _, _ = _armax_fit(y_train, p, q)
    m_residuals = _armax_residuals(a_tr, c_tr, y_test, int(num_steps))
    out.update(residual_analysis(m_residuals, y_test, 'full'))
    return out

def _whiten(y: np.ndarray, pre_proc: str, random_seed=None) -> np.ndarray:
    """
    Whiten a time series by comparing a range of preprocessings (hctsa's ``BF_Whiten``).

    ``'none'`` leaves the series alone; ``'detrend'`` removes a linear trend; ``'ar'``
    removes a linear trend and then applies (``PP_PreProcess(y, 'ar', 2, 0.05, 0)``) the
    preprocessing, from differencing (``d1`` to ``d3``), piecewise polynomial detrending
    (``p1_5`` ... ``p2_40``) and rank-mapping onto a Gaussian (``rmgd``), after which the
    z-scored series is the hardest for an AR(2) model to predict. It has to beat doing
    nothing by 5% for a preprocessing to be applied. The random draws of ``rmgd`` are
    NumPy's, seeded as ``BF_ResetSeed`` (``None``/``'default'``: seed 0, ``'none'``: NumPy's
    global state, or an integer seed).
    """
    from scipy.signal import detrend
    # (imported here: pre_process imports nonlinearity, which imports this module)
    from .pre_process import _ar_rms_error, _piecewise_poly_residual, _rank_map_gaussian
    y = np.asarray(y, dtype=float).ravel()
    if pre_proc in ('nothing', 'none'):
        return y
    y = detrend(y)
    if pre_proc == 'detrend':
        return y
    if pre_proc != 'ar':
        raise ValueError(f"Unknown preprocessing setting '{pre_proc}'")

    candidates = {'nothing': y, 'd1': np.diff(y, 1), 'd2': np.diff(y, 2), 'd3': np.diff(y, 3)}
    for order in (1, 2):
        for num_bits in (5, 10, 20, 40):
            # (as MATLAB's zscore: a constant series becomes zeros)
            candidates[f'p{order}_{num_bits}'] = _zscore_matlab(_piecewise_poly_residual(y, order, num_bits))
    candidates['rmgd'] = _rank_map_gaussian(y, random_seed)  # rank-map onto a Gaussian (stochastic)
    # (hctsa's log, log returns, Box-Cox and square-root versions need a positive series,
    # which a detrended series never is)

    names = list(candidates)
    rmse = np.array([_ar_rms_error(_zscore_matlab(candidates[k]), 2) for k in names])
    if np.any(rmse > rmse[0] * 1.05):
        return candidates[names[int(np.argmax(rmse))]]
    return candidates['nothing']


def _arch_test_pvalues(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    p-values of Engle's ARCH test at lags ``1..max_lag`` (MATLAB's ``archtest``): the
    test statistic is ``(N - lag) R^2`` for the regression of the squared series on a
    constant and ``lag`` of its own lags, referred to a chi-squared distribution with
    ``lag`` degrees of freedom.
    """
    from scipy.stats import chi2
    x2 = np.asarray(x, dtype=float) ** 2
    n = len(x2)
    pvals = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        resp = x2[lag:]
        design = np.column_stack([np.ones(n - lag)] + [x2[lag - j:n - j] for j in range(1, lag + 1)])
        fitted = design @ np.linalg.lstsq(design, resp, rcond=None)[0]
        fitted = fitted - np.mean(fitted)
        resp = resp - np.mean(resp)
        r2 = (fitted @ fitted) / (resp @ resp)
        pvals[lag - 1] = chi2.sf(r2 * (n - lag), lag)
    return pvals


def _lbq_test_pvalues(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    p-values of the Ljung-Box Q-test at lags ``1..max_lag`` (MATLAB's ``lbqtest``, with the
    degrees of freedom equal to the lag).
    """
    from scipy.stats import chi2
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - np.mean(x)
    denom = xc @ xc
    acf = np.array([xc[k:] @ xc[:n - k] for k in range(1, max_lag + 1)]) / denom
    stat = n * (n + 2) * np.cumsum(acf ** 2 / (n - np.arange(1, max_lag + 1)))
    return chi2.sf(stat, np.arange(1, max_lag + 1))


def _garch_estimate(y: np.ndarray, P: int, Q: int, model_type: str = 'garch',
                    innovation_dist: str = 'gaussian') -> dict:
    """
    Maximum-likelihood fit of a zero-mean conditional variance model (the equivalent of
    MATLAB's ``estimate`` on ``garch(P, Q)``, ``gjr(P, Q)`` or ``egarch(P, Q)`` with a free
    constant), using the ``arch`` package.

    ``P`` is the GARCH degree (lagged conditional variances) and ``Q`` the ARCH degree (lagged
    squared innovations; ``arch``'s ``p``). As ``estimate``, the presample variance and squared
    innovation are the mean of the squared series, and the covariance of the estimates is the
    inverse of the outer product of the per-observation score vectors.

    Returns a dict with the parameters in MATLAB's order (``constant``, ``garch``, ``arch``,
    ``leverage``, ``dof``), their covariance matrix ``cov`` (in the same order), the
    log-likelihood ``llf``, the conditional variances ``sigma2``, and ``exitflag``.
    """
    from arch import arch_model
    from statsmodels.tools.numdiff import approx_fprime
    dist = {'gaussian': 'normal', 't': 't'}.get(innovation_dist)
    if dist is None:
        raise ValueError(f"Unknown innovationDist '{innovation_dist}' (should be 'gaussian' or 't')")
    if model_type not in ('garch', 'gjr', 'egarch'):
        raise ValueError(f"Unknown modelType '{model_type}' (should be 'garch', 'gjr', or 'egarch')")
    asym = model_type in ('gjr', 'egarch')
    model = arch_model(y, mean='Zero', vol='EGARCH' if model_type == 'egarch' else 'GARCH',
                       p=Q, o=Q if asym else 0, q=P, dist=dist, rescale=False)
    backcast = np.mean(y ** 2)
    if model_type == 'egarch':
        backcast = np.log(backcast)  # (arch expects the log variance for an EGARCH backcast)
    res = model.fit(disp='off', show_warning=False, backcast=backcast)
    params = res.params
    if not np.all(np.isfinite(params.values)):
        raise ValueError('GARCH fit failed (non-finite parameter estimates)')

    # Standard errors: outer product of gradients of the per-observation log-likelihood
    resids = model.resids(model.starting_values())
    kwargs = dict(sigma2=np.zeros(len(y)), backcast=backcast,
                  var_bounds=model.volatility.variance_bounds(resids), individual=True)
    scores = approx_fprime(params.values, model._loglikelihood, kwargs=kwargs)
    cov = np.linalg.pinv(scores.T @ scores)

    # Reorder from arch's (omega, alpha, gamma, beta, nu) to MATLAB's (K, GARCH, ARCH, Leverage, DoF)
    names = list(params.index)
    order = ([names.index('omega')]
             + [names.index(f'beta[{i}]') for i in range(1, P + 1)]
             + [names.index(f'alpha[{i}]') for i in range(1, Q + 1)]
             + ([names.index(f'gamma[{i}]') for i in range(1, Q + 1)] if asym else [])
             + ([names.index('nu')] if dist == 't' else []))
    p = params.values[order]
    return {'constant': p[0], 'garch': p[1:1 + P], 'arch': p[1 + P:1 + P + Q],
            'leverage': p[1 + P + Q:1 + P + 2 * Q] if asym else np.array([]),
            'dof': p[-1] if dist == 't' else np.nan,
            'cov': cov[np.ix_(order, order)], 'llf': float(res.loglikelihood),
            'sigma2': np.asarray(res.conditional_volatility) ** 2,
            'exitflag': 1 if res.convergence_flag == 0 else 0}


def garch_fit(y: ArrayLike, preproc: str = 'ar', P: int = 1, Q: int = 1,
              random_seed: Union[int, str, None] = None, model_type: str = 'garch',
              innovation_dist: str = 'gaussian') -> dict:
    """
    Fits a GARCH-family model to the series and reports the fit and the standardized residuals.

    The series is whitened and z-scored, so the model is of the variance around a zero mean
    (hctsa's ``MF_GARCHfit``; MATLAB's ``garch``/``gjr``/``egarch`` and ``estimate``, here fitted
    with the ``arch`` package). Statistics are the fitted parameters and their errors, the
    log-likelihood and information criteria, the persistence of the fitted variance process,
    summaries of the conditional variance series, and how well the standardized residuals pass
    tests for remaining ARCH effects (Engle's ARCH test and the Ljung-Box Q-test) compared to
    the series itself, and :func:`residual_analysis` of the standardized residuals.

    Parameters
    ----------
    y : array-like
        The input time series.
    preproc : {'ar', 'none'}, optional
        The preprocessing applied before the fit: ``'ar'`` (default) replaces the series by
        the preprocessing (from differencing, piecewise polynomial detrending, rank-mapping to
        a Gaussian) that maximizes whiteness under an AR(2) model, if it beats the
        unprocessed series by 5%; ``'none'`` applies none (the series is linearly
        detrended and z-scored either way).
    P : int, optional
        The GARCH degree: the number of lagged conditional variances. Default is 1.
    Q : int, optional
        The ARCH degree: the number of lagged squared innovations. Default is 1.
    random_seed : int, 'default', 'none' or None, optional
        How to seed the random draws used by the whitening (``rmgd``), as ``BF_ResetSeed``.
        Default is ``None`` (seed 0). The draws are NumPy's, not MATLAB's ``randn`` stream.
    model_type : {'garch', 'gjr', 'egarch'}, optional
        The conditional variance model: ``'garch'`` (default, symmetric response to shocks),
        ``'gjr'`` (GJR-GARCH, adds a leverage term so negative and positive shocks can have
        different effects on variance) or ``'egarch'`` (exponential GARCH, models the log
        variance, also asymmetric).
    innovation_dist : {'gaussian', 't'}, optional
        The assumed innovation distribution: ``'gaussian'`` (default) or ``'t'`` (Student's t,
        which estimates a degrees-of-freedom parameter to capture fat tails).

    Returns
    -------
    dict
        - ``constant``, ``constanterr``: the constant term of the variance equation and its error
        - ``offset``: the mean offset of the model (0 for a z-scored series)
        - ``GARCH_i``, ``GARCHerr_i`` (i = 1..P), ``ARCH_i``, ``ARCHerr_i`` (i = 1..Q): the
          coefficients of the lagged variances and innovations, and their errors
        - ``leverage``, ``leverageerr``: the leverage coefficient and its error (``'gjr'`` and
          ``'egarch'`` only, otherwise NaN)
        - ``distDoF``: the degrees of freedom of a Student's t distribution (otherwise NaN)
        - ``LLF``, ``aic``, ``bic``: the log-likelihood, AIC and BIC per observation
        - ``summaryexitflag``: 1 if the optimizer converged, otherwise 0 (MATLAB's ``estimate``
          reports 1 or 2 on convergence)
        - ``persistence``: the sum of the ARCH and GARCH coefficients (plus half the leverage
          coefficient for ``'gjr'``; NaN for ``'egarch'``)
        - ``uncondVar``: the implied long-run variance (NaN if persistence is 0.999 or more,
          or for ``'egarch'``)
        - ``maxsigma``, ``minsigma``, ``rangesigma``, ``stdsigma``, ``meansigma``: summaries of
          the conditional variance series
        - ``engle_mean_diff_p``, ``engle_max_diff_p``, ``lbq_mean_diff_p``, ``lbq_max_diff_p``:
          the mean and maximum, over lags 1 to 20, of the change in p-value of Engle's ARCH test
          (series to standardized residuals) and of the Ljung-Box Q-test (squared series to
          squared standardized residuals)
        - ``engle_pval_stde_1``, ``_5``, ``_10``, ``minenglepval_stde``, ``maxenglepval_stde``:
          p-values of Engle's ARCH test on the standardized residuals
        - ``lbq_pval_stde_1``, ``_5``, ``_10``, ``minlbqpval_stde2``, ``maxlbqpval_stde2``:
          p-values of the Ljung-Box Q-test on the squared standardized residuals
        - ``ac1_stde2``: lag-1 autocorrelation of the squared standardized residuals
        - ``diff_ac1``: lag-1 autocorrelation of the squared series minus ``ac1_stde2``
        - ``zres_*``: :func:`residual_analysis` of the standardized residuals
          (``zres_meane``, ``zres_meanabs``, ``zres_stde``, ``zres_maxonstd``, ``zres_ac1``,
          ``zres_ac2``, ``zres_ac3``, ``zres_propbth``, ``zres_ftbth``, ``zres_taurat``,
          ``zres_normksstat``, ``zres_sws``, ``zres_swm``, ``zres_popt``, ``zres_minsbc``)

    Notes
    -----
    The fit uses ``arch`` rather than MATLAB's Econometrics Toolbox, so the optimizer and the
    point it stops at differ slightly: parameters agree to about 1e-3 where the likelihood is
    well-determined, but poorly identified fits (series with no volatility clustering, where
    the coefficients sit at their bounds) can land on different points of a flat likelihood
    and give very different errors. The log-likelihood agrees closely. The presample variance
    is the mean of the squared series and the errors come from the outer product of
    gradients, as in MATLAB.
    """
    y = np.asarray(y, dtype=float).ravel()
    P, Q = int(P), int(Q)
    out = {}

    # (1) Whiten and z-score
    y = _whiten(y, preproc, random_seed)
    y = z_score(y)
    n = len(y)

    # (2) Pre-estimation tests on the (whitened) series
    engle_y = _arch_test_pvalues(y)
    lbq_y2 = _lbq_test_pvalues(y ** 2)

    # (3) Fit the model
    try:
        fit = _garch_estimate(y, P, Q, model_type, innovation_dist)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as err:
        raise ValueError('GARCH fit failed (data does not allow a valid GARCH model '
                         f'to be estimated): {err}') from err
    errors = np.sqrt(np.abs(np.diag(fit['cov'])))
    llf = fit['llf']
    garch_c, arch_c, lev_c = fit['garch'], fit['arch'], fit['leverage']

    # (4) Statistics on the fit
    out['constant'] = fit['constant']
    out['constanterr'] = errors[0]
    out['offset'] = 0.0
    for i in range(1, P + 1):
        out[f'GARCH_{i}'] = garch_c[i - 1]
        # (a coefficient estimated at exactly zero has no error: NaN, as hctsa)
        out[f'GARCHerr_{i}'] = np.nan if garch_c[i - 1] == 0 else errors[i]
    for i in range(1, Q + 1):
        out[f'ARCH_{i}'] = arch_c[i - 1]
        out[f'ARCHerr_{i}'] = np.nan if arch_c[i - 1] == 0 else errors[P + i]
    if lev_c.size > 0:
        out['leverage'] = lev_c[0]
        out['leverageerr'] = errors[1 + P + Q]
    else:
        out['leverage'] = np.nan
        out['leverageerr'] = np.nan
    out['distDoF'] = fit['dof'] if innovation_dist == 't' else np.nan

    out['LLF'] = llf / n  # log-likelihood per observation
    out['summaryexitflag'] = fit['exitflag']

    n_params = int(np.sum(np.any(fit['cov'] != 0, axis=0)))
    out['aic'] = (-2 * llf + 2 * n_params) / n
    out['bic'] = (-2 * llf + n_params * np.log(n)) / n

    # Persistence of the variance process and the implied unconditional variance
    if model_type == 'garch':
        persistence = np.sum(garch_c) + np.sum(arch_c)
    elif model_type == 'gjr':
        persistence = np.sum(garch_c) + np.sum(arch_c) + np.sum(lev_c) / 2
    else:  # egarch: no simple coefficient sum
        persistence = np.nan
    out['persistence'] = persistence
    out['uncondVar'] = fit['constant'] / (1 - persistence) if persistence < 0.999 else np.nan

    # Sigmas, the time series of conditional variances
    sigmas = fit['sigma2']
    out['maxsigma'] = np.max(sigmas)
    out['minsigma'] = np.min(sigmas)
    out['rangesigma'] = np.max(sigmas) - np.min(sigmas)
    out['stdsigma'] = np.std(sigmas, ddof=1)
    out['meansigma'] = np.mean(sigmas)

    # Check the standardized residuals: residuals (mean process minus data)
    stde = (0.0 - y) / np.sqrt(sigmas)
    stde2 = stde ** 2
    engle_stde = _arch_test_pvalues(stde)
    lbq_stde2 = _lbq_test_pvalues(stde2)

    out['engle_mean_diff_p'] = np.mean(engle_stde - engle_y)
    out['engle_max_diff_p'] = np.max(engle_stde - engle_y)
    out['lbq_mean_diff_p'] = np.mean(lbq_stde2 - lbq_y2)
    out['lbq_max_diff_p'] = np.max(lbq_stde2 - lbq_y2)

    out['engle_pval_stde_1'] = engle_stde[0]
    out['engle_pval_stde_5'] = engle_stde[4]
    out['engle_pval_stde_10'] = engle_stde[9]
    out['minenglepval_stde'] = np.min(engle_stde)
    out['maxenglepval_stde'] = np.max(engle_stde)

    out['lbq_pval_stde_1'] = lbq_stde2[0]
    out['lbq_pval_stde_5'] = lbq_stde2[4]
    out['lbq_pval_stde_10'] = lbq_stde2[9]
    out['minlbqpval_stde2'] = np.min(lbq_stde2)
    out['maxlbqpval_stde2'] = np.max(lbq_stde2)

    # Statistics on the standardized innovations, prefixed zres_
    for key, value in residual_analysis(stde, y, 'full').items():
        out[f'zres_{key}'] = value

    out['ac1_stde2'] = autocorr(stde2, [1], 'Fourier')[0]
    out['diff_ac1'] = autocorr(y ** 2, [1], 'Fourier')[0] - out['ac1_stde2']
    return out


def garch_compare(y: ArrayLike, pre_proc: str = 'none', pr: ArrayLike = (1, 2, 3),
                  qr: ArrayLike = (1, 2, 3), random_seed: Union[int, str, None] = None) -> dict:
    """
    How well GARCH models of different orders describe the changing variance of the series.

    Fits a set of zero-mean GARCH(p, q) models with Gaussian innovations to the (whitened and
    z-scored) time series (hctsa's ``MF_GARCHcompare``) and returns statistics on the goodness
    of fit across a range of p (the number of lagged variances) and q (the number of lagged
    squared innovations): summaries across the grid of fitted models, and the orders that fit
    best. See :func:`garch_fit` for how the models are fitted.

    Parameters
    ----------
    y : array-like
        The input time series.
    pre_proc : {'none', 'ar'}, optional
        A preprocessing to apply after detrending: ``'none'`` (default), or ``'ar'``, which
        applies the preprocessing that maximizes AR(2) whiteness (see :func:`garch_fit`).
    pr : array-like of int, optional
        The model orders p to compare. Default is ``(1, 2, 3)``.
    qr : array-like of int, optional
        The model orders q to compare. Default is ``(1, 2, 3)``.
    random_seed : int, 'default', 'none' or None, optional
        How to seed the random draws used by the whitening, as in :func:`garch_fit`.

    Returns
    -------
    dict or float
        NaN if no model could be fitted. Otherwise, statistics across the (p, q) models that
        fitted (the log-likelihood, AIC and BIC are per observation):

        - ``minLLF``, ``maxLLF``, ``meanLLF``: the log-likelihood
        - ``minBIC``, ``maxBIC``, ``meanBIC``: the Bayesian information criterion
        - ``minAIC``, ``maxAIC``, ``meanAIC``: Akaike's information criterion
        - ``minK``, ``maxK``, ``meanK``: the constant term of the variance equation
        - ``min_meanarchps``, ``max_meanarchps``, ``mean_meanarchps``: across models, the mean
          p-value (over lags 1 to 20) of Engle's ARCH test on the standardized residuals
        - ``min_maxarchps``, ``max_maxarchps``, ``mean_maxarchps``: the same for the maximum
          p-value over the 20 lags
        - ``min_meanlbqps``, ``max_meanlbqps``, ``mean_meanlbqps``: across models, the mean
          p-value of the Ljung-Box Q-test on the squared standardized residuals
        - ``min_maxlbqps``, ``max_maxlbqps``, ``mean_maxlbqps``: the same for the maximum
        - ``bestpLLF``, ``bestqLLF``: the orders p and q of the model with the maximum
          log-likelihood
        - ``bestpAIC``, ``bestqAIC``, ``bestpBIC``, ``bestqBIC``: the orders of the models with
          the minimum AIC and BIC
        - ``Ks_vary_p``, ``Ks_vary_q``: how much the constant term varies with p and with q: the
          standard deviation of the constant across one order, averaged over the other
    """
    y = np.asarray(y, dtype=float).ravel()
    pr = [int(v) for v in np.atleast_1d(pr)]
    qr = [int(v) for v in np.atleast_1d(qr)]

    y = z_score(_whiten(y, pre_proc, random_seed))
    n = len(y)

    shape = (len(pr), len(qr))
    llfs, aics, bics, ks, mean_arch, max_arch, mean_lbq, max_lbq = (np.full(shape, np.nan) for _ in range(8))
    for i, p in enumerate(pr):
        for j, q in enumerate(qr):
            try:
                fit = _garch_estimate(y, p, q)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                logger.warning(f'Bad fit at p = {p}, q = {q}')
                continue  # didn't fit successfully; everything stays NaN
            n_params = int(np.sum(np.any(fit['cov'] != 0, axis=0)))
            if n_params < p + q + 1:
                logger.warning(f'Bad fit at p = {p}, q = {q}')
                continue
            llfs[i, j] = fit['llf']
            aics[i, j] = -2 * fit['llf'] + 2 * n_params
            bics[i, j] = -2 * fit['llf'] + n_params * np.log(n)
            ks[i, j] = fit['constant']
            stde = (0.0 - y) / np.sqrt(fit['sigma2'])
            engle = _arch_test_pvalues(stde)
            lbq = _lbq_test_pvalues(stde ** 2)
            mean_arch[i, j], max_arch[i, j] = np.mean(engle), np.max(engle)
            mean_lbq[i, j], max_lbq[i, j] = np.mean(lbq), np.max(lbq)

    if np.all(np.isnan(llfs)):
        logger.warning('None of the ARCH or GARCH models could be fit.')
        return np.nan

    # Log-likelihoods and information criteria per observation
    llfs, aics, bics = llfs / n, aics / n, bics / n

    out = {}
    for name, values in (('LLF', llfs), ('BIC', bics), ('AIC', aics), ('K', ks)):
        out[f'min{name}'] = np.nanmin(values)
        out[f'max{name}'] = np.nanmax(values)
        out[f'mean{name}'] = np.nanmean(values)
    for name, values in (('meanarchps', mean_arch), ('maxarchps', max_arch),
                         ('meanlbqps', mean_lbq), ('maxlbqps', max_lbq)):
        out[f'min_{name}'] = np.nanmin(values)
        out[f'max_{name}'] = np.nanmax(values)
        out[f'mean_{name}'] = np.nanmean(values)

    # The orders of the best models (first in column-major order, as MATLAB's find)
    for name, values, pick in (('LLF', llfs, np.nanargmax), ('AIC', aics, np.nanargmin),
                               ('BIC', bics, np.nanargmin)):
        a, b = np.unravel_index(pick(values.ravel(order='F')), shape, order='F')
        out[f'bestp{name}'] = pr[a]
        out[f'bestq{name}'] = qr[b]

    # How much the constant varies with each order
    def _std_omitnan(values):
        std = np.full(values.shape[1], np.nan)
        for k in range(values.shape[1]):
            col = values[:, k][~np.isnan(values[:, k])]
            if col.size == 1:
                std[k] = 0.0
            elif col.size > 1:
                std[k] = np.std(col, ddof=1)
        return std

    out['Ks_vary_p'] = np.nanmean(_std_omitnan(ks))
    out['Ks_vary_q'] = np.nanmean(_std_omitnan(ks.T))
    return out

def _seeded_rng(random_seed) -> np.random.RandomState:
    """
    The random generator after hctsa's ``BF_ResetSeed(random_seed)``: an integer seed, or
    ``'default'`` (seed 0), seeds MATLAB's Mersenne Twister (so ``rand`` draws are MATLAB's);
    ``'none'`` or ``None`` gives a generator that is not reset (fresh entropy).
    """
    if isinstance(random_seed, str) and random_seed == 'default':
        return _ml_rng(0)
    if random_seed is None or (isinstance(random_seed, str) and random_seed == 'none'):
        return np.random.RandomState()
    return _ml_rng(int(random_seed))


def _n4_fpe(loss: float, n_order: int, n_obs: int) -> float:
    """Akaike's final prediction error of an ``n4sid`` fit of order ``n_order`` to ``n_obs`` samples
    (3 * order free parameters, as MATLAB counts them in ``EstimationInfo.FPE``)."""
    n_eff = 3 * n_order
    return loss * (1 + n_eff / n_obs) / (1 - n_eff / n_obs)


def _n4_arx_order(y: np.ndarray, n_max: int) -> int:
    """
    The best order of an ARX model by AIC, which sets n4sid's automatic past horizon (the
    ``localAIC`` step of MATLAB's ``n4sid``). All orders are compared on the same samples,
    through the R factor of the QR decomposition of the matrix of ``n_max`` consecutive values.
    """
    n = len(y)
    r_fac = np.linalg.qr(sliding_window_view(y, n_max), mode='r')
    n_eff = n - n_max + 1
    m = min(n_max - 1, n_eff - 2)
    v = np.zeros(m + 2)  # (the final element, 0, is part of the search as in MATLAB)
    for k in range(m + 1):
        v[k] = np.log((r_fac[k, k] / n_eff) ** 2) + 2 * k / n_eff
    return int(np.argmin(v))


def _n4_horizons(y: np.ndarray, order: int) -> tuple:
    """
    n4sid's automatic horizons for a time series (no input): the future horizon
    ``ceil(1.5 order)`` and a past horizon from the best ARX order, adjusted for the series
    length and the order (as ``n4sid``).
    """
    n = len(y)
    r = int(np.ceil(1.5 * order))
    n_max = int(np.ceil(min(4 * order, (n - 1) / 2, max(n // 2 - 1 + order, 1))))
    s = _n4_arx_order(y, n_max)
    if n - 2 * r - 2 * s < 0:  # too few samples: shrink the horizons
        s = min(s, 2 * order)
        r0, s0 = r, s
        count = 1
        while n - 2 * r - 2 * s < 0 and count < r0 + 2 * s0:
            r, s = max(r - 1, order + 1), min(max(s - 1, order), s0)
            count += 1
    r = max(r, order + 1)
    if s + 1 <= order:  # the past must carry at least as many values as the order
        s = order
    return r, s


def _stabilize_matrix(a: np.ndarray, thresh: float = 1 + np.sqrt(np.finfo(float).eps)) -> np.ndarray:
    """
    Reflect eigenvalues of ``a`` that lie outside the unit circle (beyond ``thresh``) to
    ``thresh^2/lambda``, as MATLAB's ``fstab``.
    """
    from scipy.linalg import rsf2csf, schur
    eigval, eigvec = np.linalg.eig(a)
    if np.linalg.cond(eigvec) > 1e8:
        t_mat, z_mat = schur(a.astype(complex), output='complex')
        eigval, eigvec = np.diag(t_mat).copy(), z_mat
        diag_mat = t_mat
    else:
        diag_mat = np.diag(eigval)
    if np.max(np.abs(eigval)) < thresh:
        return a
    for k in range(len(eigval)):
        if abs(diag_mat[k, k]) > thresh:
            diag_mat[k, k] = thresh ** 2 / diag_mat[k, k]
    return np.real(eigvec @ diag_mat @ np.linalg.inv(eigvec))


def _ss_predict(a: np.ndarray, k: np.ndarray, c: np.ndarray, y: np.ndarray, x0: np.ndarray,
                steps: int = 1) -> np.ndarray:
    """
    ``steps``-ahead predictions of ``y`` from the innovations-form model
    ``x(t+1) = A x(t) + K e(t)``, ``y(t) = C x(t) + e(t)``, starting from state ``x0``: the
    predictor state is ``x(t+1) = (A - K C) x(t) + K y(t)`` and the prediction of
    ``y(t)`` from data up to ``t - steps`` is ``C A^(steps-1) x(t - steps + 1)``.
    """
    n_obs = len(y)
    f = a - k @ c
    states = np.zeros((n_obs, len(x0)))
    x = np.asarray(x0, dtype=float)
    for t in range(n_obs):
        states[t] = x
        x = f @ x + k[:, 0] * y[t]
    pred = np.zeros(n_obs)
    a_pow = np.linalg.matrix_power(a, steps - 1)
    for t in range(n_obs):
        if t >= steps - 1:
            pred[t] = (c @ a_pow @ states[t - steps + 1])[0]
        else:  # before the first prediction can be updated by data: run from the initial state
            pred[t] = (c @ np.linalg.matrix_power(a, t) @ states[0])[0]
    return pred


def _ss_initial_state(a: np.ndarray, k: np.ndarray, c: np.ndarray, y: np.ndarray,
                      steps: int = 1) -> tuple:
    """
    The initial state of a state-space model that minimizes the squared ``steps``-ahead
    prediction error (``'init', 'e'`` in MATLAB), and the prediction errors ``y - yp`` it gives.
    """
    n = a.shape[0]
    zero = _ss_predict(a, k, c, y, np.zeros(n), steps)
    basis = np.column_stack([_ss_predict(a, k, c, np.zeros_like(y), e_j, steps) for e_j in np.eye(n)])
    x0 = np.linalg.lstsq(basis, y - zero, rcond=None)[0]
    return x0, y - zero - basis @ x0


def _n4_state_space(y: np.ndarray, order: Union[int, str]) -> dict:
    """
    Subspace identification of a state-space model of a time series (no input),
    ``x(t+1) = A x(t) + K e(t)``, ``y(t) = C x(t) + e(t)``: the N4SID algorithm with canonical
    variate analysis (CVA) weighting, as MATLAB's ``n4sid`` with default options.

    The past and future horizons are set automatically, the gain K comes from the Riccati
    equation for the estimated state and noise covariances, and the initial state is the
    least-squares estimate for the one-step prediction errors. ``order`` can be ``'best'``,
    which takes the order from the singular values of the CVA decomposition (orders 1 to 10
    are considered).

    Returns a dict with ``A``, ``K``, ``C``, the initial state ``x0``, the loss function
    ``loss`` (mean squared one-step prediction error) and the model ``order``.
    """
    from scipy.linalg import solve_discrete_are
    y = np.asarray(y, dtype=float).ravel()
    n_obs = len(y)
    n_try = 10 if order == 'best' else int(order)
    r, s = _n4_horizons(y, n_try)

    # LQ decomposition of the block Hankel matrix of past (s rows) and future (r rows) outputs
    j = n_obs - s - r + 1
    if j < s + r:
        raise ValueError('Too few samples for the state-space model')
    hankel = sliding_window_view(y, j)[:s + r]
    lfac = np.linalg.qr(hankel.T, mode='r').T
    lfp = lfac[s:, :s]                  # the future rows' dependence on the past
    u_w, s_w, _ = np.linalg.svd(lfac[s:, :])
    s_w = s_w[:r]
    u_c, _, _ = np.linalg.svd((u_w[:, :r].T @ lfp) / s_w[:, None])
    u_n = (u_w[:, :r] * s_w) @ u_c      # CVA weighting
    # The sign of each singular vector, and so of each state, is arbitrary (it depends on the
    # linear algebra library, and MATLAB's cannot be reproduced): fix it so that the output
    # matrix C has no negative entries
    u_n = u_n * np.where(u_n[0, :] < 0, -1.0, 1.0)

    if order == 'best':
        sv = np.linalg.svd(lfp, compute_uv=False)
        if sv.max() / sv.min() < 1 + np.sqrt(np.finfo(float).eps):
            n = 1
        else:
            above = np.nonzero(np.log(sv) > (np.log(sv).max() + np.log(sv).min()) / 2)[0]
            n = max(1, min(n_try, above[-1] + 1))
    else:
        n = n_try

    # System matrices from the shift structure of the extended observability matrix
    a_mat = np.linalg.lstsq(u_n[:r - 1, :n], u_n[1:r, :n], rcond=None)[0]
    c_mat = u_n[:1, :n]
    if not np.all(np.isfinite(a_mat)):
        raise ValueError('n4sid failed: the data are not persistently exciting')
    a_mat = _stabilize_matrix(a_mat)

    # The gain: regress the next state and the output on the state, then solve the Riccati equation
    r2 = lfac[s:, :s + 1]
    vl = np.vstack([np.linalg.lstsq(u_n[:r - 1, :n], r2[1:r, :], rcond=None)[0], r2[:1, :]])
    hl = np.linalg.lstsq(u_n[:, :n], np.hstack([r2[:, :s], np.zeros((r, 1))]), rcond=None)[0]
    resid = vl - (np.linalg.lstsq(hl.T, vl.T, rcond=None)[0].T) @ hl
    w_cov = resid @ resid.T
    try:
        p_are = solve_discrete_are(a_mat.T, c_mat.T, w_cov[:n, :n], w_cov[n:, n:], s=w_cov[:n, n:])
        k_mat = np.linalg.solve(w_cov[n:, n:] + c_mat @ p_are @ c_mat.T,
                                c_mat @ p_are @ a_mat.T + w_cov[:n, n:].T).T
    except (np.linalg.LinAlgError, ValueError):
        k_mat = np.zeros((n, 1))
    if not np.all(np.isfinite(k_mat)):
        k_mat = np.zeros((n, 1))

    x0, err = _ss_initial_state(a_mat, k_mat, c_mat, y)
    return {'A': a_mat, 'K': k_mat, 'C': c_mat, 'x0': x0, 'loss': float(np.mean(err ** 2)),
            'order': n}


def state_space_n4sid(y: ArrayLike, ord: Union[int, str] = 2, ptrain: float = 0.5,
                      steps: int = 1) -> dict:
    """
    A fitted state-space model of the series, and how well it predicts the later part of the series.

    First fits the model to the whole time series, then trains it on the first portion and
    tries to predict the rest (hctsa's ``MF_StateSpace_n4sid``, which uses ``n4sid`` from MATLAB's
    System Identification Toolbox; here the same subspace algorithm is implemented directly).
    The state-space model has the form (discrete time, no input, sampling interval 1)
    ``x(t+1) = A x(t) + K e(t)``, ``y(t) = C x(t) + e(t)``,
    for state-transition matrix ``A`` and output matrix ``C``, noise-input vector ``K``,
    vector of states ``x`` and disturbance (noise) ``e``. The model fitted to the first
    ``ptrain`` proportion of the series is used to predict the remaining samples ``steps`` ahead,
    and the prediction residuals (prediction minus data) are summarized with
    :func:`residual_analysis`.

    Parameters
    ----------
    y : array-like
        The input time series.
    ord : int or 'best', optional
        The order of the state-space model (the number of states), or ``'best'`` to choose it from
        the singular values of the subspace decomposition (orders 1 to 10 are considered).
        Default is 2.
    ptrain : float, optional
        The proportion of the time series to use for training. Default is 0.5.
    steps : int, optional
        The number of steps ahead to predict. Default is 1.

    Returns
    -------
    dict
        From the model fitted to the entire time series:

        - ``A_1``, ..., ``A_(ord^2)``: the entries of the state-transition matrix ``A``, counted
          down each column in turn
        - ``k_1``, ..., ``k_ord``: the entries of the noise-input vector ``K``
        - ``c_1``, ..., ``c_ord``: the entries of the output vector ``C``
        - ``x0mod``: the length of the initial state vector
        - ``np``: the number of parameters fitted
        - ``Ts``: the sampling interval of the model (always 1)
        - ``noisevar``: the estimated noise variance
        - ``lossfn``: the loss function (the estimated prediction-error variance)
        - ``fpe``: Akaike's final prediction error
        - ``bestorder``: the order chosen (only when ``ord`` is ``'best'``)

        From the residuals of the predictions of the held-out portion (see
        :func:`residual_analysis`, ``'full'``): ``meane``, ``meanabs``, ``stde``, ``maxonstd``,
        ``ac1``, ``ac2``, ``ac3``, ``propbth``, ``ftbth``, ``taurat``, ``sws``, ``swm``,
        ``normksstat``, ``popt``, ``minsbc``; and ``ac1diff``, the absolute lag-1
        autocorrelation of the whole time series minus that of the prediction residuals.

    Notes
    -----
    The individual entries of ``A``, ``K`` and ``C`` depend on the (arbitrary) coordinates of the
    hidden state, so they are not directly comparable between time series. Here the sign of each
    state is fixed so that ``C`` has no negative entries (the sign of a singular vector depends
    on the linear algebra library); MATLAB's signs are not reproducible, so ``A``, ``K`` and ``C``
    can differ from MATLAB's by a sign change of states. Everything else (``x0mod``, the loss
    function, and the residual summaries) is the same as MATLAB's to numerical precision, as long
    as the subspace decomposition is well-conditioned.

    The residuals are prediction minus data. The held-out portion starts at sample
    ``floor(ptrain*N)``, overlapping the training portion by one sample.
    """
    y = np.asarray(y, dtype=float).ravel()
    n_obs = len(y)
    if ord != 'best':
        ord = int(ord)

    # The model of the whole time series
    fit = _n4_state_space(y, ord)
    n = fit['order']
    out = {}
    if ord == 'best':
        out['bestorder'] = n
    for i, v in enumerate(fit['A'].ravel(order='F'), start=1):
        out[f'A_{i}'] = v
    for i, v in enumerate(fit['K'].ravel(), start=1):
        out[f'k_{i}'] = v
    for i, v in enumerate(fit['C'].ravel(), start=1):
        out[f'c_{i}'] = v
    out['x0mod'] = np.sqrt(np.sum(fit['x0'] ** 2))
    out['np'] = n * n + 3 * n   # A, K, C and the initial state
    out['Ts'] = 1
    # (3n: the parameters that remain after removing the freedom in the choice of coordinates)
    n_eff = 3 * n
    out['noisevar'] = fit['loss'] * n_obs / (n_obs - n_eff)
    out['lossfn'] = fit['loss']
    out['fpe'] = _n4_fpe(fit['loss'], n, n_obs)

    # Train on the first portion, predict the rest (overlapping by one sample)
    n_cut = int(np.floor(ptrain * n_obs))
    y_test = y[n_cut - 1:]
    try:
        train = _n4_state_space(y[:n_cut], ord)
    except (ValueError, np.linalg.LinAlgError) as err:
        raise ValueError(f"Couldn't fit the model to this time series: {err}") from err
    m_residuals = -_ss_initial_state(train['A'], train['K'], train['C'], y_test, int(steps))[1]
    out.update(residual_analysis(m_residuals, y_test, 'full'))
    out['ac1diff'] = abs(autocorr(y, [1], 'Fourier')[0]) - abs(autocorr(m_residuals, [1], 'Fourier')[0])
    return out


def state_space_comp_order(y: ArrayLike, max_order: int = 10) -> Union[dict, float]:
    """
    How the fit of a state-space model improves as its order increases.

    Fits state-space models (subspace identification, as ``n4sid`` in MATLAB's System
    Identification Toolbox) of orders 1, 2, ..., ``max_order`` to the whole time series (all fits
    are within the sample), and returns statistics on how the goodness of fit changes across this
    range, measured by Akaike's information criterion (AIC) and by the loss function (the
    estimated variance of the one-step prediction error), hctsa's ``MF_StateSpaceCompOrder``.
    An order at which the model cannot be fitted is left out of the summaries, and the output is
    NaN only if no order can be fitted.

    Parameters
    ----------
    y : array-like
        The input time series.
    max_order : int, optional
        The maximum model order to consider. Default is 10.

    Returns
    -------
    dict or float
        NaN if no order could be fitted. Otherwise:

        - ``minaic``: the lowest AIC across orders 1 to ``max_order``
        - ``aicopt``: the order with the lowest AIC
        - ``minlossfn``: the lowest loss function across orders
        - ``lossfnopt``: the order with the lowest loss function
        - ``meandiffaic``: the mean change in AIC when the order increases by one
        - ``maxdiffaic``: the largest increase in AIC when the order increases by one
        - ``mindiffaic``: the largest decrease (most negative change) in AIC when the order
          increases by one
        - ``ndownaic``: the number of order increases at which the AIC decreases

        If some orders cannot be fitted, these are taken over the orders that can; the change
        statistics use only adjacent pairs of orders that both fitted.

    Notes
    -----
    The AIC is normalized by the series length, ``log(loss) + 2 d/N``, where ``d`` is
    three times the order (the number of parameters of the model once the freedom in the choice
    of coordinates is removed, with the initial state), as MATLAB's ``aic``.
    """
    y = np.asarray(y, dtype=float).ravel()
    n_obs = len(y)
    lossfns = np.full(max_order, np.nan)
    aics = np.full(max_order, np.nan)
    for k in range(1, max_order + 1):
        try:
            fit = _n4_state_space(y, k)
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.warning(f'State-space model fitting failed for k = {k}: {err}')
            continue
        lossfns[k - 1] = fit['loss']
        aics[k - 1] = np.log(fit['loss']) + 2 * 3 * k / n_obs

    if np.all(np.isnan(aics)):
        return np.nan
    out = {}
    out['minaic'] = np.nanmin(aics)
    out['aicopt'] = int(np.nanargmin(aics)) + 1
    out['minlossfn'] = np.nanmin(lossfns)
    out['lossfnopt'] = int(np.nanargmin(lossfns)) + 1
    daics = np.diff(aics)
    daics = daics[~np.isnan(daics)]
    if daics.size == 0:
        out['meandiffaic'] = out['maxdiffaic'] = out['mindiffaic'] = np.nan
    else:
        out['meandiffaic'] = np.mean(daics)
        out['maxdiffaic'] = np.max(daics)
        out['mindiffaic'] = np.min(daics)
    out['ndownaic'] = int(np.sum(daics < 0))
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

def _fpe_stats(fpes: np.ndarray) -> dict:
    """Spread statistics of the final prediction errors across segments (``fpe_*`` outputs)."""
    return {'fpe_std': np.std(fpes, ddof=1), 'fpe_mean': np.mean(fpes), 'fpe_max': np.max(fpes),
            'fpe_min': np.min(fpes), 'fpe_range': np.ptp(fpes)}


def fit_subsegments(y: ArrayLike, model: str = 'ss', order: Union[int, list, None] = 2,
                    subset_how: str = 'rand', sample_p: Union[list, tuple, int] = (20, 0.1),
                    random_seed: Union[int, str, None] = 'default') -> dict:
    """
    Robustness of model parameters across different segments of a time series.

    The spread of parameters obtained (including in-sample goodness of fit statistics)
    provides some indication of stationarity. Values of goodness of fit provide some
    indication of model suitability. Inherits strongly from :func:`compare_test_sets`
    (hctsa's ``MF_FitSubsegments``).

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
        - 'ss': fits a state-space model of the given order by subspace identification
            (``n4sid``; the order can be ``'best'``). Outputs are how Akaike's final
            prediction error (``fpe_*``) varies across segments.
        - 'arma': fits an ARMA model by prediction-error minimization (``armax``; ``order``
            is ``[p, q]``). Outputs are how the FPE (``fpe_*``) and the fitted AR (``p_k_*``)
            and MA (``q_k_*``) coefficients vary across segments. (Deregistered in hctsa,
            as it is much like 'ar' and slow.)

        Default is ``'ss'``.

    order : int or two-vector, optional
        The order of the model to fit (used for 'ar', 'ss', or 'arma' models; a two-element
        vector ``[p, q]`` for 'arma'). Default is 2.
    subset_how : str, optional
        How to choose segments from the time series, either:

        - 'uniform' (evenly spaced)
        - 'rand' (at random).

        Default is ``'rand'``.

    sample_p : list, tuple or int, optional
        A two-vector specifying how many segments to take and of what length.
        Of the form [n_samples, length], where length can be a proportion of the time-series length.
        For example, [20, 0.1] takes 20 segments of 10% the time-series length.
        For ``model='arcrosspred'``, an integer (or length-1 list): the number of
        non-overlapping segments to partition the series into.
        Default is [20, 0.1].
    random_seed : int, 'default', 'none' or None, optional
        How to reset the random seed that picks the segment starts when ``subset_how`` is
        ``'rand'``, as hctsa's ``BF_ResetSeed``: an integer seed, or ``'default'`` for seed
        0, seeding a Mersenne Twister so that the draws are MATLAB's; ``'none'`` or
        ``None`` for a generator that is not reset. Default is ``'default'``.

    Returns
    -------
    dict
        Dictionary of statistics on the spread and mean of fitted model parameters 
        and goodness of fit across segments. For ``'ar'``, ``'ss'`` and ``'arma'``,
        ``fpe_std``, ``fpe_mean``, ``fpe_max``, ``fpe_min``, ``fpe_range`` (of the final
        prediction error); for ``'ar'`` ``a_k_std``, ``a_k_mean``, ``a_k_max``, ``a_k_min`` for each
        lag ``k``, and for ``'arma'`` the same for ``p_k_*`` and ``q_k_*``. For ``'arcrosspred'``: ``std``, ``range``,
        ``iqr`` (over all entries of the cross-prediction error matrix), ``stdoffdiag``,
        ``rangeoffdiag``, ``iqroffdiag`` (over the positive off-diagonal entries),
        ``stdmean``, ``rangemean``, ``stdmedian``, ``rangemedian`` (across predicted
        segments, of the mean and of the median error), ``rangerange``, ``stdrange``,
        ``rangestd``, ``stdstd`` (across predicted segments, of the range or standard
        deviation of the errors) and ``mineig`` (smallest real part of the eigenvalues
        of the matrix).
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    sample_p = np.atleast_1d(sample_p)
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
        if sample_p[1] < 1:  # specified a fraction of time series
            l = int(np.floor(N * sample_p[1]))
        else:  # specified an absolute interval
            l = int(sample_p[1])
        # reset the random seed (BF_ResetSeed), then numPred random starting points (randi)
        rng = _seeded_rng(random_seed)
        spts = 1 + np.floor((N - l + 1) * rng.random_sample(num_pred)).astype(int)
        r = np.column_stack([spts, spts + l - 1])
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
        out.update(_fpe_stats(fpes))
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
    elif model == 'ss':
        # state-space models of the specified order: statistics on goodness of fit
        fpes = np.zeros(num_pred)
        for i in range(num_pred):
            seg = y[r[i, 0] - 1:r[i, 1]]
            try:
                fit = _n4_state_space(seg, order if isinstance(order, str) else int(order))
            except (np.linalg.LinAlgError, ValueError) as err:
                raise ValueError("Couldn't fit this state space model") from err
            fpes[i] = _n4_fpe(fit['loss'], fit['order'], len(seg))
        out.update(_fpe_stats(fpes))
    elif model == 'arma':
        # ARMA models of the specified orders: goodness of fit, and the AR (p) and MA (q) coefficients
        p_ord, q_ord = int(order[0]), int(order[1])
        fpes = np.zeros(num_pred)
        ps = np.zeros((num_pred, p_ord + 1))
        qs = np.zeros((num_pred, q_ord + 1))
        for i in range(num_pred):
            seg = y[r[i, 0] - 1:r[i, 1]]
            try:
                ps[i], qs[i], loss = _armax_fit(seg, p_ord, q_ord)[:3]
            except (np.linalg.LinAlgError, ValueError) as err:
                raise ValueError("Couldn't fit this ARMA model") from err
            n_par = p_ord + q_ord
            fpes[i] = loss * (1 + n_par / len(seg)) / (1 - n_par / len(seg))
        out.update(_fpe_stats(fpes))
        for letter, coefs in (('p', ps), ('q', qs)):
            for i in range(1, coefs.shape[1]):  # (the first column is 1)
                out[f'{letter}_{i}_std'] = np.std(coefs[:, i], ddof=1)
                out[f'{letter}_{i}_mean'] = np.mean(coefs[:, i])
                out[f'{letter}_{i}_max'] = np.max(coefs[:, i])
                out[f'{letter}_{i}_min'] = np.min(coefs[:, i])
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
        the last four, ``_stdn``; ``sws_fexp_b`` (the rate of an exponential fit
        a*exp(b*l) + c to the ``sws`` curve: negative for a decay with training length),
        ``sws_fexp_r2`` (between 0 and 1), ``sws_fexp_adjr2`` and ``sws_fexp_rmse`` of that
        fit. The fit is the global least-squares optimum over b, with a and c found by
        linear least squares (:func:`pyhctsa.robust.bf_exp_fit`); a and c are not output
        because they are poorly determined when the curve is close to a straight line.
        The four ``sws_fexp_*`` outputs are NaN if the ``sws`` curve is constant.
        ``stde_peakpos`` (1-based position in the list
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
            # global least-squares exponential fit f(l) = a exp(b l) + c to the sws curve
            f_exp = bf_exp_fit(train_length_range.astype(float), curve, True)
            out['sws_fexp_b'] = f_exp['b']
            out['sws_fexp_r2'] = f_exp['r2']
            out['sws_fexp_adjr2'] = f_exp['adjr2']
            out['sws_fexp_rmse'] = f_exp['rmse']

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

        NaN for a (nearly) exactly predictable series: when the fitted noise variance is
        below 1e-12 of the variance of the series, the design is singular and the fitted
        coefficients and residuals are not meaningful (e.g., an exact sinusoid with p > 2).
    """
    y = np.asarray(y, dtype=float).ravel()
    p = int(p)
    n = len(y)
    # covariance method: least squares fit of y(t) on its p past values, over t = p+1, ..., N
    # (a minimum-norm least-squares solve, so that a singular design is fitted exactly rather
    # than regularized)
    x_design = np.column_stack([y[p - k:n - k] for k in range(1, p + 1)])
    phi = np.linalg.lstsq(x_design, y[p:], rcond=None)[0]
    noise_var = np.sum((y[p:] - x_design @ phi) ** 2) / (n - p)
    var_y = np.var(y, ddof=1)
    if noise_var < 1e-12 * var_y or not var_y > 0:
        return np.nan
    a = np.concatenate(([1], -phi))
    out = {}
    out['noisevar'] = noise_var
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

        NaN if the series is too short for ARFIT, and for a (nearly) exactly predictable
        series (estimated noise variance below 1e-12 of the variance of the series, e.g. an
        exact sinusoid): the coefficients are not determined and the residuals are
        rounding noise.
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
    # An exactly predictable series has a singular design and a noise variance at the level
    # of rounding error: nothing to report
    var_y = np.var(y, ddof=1)
    if Cest < 1e-12 * var_y or not var_y > 0:
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

def _ml_std(a: np.ndarray) -> float:
    """MATLAB's ``std``: the sample standard deviation, 0 for a single element."""
    return float(np.std(a, ddof=1)) if np.size(a) > 1 else 0.0


def _ml_max(a, axis=None):
    """MATLAB ``max``: NaNs are omitted (NaN only if every element is NaN)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmax(a, axis=axis)


def _ml_min(a, axis=None):
    """MATLAB ``min``: NaNs are omitted (NaN only if every element is NaN)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmin(a, axis=axis)


def _gp_learn_hyperp(tt: np.ndarray, yt: np.ndarray, cov, nfevals: int = -50,
                     hyp0: Union[np.ndarray, None] = None,
                     noise_pos: Union[tuple, list, None] = None) -> np.ndarray:
    """
    learn GP hyperparameters for the time series ``(tt, yt)``.

    The GP is a mean-zero process with a Gaussian likelihood and exact Gaussian
    inference (gpml ``infGaussLik``, as hctsa's ``MF_GP_LearnHyperp`` uses);
    ``nfevals`` is negative, so it caps the number of function evaluations rather
    than the number of line searches.

    Returns the flattened hyperparameter vector ``[cov..., lik]`` -- gpml
    unwraps the hyperparameter struct with its fields alphabetised (cov, lik,
    mean), and the mean is empty for a mean-zero process.

    As in gpml's ``gp``, a failed inference (e.g., a covariance that loses
    positive definiteness) gives a NaN marginal likelihood with zero
    derivatives, which ``minimize`` treats as a bad step and backs away from, so
    it does not abort the fit. ``numpy.linalg.LinAlgError`` is only raised if
    that leaves non-finite hyperparameters, the counterpart of hctsa returning
    NaN.

    ``hyp0`` is the initial hyperparameter vector ``[cov..., lik]``; the default is
    the initialization for ``covSum{covSEiso, covNoise}`` (``cov`` must then be
    :class:`CovSEisoNoise`), see :func:`_gp_init_hyp` for the other covariances.

    Noise floor (as hctsa's ``MF_GP_LearnHyperp``): the noise standard deviations (the Gaussian
    likelihood's, and those of the ``covNoise`` terms, at the 0-based positions ``noise_pos``
    among the covariance hyperparameters; ``(2,)`` for :class:`CovSEisoNoise`) are bounded
    below by 1% of the standard deviation of the data. Without a bound the marginal
    likelihood of a smooth series is nearly flat along a valley in which the noise runs to
    e^-12 or less, so the fitted noise depends on where the optimizer stops. The bound is a
    clamp of those hyperparameters inside the objective, with a zero gradient for a clamped
    coordinate.
    """
    nhps = cov.n_hyp
    if noise_pos is None:
        if cov is not CovSEisoNoise:
            raise ValueError('noise_pos is required for this covariance function')
        noise_pos = (2,)
    noise_pos = np.asarray(noise_pos, dtype=int)
    # Initial values, set component by component as in MF_GP_LearnHyperp for
    # covSum{covSEiso, covNoise}: the SE length scale is in the ballpark of the
    # difference between time elements, its log-magnitude starts at zero, the noise
    # covariance at log(0.1), and so does the likelihood noise.
    if hyp0 is None:
        hyp0 = np.array([np.log(np.mean(np.diff(tt))), 0.0, np.log(0.1), np.log(0.1)])
        assert nhps == 3

    with np.errstate(divide='ignore'):
        noise_floor = np.log(0.01 * np.std(yt, ddof=1))  # lower bound on the log noise standard deviations

    def _clamp(theta):
        theta = np.array(theta, dtype=float)
        theta[nhps] = max(theta[nhps], noise_floor)
        theta[noise_pos] = np.maximum(theta[noise_pos], noise_floor)
        return theta

    def _nlz(theta):
        # gpml's negative log marginal likelihood with the noise hyperparameters clamped at
        # the floor (and zero gradient wherever they are clamped)
        clamp_lik = theta[nhps] < noise_floor
        clamp_cov = noise_pos[theta[noise_pos] < noise_floor]
        theta = _clamp(theta)
        hyp = {'cov': theta[:nhps], 'lik': theta[nhps], 'mean': np.zeros(0)}
        nlZ, dnlZ = gp_train(hyp, cov, tt, yt)   # NaN if the inference fails (as gp.m)
        d_cov = np.array(dnlZ['cov'], dtype=float)
        d_lik = np.array(dnlZ['lik'], dtype=float)
        d_cov[clamp_cov] = 0
        if clamp_lik:
            d_lik[:] = 0
        return nlZ, np.concatenate([d_cov, d_lik, dnlZ['mean']])

    theta, _, _ = minimize(_clamp(hyp0), _nlz, nfevals)
    theta = _clamp(theta)  # (clamped coordinates can drift: same objective value)
    if not np.all(np.isfinite(theta)):
        raise np.linalg.LinAlgError('GP hyperparameters are not finite')
    return theta


def _gp_noise_pos(components: list) -> list:
    """
    0-based positions of the ``covNoise`` standard deviations among the covariance
    hyperparameters, found as ``MF_GP_LearnHyperp`` does (while it sets the initial values):
    a degree-parameterized component (``covMaterniso``) advances the position by one only, so
    for ``covMaterniso3_covNoise`` the position found for the noise is the Matern's second
    hyperparameter (hctsa's convention, as in :func:`_gp_init_hyp`).
    """
    pos = 0
    noise = []
    for name, degree in components:
        if degree is not None:
            pos += 1
        elif name == 'covSEiso':
            pos += 2
        elif name in ('covPeriodic', 'covRQiso'):
            pos += 3
        elif name == 'covNoise':
            noise.append(pos)
            pos += 1
        else:
            pos += 1
    return noise


def _gp_cov(cov_func) -> tuple:
    """
    The covariance function for ``cov_func``, a function giving the initial
    hyperparameters for a set of times, the 0-based positions of its noise standard deviations
    (:func:`_gp_noise_pos`) and its components (a list of ``(name, degree)``). The default
    ``'covSEiso_covNoise'`` is the closed-form :class:`CovSEisoNoise` (initialized by
    :func:`_gp_learn_hyperp`); others are built by ``parse_cov``, initialized by
    :func:`_gp_init_hyp`.
    """
    from ..toolboxes.matlab.gpml.cov import parse_cov
    if isinstance(cov_func, str) and cov_func == 'covSEiso_covNoise':
        return CovSEisoNoise, (lambda tt: None), [2], [('covSEiso', None), ('covNoise', None)]
    cov, components = parse_cov(cov_func)
    return cov, (lambda tt: _gp_init_hyp(components, tt)), _gp_noise_pos(components), components


def gp_fit_across(y: ArrayLike, cov_func: str = 'covSEiso_covNoise',
                  npoints: int = 20) -> dict:
    """
    Gaussian Process time-series modeling for local prediction.

    Trains a Gaussian Process model on equally-spaced points throughout the time
    series and uses the model to predict its intermediate values. The hyperparameters
    are learned by maximizing the marginal likelihood; the noise standard deviation is
    bounded below by 1% of that of the data (see :func:`_gp_learn_hyperp`).

    Parameters
    ----------
    y : array-like
        The input time series.
    cov_func : str or list
        The covariance function, a gpml ``covSum``: the names of its components joined
        with underscores (``'covSEiso_covNoise'``, ``'covSEiso_covPeriodic_covNoise'``,
        ``'covMaterniso3_covNoise'``, ``'covRQiso_covNoise'``) or in the gpml form
        ``['covSum', ['covSEiso', 'covNoise']]``. The only configuration hctsa
        instantiates is the default, ``'covSEiso_covNoise'`` (squared exponential plus noise).
    npoints : int
        The number of points through the time series to fit the GP model to.
        Default is 20.

    Returns
    -------
    dict
        Dictionary summarising the error and the fitted hyperparameters:

        - ``stde``: the root-mean-square error of the predictive mean, compared
          with the series,
        - ``meanabs_std``: the mean absolute error of the predictive mean, in
          units of the predictive standard deviation at each time,
        - ``stdmu``: the standard deviation of the predictive mean,
        - ``meanS``, ``stdS``: the mean and standard deviation of the predictive
          standard deviation,
        - ``nlml``: the negative log marginal likelihood (gpml's ``nlZ``) of the
          whole series (or of the 2000 resampled points), divided by the number
          of points so it does not grow with the series length,
        - ``logh1``, ``logh2``, ...: the log hyperparameters of the covariance
          function (for ``'covSEiso_covNoise'``: length scale, signal amplitude,
          noise standard deviation),
        - ``h_lonN``: the fitted length scale divided by the series length (only for
          ``'covSEiso_covNoise'``).

        All values are NaN if the fit fails.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    npoints = int(npoints)

    cov, init_hyp, noise_pos, components = _gp_cov(cov_func)
    nhps = cov.n_hyp
    is_se_noise = [c[0] for c in components] == ['covSEiso', 'covNoise'] and \
        all(c[1] is None for c in components)
    nan_out = {k: np.nan for k in
               ('stde', 'meanabs_std', 'stdmu', 'meanS', 'stdS', 'nlml',
                *(f'logh{i + 1}' for i in range(nhps)),
                *(('h_lonN',) if is_se_noise else ()))}

    tt = np.floor(_linspace(1, N, npoints))
    yt = y[tt.astype(int) - 1]

    try:
        theta = _gp_learn_hyperp(tt, yt, cov, hyp0=init_hyp(tt), noise_pos=noise_pos)
    except np.linalg.LinAlgError:
        logger.warning('Lack of positive definite matrix for this time series')
        return nan_out

    loghyper = theta[:nhps]
    hyp = {'cov': loghyper, 'lik': theta[nhps], 'mean': np.zeros(0)}

    # Evaluate over the whole space now, predicting at the test times ts
    if N <= 2000:
        ts = np.arange(1, N + 1, dtype=float)
    else:  # memory constraints force us to crudely resample
        ts = np.floor(_linspace(1, N, 2000) + 0.5)  # MATLAB round()
    y_ts = y[ts.astype(int) - 1]

    try:
        mu, S2, _, _ = gp_predict(hyp, cov, tt, yt, ts)
    except np.linalg.LinAlgError:
        logger.warning('Gaussian process regression failed for this time series')
        return nan_out

    # Output statistics
    S = np.sqrt(S2)  # standard deviation function, S
    out = {}
    # rms error from mean function, mu
    out['stde'] = np.sqrt(np.mean((y_ts - mu) ** 2))
    out['meanabs_std'] = np.mean(np.abs(y_ts - mu) / S)
    out['stdmu'] = np.std(mu, ddof=1)
    out['meanS'] = np.mean(S)
    out['stdS'] = np.std(S, ddof=1)

    # Negative log marginal likelihood per point (gpml's nlZ divided by the number of
    # points, so that it does not grow with the number of points, up to 2000)
    try:
        out['nlml'] = gp_train(hyp, cov, ts, y_ts, want_dnlZ=False)[0] / len(ts)
    except Exception:
        out['nlml'] = np.nan

    # Log-hyperparameters
    for i in range(nhps):
        out[f'logh{i + 1}'] = loghyper[i]

    # Give extra output based on length parameter on length of time series
    # (only for the squared exponential plus noise covariance)
    if is_se_noise:
        out['h_lonN'] = np.exp(loghyper[0]) / N

    return out


def gp_local_prediction(y: ArrayLike, cov_func: str = 'covSEiso_covNoise',
                        num_train: int = 20, num_test: int = 5,
                        num_preds: int = 10, pmode: str = 'frombefore',
                        random_seed: int = 0) -> dict:
    """
    Gaussian Process time-series model for local prediction.

    Fits a Gaussian Process model to a section of the time series and uses it to
    predict the subsequent datapoints, repeated at equally-spaced positions
    through the time series. The noise standard deviation of each fit is bounded below by
    1% of that of its training data (see :func:`_gp_learn_hyperp`). Windows whose training
    data are constant (standard deviation below 1e-8 of that of the series) cannot be
    standardized and are left out of every statistic.

    Parameters
    ----------
    y : array-like
        The input time series.
    cov_func : str or list
        The covariance function, a gpml ``covSum``: the names of its components joined
        with underscores (``'covSEiso_covNoise'``, ``'covSEiso_covPeriodic_covNoise'``,
        ``'covMaterniso3_covNoise'``, ``'covRQiso_covNoise'``) or in the gpml form
        ``['covSum', ['covSEiso', 'covNoise']]``. The only configuration hctsa
        instantiates is the default, ``'covSEiso_covNoise'`` (squared exponential plus noise).
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
        the obtained hyperparameter values, and the marginal likelihoods. Each
        window is first standardized using its training data. The error bars are
        95% error bars (twice the predictive standard deviation), so the outputs
        ending in ``_std`` are in units of these.

        - ``meanabs``, ``maxabs``, ``minabs``: mean, maximum and minimum over
          all predicted points of the absolute prediction error,
        - ``meanabs_std``, ``maxabs_std``, ``minabs_std``: the same, in units of
          the error bar,
        - ``meanabs_run``, ``maxabs_run``, ``minabs_run``: mean, maximum and
          minimum over windows of the mean absolute error in a window,
        - ``meanabs_std_run``, ``maxabs_std_run``, ``minabs_std_run``: the same,
          in units of the error bar,
        - ``maxerrbar``, ``meanerrbar``, ``minerrbar``: maximum, mean and
          minimum error-bar half-width over all predicted points,
        - ``meanlogh1``, ..., ``stdlogh1``, ...: mean and standard deviation
          across windows of each log hyperparameter,
        - ``maxnlml``, ``minnlml``, ``stdnlml``: maximum, minimum and standard
          deviation across windows of the negative log marginal likelihood on
          the window's training data, divided by the number of training points.

        All values are NaN if hyperparameters cannot be learned, or if the training data
        of every window are constant.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    num_train, num_test, num_preds = int(num_train), int(num_test), int(num_preds)

    cov, init_hyp, noise_pos, _ = _gp_cov(cov_func)
    nhps = cov.n_hyp

    if pmode in ('frombefore', 'randomgap'):
        spns = np.floor(_linspace(1, N - (num_test + num_train), num_preds))
    elif pmode == 'beforeafter':
        spns = np.floor(_linspace(1, N - (num_test + num_train * 2), num_preds))
    else:
        raise ValueError(f"Unknown prediction mode {pmode!r}")
    spns = spns.astype(int)

    out_keys = (
        'maxabs_std', 'maxabs', 'minabs_std', 'minabs', 'meanabs_std',
        'meanabs', 'meanabs_std_run', 'meanabs_run', 'maxabs_std_run',
        'maxabs_run', 'minabs_std_run', 'minabs_run', 'maxerrbar',
        'meanerrbar', 'minerrbar',
        *(f'{s}logh{i + 1}' for i in range(nhps) for s in ('mean', 'std')),
        'maxnlml', 'minnlml', 'stdnlml',
    )

    mus = np.zeros((num_test, num_preds))        # predicted values
    stderrs = np.zeros((num_test, num_preds))    # standard errors on predictions
    yss = np.zeros((num_test, num_preds))        # test values
    nlmls = np.full(num_preds, np.nan)           # per-point negative log marginal likelihoods (NaN: skipped window)
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

        # A window whose training data are constant (to rounding error, relative to the
        # series) cannot be standardized and carries no information about a GP: skip it
        yt_mean, yt_std = np.mean(yt), np.std(yt, ddof=1)
        if not yt_std > 1e-8 * np.std(y, ddof=1):
            continue

        # Process to normalize scales (the same transformation for both sets)
        ys = (ys - yt_mean) / yt_std
        yt = (yt - yt_mean) / yt_std

        # (1) Learn hyperparameters from the training set
        try:
            theta = _gp_learn_hyperp(tt, yt, cov, hyp0=init_hyp(tt), noise_pos=noise_pos)
        except np.linalg.LinAlgError:
            logger.warning('Unable to learn hyperparameters for this time series')
            return {k: np.nan for k in out_keys}

        loghyper = theta[:nhps]
        loghypers[:, i] = loghyper
        hyp = {'cov': loghyper, 'lik': theta[nhps], 'mean': np.zeros(0)}

        # Negative log marginal likelihood for this model (gpml's nlZ divided by the
        # number of training points), with hyperparameters optimized over the
        # training data
        nlmls[i] = gp_train(hyp, cov, tt, yt, want_dnlZ=False)[0] / len(tt)

        # (2) Evaluate at the test points, based on the training time/data
        mu, S2, _, _ = gp_predict(hyp, cov, tt, yt, ts)

        mus[:, i] = mu                     # ~predicted values for time-series points
        stderrs[:, i] = 2 * np.sqrt(S2)    # ~errors on those predictions
        yss[:, i] = ys

    # Drop the skipped windows (those with constant training data)
    keep = ~np.isnan(nlmls)
    if not np.any(keep):
        return {k: np.nan for k in out_keys}
    mus, stderrs, yss = mus[:, keep], stderrs[:, keep], yss[:, keep]
    loghypers, nlmls = loghypers[:, keep], nlmls[keep]

    # (1) Prediction error measures
    allabserrs = np.abs(mus - yss)                 # absolute errors
    allstderrs = allabserrs / stderrs   # in units of 95% confidence-interval bars

    out = {}
    # Largest/smallest/mean error across all runs:
    out['maxabs_std'] = _ml_max(allstderrs)
    out['maxabs'] = _ml_max(allabserrs)
    out['minabs_std'] = _ml_min(allstderrs)
    out['minabs'] = _ml_min(allabserrs)
    out['meanabs_std'] = np.mean(allstderrs)
    out['meanabs'] = np.mean(allabserrs)

    # Summary of how it did on each run:
    stderr_run = np.mean(allstderrs, axis=0)
    abserr_run = np.mean(allabserrs, axis=0)

    out['meanabs_std_run'] = np.mean(stderr_run)
    out['meanabs_run'] = np.mean(abserr_run)
    out['maxabs_std_run'] = _ml_max(stderr_run)
    out['maxabs_run'] = _ml_max(abserr_run)
    out['minabs_std_run'] = _ml_min(stderr_run)
    out['minabs_run'] = _ml_min(abserr_run)

    # Error bar stats:
    out['maxerrbar'] = _ml_max(stderrs)     # largest error bar
    out['meanerrbar'] = np.mean(stderrs)   # mean error bar length
    out['minerrbar'] = _ml_min(stderrs)     # minimum error bar length

    # (2) Hyperparameter measures: mean and std for each hyperparameter
    for i in range(nhps):
        out[f'meanlogh{i + 1}'] = np.mean(loghypers[i, :])
        out[f'stdlogh{i + 1}'] = _ml_std(loghypers[i, :])

    # (3) Negative log marginal likelihood measures
    out['maxnlml'] = _ml_max(nlmls)
    out['minnlml'] = _ml_min(nlmls)
    out['stdnlml'] = _ml_std(nlmls)

    return out


def _arx_losses(y_train: np.ndarray, y_test: np.ndarray, orders) -> tuple:
    """
    Out-of-sample loss of AR models of a range of orders (MATLAB's ``arxstruc``).

    For each order ``p`` an AR(p) model (no mean, no windowing) is fitted by least
    squares to ``y_train`` and applied to ``y_test``. As in ``arxstruc``, the same
    points are scored for every order: the first ``max(orders) + 1`` samples of the
    training and test segments are excluded from the fit and from the sum of squared
    one-step prediction errors, which is nonetheless divided by the full test length.

    Returns the losses (one per order) and the test length.
    """
    m = int(np.max(orders)) + 1
    n_tr, n_te = len(y_train), len(y_test)
    if n_tr <= m or n_te <= m:
        raise ValueError('time series too short for the model orders')
    loss = np.zeros(len(orders))
    for i, p in enumerate(orders):
        p = int(p)
        X = np.column_stack([y_train[m - k:n_tr - k] for k in range(1, p + 1)])
        a = np.linalg.lstsq(X, y_train[m:], rcond=None)[0]
        Xe = np.column_stack([y_test[m - k:n_te - k] for k in range(1, p + 1)])
        loss[i] = np.sum((y_test[m:] - Xe @ a) ** 2) / n_te
    return loss, n_te


def compare_ar(y: ArrayLike, orders: ArrayLike = np.arange(1, 11),
               test_how: Union[float, str] = 'all') -> dict:
    """
    How the out-of-sample error of an AR model changes with its order.

    Fits autoregressive (AR) models of a range of orders and compares the loss of
    each (the sum of squared one-step prediction errors on the test segment divided
    by the test length) when the model fitted to a training segment is applied to a
    test segment (the counterpart of MATLAB's ``arxstruc`` and ``selstruc``).
    Statistics are taken over the loss as a function of model order, ``v``.

    The first ``max(orders) + 1`` points of the training and test segments are
    excluded from the fit and from the sum, so that every order is scored on the same
    points, but the sum is still divided by the full test length. The loss is
    therefore the mean squared error scaled by about
    ``1 - (max(orders) + 1) / (test length)``, the same for every order.

    With ``test_how = 'all'`` the models are tested on the data they were trained on,
    so the loss measures in-sample fit: it cannot rise with the model order, and
    features such as ``minv``, ``firstonmin`` and ``where01max`` mostly describe how
    fast the fit improves with order. Use a training fraction (e.g. 0.5) for a
    genuine out-of-sample comparison.

    Parameters
    ----------
    y : array-like
        The input time series.
    orders : array-like, optional
        The model orders to compare. Default is 1 to 10.
    test_how : float or str, optional
        A fraction of the time series to train on (the model is tested on the
        remaining portion), or ``'all'`` to train and test on all the data. Default
        is ``'all'``.

    Returns
    -------
    dict
        - ``maxv``, ``minv``, ``meanv``, ``medianv``: the maximum, minimum, mean and
          median of the loss over orders,
        - ``firstonmin``: the loss of the first order divided by the minimum loss,
        - ``maxonmed``: the maximum loss divided by the median loss,
        - ``meandiff``, ``stddiff``, ``maxdiff``, ``meddiff``: the mean, standard
          deviation, maximum absolute value and median of the change in loss from
          one order to the next,
        - ``minstdfromi``: the minimum (over starting orders ``i``) of the standard
          error of the loss over orders ``i`` onward,
          ``std(v[i:]) / sqrt(len(v) - i)``, ignoring zeros,
        - ``where01max``: the first position in the list of orders (from 1) from
          which that standard error is below 10% of its maximum (NaN if none),
        - ``whereen4``: the first position from which it is below 1e-4 (NaN if none),
        - ``best_n``: the order with the smallest loss,
        - ``aic_n``: the order that minimizes Akaike's Information Criterion,
          ``log(loss * (1 + 2 * order / test length))``,
        - ``bestaic``: the minimum value of that criterion over orders.

        NaN if the series is too short for the largest order.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    orders = np.atleast_1d(np.asarray(orders)).ravel().astype(int)

    if isinstance(test_how, str):
        if test_how != 'all':
            raise ValueError(f"Unknown testing set specifier '{test_how}'")
        y_train, y_test = y, y
    else:
        co = int(np.floor(N * test_how))  # cutoff
        y_train, y_test = y[:co], y[co:]

    try:
        v, n_test = _arx_losses(y_train, y_test, orders)
    except (ValueError, np.linalg.LinAlgError):
        logger.warning('Time series too short to compare AR models of these orders')
        return np.nan

    out = {}
    out['maxv'] = np.max(v)
    out['minv'] = np.min(v)
    out['meanv'] = np.mean(v)
    out['medianv'] = np.median(v)
    out['firstonmin'] = v[0] / np.min(v)
    out['maxonmed'] = np.max(v) / np.median(v)
    dv = np.diff(v)
    if len(dv) > 0:
        out['meandiff'] = np.mean(dv)
        out['stddiff'] = np.std(dv, ddof=1) if len(dv) > 1 else 0.0
        out['maxdiff'] = np.max(np.abs(dv))
        out['meddiff'] = np.median(dv)
    else:
        out['meandiff'] = out['stddiff'] = out['maxdiff'] = out['meddiff'] = np.nan

    # where does it steady off?
    nv = len(v)
    stdfromi = np.array([(np.std(v[i:], ddof=1) if nv - i > 1 else 0.0) / np.sqrt(nv - i)
                         for i in range(nv)])
    pos = stdfromi[stdfromi > 0]
    out['minstdfromi'] = np.min(pos) if len(pos) > 0 else np.nan
    w01 = np.flatnonzero(stdfromi < np.max(stdfromi) * 0.1)
    out['where01max'] = w01[0] + 1 if len(w01) > 0 else np.nan
    wen4 = np.flatnonzero(stdfromi < 1e-4)
    out['whereen4'] = wen4[0] + 1 if len(wen4) > 0 else np.nan

    # 'best' order measures (selstruc): by loss, and by AIC = log(loss (1 + 2 p / Nc))
    out['best_n'] = orders[np.argmin(v)]
    aic = np.log(v * (1 + 2 * orders / n_test))
    out['aic_n'] = orders[np.argmin(aic)]
    out['bestaic'] = np.min(aic)

    return out


def _kstep_residuals_ss(F: np.ndarray, K: np.ndarray, C: np.ndarray, y: np.ndarray,
                        steps: int) -> np.ndarray:
    """
    Errors, prediction minus data, of the ``steps``-ahead predictor of the innovations-form
    state-space model ``x(t+1) = F x(t) + K e(t)``, ``y(t) = C x(t) + e(t)`` (MATLAB's
    ``predict(m, y, steps)``).

    The one-step predictor state ``x(u)`` is updated from the data, ``x(u+1) = (F - K C) x(u) +
    K y(u)``, and the prediction of ``y(t)`` is ``C F^(steps-1) x(t - steps + 1)``. As in
    ``predict`` (``'InitialCondition'`` ``'e'``), the state at the first sample ``x(0)`` is the
    one that minimizes the squared prediction error, and the first ``steps - 1`` predictions
    are the free run ``C F^t x(0)`` from it (the predictor has no earlier information).
    """
    n = len(y)
    r = F.shape[0]
    K = np.reshape(K, (r, 1))
    Cr = np.reshape(C, (1, r))
    Phi = F - K @ Cr                       # state update of the one-step predictor
    CFk = Cr @ np.linalg.matrix_power(F, steps - 1)
    # states x(u) = xz(u) + Phi^u x(0), u = 0..n-1: zero initial state response and homogeneous part
    xz = np.zeros((n, r))
    Pw = np.zeros((n, r, r))               # Phi^u
    x, P = np.zeros(r), np.eye(r)
    for u in range(n):
        xz[u], Pw[u] = x, P
        x = Phi @ x + K[:, 0] * y[u]
        P = Phi @ P
    yz = np.zeros(n)
    B = np.zeros((n, r))
    for t in range(n):
        u = t - steps + 1
        if u >= 0:
            yz[t] = (CFk @ xz[u])[0]
            B[t] = (CFk @ Pw[u])[0]
        else:                              # free run from the initial state
            B[t] = (Cr @ np.linalg.matrix_power(F, t))[0]
    x0 = np.linalg.lstsq(B, y - yz, rcond=None)[0]
    return yz + B @ x0 - y


def _kstep_residuals(a: np.ndarray, c: np.ndarray, y: np.ndarray, steps: int) -> np.ndarray:
    """
    Errors, prediction minus data, of the ``steps``-ahead predictor of the polynomial model
    ``a(q) y(t) = c(q) e(t)`` (MATLAB's ``predict(m, y, steps)``).

    ``a`` and ``c`` are the coefficient vectors including the leading 1. The model is put in
    innovations form (observer canonical form, ``x(t+1) = F x(t) + K e(t)``, ``y(t) = x_1(t) +
    e(t)``) and passed to :func:`_kstep_residuals_ss`.
    """
    r = max(len(a), len(c)) - 1
    if r == 0:
        return -np.asarray(y, dtype=float)
    ap = np.r_[a, np.zeros(r + 1 - len(a))]
    cp = np.r_[c, np.zeros(r + 1 - len(c))]
    F = np.zeros((r, r))
    F[:, 0] = -ap[1:]
    F[:-1, 1:] = np.eye(r - 1)
    C = np.zeros(r)
    C[0] = 1.0
    return _kstep_residuals_ss(F, cp[1:] - ap[1:], C, y, steps)


def _fit_predictor_model(y: np.ndarray, model: str, order):
    """
    Fit the model of MF_steps_ahead / MF_CompareTestSets to the whole series ``y``.

    ``model`` is ``'ar'`` (forward-backward least squares as MATLAB's ``ar``; ``order`` an
    integer, or ``'best'`` for the order from 1 to 10 chosen by Schwarz's Bayesian criterion,
    ARFIT), ``'arma'`` (``armax``; ``order`` is ``[p, q]``) or ``'ss'`` (``n4sid``; ``order``
    an integer or ``'best'``). Returns a function ``predict_errors(y_seg, steps)`` giving the
    errors, prediction minus data, of the model's ``steps``-ahead predictions of a series
    (MATLAB's ``predict(m, y_seg, steps)``, with the initial state estimated), or None if the
    fit fails.
    """
    if model == 'ss':
        try:
            fit = _n4_state_space(y, order if isinstance(order, str) else int(order))
        except (np.linalg.LinAlgError, ValueError):
            return None
        return lambda y_seg, steps: _kstep_residuals_ss(fit['A'], fit['K'], fit['C'], y_seg, steps)
    if model == 'ar':
        if isinstance(order, str) and order == 'best':
            try:
                order = len(_arfit(y, 1, 10, 'sbc', zero=True)[1])
            except ValueError:
                return None
        try:
            a, c = np.r_[1.0, _ar_fb(y, int(order))[0]], np.ones(1)
        except np.linalg.LinAlgError:
            return None
    elif model == 'arma':
        try:
            a, c = _armax_fit(y, int(order[0]), int(order[1]))[:2]
        except (np.linalg.LinAlgError, ValueError):
            return None
    else:
        raise ValueError(f"Unknown model '{model}'")
    return lambda y_seg, steps: _kstep_residuals(a, c, y_seg, steps)


def steps_ahead(y: ArrayLike, model: str = 'ar', order: Union[int, str, list] = 2,
                max_steps: int = 6) -> dict:
    """
    How the accuracy of multi-step-ahead model predictions compares with trivial
    predictors and changes with the horizon.

    Given a model, characterizes the variation in goodness of model predictions across
    a range of prediction lengths, ``l``, from 1-step-ahead to ``max_steps``-steps-ahead
    predictions. The model is fitted on the full time series and then used to predict the
    same data (so all predictions are within the sample).

    At each horizon, the errors of the model are compared with those of three trivial
    predictors: (i) the value ``l`` samples earlier (a sliding mean of length 1),
    (ii) the average of the last two values, iterated forward ``l`` steps (a sliding mean
    of length 2), and (iii) the mean of the full time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    model : {'ar', 'arma', 'ss'}, optional
        The time-series model to fit: an AR model (forward-backward least squares, as
        MATLAB's ``ar``), an ARMA model (``armax``), or a state-space model (``n4sid``). Default is ``'ar'``.
        The predictions of the fitted model are those of MATLAB's ``predict(m, y, l)``, with the initial state of
        the predictor estimated.
    order : int, 'best' or two-vector, optional
        The order of the model to fit: an integer for ``'ar'`` and ``'ss'``, a two-vector
        ``[p, q]`` for ``'arma'``, or ``'best'``. For ``'ar'``, ``'best'`` picks the order
        (1 to 10) by Schwarz's Bayesian criterion using ARfit; for ``'ss'``, n4sid chooses
        the order from 1 to 10 by a gap rule on its Hankel singular values. Default is 2.
    max_steps : int, optional
        The maximum number of steps ahead to predict. Default is 6.

    Returns
    -------
    dict
        - ``stde_h1``, ..., ``stde_h<max_steps>``: the root-mean-square error of the model
          at horizon ``l``, divided by the lowest root-mean-square error of the three
          trivial predictors at that horizon,
        - ``meanabs_h1``, ...: the same for the mean absolute error,
        - ``ac1_h1``, ...: the absolute lag-1 autocorrelation of the model's errors at each
          horizon (not a ratio),
        - ``stde_meanabs_diff``: the absolute value of the mean difference between the
          model's root-mean-square and mean absolute errors across horizons,
        - ``stde_meandiff``, ``stde_maxdiff``, ``stde_stddiff``: the mean, maximum, and
          standard deviation of the change in the model's root-mean-square error from one
          horizon to the next,
        - ``stde_ndown``: the number of horizon steps at which the model's
          root-mean-square error falls.

        The last five outputs use the model's raw errors, not the ratios to the trivial
        predictors. NaN if the model cannot be fitted.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    max_steps = int(max_steps)
    if order is None:
        order = 2

    # Fit the model on the whole time series
    predict_errors = _fit_predictor_model(y, model, order)
    if predict_errors is None:
        return np.nan

    def model_residuals(k):
        return predict_errors(y, k)

    # Statistics of the predictions at each horizon
    mf_rms, mf_abs, mf_ac1 = (np.zeros(max_steps) for _ in range(3))
    sm1_rms, sm1_abs = np.zeros(max_steps), np.zeros(max_steps)
    sm2_rms, sm2_abs = np.zeros(max_steps), np.zeros(max_steps)
    for j in range(max_steps):
        i = j + 1

        # (1) *** Model ***
        mres = model_residuals(i)[i - 1:]
        mf_rms[j] = np.sqrt(np.mean(mres ** 2))
        mf_abs[j] = np.mean(np.abs(mres))
        mf_ac1[j] = np.ravel(autocorr(mres, 1, 'Fourier'))[0]

        # (2) *** Sliding mean 1 ***: predicts with the value i steps before it
        mres = y[i:] - y[:N - i]
        sm1_rms[j] = np.sqrt(np.mean(mres ** 2))
        sm1_abs[j] = np.mean(np.abs(mres))

        # (3) *** Sliding mean 2 ***: closed-form solution of the order-2 linear recurrence
        # p(n) = (p(n-1) + p(n-2)) / 2 that iterating the average of the last two values
        # i steps ahead converges to
        weights = np.array([1 + (-1) ** (i + 1) / 2 ** i, 2 + (-1) ** i / 2 ** i]) / 3
        sm2p = np.column_stack([y[:N - i - 1], y[1:N - i]]) @ weights
        mres = y[i + 1:] - sm2p
        sm2_rms[j] = np.sqrt(np.mean(mres ** 2))
        sm2_abs[j] = np.mean(np.abs(mres))

    # (global) sample mean predictor
    sminf_res = y - np.mean(y)
    sminf_rms = np.sqrt(np.mean(sminf_res ** 2))
    sminf_abs = np.mean(np.abs(sminf_res))

    out = {}
    for j in range(max_steps):
        # relative to the best null (dumb) predictor
        out[f'stde_h{j + 1}'] = mf_rms[j] / min(sm1_rms[j], sm2_rms[j], sminf_rms)
        out[f'meanabs_h{j + 1}'] = mf_abs[j] / min(sm1_abs[j], sm2_abs[j], sminf_abs)
        # raw ac1 values -- ratios don't really make sense
        out[f'ac1_h{j + 1}'] = abs(mf_ac1[j])

    out['stde_meanabs_diff'] = abs(np.mean(mf_rms - mf_abs))

    # Quantify shape, other than being a boring increasing curve
    d = np.diff(mf_rms)
    out['stde_meandiff'] = np.mean(d)
    out['stde_maxdiff'] = np.max(d)
    out['stde_stddiff'] = np.std(d, ddof=1)
    out['stde_ndown'] = int(np.sum(d < 0))
    return out


def compare_test_sets(y: ArrayLike, the_model: str = 'ss', ord: Union[int, str, list] = 2,
                      subset_how: str = 'rand', sample_p: Union[list, tuple] = (20, 0.1),
                      steps: int = 2, random_seed: Union[int, str, None] = 0) -> dict:
    """
    How well a model fitted to the whole series predicts short stretches of it.

    Fits a time-series model to the full series, then uses it to predict a set of short
    test segments of the series (``steps`` samples ahead), and summarizes how the
    prediction quality varies across the segments. For each segment it records the
    root-mean-square prediction error, the lag-1 autocorrelation of the errors, the
    absolute difference between the mean prediction and the mean of the data, and the
    ratio of the standard deviations of the predictions and the data. It says something
    about stationarity in the spread of values, and about the suitability of the model in
    the level of values.

    Similar to :func:`fit_subsegments`, except that the model is fitted on the full time
    series and tested on different local segments. The predictions are those of MATLAB's
    ``predict(m, segment, steps)``, with the initial state of the predictor estimated for
    each segment.

    Parameters
    ----------
    y : array-like
        The input time series.
    the_model : {'ss', 'ar', 'arma'}, optional
        The type of time-series model to fit: a state-space model (``'ss'``), an AR model (``'ar'``) or an ARMA model (``'arma'``). Default is
        ``'ss'``.
    ord : int, 'best' or two-vector, optional
        The order of the model to fit (a two-element vector for ``'arma'``), or ``'best'``
        to select it automatically: for ``'ar'``, the order from 1 to 10 minimizing the
        Schwarz Bayesian criterion (ARFIT); for ``'ss'``, as chosen by n4sid. Default is 2.
    subset_how : {'rand', 'uniform'}, optional
        How to select the test segments: at random, or evenly spaced throughout the time
        series. Default is ``'rand'``.
    sample_p : two-vector, optional
        ``[number of segments, segment length]``. A segment length below 1 is a fraction of
        the series length, capped to between 10 and 20 samples (so ``[25, 0.1]`` takes 25
        segments of 10 to 20 samples); otherwise it is a number of samples. A single value
        (with ``'uniform'``) partitions the series into that many segments. Default is
        ``[20, 0.1]``.
    steps : int, optional
        The number of steps ahead to predict in each segment. Default is 2.
    random_seed : int, 'default', 'none' or None, optional
        Seed for the Mersenne Twister that picks the random segments, reset first as
        ``BF_ResetSeed`` does (0, or ``'default'``, is MATLAB's default); ``'none'`` or
        ``None`` leaves the stream alone. Default is 0.

    Returns
    -------
    dict
        - ``stde_mean``, ``stde_std``, ``stde_iqr``: the mean, standard deviation and
          interquartile range over segments of the root-mean-square prediction error,
        - ``ac1_mean``, ``ac1_median``: the absolute value of the mean, and of the median,
          over segments of the lag-1 autocorrelation of the prediction errors,
        - ``ac1_std``, ``ac1_iqr``: the standard deviation and interquartile range over
          segments of that autocorrelation,
        - ``meane_mean``, ``meane_std``, ``meane_iqr``: the mean, standard deviation and
          interquartile range over segments of the absolute difference between the mean
          prediction and the mean of the data,
        - ``stdrat_mean``, ``stdrat_median``, ``stdrat_std``, ``stdrat_iqr``: the mean,
          median, standard deviation and interquartile range over segments of the ratio of
          the standard deviation of the predictions to that of the data (segments in which
          the data are near-constant are excluded).

        NaN if the model cannot be fitted.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    sigma_y = np.std(y, ddof=1)  # used to flag degenerate (near-constant) test segments
    if ord is None:
        ord = 2
    sample_p = np.atleast_1d(np.asarray(sample_p, dtype=float))
    steps = int(steps)
    num_pred = int(sample_p[0])

    # Fit the model on the whole time series (the test sets are smaller chunks of it)
    predict_errors = _fit_predictor_model(y, the_model, ord)
    if predict_errors is None:
        return np.nan

    # Set the ranges of the test segments (1-based, inclusive)
    r = np.zeros((num_pred, 2), dtype=int)
    if subset_how in ('rand', 'uniform') and len(sample_p) > 1:
        if sample_p[1] < 1:  # a fraction of the time series, capped to between 10 and 20
            seg_len = int(max(min(20, np.floor(N * sample_p[1])), 10))
        else:  # an absolute interval
            seg_len = int(sample_p[1])
    if subset_how == 'rand':
        # reset the random seed (BF_ResetSeed), then numPred starting points
        rng = _seeded_rng(random_seed)
        spts = 1 + np.floor((N - seg_len + 1) * rng.random_sample(num_pred)).astype(int)  # randi
        r[:, 0] = spts
        r[:, 1] = spts + seg_len - 1
    elif subset_how == 'uniform':
        if len(sample_p) == 1:  # size will depend on number of unique subsegments
            spts = np.floor(_linspace(0, N, num_pred + 1) + 0.5).astype(int)  # MATLAB round()
            r[:, 0] = spts[:num_pred] + 1
            r[:, 1] = spts[1:]
        else:
            spts = np.floor(_linspace(1, N - seg_len + 1, num_pred) + 0.5).astype(int)
            r[:, 0] = spts
            r[:, 1] = spts + seg_len - 1
    else:
        raise ValueError(f"Unknown subset method '{subset_how}'")

    # Quickly check that ranges are valid
    if np.any(r[:, 0] >= r[:, 1]):
        raise ValueError('Invalid settings')

    # Do the series of predictions
    rmserrs = np.zeros(num_pred)
    ac1s = np.zeros(num_pred)
    meandiffs = np.zeros(num_pred)
    stdrats = np.zeros(num_pred)
    for i in range(num_pred):
        y_test = y[r[i, 0] - 1:r[i, 1]]
        # step-ahead predictions across the test set, using the model fitted to all the data
        mres = predict_errors(y_test, steps)  # prediction minus data
        yp = y_test + mres

        # statistics on the residuals
        rmserrs[i] = np.sqrt(np.mean(mres ** 2))
        ac1s[i] = np.ravel(autocorr(mres, 1, 'Fourier'))[0]

        # statistics on the output time series
        meandiffs[i] = abs(np.mean(yp) - np.mean(y_test))
        # near-constant test segments: the ratio of standard deviations is undefined, not just large
        if np.std(y_test, ddof=1) < 1e-6 * sigma_y:
            stdrats[i] = np.nan
        else:
            stdrats[i] = np.std(yp, ddof=1) / np.std(y_test, ddof=1)

    def iqr(x):
        return np.diff(matlab_quantile(x, [0.25, 0.75]))[0] if len(x) > 0 else np.nan

    def std(x):
        return np.std(x, ddof=1) if len(x) > 1 else (0.0 if len(x) == 1 else np.nan)

    def mean(x):
        return np.mean(x) if len(x) > 0 else np.nan

    def median(x):
        return np.median(x) if len(x) > 0 else np.nan

    out = {}
    out['stde_mean'] = mean(rmserrs)
    out['stde_std'] = std(rmserrs)
    out['stde_iqr'] = iqr(rmserrs)

    # absolute values of operations on the raw ac1s (not the absolute values of ac1s)
    out['ac1_mean'] = abs(mean(ac1s))
    out['ac1_median'] = abs(median(ac1s))
    out['ac1_std'] = std(ac1s)
    out['ac1_iqr'] = iqr(ac1s)

    # differences in mean between the predictions and the data
    out['meane_mean'] = mean(meandiffs)
    out['meane_std'] = std(meandiffs)
    out['meane_iqr'] = iqr(meandiffs)

    # ratio of standard deviations (omitting segments flagged as degenerate above)
    valid = stdrats[~np.isnan(stdrats)]
    out['stdrat_mean'] = mean(valid)
    out['stdrat_median'] = median(valid)
    out['stdrat_std'] = std(valid)
    out['stdrat_iqr'] = iqr(valid)

    return out


def hmm_compare_n_states(y: ArrayLike, train_p: float = 0.6,
                         n_states: ArrayLike = (2, 3, 4)) -> dict:
    """
    How the fit of hidden Markov models to the series changes with the number of hidden
    states.

    Fits Gaussian hidden Markov models (HMMs) with different numbers of states to the first
    ``train_p`` proportion of the time series (each with at most 30 cycles of EM), and
    compares the resulting log-likelihoods per sample on the training part and on the
    held-out remainder (hctsa's ``MF_hmm_CompareNStates``). Each model is fitted
    deterministically, by the best of six fixed starting points with a floor on the shared
    variance (:func:`_zg_hmm_fit`, as in :func:`hmm_fit`).

    Parameters
    ----------
    y : array-like
        The input time series.
    train_p : float, optional
        The initial proportion of the time series to train the model on. Default is 0.6.
    n_states : array-like of int, optional
        The numbers of states to compare. Default is 2 to 4.

    Returns
    -------
    dict
        - ``meanLLtrain``, ``maxLLtrain``: mean and maximum across models of the
          log-likelihood per sample on the training part,
        - ``meanLLtest``, ``maxLLtest``: the same on the test part,
        - ``chLLtrain``, ``chLLtest``: change in training and test log-likelihood per sample
          from the model with the fewest states to the one with the most,
        - ``meandiffLLtt``: mean across models of the absolute difference between the test
          and training log-likelihoods per sample,
        - ``LLtestdiff1``, ``LLtestdiff2``, ...: change in test log-likelihood per sample
          from the i-th to the (i+1)-th number of states in ``n_states``.

        NaN if the series is too short to train on.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    n_states = np.atleast_1d(np.asarray(n_states)).astype(int)
    n_train = int(np.floor(train_p * N))  # number of initial samples to train the model on

    if n_train >= N:
        raise ValueError(f'train_p = {train_p:g} leaves no test data for a series of length {N}')
    if n_train < 2:
        logger.warning(f'Time series (N = {N}) too short to train on {train_p:g} of it')
        return np.nan
    y_train = y[:n_train]
    y_test = y[n_train:]
    n_test = len(y_test)

    ll_trains = np.zeros(len(n_states))
    ll_tests = np.zeros(len(n_states))
    for j, k in enumerate(n_states):
        # train an HMM with k states for 30 cycles of EM (or until convergence)
        mu, cov, p_matrix, pi, LL = _zg_hmm_fit(y_train, k)
        ll_trains[j] = LL[-1] / n_train
        # log likelihood of the test data
        ll_tests[j] = _zg_hmm_loglik(y_test, mu, cov, p_matrix, pi) / n_test

    out = {}
    out['meanLLtrain'] = np.mean(ll_trains)
    out['meanLLtest'] = np.mean(ll_tests)
    out['maxLLtrain'] = np.max(ll_trains)
    out['maxLLtest'] = np.max(ll_tests)
    out['chLLtrain'] = ll_trains[-1] - ll_trains[0]
    out['chLLtest'] = ll_tests[-1] - ll_tests[0]
    out['meandiffLLtt'] = np.mean(np.abs(ll_tests - ll_trains))
    for i in range(len(n_states) - 1):
        out[f'LLtestdiff{i + 1}'] = ll_tests[i + 1] - ll_tests[i]
    return out


def _gp_init_hyp(components: list, tt: np.ndarray) -> np.ndarray:
    """
    Initial hyperparameters ``[cov..., lik]`` for ``MF_GP_LearnHyperp``, set component by
    component of the ``covSum``: the length scale at the typical time step, log-magnitudes at
    zero, the period of ``covPeriodic`` at a tenth of the time span (about ten cycles across
    the data), ``covRQiso``'s log-shape at zero, and ``covNoise`` and the likelihood noise at
    log(0.1).

    As in hctsa, a component with a degree (``covMaterniso``) is not initialized and advances
    the position by one only (not by its two hyperparameters), so the next component is
    written over the Matern's second hyperparameter and the vector is zero elsewhere.
    ``components`` is the list of ``(name, degree)`` from ``parse_cov``.
    """
    n_cov = int(sum({'covSEiso': 2, 'covPeriodic': 3, 'covRQiso': 3, 'covNoise': 1,
                     'covMaterniso': 2}[name] for name, _ in components))
    typical_dt = np.mean(np.diff(tt))  # typical time step: a length-scale prior
    data_span = np.max(tt) - np.min(tt)
    hyp = np.zeros(n_cov)
    pos = 0
    for name, degree in components:
        if degree is not None:  # degree-parameterized component: left at zero
            pos += 1
        elif name == 'covSEiso':
            hyp[pos] = np.log(typical_dt)       # length-scale
            hyp[pos + 1] = 0.0                  # log-magnitude
            pos += 2
        elif name == 'covPeriodic':
            hyp[pos] = np.log(typical_dt)       # length-scale
            hyp[pos + 1] = np.log(data_span / 10)  # period (guess: ~10 cycles across the data)
            hyp[pos + 2] = 0.0                  # log-magnitude
            pos += 3
        elif name == 'covRQiso':
            hyp[pos] = np.log(typical_dt)       # length-scale
            hyp[pos + 1] = 0.0                  # log-magnitude
            hyp[pos + 2] = 0.0                  # log-alpha (shape)
            pos += 3
        elif name == 'covNoise':
            hyp[pos] = np.log(0.1)              # noise magnitude
            pos += 1
        else:
            pos += 1                            # unrecognized component: leave at zero
    return np.r_[hyp, np.log(0.1)]


def _ml_randi(imax: int, rng: np.random.RandomState) -> int:
    """MATLAB's scalar ``randi(imax)``: ``1 + floor(imax * rand)``."""
    return 1 + int(np.floor(imax * rng.random_sample()))


def _ml_randsample(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """MATLAB's ``randsample(n, k)`` (without replacement): 1-based indices."""
    if 4 * k > n:
        return _ml_randperm(n, rng)[:k]
    selected = np.zeros(n, dtype=bool)
    out = np.zeros(k, dtype=int)
    nsel = 0
    while nsel < k:
        r = _ml_randi(n, rng)
        if not selected[r - 1]:
            selected[r - 1] = True
            out[nsel] = r
            nsel += 1
    return out


def gp_hyperparameters(y: ArrayLike, cov_func: Union[str, list] = 'covSEiso_covNoise',
                       squish_or_squash: int = 1, max_n: Union[int, float, str] = 500,
                       resample_how: str = 'resample',
                       random_seed: Union[int, str, None] = 0) -> dict:
    """
    Fits a Gaussian process to the series and reports its fitted kernel parameters and
    goodness of fit.

    Models the series as a smooth function of time using a Gaussian process (GP). A
    zero-mean GP with a Gaussian likelihood is fitted using the covariance function
    ``cov_func``, e.g., (i) a sum of squared exponential and noise terms, or (ii) a sum of
    squared exponential, periodic, and noise terms. The log hyperparameters are found by
    maximizing the marginal likelihood (at most 50 function evaluations), starting from a
    data-informed initial guess (:func:`_gp_init_hyp`). Goodness of fit is summarized by the
    per-point negative log marginal likelihood, the error of the fitted mean, and the GP's
    predictive standard deviation.

    Fitting is O(N^3), so the model is fitted to at most ``max_n`` samples from the time
    series, chosen by (i) resampling the time series down to this many points, (ii) taking
    the first ``max_n`` samples, or (iii) taking random samples. Times are the sample indices
    (``squish_or_squash = 1``), so length scales and periods are in samples of the cut
    series. The output is NaN if the fit fails or if the fitted mean is nearly constant
    (standard deviation below 0.01).

    Parameters
    ----------
    y : array-like
        The input time series (should be z-scored).
    cov_func : str or list, optional
        The covariance function: the names of the components of a gpml ``covSum``
        joined with underscores (``'covSEiso_covNoise'``, ``'covSEiso_covPeriodic_covNoise'``,
        ``'covMaterniso3_covNoise'``, ``'covRQiso_covNoise'``), or the gpml form
        ``['covSum', ['covSEiso', 'covNoise']]`` (``['covMaterniso', 3]`` for a component
        with a degree). Default is ``'covSEiso_covNoise'``.
    squish_or_squash : int, optional
        How to set the time index: if nonzero (default), t = 1, ..., N; if zero, t is
        spread across the unit interval.
    max_n : int, float or 'full', optional
        The maximum length of time series to consider -- longer inputs are cut down to
        ``max_n`` samples. A value below 1 is a proportion of the length. 0 or ``'full'``
        disables the cut and uses the whole series. Default is 500.
    resample_how : str, optional
        How to cut time series longer than ``max_n`` down to ``max_n`` points:

        - ``'resample'`` (default): resample the whole series down (``scipy.signal.resample_poly``),
        - ``'first'``: take the first ``max_n`` samples,
        - ``'random_i'``: take ``max_n`` random samples (unevenly spaced),
        - ``'random_consec'``: take ``max_n`` consecutive samples from a random position,
        - ``'random_both'``: take ``max_n`` consecutive samples from a random position, then
          a random fifth of them.
    random_seed : int, 'default', 'none' or None, optional
        Seed for the Mersenne Twister, reset first (as ``BF_ResetSeed``) for the settings of
        ``resample_how`` that use random numbers; ``'none'`` or ``None`` leaves the stream
        alone. Default is 0.

    Returns
    -------
    dict
        - ``logh1``, ``logh2``, ...: the log hyperparameters of the fitted covariance
          function, in the order of its components (the number depends on ``cov_func``):
          covSEiso: [log length scale, log amplitude]; covPeriodic: [log length scale,
          log period, log amplitude]; covMaterniso(3): [log length scale, log amplitude];
          covRQiso: [log length scale, log amplitude, log shape parameter alpha]; covNoise:
          [log noise standard deviation],
        - ``nlml``: the negative log marginal likelihood of the fitted model, divided by the
          number of points it was fitted to,
        - ``stde``: root-mean-square error of the GP mean at the sampled times,
        - ``meanabs_std``: mean absolute error of the GP mean, in units of the GP's
          predictive standard deviation at each sampled time,
        - ``std_mu_data``: standard deviation of the GP mean at the sampled times (if not
          close to one, the GP has not followed the z-scored data),
        - ``std_S_data``: standard deviation of the GP's predictive standard deviation at the
          sampled times,
        - ``maxS``, ``minS``, ``meanS``: maximum, minimum, and mean of the GP's predictive
          standard deviation over 1000 equally spaced times spanning the sampled series.
    """
    from scipy.signal import resample_poly
    from ..toolboxes.matlab.gpml.cov import parse_cov

    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    cov, components = parse_cov(cov_func)
    num_hps = cov.n_hyp

    if isinstance(max_n, str):
        if max_n.lower() != 'full':
            raise ValueError(f"Invalid max_n '{max_n}'")
        max_n = 0
    if 0 < max_n < 1:  # a proportion of the time series length
        max_n = int(np.ceil(N * max_n))
    max_n = int(max_n)

    def set_time_index(n):
        return np.arange(1, n + 1, dtype=float) if squish_or_squash else _linspace(0, 1, n)

    def reset_seed():  # BF_ResetSeed
        return _seeded_rng(random_seed)

    # Downsample long time series
    if max_n == 0:
        t = set_time_index(N)  # no resampling requested
    elif N > max_n:
        if resample_how == 'resample':  # resamples the whole time series down
            f = max_n / N
            y = resample_poly(y, int(np.ceil(f * 10000)), 10000)
            if len(y) > max_n:
                y = y[:max_n]
            N = len(y)
            t = set_time_index(N)
        elif resample_how == 'random_i':  # max_n random indices (unevenly spaced)
            t = set_time_index(N)
            rng = reset_seed()
            ii = np.sort(_ml_randsample(N, max_n, rng))
            t = t[ii - 1]
            t = (t - np.min(t)) / np.ptp(t) * (max_n - 1) + 1  # respace from 1:max_n
            y = y[ii - 1]
        elif resample_how == 'random_consec':  # max_n consecutive samples from a random position
            rng = reset_seed()
            sind = _ml_randi(N - max_n + 1, rng)  # start index
            y = y[sind - 1:sind - 1 + max_n]
            t = set_time_index(max_n)
        elif resample_how == 'first':  # the first max_n samples
            y = y[:max_n]
            t = set_time_index(max_n)
        elif resample_how == 'random_both':  # random start, then a random fifth of those samples
            rng = reset_seed()
            sind = _ml_randi(N - max_n + 1, rng)
            y = y[sind - 1:sind - 1 + max_n]
            N = len(y)
            t = set_time_index(N)
            ii = np.sort(_ml_randsample(N, int(np.ceil(max_n / 5)), rng))
            t = t[ii - 1]
            y = y[ii - 1]
        else:
            raise ValueError(f"Invalid sampling method '{resample_how}'.")
    else:
        t = set_time_index(N)

    # Learn the hyperparameters (mean-zero process, Gaussian likelihood, exact inference)
    try:
        theta = _gp_learn_hyperp(t, y, cov, hyp0=_gp_init_hyp(components, t),
                                 noise_pos=_gp_noise_pos(components))
    except np.linalg.LinAlgError:
        logger.warning('Lack of positive definite matrix for this time series')
        return np.nan
    log_hyper = theta[:num_hps]
    hyp = {'cov': log_hyper, 'lik': theta[num_hps], 'mean': np.zeros(0)}

    out = {}
    for i in range(num_hps):
        out[f'logh{i + 1}'] = log_hyper[i]

    # negative log marginal likelihood using the optimized hyperparameters, per point
    out['nlml'] = gp_train(hyp, cov, t, y, want_dnlZ=False)[0] / len(t)

    # mean error from the fit, evaluated at the data points
    try:
        mu, S2, _, _ = gp_predict(hyp, cov, t, y, t)
    except np.linalg.LinAlgError:
        return np.nan
    if np.std(mu, ddof=1) < 0.01:  # hasn't fit the time series well at all -- too constant
        logger.warning('This time series is not suited to Gaussian Process fitting')
        return np.nan

    # root-mean-square error of the mean function, mu
    out['stde'] = np.sqrt(np.mean((y - mu) ** 2))
    # better to look at the mean distance away in units of std
    out['meanabs_std'] = np.mean(np.abs((y - mu) / np.sqrt(S2)))
    out['std_mu_data'] = np.std(mu, ddof=1)  # std of the mean function at the datapoints
    out['std_S_data'] = np.std(np.sqrt(S2), ddof=1)  # should vary a fair bit

    # statistics on the predictive variance
    xstar = _linspace(np.min(t), np.max(t), 1000)
    try:
        _, S2, _, _ = gp_predict(hyp, cov, t, y, xstar)
    except np.linalg.LinAlgError:
        return np.nan
    S = np.sqrt(S2)  # standard deviation function (S2 is the variance)
    out['maxS'] = np.max(S)
    out['minS'] = np.min(S)
    out['meanS'] = np.mean(S)
    return out
