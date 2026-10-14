"""
Stepwise least-squares fitting of univariate autoregressive (AR) models, with order selection,
a residual whiteness test and the eigenmodes (periods, damping times, excitations) of the fit.

The methods are those of Neumaier & Schneider [1]_ for the regularized least-squares fit of all
orders from one QR factorization, the order criteria, the confidence intervals and the
eigenmodes; and of Li & McLeod [2]_ for the portmanteau test on the residuals.  This module
was written from those papers only.

References
----------
.. [1] A. Neumaier and T. Schneider, "Estimation of parameters and eigenmodes of multivariate
       autoregressive models", ACM Trans. Math. Softw. 27(1):27-57 (2001).
.. [2] W. K. Li and A. I. McLeod, "Distribution of the residual autocorrelations in
       multivariate ARMA time series models", J. R. Stat. Soc. B 43:231-239 (1981).
"""
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from scipy.stats import chi2, t as t_dist


def _n_params(p: int, zero: bool) -> int:
    """Number of fitted parameters of an order-``p`` model (``p``, plus an intercept if not ``zero``)."""
    return p if zero else p + 1


def _arfit(v: ArrayLike, pmin: int, pmax: int, selector: str = 'sbc',
           zero: bool = True) -> tuple:
    """
    Stepwise least-squares AR model for a univariate series (Neumaier & Schneider 2001).

    All orders ``pmin`` to ``pmax`` are compared on the same ``N - pmax`` equations (the rows
    ``t = pmax, ..., N - 1``), so that the criteria are comparable between orders.  The data matrix
    (with the intercept column if ``zero`` is False, lagged values, and the response) is regularized
    and triangularized once with a QR factorization; the residual sum of squares of every
    lower order follows from trailing entries of the last column of the triangular factor.
    The order is chosen by Schwarz's Bayesian criterion (``'sbc'``) or the log of Akaike's final
    prediction error (``'fpe'``).

    Parameters
    ----------
    v : array-like
        The time series (1-D).
    pmin, pmax : int
        The smallest and largest model orders to consider, ``1 <= pmin <= pmax``.
    selector : {'sbc', 'fpe'}
        The order-selection criterion; the first minimizer is chosen.
    zero : bool, optional
        If True (default) fit the model ``v_t = A_1 v_{t-1} + ... + A_p v_{t-p} + e_t`` with no
        intercept; if False include an intercept ``w``.

    Returns
    -------
    w, A, C, sbc, fpe, th
        The intercept (0 if ``zero``), the coefficients ``A1, ..., Ap`` at the selected
        order, the noise variance, the criteria for orders ``pmin`` to ``pmax``, and
        ``th = (dof, Uinv)`` for the confidence intervals and eigenmodes: the degrees of
        freedom, and the matrix such that ``sqrt(diag(Uinv) * C)`` are the standard errors of the
        parameters (the intercept first, if present).

    Raises
    ------
    ValueError
        If the orders are invalid, the series is too short, or a criterion is not finite.

    Notes
    -----
    The regression matrix ``K`` is stabilized by stacking a diagonal damping block under it: the
    triangular factor ``R`` of ``[K; sqrt(delta) D]`` is used, where ``D`` holds the column norms
    of ``K`` and ``delta = (q^2 + q + 1) eps`` for ``q`` columns (Neumaier & Schneider, sec. 3.1).
    The final estimate is read from the leading block of this one factorization.
    """
    v = np.asarray(v, dtype=float).ravel()
    pmin, pmax = int(pmin), int(pmax)
    if pmin < 1 or pmax < pmin:
        raise ValueError(f'Invalid orders pmin = {pmin}, pmax = {pmax}: need 1 <= pmin <= pmax.')
    if selector not in ('sbc', 'fpe'):
        raise ValueError(f"Unknown order selector '{selector}'.")
    n_obs = v.size
    n_rows = n_obs - pmax  # number of regression equations, common to every order
    n_coef_max = _n_params(pmax, zero)
    if n_rows <= n_coef_max:
        raise ValueError(f'A series of {n_obs} samples gives {n_rows} equations, too few to fit up '
                         f'to order {pmax} ({n_coef_max} parameters).')

    # Regression matrix with columns [intercept] [v_{t-1} ... v_{t-pmax}] [v_t], t = pmax..n_obs-1
    windows = sliding_window_view(v, pmax + 1)[:, ::-1]  # row t: v_t, v_{t-1}, ..., v_{t-pmax}
    blocks = [windows[:, 1:], windows[:, :1]]
    if not zero:
        blocks.insert(0, np.ones((n_rows, 1)))
    design = np.hstack(blocks)
    n_cols = design.shape[1]  # n_coef_max + 1

    # Regularize (a damping term sized to the factorization's rounding error) and triangularize
    damping = (n_cols * n_cols + n_cols + 1) * np.finfo(float).eps
    col_norms = np.linalg.norm(design, axis=0)
    tri = np.linalg.qr(np.vstack([design, np.sqrt(damping) * np.diag(col_norms)]), mode='r')

    # Residual sum of squares of the order-p model: the squared last-column entries from row
    # n_coef(p) downwards.  Accumulating them from the bottom gives every order at once.
    orders = np.arange(pmin, pmax + 1)
    n_coef = orders if zero else orders + 1
    rss_from_row = np.cumsum((tri[:, -1] ** 2)[::-1])[::-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_rss = np.log(rss_from_row[n_coef])
        sbc = log_rss - (1 - n_coef / n_rows) * np.log(n_rows)
        fpe = log_rss - np.log(n_rows * (n_rows - n_coef) / (n_rows + n_coef))
    if not (np.all(np.isfinite(sbc)) and np.all(np.isfinite(fpe))):
        raise ValueError('Order-selection criteria are not finite.')

    criterion = sbc if selector == 'sbc' else fpe
    best_order = int(orders[np.argmin(criterion)])
    n_best = _n_params(best_order, zero)
    resid_dof = n_rows - n_best
    noise_var = float(rss_from_row[n_best] / resid_dof)

    # Coefficients, and the unit-variance parameter covariance, from the leading block of the
    # same factorization
    lead = tri[:n_best, :n_best]
    params = np.linalg.solve(lead, tri[:n_best, -1])
    lead_inv = np.linalg.inv(lead)
    param_cov_unit = lead_inv @ lead_inv.T
    if zero:
        intercept, coefs = 0.0, params
    else:
        intercept, coefs = float(params[0]), params[1:]
    return intercept, np.asarray(coefs, dtype=float), noise_var, sbc, fpe, (resid_dof, param_cov_unit)


def _arfit_residuals(w: float, A: np.ndarray, v: np.ndarray, k: Optional[int] = None) -> tuple:
    """
    Residuals of a fitted AR model and the Li-McLeod portmanteau test of their whiteness.

    The residuals are the data minus the fit, ``e_t = v_t - w - sum_j A_j v_{t-j}`` for
    ``t = p, ..., N - 1``.  With ``r_l`` the autocorrelations of the (demeaned) residuals, the modified
    portmanteau statistic is ``Q = n sum_{l=1}^k r_l^2 + k (k + 1) / (2 n)``, asymptotically
    chi-squared with ``k - p`` degrees of freedom under whiteness (Li & McLeod 1981).

    Parameters
    ----------
    w : float
        The intercept (0 for a zero-mean model).
    A : array-like
        The AR coefficients ``A_1, ..., A_p``.
    v : array-like
        The time series.
    k : int, optional
        The largest lag of the test; default ``min(20, n - 1)`` with ``n = N - p`` the number of residuals.

    Returns
    -------
    siglev, res
        The significance level of the test (upper-tail probability of ``Q``) and the residuals.

    Raises
    ------
    ValueError
        If ``k <= p`` or ``k >= n``.
    """
    v = np.asarray(v, dtype=float).ravel()
    A = np.atleast_1d(np.asarray(A, dtype=float))
    p = A.size
    n = v.size - p
    if n < 1:
        raise ValueError('Time series too short for the model order.')
    past = sliding_window_view(v, p + 1)  # row: v_{t-p}, ..., v_t
    res = past[:, p] - w - past[:, :p][:, ::-1] @ A
    if k is None:
        k = min(20, n - 1)
    k = int(k)
    if k <= p or k >= n:
        raise ValueError(f'Number of lags k = {k} must satisfy p < k < n (p = {p}, n = {n}).')
    e = res - res.mean()
    denom = e @ e
    r = np.array([e[:-l] @ e[l:] for l in range(1, k + 1)]) / denom
    Q = n * np.sum(r ** 2) + k * (k + 1) / (2 * n)
    siglev = float(chi2.sf(Q, k - p))
    return siglev, res


def _arfit_modes(A: np.ndarray, C: float, th: tuple, conf: float = 0.95) -> tuple:
    """
    Eigenmodes of a univariate AR model, with confidence intervals.

    The model is written as a first-order process with the companion matrix of ``A``.  Each
    eigenvalue ``lambda_k`` gives a damping time ``tau_k = -1 / log|lambda_k|`` and a period
    ``T_k = 2 pi / |arg lambda_k|`` (infinite for a positive real eigenvalue); the relative
    excitation of the mode is its share of the variance contributed by the noise,
    ``C |(S^{-1})_{k1}|^2 / (1 - |lambda_k|^2)`` normalized to sum to one, with ``S`` the matrix of
    eigenvectors.  Margins of error of the period and damping time follow from the first-order
    propagation of the coefficient covariance ``C * Uinv`` through the eigenvalues, with a Student-t
    quantile at ``dof`` degrees of freedom (Neumaier & Schneider 2001, sec. 2 and 4).

    Parameters
    ----------
    A : array-like
        The AR coefficients.
    C : float
        The noise variance.
    th : tuple
        ``(dof, Uinv)`` as returned by :func:`_arfit`.
    conf : float
        The confidence level of the margins of error (default 0.95).

    Returns
    -------
    per, tau, exctn, lam
        Periods and damping times with their margins of error (each of shape ``(2, p)``: the estimate,
        then the margin of error at level ``conf``), the relative excitations (length ``p``, summing
        to 1) and the complex eigenvalues.  The modes appear in the same order throughout.
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    order = A.size
    resid_dof, param_cov_unit = th
    param_cov_unit = np.atleast_2d(np.asarray(param_cov_unit, dtype=float))
    if param_cov_unit.shape[0] == order + 1:  # leading intercept entry: not needed here
        param_cov_unit = param_cov_unit[1:, 1:]
    coef_cov = C * param_cov_unit
    t_crit = t_dist.ppf(0.5 + conf / 2, df=resid_dof)

    # Eigen-decomposition of the companion matrix (coefficients in the top row, shift below)
    companion = np.diag(np.ones(order - 1), k=-1)
    companion[0, :] = A
    eigvals, eigvecs = np.linalg.eig(companion)
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
    eigvecs_inv = np.linalg.inv(eigvecs)

    with np.errstate(divide='ignore', invalid='ignore'):
        modulus = np.abs(eigvals)
        damping_time = -1.0 / np.log(modulus)
        period = 2 * np.pi / np.abs(np.angle(eigvals))  # inf for positive real eigenvalues

        # Sensitivity of each eigenvalue to each coefficient: row k, column j is
        # (S^{-1})_{k,1} S_{j,k}; split into real and imaginary parts
        d_eig = eigvecs_inv[:, [0]] * eigvecs.T
        re_part, im_part = eigvals.real[:, None], eigvals.imag[:, None]
        d_re, d_im = d_eig.real, d_eig.imag
        mod_sq = re_part ** 2 + im_part ** 2
        damping_grad = damping_time[:, None] ** 2 * (re_part * d_re + im_part * d_im) / mod_sq
        period_grad = period[:, None] ** 2 / (2 * np.pi) * (im_part * d_re - re_part * d_im) / mod_sq
        period_grad = np.where(np.isfinite(period)[:, None], period_grad, 0.0)  # infinite period: no margin

        # First-order error propagation: margin = t * sqrt(g' Cov g) for each mode
        damping_margin = t_crit * np.sqrt(np.einsum('kj,jl,kl->k', damping_grad, coef_cov, damping_grad))
        period_margin = t_crit * np.sqrt(np.einsum('kj,jl,kl->k', period_grad, coef_cov, period_grad))

        # Relative excitation: share of the noise-driven variance carried by each mode
        mode_variance = C * np.abs(eigvecs_inv[:, 0]) ** 2 / (1 - modulus ** 2)
        excitation = mode_variance / mode_variance.sum()
    period_margin = np.where(np.isfinite(period), period_margin, 0.0)

    return (np.vstack([period, period_margin]), np.vstack([damping_time, damping_margin]),
            excitation, eigvals)
