"""Robust linear regression by iteratively reweighted least squares with Tukey's bisquare weights.

The estimator is the M-estimator of Holland & Welsch (1977) with the bisquare (biweight) score
function, with the leverage adjustment of the residuals and the final scale and covariance
estimates of DuMouchel & O'Brien (1989) and Street, Carroll & Ruppert (1988); see also
Huber (1981), *Robust Statistics*.

Notation:

* ``X`` is the n-by-p design matrix, ``h_i`` the leverage (diagonal of the hat matrix of the
  *unweighted* design), ``r_i = y_i - x_i b`` the residuals, ``r_i / sqrt(1 - h_i)`` the
  leverage-adjusted residuals, ``sigma`` a robust scale and ``c`` the tuning constant.
* ``u_i = r_i / (sqrt(1 - h_i) * c * sigma)`` is the scaled residual, ``psi(u) = u (1 - u^2)^2``
  the bisquare score function and ``w(u) = psi(u) / u = (1 - u^2)^2`` its weight function, both
  zero for ``|u| >= 1``.

Algorithm. Start from the ordinary least-squares fit. Repeat: compute the scale ``sigma`` as the
median absolute leverage-adjusted residual (after discarding the ``p - 1`` smallest) divided by
0.6745, set the weights ``w(u_i)``, and solve the weighted least-squares problem, until the
coefficients change by at most ``1e-10 (1 + max|b|)`` in the maximum norm (at most 200
iterations). Afterwards the residual scale is estimated by the Huber/Street-Carroll-Ruppert
formula (``robust_s``) and combined with the least-squares scale into the final ``s``; the
coefficient covariance is ``s^2 (X'X)^{-1}`` using the unweighted design.

Numerical guards (not part of the published estimator): a row whose leverage is 1 up to rounding
is fitted exactly by any weighting, so its adjusted residual is 0/0; such rows get an adjusted
residual of 0 (weight 1) and are left out of the scale estimate. A robust scale of exactly zero
(at least half the adjusted residuals are exactly zero) is kept as zero: residuals that are
exactly zero keep weight 1 and all others get weight 0.
"""
import warnings

import numpy as np
import scipy.linalg as sla
from scipy import stats as _sps

_EPS = np.finfo(float).eps

BISQUARE_TUNE = 4.685          # Holland & Welsch (1977) 95%-efficiency constant for the bisquare
MAD_TO_SIGMA = 0.6745          # MAD / 0.6745 estimates sigma for Gaussian data (Holland & Welsch)
LEVERAGE_TOL = 1e-10           # rows with 1 - h_i below this are treated as exactly fitted (h_i = 1)
CONVERGENCE_TOL = 1e-10        # stop when max|db| <= CONVERGENCE_TOL * (1 + max|b|)
MAX_ITER = 200                 # iteration limit


def bisquare_weights(u):
    """Tukey bisquare weight ``(1 - u^2)^2`` for ``|u| < 1`` and 0 otherwise."""
    u = np.asarray(u, dtype=float)
    return np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)


def bisquare_psi_slope(u):
    """Derivative of the bisquare score ``psi(u) = u (1 - u^2)^2``: ``(1 - u^2)(1 - 5 u^2)``."""
    u = np.asarray(u, dtype=float)
    return np.where(np.abs(u) < 1.0, (1.0 - u * u) * (1.0 - 5.0 * u * u), 0.0)


def _as_columns(x, y):
    """Return ``(x2d, y1d)`` with ``x2d`` n-by-k; 1-D x of either orientation is one column."""
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    if x.ndim <= 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 2 and x.shape[0] != y.size and x.shape[1] == y.size and x.shape[0] == 1:
        x = x.T
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("The number of rows in X must match the number of elements in y.")
    return x, y


class BisquareIRLS:
    """Bisquare M-estimation of a linear model by iteratively reweighted least squares.

    Parameters
    ----------
    x, y : array_like
        Predictors (n or n-by-k) and response (n). Rows containing NaN are ignored.
    tune : float
        Tuning constant ``c`` of the bisquare (default 4.685).
    const : bool
        Prepend a column of ones (default True).
    """

    def __init__(self, x, y, tune=BISQUARE_TUNE, const=True):
        x, y = _as_columns(x, y)
        self.n_total = y.size
        self.tune = float(tune)
        X = np.column_stack([np.ones(y.size), x]) if const else x
        self.keep = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        self.X = X[self.keep]
        self.y = y[self.keep]
        self.n, self.p_cols = self.X.shape
        if self.n <= self.p_cols:
            raise ValueError("Too few observations for a robust fit.")
        self._rank_and_leverage()

    # ------------------------------------------------------------------ design
    def _rank_and_leverage(self):
        """Pivoted QR of X: numerical rank, kept columns, leverages.

        A rank-deficient X gives a basic solution: dropped columns get ``b = 0`` and ``se = 0``,
        the degrees of freedom are ``n - rank`` and the leverages come from the retained columns.
        The rank tolerance is ``max(n, p) * eps * max|diag R|``.
        """
        Q, R, piv = sla.qr(self.X, mode="economic", pivoting=True)
        d = np.abs(np.diag(R))
        tol = max(self.n, self.p_cols) * _EPS * (d.max() if d.size else 0.0)
        self.rank = int(np.sum(d > tol))
        self.cols = np.sort(piv[: self.rank])        # retained columns, in original order
        Xr = self.X[:, self.cols]
        Q1, _ = np.linalg.qr(Xr)
        self.h = np.clip(np.sum(Q1 * Q1, axis=1), 0.0, 1.0)
        self.exact = (1.0 - self.h) < LEVERAGE_TOL   # leverage-one rows
        self.rank_deficient = self.rank < self.p_cols
        self.Xr = Xr

    def _solve(self, w=None):
        """(Weighted) least squares on the retained columns; returns the full-length b."""
        if w is None:
            A, z = self.Xr, self.y
        else:
            sw = np.sqrt(w)
            A, z = self.Xr * sw[:, None], self.y * sw
        sol = np.linalg.lstsq(A, z, rcond=None)[0]
        b = np.zeros(self.p_cols)
        b[self.cols] = sol
        return b

    # ------------------------------------------------------------------ scale
    def adjusted_residuals(self, b):
        """Leverage-adjusted residuals ``r_i / sqrt(1 - h_i)`` (0 for leverage-one rows)."""
        r = self.y - self.X @ b
        ra = np.zeros_like(r)
        ok = ~self.exact
        ra[ok] = r[ok] / np.sqrt(1.0 - self.h[ok])
        return ra

    def mad_sigma(self, b):
        """MAD-based scale of the adjusted residuals.

        Median of the absolute adjusted residuals after discarding the ``rank - 1`` smallest
        ones (the degrees of freedom absorbed by the fit; Holland & Welsch 1977), divided by
        0.6745. No centring. Leverage-one rows are excluded and reduce the number discarded.
        """
        a = np.sort(np.abs(self.adjusted_residuals(b)[~self.exact]))
        k = max(self.rank - 1 - int(self.exact.sum()), 0)
        a = a[k:] if k < a.size else a
        return float(np.median(a)) / MAD_TO_SIGMA if a.size else 0.0

    def _scaled_residuals(self, b):
        """``u_i = r_i / (sqrt(1 - h_i) c sigma)``; with ``sigma = 0``: 0 for exact zeros, else inf."""
        sigma = self.mad_sigma(b)
        ra = self.adjusted_residuals(b)
        if sigma == 0.0:
            return np.where(ra == 0.0, 0.0, np.inf)
        return ra / (self.tune * sigma)

    # ------------------------------------------------------------------ fit
    def fit(self):
        """Run the iteration and compute all summary statistics; returns ``(b, stats)``."""
        b_ols = self._solve()
        r_ols = self.y - self.X @ b_ols
        ols_s = np.sqrt(np.sum(r_ols ** 2) / (self.n - self.rank))

        b = b_ols
        converged = False
        for _ in range(MAX_ITER):
            w = bisquare_weights(self._scaled_residuals(b))     # weights that produce b_new
            b_new = self._solve(w)
            converged = bool(np.max(np.abs(b_new - b)) <= CONVERGENCE_TOL * (1.0 + np.max(np.abs(b_new))))
            b = b_new
            if converged:
                break
        if not converged:
            warnings.warn("Iteration limit reached.", RuntimeWarning, stacklevel=3)
        if self.rank_deficient:
            warnings.warn(f"X is rank deficient, rank = {self.rank}", RuntimeWarning, stacklevel=3)

        return b, self._statistics(b, w, ols_s)

    # ------------------------------------------------------------------ statistics
    def _statistics(self, b, w, ols_s):
        n, p = self.n, self.rank
        r = self.y - self.X @ b

        # MAD scale at the final coefficients
        mad_s = self.mad_sigma(b)

        # Robust residual scale (Huber 1981; Street, Carroll & Ruppert 1988):
        #   robust_s = sqrt(sum((w r)^2) / (n - p)) / a * (1 + (p / n) (1 - a) / a),
        #   a = mean psi'(u) with u from the final adjusted residuals.
        u = self._scaled_residuals(b)
        a = float(np.mean(bisquare_psi_slope(u)))
        root_mean_psi = np.sqrt(np.sum((bisquare_weights(u) * r) ** 2) / (n - p))
        with np.errstate(divide="ignore", invalid="ignore"):
            robust_s = root_mean_psi / a * (1.0 + (p / n) * (1.0 - a) / a) if a > 0 else np.nan

        # Final scale (DuMouchel & O'Brien 1989): the larger of robust_s and the (p^2 : n)-weighted
        # average of the least-squares and robust variances.
        q = p * p / (p * p + n)
        with np.errstate(invalid="ignore"):
            s = max(robust_s, np.sqrt(q * ols_s ** 2 + (1.0 - q) * robust_s ** 2)) if np.isfinite(robust_s) else np.nan

        # Covariance from the unweighted design
        covb_r = s ** 2 * np.linalg.inv(self.Xr.T @ self.Xr)
        covb = np.zeros((self.p_cols, self.p_cols))
        covb[np.ix_(self.cols, self.cols)] = covb_r
        se = np.zeros(self.p_cols)
        se[self.cols] = np.sqrt(np.diag(covb_r))
        dfe = n - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, b / se, np.nan)
            pval = np.where(se > 0, 2.0 * _sps.t.cdf(-np.abs(t), dfe), np.nan)
            dse = np.sqrt(np.diag(covb))
            coeffcorr = covb / np.outer(dse, dse)

        def expand(v):
            out = np.full(self.n_total, np.nan)
            out[self.keep] = v
            return out

        with np.errstate(divide="ignore", invalid="ignore"):
            rstud = np.where(self.exact, np.nan, r / (s * np.sqrt(np.where(self.exact, 1.0, 1.0 - self.h))))
        return {
            "ols_s": ols_s, "robust_s": robust_s, "mad_s": mad_s, "s": s,
            "resid": expand(r), "rstud": expand(rstud), "se": se, "covb": covb,
            "coeffcorr": coeffcorr, "t": t, "p": pval, "w": expand(w), "dfe": dfe,
            "h": expand(self.h),
        }


def robust_linear_fit(x, y, tune=BISQUARE_TUNE):
    """Robust (bisquare-weighted) linear regression of ``y`` on ``x``.

    Iteratively reweighted least squares with Tukey's bisquare weights (tuning constant
    ``tune``) and a constant term (see the module docstring). ``x`` is a vector or an n-by-k
    matrix; rows with NaN are ignored.

    Returns
    -------
    b : numpy.ndarray
        Coefficients, intercept first.
    stats : dict
        ``ols_s``, ``robust_s``, ``mad_s``, ``s``, ``resid``, ``rstud``, ``se``, ``covb``,
        ``coeffcorr``, ``t``, ``p``, ``w``, ``dfe`` and ``h`` (``w``, ``h``, ``resid`` and
        ``rstud`` have NaN where an input row was ignored).

    Raises
    ------
    ValueError
        If there are not more valid rows than coefficients.
    """
    return BisquareIRLS(x, y, tune=tune).fit()
