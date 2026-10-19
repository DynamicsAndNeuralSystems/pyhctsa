"""
Maximum-likelihood estimation of zero-mean GARCH, GJR-GARCH and t-GARCH(P, Q) conditional
variance models, and a test for conditional heteroskedasticity.

The estimator is specified here from the published model definitions: Bollerslev's GARCH [1]_,
the GJR (threshold) extension of Glosten, Jagannathan and Runkle [2]_ and Bollerslev's
Student-t innovations [3]_.  Nothing is taken from an estimation package.

Model.  For a zero-mean series ``y_1, ..., y_N`` the conditional variance is::

    h_t = w + sum_{i=1}^{Q} (a_i + g_i 1[y_{t-i} < 0]) y_{t-i}^2 + sum_{j=1}^{P} b_j h_{t-j}

(``g_i = 0`` for a plain GARCH).  The innovations ``y_t / sqrt(h_t)`` are standard normal, or
Student-t with ``nu = 1/kappa`` degrees of freedom scaled to unit variance.

Presample.  ``y_s^2 = h_s = B`` for ``s <= 0``, with ``B`` the mean of ``y^2`` (the variance
the model would have if there were no clustering); in a GJR model the presample indicator is 1/2.

Parameters and bounds.  ``theta = (w, b_1..b_P, a_1..a_Q, g_1..g_Q, kappa)`` (the last two groups
only for GJR and t).  The feasible set is a polytope::

    1e-6 <= w <= 10,   b_j >= 0,   a_i >= 0,   a_i + g_i >= 0,
    sum a + sum g / 2 + sum b <= 0.9999   (persistence below one: covariance stationarity),
    0.01 <= kappa <= 1/2.2                (2.2 <= nu <= 100).

Leverage ``g_i`` may be negative (down to ``-a_i``) so that its sign is informative.

Estimation.  A deterministic grid of starting points (total persistence, share of it in the
ARCH terms, and for t the tail weight) is scored by the likelihood; the best 12 grid points are taken
through a coarse sequence of log-barrier Newton problems (barrier weights 1e-3 to 1e-5); the
two best are then taken to barrier weight 1e-10 (the Newton polish) and the better is the fit.
Newton steps use the analytic gradient, a central-difference Hessian of it, and flip the sign
of negative curvature.  Nothing is random.

References
----------
.. [1] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity",
       J. Econometrics 31, 307-327 (1986). DOI: 10.1016/0304-4076(86)90063-1
.. [2] L. R. Glosten, R. Jagannathan and D. E. Runkle, "On the relation between the expected
       value and the volatility of the nominal excess return on stocks", J. Finance 48,
       1779-1801 (1993). DOI: 10.1111/j.1540-6261.1993.tb05128.x
.. [3] T. Bollerslev, "A conditionally heteroskedastic time series model for speculative prices
       and rates of return", Rev. Econ. Stat. 69, 542-547 (1987). DOI: 10.2307/1925546
"""
from collections import OrderedDict

import numpy as np
from scipy.signal import lfilter
from scipy.special import digamma, gammaln

W_LO, W_HI = 1e-6, 10.0
RHO_MAX = 0.9999
KAPPA_LO, KAPPA_HI = 0.01, 1 / 2.2
MU_COARSE = (1e-3, 1e-4, 1e-5)
MU_FINE = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
NU_MAX = 1 / KAPPA_LO
MAX_GRID = 12   # grid starting points that are run (the best by likelihood)
FLOOR_TOL = 0.01   # the variance constant is 'at its floor' within this relative tolerance of W_LO

# Structure test: likelihood-ratio statistic needed to report a variance model, calibrated by
# simulation of white noise (3% to 6% of white-noise series of 100 to 10000 samples exceed 4;
# Gaussian likelihood: rescaled by 2 / (kurtosis - 1)); the GJR model has more room to fit
# noise, and about 5% exceed 6. Tail test: 5% point of the 50:50 mixture of chi-squared(0) and
# chi-squared(1), the 90% point of chi-squared(1).
STRUCTURE_LR = 4.0
STRUCTURE_LR_GJR = 6.0
TAIL_LR = 2.7055


class Spec:
    """Model structure and its linear constraints ``A theta <= c``."""

    def __init__(self, P, Q, gjr=False, t=False):
        self.P, self.Q, self.gjr, self.t = int(P), int(Q), bool(gjr), bool(t)
        P, Q = self.P, self.Q
        self.ib = list(range(1, 1 + P))
        self.ia = list(range(1 + P, 1 + P + Q))
        self.ig = list(range(1 + P + Q, 1 + P + 2 * Q)) if gjr else []
        self.ik = (1 + P + Q + (Q if gjr else 0)) if t else None
        self.k = 1 + P + Q + (Q if gjr else 0) + (1 if t else 0)
        rows, rhs = [], []

        def add(coefs, b):
            r = np.zeros(self.k)
            for idx, v in coefs:
                r[idx] = v
            rows.append(r)
            rhs.append(b)

        add([(0, -1.0)], -W_LO)
        add([(0, 1.0)], W_HI)
        for j in self.ib:
            add([(j, -1.0)], 0.0)
        for i in self.ia:
            add([(i, -1.0)], 0.0)
        for a, g in zip(self.ia, self.ig):
            add([(a, -1.0), (g, -1.0)], 0.0)
        if P + Q > 0:
            add([(j, 1.0) for j in self.ib] + [(i, 1.0) for i in self.ia]
                + [(g, 0.5) for g in self.ig], RHO_MAX)
        if t:
            add([(self.ik, -1.0)], -KAPPA_LO)
            add([(self.ik, 1.0)], KAPPA_HI)
        self.A, self.c = np.array(rows), np.array(rhs)


class Data:
    """The series with its squares and lag matrices (computed once per series)."""

    def __init__(self, y):
        y = np.asarray(y, dtype=float).ravel()
        self.y = y
        self.N = len(y)
        self.y2 = y * y
        self.B = float(np.mean(self.y2))
        self.y2g = self.y2 * (y < 0)
        self._lag = {}
        self._filtered = OrderedDict()

    def lags(self, Q):
        """Lagged squares (N x Q; presample B) and lagged squares of negative values (presample B/2)."""
        if Q not in self._lag:
            N, B = self.N, self.B
            L = np.empty((N, Q))
            Lg = np.empty((N, Q))
            for i in range(1, Q + 1):
                L[:i, i - 1] = B
                L[i:, i - 1] = self.y2[:N - i]
                Lg[:i, i - 1] = B / 2
                Lg[i:, i - 1] = self.y2g[:N - i]
            self._lag[Q] = (L, Lg)
        return self._lag[Q]

    def filtered_inputs(self, A, Q, gjr):
        """The constant and lagged-square columns of the variance derivatives (ones, the lagged
        squares and, for GJR, the lagged squares of negative values), filtered by ``1/A``.

        They depend on theta only through the GARCH coefficients in ``A``, which are unchanged
        by a step in the constant or the ARCH coefficients (as in most of the coordinate steps
        of the finite-difference Hessian), so the most recent few are kept, keyed on the exact
        coefficients. (Each column is filtered on its own, which gives the same values as
        filtering it with the others.)
        """
        key = (A.tobytes(), int(Q), bool(gjr))
        if key in self._filtered:
            self._filtered.move_to_end(key)
            return self._filtered[key]
        L, Lg = self.lags(Q)
        M = np.column_stack([np.ones(self.N)] + [L[:, i] for i in range(Q)]
                            + ([Lg[:, i] for i in range(Q)] if gjr else []))
        out = lfilter([1.0], A, M, axis=0)
        self._filtered[key] = out
        while len(self._filtered) > 8:
            self._filtered.popitem(last=False)
        return out


def _variance(theta, D, S, want_grad=False):
    """Conditional variances (and their derivatives with respect to theta, an N x k array)."""
    N, B, P, Q = D.N, D.B, S.P, S.Q
    L, Lg = D.lags(Q)
    d = np.full(N, theta[0])
    if Q:
        d = d + L @ theta[S.ia]
    if S.gjr:
        d = d + Lg @ theta[S.ig]
    if P:
        b = theta[S.ib]
        A = np.concatenate(([1.0], -b))
        zi = B * np.cumsum(b[::-1])[::-1]       # filter state for a presample of B
        h = lfilter([1.0], A, d, zi=zi)[0]
    else:
        A = None
        h = d
    if not want_grad:
        return h
    # columns: ones, the lagged variances (P), the lagged squares (Q) and, for GJR, the lagged
    # squares of negative values (Q), each filtered by 1/A (the recursion of h); then kappa (zero)
    n_in = 1 + Q + (Q if S.gjr else 0)
    dh = np.zeros((N, 1 + P + Q + (Q if S.gjr else 0) + (1 if S.t else 0)))
    if not P:
        dh[:, 0] = 1.0
        dh[:, 1:n_in] = np.column_stack([L] + ([Lg] if S.gjr else []))
        return h, dh
    hl = np.empty((N, P))
    for j in range(1, P + 1):
        hl[:j, j - 1] = B
        hl[j:, j - 1] = h[:N - j]
    const = D.filtered_inputs(A, Q, S.gjr)
    dh[:, 0] = const[:, 0]
    dh[:, 1:1 + P] = lfilter([1.0], A, hl, axis=0)
    dh[:, 1 + P:P + n_in] = const[:, 1:]
    return h, dh


def _nll(theta, D, S, want_grad=True, want_scores=False):
    """Negative log-likelihood per observation, with its gradient (and per-observation scores)."""
    y2, N = D.y2, D.N
    if want_grad:
        h, dh = _variance(theta, D, S, True)
    else:
        h = _variance(theta, D, S, False)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return (np.inf, None, None) if want_scores else ((np.inf, None) if want_grad else np.inf)
    if not S.t:
        f = 0.5 * (np.log(2 * np.pi) + np.mean(np.log(h) + y2 / h))
        if not want_grad:
            return f
        w1 = 0.5 * (1 / h - y2 / h ** 2)
        g = (w1 @ dh) / N
        if want_scores:
            return f, g, w1[:, None] * dh
        return f, g
    kappa = theta[S.ik]
    nu = 1 / kappa
    w = y2 / (h * (nu - 2))
    lw = np.log1p(w)
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    f = -c + 0.5 * np.mean(np.log(h)) + 0.5 * (nu + 1) * np.mean(lw)
    if not want_grad:
        return f
    w1 = 0.5 / h * (1 - (nu + 1) * w / (1 + w))
    g = (w1 @ dh) / N
    cnu = 0.5 * digamma((nu + 1) / 2) - 0.5 * digamma(nu / 2) - 0.5 / (nu - 2)
    dnu_obs = -cnu + 0.5 * lw - 0.5 * (nu + 1) * w / ((nu - 2) * (1 + w))
    g[S.ik] = np.mean(dnu_obs) * (-nu ** 2)
    if want_scores:
        sc = w1[:, None] * dh
        sc[:, S.ik] = dnu_obs * (-nu ** 2)
        return f, g, sc
    return f, g


def _slack_steps(theta, S):
    """The largest step in each coordinate that keeps every constraint satisfied."""
    s = S.c - S.A @ theta
    out = np.full(len(theta), np.inf)
    for r, si in zip(S.A, s):
        for i in np.nonzero(r)[0]:
            out[i] = min(out[i], si / abs(r[i]))
    return out


def _hessian(theta, D, S):
    k = len(theta)
    room = _slack_steps(theta, S)
    H = np.empty((k, k))
    g0 = None
    for i in range(k):
        step = max(min(1e-5 * max(1.0, abs(theta[i])), 0.4 * room[i]), 1e-13)
        e = np.zeros(k)
        e[i] = step
        gp = _nll(theta + e, D, S)[1]
        gm = _nll(theta - e, D, S)[1]
        if gp is None or gm is None:
            if g0 is None:
                g0 = _nll(theta, D, S)[1]
            gp = g0 if gp is None else gp
            gm = g0 if gm is None else gm
        H[:, i] = (gp - gm) / (2 * step)
    return 0.5 * (H + H.T)


def _barrier(theta, D, S, mu, want_grad=True):
    """The log-barrier objective (with its gradient and the constraint slacks if ``want_grad``).

    The value is the same whether or not the gradient is computed (the conditional variances
    are the same filter of the same inputs), so a line search, which compares values only,
    asks for the value alone.
    """
    s = S.c - S.A @ theta
    if np.any(s <= 0):
        return (np.inf, None, None) if want_grad else np.inf
    if not want_grad:
        return _nll(theta, D, S, want_grad=False) - mu * np.sum(np.log(s))
    f, g = _nll(theta, D, S)
    if g is None:
        return np.inf, None, None
    return f - mu * np.sum(np.log(s)), g + mu * (S.A.T @ (1 / s)), s


def _newton(theta, D, S, mus, max_it=60, tol=1e-13):
    """Log-barrier Newton method through the barrier weights ``mus``; returns theta and the last decrement."""
    theta = theta.copy()
    dec = np.inf
    for mu in mus:
        for _ in range(max_it):
            phi, g, s = _barrier(theta, D, S, mu)
            H = _hessian(theta, D, S) + mu * (S.A.T * (1 / s ** 2)) @ S.A
            ev, V = np.linalg.eigh(H)
            ev = np.maximum(np.abs(ev), 1e-9 * max(1.0, np.abs(ev).max()))
            step = -V @ ((V.T @ g) / ev)
            dec = -g @ step
            if dec < tol:
                break
            Ad = S.A @ step
            pos = Ad > 0
            a = 1.0 if not pos.any() else min(1.0, 0.995 * np.min(s[pos] / Ad[pos]))
            ok = False
            for _ in range(60):
                if _barrier(theta + a * step, D, S, mu, want_grad=False) <= phi - 1e-4 * a * dec:
                    ok = True
                    break
                a *= 0.5
            if not ok:
                break
            theta = theta + a * step
    return theta, dec


def _start_grid(D, S):
    """Grid of starting points sorted by negative log-likelihood (ties in grid order)."""
    P, Q, B = S.P, S.Q, D.B
    rhos = [0.0] if P + Q == 0 else [0.3, 0.6, 0.9, 0.98]
    shares = [0.1, 0.25, 0.5] if (P and Q) else [1.0]
    kappas = [0.02, 0.1, 0.2] if S.t else [None]
    pts = []
    for rho in rhos:
        for sh in shares:
            th = np.zeros(S.k)
            if P and Q:
                arch, gar = rho * sh, rho * (1 - sh)
            elif Q:
                arch, gar = rho, 0.0
            else:
                arch, gar = 0.0, rho
            if P:
                th[S.ib] = gar / P
            if Q:
                th[S.ia] = arch / Q
            th[0] = max(B * (1 - rho), 2 * W_LO)
            for kp in kappas:
                t2 = th.copy()
                if kp is not None:
                    t2[S.ik] = kp
                pts.append(t2)
    sc = np.array([_nll(p, D, S, want_grad=False) for p in pts])
    order = np.argsort(sc, kind='stable')
    return [pts[i] for i in order]


def _interior(theta, S, eps=1e-6):
    """Move a start strictly inside the constraints (towards a fixed interior reference point)."""
    ref = np.zeros(S.k)
    ref[0] = 0.5
    ref[S.ib] = 0.05 / max(S.P, 1)
    ref[S.ia] = 0.05 / max(S.Q, 1)
    if S.t:
        ref[S.ik] = 0.1
    th = theta.copy()
    for _ in range(50):
        if np.all(S.c - S.A @ th > eps / 2):
            return th
        th = 0.9 * th + 0.1 * ref
    return th


def mle(D, P=1, Q=1, gjr=False, t=False, extra_starts=(), grid=True, npolish=2):
    """
    Maximum-likelihood fit of the specified model.

    Without ``grid`` only ``extra_starts`` are used (the grid if there are none).

    Returns a dict with ``theta``, ``llf`` (total log-likelihood), ``h`` (conditional
    variances), ``spec``, ``dec`` (the last Newton decrement, small when converged) and
    ``k`` (the number of parameters).
    """
    S = Spec(P, Q, gjr, t)
    starts = list(extra_starts) + (_start_grid(D, S)[:MAX_GRID] if grid or not extra_starts else [])
    runs = []
    for st in starts:
        th, _ = _newton(_interior(st, S), D, S, MU_COARSE)
        runs.append((_barrier(th, D, S, MU_COARSE[-1], want_grad=False), th))
    order = np.argsort([r[0] for r in runs], kind='stable')
    best = None
    for i in order[:npolish]:
        th, dec = _newton(runs[i][1], D, S, MU_FINE)
        f = _nll(th, D, S, want_grad=False)
        if best is None or f < best[1]:
            best = (th, f, dec)
    th, f, dec = best
    return dict(theta=th, llf=-f * D.N, h=_variance(th, D, S), spec=S, dec=dec, k=S.k)


def embed(theta, S_from, S_to):
    """Map the parameters of a smaller (P, Q) model onto a larger one (new coefficients at zero)."""
    th = np.zeros(S_to.k)
    th[0] = theta[0]
    for n, i in enumerate(S_from.ib):
        th[S_to.ib[n]] = theta[i]
    for n, i in enumerate(S_from.ia):
        th[S_to.ia[n]] = theta[i]
    for n, i in enumerate(S_from.ig):
        th[S_to.ig[n]] = theta[i]
    if S_from.t and S_to.t:
        th[S_to.ik] = theta[S_from.ik]
    return th


def null_fit(D, t=False):
    """The constant-variance (white-noise) special case: Gaussian in closed form, t by ML."""
    if not t:
        llf = -0.5 * D.N * (np.log(2 * np.pi) + np.log(D.B) + 1)
        return dict(theta=np.array([D.B]), llf=llf, h=np.full(D.N, D.B), spec=Spec(0, 0), dec=0.0, k=1)
    return mle(D, 0, 0, False, True)


def structure_test(alt, null, D, gjr=False):
    """
    Test for conditional heteroskedasticity: the likelihood-ratio statistic of the variance model
    ``alt`` against the constant-variance model ``null``, and whether it reaches the critical value.

    For Gaussian innovations the statistic is rescaled by ``2 / (kurtosis - 1)`` of the
    standardized residuals (the Gaussian likelihood-ratio statistic is inflated by heavy-tailed
    innovations); for t innovations the likelihood-ratio statistic is used as it is.
    """
    lr = max(2 * (alt['llf'] - null['llf']), 0.0)
    if not alt['spec'].t:
        z2 = D.y2 / alt['h']
        lr = lr * 2 / max(np.mean(z2 ** 2) / np.mean(z2) ** 2 - 1, 1e-3)
    return lr, lr >= (STRUCTURE_LR_GJR if gjr else STRUCTURE_LR)


def opg_errors(theta, D, S):
    """Standard errors from the outer product of the per-observation scores at ``theta``."""
    sc = _nll(theta, D, S, True, True)[2]
    if sc is None:
        return np.full(len(theta), np.nan)
    cov = np.linalg.pinv(sc.T @ sc, rcond=1e-10)
    return np.sqrt(np.abs(np.diag(cov)))


def tail_starts(fit_gauss, S_t):
    """Starting points of a t model from a Gaussian fit of the same orders (tail weights on a grid)."""
    out = []
    for kp in (0.02, 0.1, 0.2):
        th = embed(fit_gauss['theta'], fit_gauss['spec'], S_t)
        th[S_t.ik] = kp
        out.append(th)
    return out


def at_floor(theta):
    """Whether the variance constant of a fit sits at its lower bound (within ``FLOOR_TOL`` of it).

    The Gaussian likelihood of a series with long stretches of (near) zero values has no maximum
    (the variance can shrink without limit where the series is zero), so a fit at the floor has a
    likelihood and a variance path that depend on the bound rather than on the data.
    """
    return bool(theta[0] <= W_LO * (1 + FLOOR_TOL))
