"""
A port of the nonlinear least-squares machinery behind MATLAB's ``fit(x, y, fittype(...))``.

This module ports that path for the unconstrained case, which is all the Curve Fitting
Toolbox uses when no ``Lower``/``Upper`` bounds are given. Without finite bounds the
reflective machinery collapses considerably: the scaling matrix ``D`` is the identity,
``DG`` vanishes, no step is ever truncated at a bound, and the reflected trial direction is
never evaluated. What remains is the two-dimensional subspace trust-region iteration.
"""

import numpy as np
from scipy.linalg import solve_triangular

__all__ = ['lsqcurvefit_trr', 'fit_exp1', 'fit_poly1', 'goodness_of_fit']

_EPS = np.finfo(float).eps
_SQRT_EPS = np.sqrt(_EPS)


# ------------------------------------------------------------------------------
# Trust-region subproblem: min { g'*s + 0.5*s'*H*s : ||s|| <= delta }
# ------------------------------------------------------------------------------

def _seceqn(lam, eigval, alpha, delta):
    """The secular equation 1/delta - 1/||s(lambda)||, evaluated at a scalar lambda."""
    w = eigval + lam
    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where(w != 0, alpha / np.where(w != 0, w, 1), np.inf)
    m = m * m
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.sqrt(1.0 / np.sum(m))
    if np.isnan(value):
        value = 0.0
    return 1.0 / delta - value


def _secular_eqn_root(x, itbnd, eigval, alpha, delta, tol):
    """
    Zero of the secular equation to the RIGHT of the starting point x.
    A port of the modified `fzero` used by MATLAB's `trust`.
    """
    itfun = 0
    dx = abs(x) / 2 if x != 0 else 0.5

    a = x
    c = a
    fa = _seceqn(a, eigval, alpha, delta)
    itfun += 1

    b = x + 1
    fb = _seceqn(b, eigval, alpha, delta)
    itfun += 1

    # Find change of sign
    while (fa > 0) == (fb > 0):
        dx = 2 * dx
        if (fa > 0) != (fb > 0):
            break
        b = x + dx
        fb = _seceqn(b, eigval, alpha, delta)
        itfun += 1
        if itfun > itbnd:
            break

    fc = fb
    d = e = b - a
    # Main loop, exit from middle of the loop
    while fb != 0:
        # Insure that b is the best result so far, a is the previous value of b,
        # and c is on the opposite side of the zero from b
        if (fb > 0) == (fc > 0):
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        if itfun > itbnd:
            break
        m = 0.5 * (c - b)
        toler = 2.0 * tol * max(abs(b), 1.0)
        if abs(m) <= toler or fb == 0.0:
            break

        # Choose bisection or interpolation
        if abs(e) < toler or abs(fa) <= abs(fb):
            d = m
            e = m
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s  # linear interpolation
                q = 1.0 - s
            else:
                q = fa / fc  # inverse quadratic interpolation
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if (2.0 * p < 3.0 * m * q - abs(toler * q)) and (p < abs(0.5 * e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m

        a = b
        fa = fb
        if abs(d) > toler:
            b = b + d
        elif b > c:
            b = b - toler
        else:
            b = b + toler
        fb = _seceqn(b, eigval, alpha, delta)
        itfun += 1

    return b, itfun


def _coeff_from_w(w, alpha):
    """The `coeff = alpha ./ w` step of `trust`, with its handling of zeros in w."""
    coeff = np.zeros_like(alpha)
    nz = w != 0
    coeff[nz] = alpha[nz] / w[nz]
    coeff[~nz & (alpha != 0)] = np.inf
    coeff[np.isnan(coeff)] = 0.0
    return coeff


def _trust(g, H, delta):
    """
    Exact solution of the trust region problem min{g's + 0.5 s'Hs : ||s|| <= delta},
    via the full eigen-decomposition and the secular equation.
    """
    tol, tol2, itbnd = 1e-12, 1e-8, 50
    key = 0
    lam_val = 0.0
    g = np.atleast_1d(np.asarray(g, dtype=float))
    H = np.atleast_2d(np.asarray(H, dtype=float))

    # MATLAB's `eig` dispatches to the symmetric solver only for exactly symmetric input
    if np.array_equal(H, H.T):
        eigval, V = np.linalg.eigh(H)
    else:
        eigval, V = np.linalg.eig(H)
        eigval, V = np.real(eigval), np.real(V)

    jmin = int(np.argmin(eigval))
    mineig = eigval[jmin]
    alpha = -(V.T @ g)
    sig = np.sign(alpha[jmin]) + (alpha[jmin] == 0)

    s = None
    laminit = 0.0
    if mineig > 0:  # positive definite case
        coeff = alpha / eigval
        lam_val = 0.0
        s = V @ coeff
        if np.linalg.norm(s) <= 1.2 * delta:
            key = 1
        else:
            laminit = 0.0
    else:  # indefinite case
        laminit = -mineig

    if key == 0:
        if _seceqn(laminit, eigval, alpha, delta) > 0:
            b, _ = _secular_eqn_root(laminit, itbnd, eigval, alpha, delta, tol)
            if abs(_seceqn(b, eigval, alpha, delta)) <= tol2:
                lam_val = b
                key = 2
                s = V @ _coeff_from_w(eigval + lam_val, alpha)
                nrms = np.linalg.norm(s)
                if nrms > 1.2 * delta or nrms < 0.8 * delta:
                    key = 5
                    lam_val = -mineig
            else:
                lam_val = -mineig
                key = 3
        else:
            lam_val = -mineig
            key = 4

        if key > 2:
            arg = np.abs(eigval + lam_val) < 10 * _EPS * np.maximum(np.abs(eigval), 1)
            alpha = np.where(arg, 0.0, alpha)
        s = V @ _coeff_from_w(eigval + lam_val, alpha)
        nrms = np.linalg.norm(s)
        if key > 2 and nrms < 0.8 * delta:
            s = s + np.sqrt(delta**2 - nrms**2) * sig * V[:, jmin]
        if key > 2 and nrms > 1.2 * delta:
            b, _ = _secular_eqn_root(laminit, itbnd, eigval, alpha, delta, tol)
            lam_val = b
            s = V @ _coeff_from_w(eigval + lam_val, alpha)

    val = g @ s + 0.5 * (s @ (H @ s))
    return s, val


# ------------------------------------------------------------------------------
# The 2-D subspace trial step (trdog.m, specialized to the unbounded case)
# ------------------------------------------------------------------------------

def _newton_direction(A, g):
    """
    The (Gauss-)Newton direction, i.e. the solution of (A'A)p = -g.

    `pcgr` preconditions with the exact QR factor of A produced by `aprecon`
    (`PrecondBandWidth = Inf`), so its single permitted CG iteration reproduces the
    direct solve. Only the direction survives -- `trdog` normalizes p immediately.
    """
    R = np.linalg.qr(A, mode='r')
    r = -g
    try:
        z = solve_triangular(R, solve_triangular(R, r, trans='T', lower=False), lower=False)
    except (np.linalg.LinAlgError, ValueError):
        return r, 1
    if not np.all(np.isfinite(z)):
        return r, 1
    ww = A.T @ (A @ z)
    denom = z @ ww
    if denom <= 0:
        nz = np.linalg.norm(z)
        return (z if nz == 0 else z / nz), 0
    return (r @ z) / denom * z, 1


def _trdog(g, A, delta, Z):
    """Reflected (2-D) trust region trial step, with no finite bounds."""
    n = len(g)
    grad = g  # D is the identity, so the scaled gradient is just g
    posdef = 1
    tol2 = _SQRT_EPS

    if Z is None:
        v1, posdef = _newton_direction(A, g)
        nv1 = np.linalg.norm(v1)
        if nv1 > 0:
            v1 = v1 / nv1
        cols = [v1]
        if n > 1:
            # posdef < 1 would use D*sign(grad); D is the identity and, with A'A
            # positive semi-definite, pcgr never reports negative curvature anyway
            ngrad = np.linalg.norm(grad)
            v2 = grad / ngrad if ngrad > 0 else grad
            v2 = v2 - v1 * (v1 @ v2)
            nrmv2 = np.linalg.norm(v2)
            if nrmv2 > tol2:
                cols.append(v2 / nrmv2)
        Z = np.column_stack(cols)

    # Reduce to the chosen subspace. D = I and DG = 0, so M = Z'(A'A)Z
    MM = Z.T @ (A.T @ (A @ Z))
    rhs = Z.T @ grad

    # Determine the 2-D trust region solution. Unbounded, so it is never truncated
    st, _ = _trust(rhs, MM, delta)
    ss = Z @ st
    s = ss
    if np.any(np.isnan(s)):
        raise FloatingPointError('NaN in the trust region step')
    qpval1 = rhs @ st + 0.5 * (st @ (MM @ st))

    # The reflected direction is only evaluated when the step is truncated at a
    # bound, which cannot happen here, so qpval3 stays infinite
    qpval2 = np.inf
    sg = ssg = None
    if n > 1:
        # Evaluate along the gradient direction
        gnorm = np.linalg.norm(grad)
        ZZ = (grad / (gnorm + (gnorm == 0))).reshape(-1, 1)
        MMg = ZZ.T @ (A.T @ (A @ ZZ))
        rhsg = ZZ.T @ grad
        stg, _ = _trust(rhsg, MMg, delta)
        ssg = ZZ @ stg
        sg = ssg
        qpval2 = rhsg @ stg + 0.5 * (stg @ (MMg @ stg))

    # Choose the best of the two steps
    if qpval2 <= qpval1:
        return sg, ssg, qpval2, posdef, Z
    return s, ss, qpval1, posdef, Z


# ------------------------------------------------------------------------------
# The main solver (snls.m, specialized to the unbounded case)
# ------------------------------------------------------------------------------

def _findiff_jac(resid_fn, x, fvec, diff_min_change, diff_max_change):
    """
    Forward finite-difference Jacobian, matching MATLAB's `finitedifferences`:
    a step of sign'(x_j) * sqrt(eps) * max(|x_j|, typicalx_j) with typicalx = 1,
    clamped in magnitude to [DiffMinChange, DiffMaxChange].
    """
    n = len(x)
    J = np.empty((len(fvec), n))
    for j in range(n):
        sgn = -1.0 if x[j] < 0 else 1.0  # sign'(0) is +1
        chg = sgn * _SQRT_EPS * max(abs(x[j]), 1.0)
        chg = np.sign(chg) * min(max(abs(chg), diff_min_change), diff_max_change)
        xp = x.copy()
        xp[j] = x[j] + chg
        J[:, j] = (resid_fn(xp) - fvec) / chg
    return J, n


def lsqcurvefit_trr(fun, x0, xdata, ydata, tol_fun=1e-6, tol_x=1e-6,
                    max_iter=400, max_fun_evals=600,
                    diff_min_change=1e-8, diff_max_change=0.1):
    """
    Minimize ``sum((fun(p, xdata) - ydata)**2)`` over ``p`` the way MATLAB's
    ``lsqcurvefit`` does with ``Algorithm = 'trust-region-reflective'`` and no bounds.

    The defaults are the ones MATLAB's Curve Fitting Toolbox passes down from
    ``fitoptions('Method', 'NonlinearLeastSquares')``.

    Parameters
    ----------
    fun : callable
        ``fun(p, xdata)`` returning the model values.
    x0 : array-like
        Starting point for the coefficients (MATLAB's ``StartPoint``).
    xdata, ydata : array-like
        The data to fit.

    Returns
    -------
    numpy.ndarray
        The fitted coefficients.
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    x = np.asarray(x0, dtype=float).ravel().copy()
    n = len(x)

    def resid(p):
        return np.asarray(fun(p, xdata), dtype=float).ravel() - ydata

    iter_count = 0
    num_fun_evals = 1
    delta = 10.0
    nrmsx = 1.0
    ratio = 0.0
    oval = np.inf
    Z = None
    posdef = 1

    fvec = resid(x)
    A, fdevals = _findiff_jac(resid, x, fvec, diff_min_change, diff_max_change)
    num_fun_evals += fdevals
    if not np.all(np.isfinite(A)):
        raise FloatingPointError('Jacobian undefined at the initial point')

    g = A.T @ fvec
    val = fvec @ fvec
    if not np.isfinite(val) or not np.all(np.isfinite(g)):
        raise FloatingPointError('Objective or its gradient undefined at the initial point')

    ex = 0
    while not ex:
        # With no finite bounds definev returns v = +/-1 and dv = 0, so |v| = 1
        optnrm = np.max(np.abs(g))

        # Test for convergence
        diff = abs(oval - val)
        oval = val
        if optnrm < tol_fun and posdef == 1:
            ex = 1
        elif nrmsx < 0.9 * delta and ratio > 0.25 and diff < tol_fun * (1 + abs(oval)):
            ex = 2
        elif iter_count > 1 and nrmsx < tol_x:
            ex = 3
        elif iter_count > max_iter or num_fun_evals > max_fun_evals:
            ex = 4
        if ex:
            break

        # Determine the trust region correction
        sx, snod, qp, posdef, Z = _trdog(g, A, delta, Z)
        nrmsx = np.linalg.norm(snod)
        newx = x + sx

        newfvec = resid(newx)
        newA, fdevals = _findiff_jac(resid, newx, newfvec, diff_min_change, diff_max_change)
        num_fun_evals += 1 + fdevals
        newval = newfvec @ newfvec

        # Update the trust region radius
        if not np.all(np.isfinite(newfvec)):
            # Shrink if any element of the function vector is not defined
            delta = min(nrmsx / 20, delta / 20)
        else:
            newgrad = newA.T @ newfvec
            # `aug` involves dv, which is zero without finite bounds
            ratio = (0.5 * (newval - val)) / qp
            if ratio >= 0.75 and nrmsx >= 0.9 * delta:
                delta = 2 * delta
            elif ratio <= 0.25:
                delta = min(nrmsx / 4, delta / 4)

            # Accept or reject the trial point
            if newval < val:
                x = newx
                val = newval
                g = newgrad
                A = newA
                Z = None
                fvec = newfvec

        iter_count += 1

    return x


def fit_exp1(x, y, start_point=(1.0, -0.2)):
    """
    Fit ``a*exp(b*x)`` as MATLAB's ``fit(x, y, fittype('a*exp(b*x)', 'options', s))``
    does, where ``s`` sets ``Method`` to ``'NonlinearLeastSquares'`` and ``StartPoint``
    to `start_point`.

    Returns
    -------
    tuple of float
        The fitted ``(a, b)``.
    """
    p = lsqcurvefit_trr(lambda p, xd: p[0] * np.exp(p[1] * xd), start_point, x, y)
    return float(p[0]), float(p[1])


def fit_poly1(x, y, start_point=(0.1, 0.0)):
    """
    Fit ``a*x + b`` as MATLAB's ``fit(x, y, fittype('a*x+b', 'options', s))`` does with
    ``Method = 'NonlinearLeastSquares'`` and ``StartPoint`` set to `start_point`.

    The model is linear in its coefficients, but the trust-region iteration can still
    stop short of the exact least-squares solution on the toolbox's default tolerances,
    so this does not generally agree with `numpy.polyfit`.

    Returns
    -------
    tuple of float
        The fitted ``(a, b)``.
    """
    p = lsqcurvefit_trr(lambda p, xd: p[0] * xd + p[1], start_point, x, y)
    return float(p[0]), float(p[1])


def goodness_of_fit(y, y_fit, num_coeffs):
    """
    Goodness-of-fit statistics, matching the ``gof`` struct returned by MATLAB's ``fit``.
    Note that ``rmse`` is the standard error, ``sqrt(SSE/dfe)``, not the root-mean-square
    of the residuals.

    Parameters
    ----------
    y : array-like
        The observed values.
    y_fit : array-like
        The fitted values.
    num_coeffs : int
        The number of fitted coefficients, which sets the error degrees of freedom.

    Returns
    -------
    dict
        With keys ``sse``, ``rsquare``, ``adjrsquare`` and ``rmse``.
    """
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    n = len(y)
    dfe = n - num_coeffs
    sse = np.sum((y - y_fit)**2)
    sst = np.sum((y - np.mean(y))**2)
    rsquare = 1 - sse / sst
    return {
        'sse': sse,
        'rsquare': rsquare,
        'adjrsquare': 1 - (1 - rsquare) * (n - 1) / dfe,
        'rmse': np.sqrt(sse / dfe),
    }
