"""
A port of the nonlinear least-squares machinery behind MATLAB's
``fit(x, y, fittype(...))``.
"""

import math

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import get_lapack_funcs

_dtrtrs, _dgeqrf = get_lapack_funcs(
    ("trtrs", "geqrf"),
    (np.empty(0, dtype=np.float64),),
)

_TRIL_IDX = {}


def _qr_r(A):
    """
    ``np.linalg.qr(A, mode="r")``.

    ``qr`` copies ``A``, runs ``dgeqrf`` and returns ``triu`` of the leading
    ``min(m, n)`` rows. ``dgeqrf`` here is the same LAPACK routine on the same
    matrix, so the factor is identical bit for bit, and the explicit copy +
    zero-fill below reproduces ``np.triu``'s C-contiguous result exactly.
    """
    qr, _tau, _work, _info = _dgeqrf(A)

    m, n = A.shape

    R = qr[: (n if m >= n else m), :].copy()

    k = R.shape[0]

    idx = _TRIL_IDX.get(k)

    if idx is None:
        idx = _TRIL_IDX[k] = np.tril_indices(k, -1)

    R[idx] = 0.0

    return R


def _solve_tri(a1, b1, trans, overwrite_b=False):
    """
    ``scipy.linalg.solve_triangular(a1, b1, trans=trans, lower=False,
    check_finite=False)``.

    This reproduces scipy's dispatch verbatim, including the trick of solving
    the transposed system when ``a1`` is not Fortran-ordered (``dtrtrs``
    expects column-major input), so the exact same LAPACK call is made.

    The shape validation is load-bearing, not defensive: when there are fewer
    residuals than parameters the R factor is rectangular and scipy raises
    ``ValueError``, which ``_newton_direction`` catches to fall back on the
    steepest-descent direction.
    """
    if a1.shape[0] != a1.shape[1]:
        raise ValueError("expected square matrix")

    if a1.shape[0] != b1.shape[0]:
        raise ValueError(
            f"shapes of a {a1.shape} and b {b1.shape} are incompatible"
        )

    if a1.flags.f_contiguous:
        x, info = _dtrtrs(
            a1,
            b1,
            overwrite_b=overwrite_b,
            lower=False,
            trans=trans,
            unitdiag=False,
        )

    else:
        x, info = _dtrtrs(
            a1.T,
            b1,
            overwrite_b=overwrite_b,
            lower=True,
            trans=not trans,
            unitdiag=False,
        )

    if info == 0:
        return x

    if info > 0:
        raise LinAlgError(
            f"singular matrix: resolution failed at diagonal {info - 1}"
        )

    raise ValueError(
        f"illegal value in {-info}-th argument of internal trtrs"
    )


_EPS = np.finfo(float).eps
_SQRT_EPS = np.sqrt(_EPS)
_TEN_EPS = float(10 * _EPS)

_INF = math.inf
_NAN = math.nan

_V_1X1 = np.ones((1, 1), dtype=float)

_float64 = np.dtype(np.float64)
_ndarray = np.ndarray


def _norm(v):
    """
    ``np.linalg.norm(v)`` for a 1-D real array.

    ``norm`` computes ``sqrt(dot(x, x))`` for this case, so this is bitwise
    identical.
    """
    return math.sqrt(v @ v)


# ------------------------------------------------------------------------------
# The secular equation
# ------------------------------------------------------------------------------


def _make_seceqn(eigval, alpha, delta):
    """
    Build a scalar callable evaluating

        1 / delta - 1 / ||s(lambda)||.

    ``_trdog`` only ever produces subproblems of dimension 1 or 2, and those
    two cases are specialized into closures over Python floats.

    The zero/NaN handling reproduces the reference exactly:

        w == 0            -> 1 / delta
        sum == 0          -> 1 / delta - inf   (sqrt(1/0) == inf)
        sum is NaN        -> 1 / delta         (value clamped to 0)
    """
    n = eigval.size
    inv_delta = 1.0 / delta

    if n == 1:
        e0 = float(eigval[0])
        a0 = float(alpha[0])

        def seceqn(lam):
            w0 = e0 + lam

            if w0 == 0.0:
                return inv_delta

            q0 = a0 / w0
            sm = q0 * q0

            if sm != sm:
                return inv_delta

            if sm == 0.0:
                return inv_delta - _INF

            return inv_delta - math.sqrt(1.0 / sm)

        return seceqn

    if n == 2:
        e0 = float(eigval[0])
        a0 = float(alpha[0])
        e1 = float(eigval[1])
        a1 = float(alpha[1])

        def seceqn(lam):
            w0 = e0 + lam

            if w0 == 0.0:
                return inv_delta

            w1 = e1 + lam

            if w1 == 0.0:
                return inv_delta

            q0 = a0 / w0
            q1 = a1 / w1
            sm = q0 * q0 + q1 * q1

            if sm != sm:
                return inv_delta

            if sm == 0.0:
                return inv_delta - _INF

            return inv_delta - math.sqrt(1.0 / sm)

        return seceqn

    # General fallback, unreachable from _trdog.
    def seceqn(lam):
        w = eigval + lam

        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(
                w != 0,
                alpha / np.where(w != 0, w, 1),
                np.inf,
            )
            m = m * m
            value = np.sqrt(1.0 / np.sum(m))

        if np.isnan(value):
            value = 0.0

        return inv_delta - value

    return seceqn


def _secular_eqn_root(x, itbnd, seceqn, tol):
    """
    Zero of the secular equation to the RIGHT of the starting point ``x``.

    ``seceqn`` is the closure returned by ``_make_seceqn``; the control flow is
    unchanged from the reference.
    """
    itfun = 0
    dx = abs(x) / 2 if x != 0 else 0.5

    a = x
    c = a

    fa = seceqn(a)
    itfun += 1

    b = x + 1
    fb = seceqn(b)
    itfun += 1

    # Find change of sign.
    while (fa > 0) == (fb > 0):
        dx = 2 * dx

        if (fa > 0) != (fb > 0):
            break

        b = x + dx
        fb = seceqn(b)
        itfun += 1

        if itfun > itbnd:
            break

    fc = fb
    d = e = b - a

    # Main loop, exit from middle of the loop.
    while fb != 0:
        # Ensure that:
        #   b is the best result so far,
        #   a is the previous value of b,
        #   c is on the opposite side of the zero from b.
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

        # Choose bisection or interpolation.
        if abs(e) < toler or abs(fa) <= abs(fb):
            d = m
            e = m

        else:
            s = fb / fa

            if a == c:
                # Linear interpolation.
                p = 2.0 * m * s
                q = 1.0 - s

            else:
                # Inverse quadratic interpolation.
                q = fa / fc
                r = fb / fc
                p = s * (
                    2.0 * m * q * (q - r)
                    - (b - a) * (r - 1.0)
                )
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            else:
                p = -p

            if (
                2.0 * p < 3.0 * m * q - abs(toler * q)
                and p < abs(0.5 * e * q)
            ):
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

        fb = seceqn(b)
        itfun += 1

    return b, itfun


def _coeff_scalar(wi, ai):
    """One entry of ``alpha ./ w``, with the reference's zero/NaN handling."""
    if wi != 0.0:
        c = ai / wi
    elif ai != 0.0:
        c = _INF
    else:
        c = 0.0

    if c != c:
        c = 0.0

    return c


def _coeff_from_w(w, alpha):
    """
    The ``coeff = alpha ./ w`` step of ``trust``.
    """
    n = alpha.size

    if n == 1:
        return np.array(
            [_coeff_scalar(float(w[0]), float(alpha[0]))]
        )

    if n == 2:
        return np.array(
            [
                _coeff_scalar(float(w[0]), float(alpha[0])),
                _coeff_scalar(float(w[1]), float(alpha[1])),
            ]
        )

    coeff = np.zeros_like(alpha)

    nz = w != 0

    coeff[nz] = alpha[nz] / w[nz]
    coeff[~nz & (alpha != 0)] = np.inf
    coeff[np.isnan(coeff)] = 0.0

    return coeff


def _argmin2(e0, e1):
    """
    ``int(np.argmin(eigval))`` for a two-element array, NaN semantics included
    (``argmin`` returns the index of a NaN, earliest first).
    """
    if e0 != e0:
        return 0
    if e1 != e1:
        return 1
    return 0 if e0 <= e1 else 1


def _trust(g, H, delta):
    """
    Exact solution of the trust-region problem

        min { g's + 0.5 s'Hs : ||s|| <= delta }

    via the full eigen-decomposition and secular equation.
    """
    tol = 1e-12
    tol2 = 1e-8
    itbnd = 50

    key = 0
    lam_val = 0.0

    if type(g) is not _ndarray or g.dtype is not _float64 or g.ndim != 1:
        g = np.atleast_1d(np.asarray(g, dtype=float))

    if type(H) is not _ndarray or H.dtype is not _float64 or H.ndim != 2:
        H = np.atleast_2d(np.asarray(H, dtype=float))

    shape = H.shape

    # ``_trdog`` produces trust-region subproblems of dimension at most 2. The
    # 1-D problem is handled directly; the 2-D case keeps the eig/eigh split.
    if shape == (1, 1):
        h00 = H[0, 0]

        eigval = np.array([h00], dtype=float)
        V = _V_1X1

        jmin = 0
        mineig = h00

    else:
        if shape == (2, 2):
            # Equivalent to np.array_equal(H, H.T) for a 2x2 matrix.
            h00 = H[0, 0]
            h11 = H[1, 1]

            symmetric = (
                h00 == h00
                and h11 == h11
                and H[0, 1] == H[1, 0]
            )

        else:
            symmetric = np.array_equal(H, H.T)

        if symmetric:
            eigval, V = np.linalg.eigh(H)

        else:
            eigval, V = np.linalg.eig(H)
            eigval = np.real(eigval)
            V = np.real(V)

        if eigval.size == 2:
            jmin = _argmin2(eigval[0], eigval[1])
        else:
            jmin = int(np.argmin(eigval))

        mineig = eigval[jmin]

    alpha = -(V.T @ g)

    # sign(a) + (a == 0); sign(0) is 0 so the total is 1, and NaN propagates.
    aj = float(alpha[jmin])

    if aj > 0.0:
        sig = 1.0
    elif aj < 0.0:
        sig = -1.0
    elif aj == 0.0:
        sig = 1.0
    else:
        sig = _NAN

    s = None
    laminit = 0.0

    if mineig > 0:
        # Positive-definite case.
        coeff = alpha / eigval
        lam_val = 0.0

        s = V @ coeff

        if _norm(s) <= 1.2 * delta:
            key = 1
        else:
            laminit = 0.0

    else:
        # Indefinite case.
        laminit = -mineig

    if key == 0:
        laminit = float(laminit)

        seceqn = _make_seceqn(eigval, alpha, delta)

        if seceqn(laminit) > 0:
            b, _ = _secular_eqn_root(laminit, itbnd, seceqn, tol)

            if abs(seceqn(b)) <= tol2:
                lam_val = b
                key = 2

                s = V @ _coeff_from_w(eigval + lam_val, alpha)

                nrms = _norm(s)

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
            # np.where(|eigval + lam| < 10*eps*max(|eigval|, 1), 0, alpha).
            k = eigval.size

            if k <= 2:
                vals = []

                for i in range(k):
                    ei = float(eigval[i])
                    aei = abs(ei)

                    if abs(ei + lam_val) < _TEN_EPS * (aei if aei > 1.0 else 1.0):
                        vals.append(0.0)
                    else:
                        vals.append(float(alpha[i]))

                alpha = np.array(vals)

            else:
                arg = (
                    np.abs(eigval + lam_val)
                    < 10 * _EPS * np.maximum(np.abs(eigval), 1)
                )

                alpha = np.where(arg, 0.0, alpha)

        s = V @ _coeff_from_w(eigval + lam_val, alpha)

        nrms = _norm(s)

        if key > 2:
            if nrms < 0.8 * delta:
                s = (
                    s
                    + np.sqrt(delta**2 - nrms**2)
                    * sig
                    * V[:, jmin]
                )

            elif nrms > 1.2 * delta:
                # ``alpha`` was rewritten above, so the secular equation has to
                # be rebuilt against the new coefficients.
                b, _ = _secular_eqn_root(
                    laminit,
                    itbnd,
                    _make_seceqn(eigval, alpha, delta),
                    tol,
                )

                lam_val = b

                s = V @ _coeff_from_w(eigval + lam_val, alpha)

    val = g @ s + 0.5 * (s @ (H @ s))

    return s, val


# ------------------------------------------------------------------------------
# The 2-D subspace trial step (trdog.m, specialized to the unbounded case)
# ------------------------------------------------------------------------------


def _newton_direction(A, g):
    """
    The (Gauss-)Newton direction, i.e. the solution of

        (A'A)p = -g.

    ``pcgr`` preconditions with the exact QR factor of ``A`` produced by
    ``aprecon`` (``PrecondBandWidth = Inf``), so its single permitted CG
    iteration reproduces the direct solve.

    Only the direction survives -- ``trdog`` normalizes ``p`` immediately.
    """
    R = _qr_r(A)

    r = -g

    try:
        # trans="T" -> 1, trans="N" -> 0.
        tmp = _solve_tri(R, r, 1)

        z = _solve_tri(R, tmp, 0, overwrite_b=True)

    except (LinAlgError, ValueError):
        return r, 1

    if not np.isfinite(z).all():
        return r, 1

    # Preserve the original floating-point association:
    #
    #     A.T @ (A @ z)
    #
    # Do not replace this with algebraically equivalent forms such as
    # (A.T @ A) @ z.
    ww = A.T @ (A @ z)

    denom = z @ ww

    if denom <= 0:
        nz = _norm(z)

        return (z if nz == 0 else z / nz), 0

    return ((r @ z) / denom * z), 1


def _trdog(g, A, delta, Z):
    """
    Reflected (2-D) trust-region trial step, with no finite bounds.
    """
    n = len(g)

    grad = g

    # D is identity, so the scaled gradient is just g.
    posdef = 1

    multi = n > 1

    gnorm = _norm(grad) if multi else None

    if Z is None:
        v1, posdef = _newton_direction(A, g)

        nv1 = _norm(v1)

        if nv1 > 0:
            v1 = v1 / nv1

        cols = [v1]

        if multi:
            # posdef < 1 would use D*sign(grad). D is identity and, with A'A
            # positive semi-definite, pcgr never reports negative curvature
            # anyway.
            v2 = grad / gnorm if gnorm > 0 else grad

            v2 = v2 - v1 * (v1 @ v2)

            nrmv2 = _norm(v2)

            if nrmv2 > _SQRT_EPS:
                cols.append(v2 / nrmv2)

        # np.column_stack for 1-D inputs.
        if len(cols) == 1:
            Z = cols[0].reshape(-1, 1)
        else:
            Z = np.empty((n, 2))
            Z[:, 0] = cols[0]
            Z[:, 1] = cols[1]

    # Reduce to the chosen subspace.
    #
    # D = I and DG = 0, so:
    #
    #     M = Z' (A'A) Z
    #
    # Preserve the original floating-point association exactly:
    #
    #     Z.T @ (A.T @ (A @ Z))
    #
    # Do NOT simplify to (A @ Z).T @ (A @ Z).
    ZT = Z.T

    MM = ZT @ (A.T @ (A @ Z))

    rhs = ZT @ grad

    # Determine the 2-D trust-region solution.
    st, _ = _trust(rhs, MM, delta)

    ss = Z @ st
    s = ss

    if n <= 2:
        nan_step = s[0] != s[0] or (n == 2 and s[1] != s[1])
    else:
        nan_step = np.isnan(s).any()

    if nan_step:
        raise FloatingPointError("NaN in the trust region step")

    qpval1 = rhs @ st + 0.5 * (st @ (MM @ st))

    # The reflected direction is only evaluated when the step is truncated at
    # a bound, which cannot happen here, so qpval3 remains infinite.
    if not multi:
        return s, ss, qpval1, posdef, Z

    # Evaluate along the gradient direction.
    ZZ = (grad / (gnorm + (gnorm == 0))).reshape(-1, 1)

    # Again preserve:
    #
    # ZZ.T @ (A.T @ (A @ ZZ))
    ZZT = ZZ.T

    MMg = ZZT @ (A.T @ (A @ ZZ))

    rhsg = ZZT @ grad

    stg, _ = _trust(rhsg, MMg, delta)

    ssg = ZZ @ stg
    sg = ssg

    qpval2 = rhsg @ stg + 0.5 * (stg @ (MMg @ stg))

    # Choose the best of the two steps.
    if qpval2 <= qpval1:
        return sg, ssg, qpval2, posdef, Z

    return s, ss, qpval1, posdef, Z


# ------------------------------------------------------------------------------
# Finite-difference Jacobian
# ------------------------------------------------------------------------------


def _findiff_jac(
    resid_fn,
    x,
    fvec,
    diff_min_change,
    diff_max_change,
):
    """
    Forward finite-difference Jacobian.
    """
    n = len(x)

    J = np.empty((len(fvec), n))

    xp = x.copy()

    sqrt_eps = float(_SQRT_EPS)

    for j in range(n):
        xj = float(x[j])

        # MATLAB sign'(0) behaviour here is +1.
        sgn = -1.0 if xj < 0.0 else 1.0

        chg = sgn * sqrt_eps * (abs(xj) if abs(xj) > 1.0 else 1.0)

        mag = abs(chg)

        if mag < diff_min_change:
            mag = diff_min_change
        if mag > diff_max_change:
            mag = diff_max_change

        # np.sign(chg) is +/-1 here, and 0 only if chg is 0.
        if chg > 0.0:
            chg = mag
        elif chg < 0.0:
            chg = -mag
        else:
            chg = 0.0

        xp[j] = xj + chg

        col = resid_fn(xp)

        # (col - fvec) / chg.
        np.subtract(col, fvec, out=col)
        col /= chg

        J[:, j] = col

        # Restore the original value before perturbing the next parameter.
        xp[j] = xj

    return J, n


# ------------------------------------------------------------------------------
# The main solver
# ------------------------------------------------------------------------------


def lsqcurvefit_trr(
    fun,
    x0,
    xdata,
    ydata,
    tol_fun=1e-6,
    tol_x=1e-6,
    max_iter=400,
    max_fun_evals=600,
    diff_min_change=1e-8,
    diff_max_change=0.1,
):
    """
    Minimize

        sum((fun(p, xdata) - ydata)**2)

    over ``p`` the way MATLAB's ``lsqcurvefit`` does with

        Algorithm = 'trust-region-reflective'

    and no bounds.

    Parameters
    ----------
    fun : callable
        ``fun(p, xdata)`` returning the model values.

    x0 : array-like
        Starting point for the coefficients (MATLAB's ``StartPoint``).

    xdata, ydata : array-like
        Data to fit.

    tol_fun : float
        Function tolerance.

    tol_x : float
        Parameter-step tolerance.

    max_iter : int
        Maximum number of trust-region iterations.

    max_fun_evals : int
        Maximum logical number of function evaluations.

    diff_min_change : float
        Minimum finite-difference perturbation.

    diff_max_change : float
        Maximum finite-difference perturbation.

    Returns
    -------
    numpy.ndarray
        Fitted coefficients.
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)

    x = np.asarray(x0, dtype=float).ravel().copy()

    n = len(x)

    def resid(p):
        v = fun(p, xdata)

        if type(v) is not _ndarray or v.dtype is not _float64 or v.ndim != 1:
            v = np.asarray(v, dtype=float).ravel()

        return v - ydata

    iter_count = 0
    num_fun_evals = 1

    delta = 10.0

    nrmsx = 1.0
    ratio = 0.0
    oval = np.inf

    Z = None
    posdef = 1

    fvec = resid(x)

    A, fdevals = _findiff_jac(
        resid,
        x,
        fvec,
        diff_min_change,
        diff_max_change,
    )

    num_fun_evals += fdevals

    if not np.isfinite(A).all():
        raise FloatingPointError("Jacobian undefined at the initial point")

    g = A.T @ fvec
    val = fvec @ fvec

    if not np.isfinite(val) or not np.isfinite(g).all():
        raise FloatingPointError(
            "Objective or its gradient undefined at the initial point"
        )

    ex = 0

    while not ex:
        # With no finite bounds, definev returns v = +/-1 and dv = 0,
        # therefore |v| = 1.
        optnrm = np.abs(g).max()

        # --------------------------------------------------------------
        # Test for convergence
        # --------------------------------------------------------------

        diff = abs(oval - val)

        oval = val

        if optnrm < tol_fun and posdef == 1:
            ex = 1

        elif (
            nrmsx < 0.9 * delta
            and ratio > 0.25
            and diff < tol_fun * (1 + abs(oval))
        ):
            ex = 2

        elif iter_count > 1 and nrmsx < tol_x:
            ex = 3

        elif iter_count > max_iter or num_fun_evals > max_fun_evals:
            ex = 4

        if ex:
            break

        # --------------------------------------------------------------
        # Determine trust-region correction
        # --------------------------------------------------------------

        sx, snod, qp, posdef, Z = _trdog(g, A, delta, Z)

        nrmsx = _norm(snod)

        newx = x + sx

        # --------------------------------------------------------------
        # Evaluate trial point
        # --------------------------------------------------------------

        newfvec = resid(newx)

        newval = newfvec @ newfvec

        # The original implementation evaluates 1 residual plus n residuals for
        # a finite-difference Jacobian at every trial point, even when the point
        # is rejected. Match that logical evaluation count so the max_fun_evals
        # stopping behaviour of the original solver is preserved.
        num_fun_evals += 1 + n

        # --------------------------------------------------------------
        # Update trust-region radius
        # --------------------------------------------------------------

        if not np.isfinite(newfvec).all():
            delta = min(nrmsx / 20, delta / 20)

        else:
            # ``aug`` involves dv, which is zero without finite bounds.
            ratio = (0.5 * (newval - val)) / qp

            if ratio >= 0.75 and nrmsx >= 0.9 * delta:
                delta = 2 * delta

            elif ratio <= 0.25:
                delta = min(nrmsx / 4, delta / 4)

        # --------------------------------------------------------------
        # Accept or reject trial point
        # --------------------------------------------------------------

        if newval < val:
            # The finite-difference Jacobian is required only for accepted
            # trial points. On rejected points it is discarded immediately
            # by the original algorithm.
            newA, _ = _findiff_jac(
                resid,
                newx,
                newfvec,
                diff_min_change,
                diff_max_change,
            )

            g = newA.T @ newfvec

            x = newx
            val = newval
            A = newA

            # Recompute the subspace after accepting a new point.
            Z = None

            fvec = newfvec

        iter_count += 1

    return x


# ------------------------------------------------------------------------------
# Convenience fits
# ------------------------------------------------------------------------------


def _exp1_model(p, xd):
    return p[0] * np.exp(p[1] * xd)


def _poly1_model(p, xd):
    return p[0] * xd + p[1]


def fit_exp1(x, y, start_point=(1.0, -0.2)):
    """
    Fit

        a * exp(b*x)

    as MATLAB's

        fit(x, y, fittype('a*exp(b*x)', 'options', s))

    does, where ``s`` sets ``Method`` to ``'NonlinearLeastSquares'`` and
    ``StartPoint`` to ``start_point``.

    Returns
    -------
    tuple of float
        Fitted ``(a, b)``.
    """
    p = lsqcurvefit_trr(_exp1_model, start_point, x, y)

    return float(p[0]), float(p[1])


def fit_poly1(x, y, start_point=(0.1, 0.0)):
    """
    Fit

        a*x + b

    as MATLAB's

        fit(x, y, fittype('a*x+b', 'options', s))

    does with ``Method = 'NonlinearLeastSquares'`` and ``StartPoint`` set to
    ``start_point``.

    The model is linear in its coefficients, but the trust-region iteration can
    still stop short of the exact least-squares solution on the toolbox's
    default tolerances, so this does not generally agree with ``numpy.polyfit``.

    Returns
    -------
    tuple of float
        Fitted ``(a, b)``.
    """
    p = lsqcurvefit_trr(_poly1_model, start_point, x, y)

    return float(p[0]), float(p[1])


# ------------------------------------------------------------------------------
# Goodness of fit
# ------------------------------------------------------------------------------


def goodness_of_fit(y, y_fit, num_coeffs):
    """
    Goodness-of-fit statistics matching the ``gof`` struct returned by
    MATLAB's ``fit``.

    Note that ``rmse`` is the standard error

        sqrt(SSE / dfe)

    rather than the root-mean-square of the residuals.

    Parameters
    ----------
    y : array-like
        Observed values.

    y_fit : array-like
        Fitted values.

    num_coeffs : int
        Number of fitted coefficients, which determines the error degrees of
        freedom.

    Returns
    -------
    dict
        Keys are ``sse``, ``rsquare``, ``adjrsquare`` and ``rmse``.
    """
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)

    n = len(y)
    dfe = n - num_coeffs

    residual = y - y_fit

    sse = np.sum(residual**2)

    centered = y - np.mean(y)

    sst = np.sum(centered**2)

    rsquare = 1 - sse / sst

    return {
        "sse": sse,
        "rsquare": rsquare,
        "adjrsquare": 1 - (1 - rsquare) * (n - 1) / dfe,
        "rmse": np.sqrt(sse / dfe),
    }