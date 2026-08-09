"""
Line-by-line ports of the two gradient-free/gradient-based optimizers that hctsa
inherits from the gpml toolbox (``Toolboxes/gpml/util``).

Both are transcribed faithfully rather than replaced by a SciPy equivalent. hctsa
runs ``minimize`` with a hard budget of 50 function evaluations and reports the
resulting hyperparameters *as feature values*, so the optimizer trajectory --
including its truncation point, its line-search branch structure and a couple of
its quirks -- is part of the definition of the feature, not an implementation
detail. Swapping in L-BFGS-B changes the answers.

This revision is a speed pass over the transcription. Every arithmetic
expression, its operand order and its dtype at the point of rounding are
unchanged, so the float64 bit patterns of every intermediate -- and hence the
branch decisions and the returned values -- are identical to the previous
version. What changed is only the scaffolding around the arithmetic: MATLAB
compatibility shims short-circuit on the real-valued fast path, scalar
bookkeeping is carried in Python floats rather than 0-d numpy scalars,
per-iteration constants are hoisted, and the dead initialisers before the
extrapolation loop are dropped.

The one deliberate exception is documented at ``_ERRSTATE`` below.

Ported functions
----------------
``minimize``
    Conjugate gradients (Polack-Ribiere) with a Wolfe-Powell line search using
    quadratic/cubic polynomial approximations.
    Carl Edward Rasmussen, 2001-2010. gpml ``util/minimize.m``.
``brentmin``
    Brent's one-dimensional minimisation (Numerical Recipes S10.2).
    Hannes Nickisch, 2010. gpml ``util/brentmin.m``.
"""

import math

import numpy as np

__all__ = ['minimize', 'brentmin']

_EPS = np.finfo(float).eps
_REALMIN = np.finfo(float).tiny

# Python-float copies of the above, for the scalar loops. Same bit patterns,
# ~10x cheaper per operation than the 0-d numpy scalars they replace.
_EPS_F = float(_EPS)
# brentmin recomputed these on every call; they are constants.
_SEPS = math.sqrt(_EPS_F)                       # was np.sqrt(_EPS)
_GOLD = 0.5 * (3.0 - math.sqrt(5.0))            # golden ratio, was np.sqrt(5.0)

_errstate = np.errstate           # bound once; numpy 2.x forbids reuse of an
                                  # errstate instance, so these are still fresh

_isfinite = math.isfinite


def _key(v):
    # MATLAB's ``isreal`` test collapses away here: a non-complex value always
    # has ``v.imag == 0``, so the imaginary-part test alone is equivalent to the
    # original ``iscomplexobj(v) and imag(v) != 0``.
    if v.imag != 0:
        return (abs(v), np.angle(v))
    return (float(v.real),)


def _mmin(a, b):
    return a if _key(a) <= _key(b) else b


def _mmax(a, b):
    return a if _key(a) >= _key(b) else b


def _isreal(v):
    """MATLAB ``isreal``: false as soon as the value carries an imaginary part.

    ``np.complex128`` subclasses Python's ``complex``, so the single isinstance
    covers both the numpy and the builtin complex scalars that ``_sqrt`` can
    return; ``np.float64`` subclasses ``float``, not ``complex``.
    """
    return not isinstance(v, complex)


def _bad(v):
    """MATLAB ``isnan(v) || isinf(v)``, safe for complex v."""
    if isinstance(v, complex):
        return not (_isfinite(v.real) and _isfinite(v.imag))
    return not _isfinite(v)


def _sqrt(v):
    """MATLAB ``sqrt``: returns a complex root for negative arguments.

    IEEE-754 square root is correctly rounded, so the real branch agrees with
    ``np.emath.sqrt`` bit for bit. The ``v >= 0`` guard is false for NaN, which
    routes NaN through the original path (and back out as NaN).
    """
    if v >= 0.0:
        return math.sqrt(v)
    return np.emath.sqrt(v)


def _sign_or_one(v):
    """MATLAB ``sign(v) + (v == 0)``: +1 at zero rather than 0.

    NaN falls through to the final return, matching ``np.sign(nan) + 0 -> nan``.
    Note ``-0.0`` compares equal to zero and so yields +1, as before.
    """
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    if v == 0.0:
        return 1.0
    return float(v)


# ------------------------------------------------------------------------------
# minimize.m
# ------------------------------------------------------------------------------

def minimize(x0, f, length, *args):
    """
    Minimize a differentiable multivariate function using conjugate gradients.

    A port of gpml's ``minimize.m``. The ``unwrap``/``rewrap`` machinery of the
    original is dropped: ``x0`` is already a flat vector, so the caller is
    responsible for packing/unpacking any structured parameter set (gpml
    alphabetises struct fields when it flattens, i.e. ``cov``, ``lik``, ``mean``).

    Parameters
    ----------
    x0 : array-like
        Initial guess, a flat vector.
    f : callable
        ``f(x, *args) -> (fval, grad)`` with ``grad`` the same shape as ``x``.
        May raise, or return non-finite values, during extrapolation; this is
        handled the way MATLAB's ``try``/``catch`` handles it (bisect and retry).
    length : int
        Length of the run. Positive gives the maximum number of line searches,
        negative gives the maximum number of function evaluations. hctsa uses
        ``-50``.
    *args
        Extra arguments forwarded to ``f``.

    Returns
    -------
    x : np.ndarray
        The returned solution.
    fx : list of float
        Function values indicating the progress made.
    i : int
        Number of iterations (line searches or function evaluations, depending
        on the sign of ``length``) used at termination.
    """
    INT = 0.1      # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0      # extrapolate maximum 3 times the current step-size
    MAX = 20       # max 20 function evaluations per line search
    RATIO = 10     # maximum allowed slope ratio
    SIG = 0.1
    RHO = SIG / 2  # Wolfe-Powell constants

    if np.size(length) == 2:
        red = float(np.ravel(length)[1])
        length = int(np.ravel(length)[0])
    else:
        red = 1.0
        length = int(length)

    X = np.asarray(x0, dtype=float).ravel().copy()

    # Loop-invariant products, hoisted out of the line search.
    abslen = abs(length)
    count_ls = length > 0                              # count iterations?!
    count_fe = length < 0                              # count epochs?!

    i = 0                                          # zero the run length counter
    ls_failed = 0                           # no previous line search has failed
    f0, df0 = f(X, *args)                    # get function value and gradient
    f0 = float(f0)
    df0 = np.array(df0, dtype=float).ravel()
    fX = [f0]
    i = i + count_fe                                            # count epochs?!
    s = -df0
    d0 = float(-s.dot(s))  # initial search direction (steepest) and slope
    x3 = red / (1 - d0)                            # initial step is red/(|s|+1)

    while i < abslen:                                       # while not finished
        i = i + count_ls                                    # count iterations?!

        X0 = X.copy()
        F0 = f0
        dF0 = df0.copy()                        # make a copy of current values
        M = MAX if count_ls else min(MAX, -length - i)

        # SIG*d0 is fixed for the whole line search. RHO*d0 is deliberately
        # NOT hoisted: the original groups as (x3*RHO)*d0, and float multiply
        # is not associative.
        sig_d0 = SIG * d0
        neg_sig_d0 = -sig_d0

        while True:                    # keep extrapolating as long as necessary
            x2, f2, d2 = 0.0, f0, d0
            f3, df3 = f0, df0                # f0/df0 are not mutated in here
            success = 0
            while not success and M > 0:
                try:
                    M = M - 1
                    i = i + count_fe                            # count epochs?!
                    f3_try, df3_try = f(X + x3 * s, *args)
                    f3_try = float(f3_try)
                    df3_try = np.array(df3_try, dtype=float).ravel()
                    if not _isfinite(f3_try) \
                            or not np.isfinite(df3_try).all():
                        raise FloatingPointError(' ')
                    f3, df3 = f3_try, df3_try
                    success = 1
                except Exception:           # catch any error which occurred in f
                    x3 = (x2 + x3) / 2                   # bisect and try again
            if f3 < F0:                                    # keep best values
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3.copy()
            d3 = float(df3.dot(s))                                  # new slope
            if d3 > sig_d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break                             # are we done extrapolating?
            x1, f1, d1 = x2, f2, d2                # move point 2 to point 1
            x2, f2, d2 = x3, f3, d3                # move point 3 to point 2
            dx = x2 - x1
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * dx         # cubic extrapolation
            B = 3 * (f2 - f1) - (2 * d1 + d2) * dx
            with _errstate(divide='ignore', invalid='ignore'):
                # num. error possible, ok! -- division by zero yields Inf as in MATLAB
                x3 = x1 - np.float64(d1) * dx ** 2 \
                    / (B + _sqrt(B * B - A * d1 * dx))
            if (not _isreal(x3)) or _bad(x3) or x3.real < 0:
                x3 = x2 * EXT                     # extrapolate maximum amount
            elif x3 > x2 * EXT:      # new point beyond extrapolation limit?
                x3 = x2 * EXT
            elif x3 < x2 + INT * dx:         # new point too close to previous?
                x3 = x2 + INT * dx
            x3 = float(x3)
        # end extrapolation

        x4, f4, d4 = x3, f3, d3                                 # init point 4
        while (abs(d3) > neg_sig_d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:            # choose subinterval
                x4, f4, d4 = x3, f3, d3            # move point 3 to point 4
            else:
                x2, f2, d2 = x3, f3, d3            # move point 3 to point 2
            dx = x4 - x2
            with _errstate(divide='ignore', invalid='ignore'):
                if f4 > f0:                             # quadratic interpolation
                    x3 = x2 - (np.float64(0.5) * d2 * dx ** 2) \
                        / (f4 - f2 - d2 * dx)
                else:                                       # cubic interpolation
                    A = 6 * np.float64(f2 - f4) / dx + 3 * (d4 + d2)
                    B = 3 * (f4 - f2) - (2 * d2 + d4) * dx
                    x3 = x2 + (_sqrt(B * B - A * d2 * dx ** 2) - B) / A
            if _bad(x3):
                x3 = (x2 + x4) / 2      # if we had a numerical problem, bisect
            x3 = _mmax(_mmin(x3, x4 - INT * dx), x2 + INT * dx)
            x3 = float(x3.real)                   # the clamp resolves any root
            f3, df3 = f(X + x3 * s, *args)
            f3 = float(f3)
            df3 = np.array(df3, dtype=float).ravel()
            if f3 < F0:                                    # keep best values
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3.copy()
            M = M - 1
            i = i + count_fe                                # count epochs?!
            d3 = float(df3.dot(s))                                 # new slope
        # end interpolation

        if abs(d3) < neg_sig_d0 and f3 < f0 + x3 * RHO * d0:   # line search ok
            X = X + x3 * s
            f0 = f3
            fX.append(f0)                                # update variables
            s = (df3.dot(df3) - df0.dot(df3)) / (df0.dot(df0)) * s - df3
            df0 = df3                                    # swap derivatives
            d3 = d0
            d0 = float(df0.dot(s))
            if d0 > 0:                          # new slope must be negative
                s = -df0
                d0 = float(-s.dot(s))       # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3 / (d0 - _REALMIN))   # slope ratio, max RATIO
            ls_failed = 0                       # this line search did not fail
        else:
            X = X0
            f0 = F0
            df0 = dF0                            # restore best point so far
            if ls_failed or i > abslen:        # line search failed twice in a row
                break                    # or we ran out of time, so we give up
            s = -df0
            d0 = float(-s.dot(s))                             # try steepest
            x3 = 1 / (1 - d0)
            ls_failed = 1                             # this line search failed

    return X, fX, i


# ------------------------------------------------------------------------------
# brentmin.m
# ------------------------------------------------------------------------------

def brentmin(xlow, xupp, Nitmax, tol, f, nout, *args):
    """
    Brent's minimization method in one dimension.

    A port of gpml's ``brentmin.m`` (Numerical Recipes S10.2). Given a function
    ``f`` and a search interval, isolates the minimum to a fractional precision
    of about ``tol``.

    Parameters
    ----------
    xlow, xupp : float
        Search interval, such that ``xlow <= xmin <= xupp``.
    Nitmax : int
        Maximum number of function evaluations made by the routine.
    tol : float
        Fractional precision.
    f : callable
        ``f(x, *args) -> (y, extra_1, ..., extra_nout)``; a bare scalar return is
        accepted when ``nout == 0``.
    nout : int
        Number of outputs of ``f`` beyond the ``y`` value.
    *args
        Extra arguments forwarded to ``f``.

    Returns
    -------
    xmin : float
        Abscissa of the minimum found.
    fmin : float
        Minimal function value.
    funccount : int
        Number of function evaluations made.
    extras : list
        The ``nout`` additional outputs of ``f``.

        Note these come from the *last* point evaluated, not necessarily from
        ``xmin`` -- gpml overwrites its ``varargout`` cell on every call to
        ``f``, and the final evaluation is not always the accepted one. That
        quirk is load-bearing: ``infLaplace``'s IRLS takes its next ``alpha``
        from here, so the behaviour is reproduced rather than fixed.
    """
    # The nout test was invariant across every call; resolve it once.
    # ``x`` is carried through the loop as a Python float (much cheaper than a
    # 0-d numpy scalar) but handed to ``f`` as np.float64, which is what the
    # all-numpy version produced -- identical bits, identical type, in case the
    # caller's ``f`` propagates the scalar type into its extra outputs.
    _f64 = np.float64
    if nout == 0:
        def _call(x):
            res = f(x, *args)
            if isinstance(res, tuple):
                return float(res[0]), []
            return float(res), []
    else:
        _hi = 1 + nout

        def _call(x):
            res = f(x, *args)
            return float(res[0]), list(res[1:_hi])

    extras = [None] * nout

    # tolerance is no smaller than machine's floating point precision
    tol = max(float(tol), _EPS_F)
    tol_3 = tol / 3.0

    # Evaluate endpoints
    fa, _ = _call(xlow)
    fb, _ = _call(xupp)
    funccount = 2
    # Compute the start point
    seps = _SEPS
    c = _GOLD                                                  # golden ratio
    a, b = float(xlow), float(xupp)
    v = a + c * (b - a)
    w = v
    xf = v
    d = 0.0
    e = 0.0
    x = xf
    fx, extras = _call(_f64(x))
    funccount += 1

    fv = fx
    fw = fx
    xm = 0.5 * (a + b)
    tol1 = seps * abs(xf) + tol_3
    tol2 = 2.0 * tol1

    # Main loop
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        gs = 1
        # Is a parabolic fit possible
        if abs(e) > tol1:
            gs = 0                                     # yes, so fit parabola
            r = (xf - w) * (fx - fv)
            q = (xf - v) * (fx - fw)
            p = (xf - v) * q - (xf - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = d

            # Is the parabola acceptable
            if abs(p) < abs(0.5 * q * r) and p > q * (a - xf) and p < q * (b - xf):
                d = p / q                       # parabolic interpolation step
                x = xf + d
                # f must not be evaluated too close to ax or bx
                if (x - a) < tol2 or (b - x) < tol2:
                    si = _sign_or_one(xm - xf)
                    d = tol1 * si
            else:
                gs = 1              # not acceptable, must do a golden section

        if gs:                            # a golden-section step is required
            e = (a - xf) if xf >= xm else (b - xf)
            d = c * e

        # The function must not be evaluated too close to xf
        si = _sign_or_one(d)
        x = xf + si * max(abs(d), tol1)
        fu, extras = _call(_f64(x))
        funccount += 1

        # Update a, b, v, w, x, xm, tol1, tol2
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            v, fv = w, fw
            w, fw = xf, fx
            xf, fx = x, fu
        else:                                                      # fu > fx
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fw or w == xf:
                v, fv = w, fw
                w, fw = x, fu
            elif fu <= fv or v == xf or v == w:
                v, fv = x, fu
        xm = 0.5 * (a + b)
        tol1 = seps * abs(xf) + tol_3
        tol2 = 2.0 * tol1

        if funccount >= Nitmax:            # typically we should not get here
            break

    # check that endpoints are less than the minimum found
    if fa < fx and fa <= fb:
        return xlow, fa, funccount, extras     # endpoints pass through as given
    if fb < fx:
        return xupp, fb, funccount, extras

    return _f64(xf), fx, funccount, extras     # else np.float64, as before