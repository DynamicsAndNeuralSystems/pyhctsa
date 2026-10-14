# Derived from the GPML Matlab code v4.2 (http://gaussianprocess.org/gpml/code/matlab/doc/),
# Copyright (c) 2005-2018 Carl Edward Rasmussen & Hannes Nickisch. All rights reserved.
# Released under the FreeBSD License (BSD-2-Clause); see LICENSE in this directory.
# ``minimize`` is a port of minimize.m, Copyright (C) 2001-2010 Carl Edward Rasmussen.
# Python port and modifications (c) 2025-2026 the pyhctsa authors, distributed under GPL-3.0-or-later.
"""
Optimizers from the gpml Gaussian-process toolbox (Rasmussen & Nickisch), as used by hctsa.

``minimize`` is Rasmussen's conjugate-gradient minimizer. hctsa runs it with a hard budget of 50
function evaluations and reports the resulting hyperparameters *as feature values*, so the optimizer's trajectory (its line-search
branch structure and its truncation point) is part of the definition of the feature; a different
optimizer (e.g. L-BFGS-B) would give different feature values.

Algorithms
----------
``minimize``
    Conjugate gradients (Polak-Ribiere) with a Wolfe-Powell line search using quadratic and
    cubic polynomial approximations (Fletcher, *Practical Methods of Optimization*, 1987;
    Rasmussen, gpml ``util/minimize``).

The line search treats a cubic whose derivative has no real root (negative discriminant) as a
failed polynomial step and falls back to extrapolating by the maximum factor (extrapolation) or
bisecting the bracket (interpolation).
"""

import math

import numpy as np

__all__ = ['minimize']

_REALMIN = np.finfo(float).tiny

_errstate = np.errstate

_isfinite = math.isfinite


# ------------------------------------------------------------------------------
# minimize
# ------------------------------------------------------------------------------

def minimize(x0, f, length, *args):
    """
    Minimize a differentiable multivariate function using conjugate gradients.

    The gpml conjugate-gradient minimizer. ``x0`` is a flat vector; the caller packs and
    unpacks any structured parameter set (in the order ``cov``, ``lik``, ``mean``).

    Parameters
    ----------
    x0 : array-like
        Initial guess, a flat vector.
    f : callable
        ``f(x, *args) -> (fval, grad)`` with ``grad`` the same shape as ``x``.
        May raise, or return non-finite values, during extrapolation; this is
        caught and handled by bisecting the step and retrying.
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
            disc = B * B - A * d1 * dx          # discriminant of the cubic's stationary points
            if disc >= 0 or disc != disc:
                with _errstate(divide='ignore', invalid='ignore'):
                    # a zero denominator gives +-Inf or NaN, handled just below
                    x3 = x1 - np.float64(d1) * dx ** 2 \
                        / (B + math.sqrt(disc))
            else:
                x3 = np.nan                      # no real stationary point
            if not _isfinite(x3) or x3 < 0:
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
                    disc = B * B - A * d2 * dx ** 2
                    # no real root of the cubic's derivative: x3 = NaN, bisected below
                    x3 = x2 + (math.sqrt(disc) - B) / A if disc >= 0 else np.nan
            if not _isfinite(x3):
                x3 = (x2 + x4) / 2      # if we had a numerical problem, bisect
            x3 = float(max(min(x3, x4 - INT * dx), x2 + INT * dx))   # keep x3 inside the bracket
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
