# SPDX-License-Identifier: GPL-2.0-or-later
"""
Delay vector variance (DVV) curve and its IAAFT surrogate, ported from the DVV toolbox.

The DVV toolbox (``DVV_dvv.m``, ``DVV_surrogate.m``) is (c) Copyright Danilo P. Mandic 2008, with
the delay vector variance method of T. Gautama, D. P. Mandic and M. M. Van Hulle; it is distributed
under the GNU General Public License, version 2 or (at your option) any later version
(https://www.commsp.ee.ic.ac.uk/~mandic/dvv.htm). These Python ports are derived works, distributed
with pyhctsa under GPL-3.0-or-later (permitted by the "or later" clause).

Modified for pyhctsa: ported from MATLAB to Python (NumPy). The routines follow the toolbox
(including its choice of target index in ``DVV_dvv.m`` and, for the surrogate, the real-valued
branch of ``DVV_surrogate.m`` as modified in hctsa by B. D. Fulcher), vectorized, with the
reference delay vectors and starting permutations supplied by the caller, and with the target
variances computed in two passes (never negative, and 0 for targets equal up to rounding).

Reference: T. Gautama, D. P. Mandic and M. M. Van Hulle, "The delay vector variance method for
detecting determinism and nonlinearity in time series", Physica D 190(3-4), 167-176 (2004).
https://doi.org/10.1016/j.physd.2003.11.001
"""
import numpy as np


def _dvv_curve(x: np.ndarray, m: int, nsub: int, nd: float, ntv: int, tau: int,
               ref: np.ndarray) -> np.ndarray:
    # Delay vector variance (Gautama, Mandic & Van Hulle 2004): the target variance (of the value following each delay vector) of the sets of
    # delay vectors within distance rd of each of the nsub reference vectors ``ref`` (indices of
    # delay vectors), averaged over references and normalized by the variance of x, at ntv
    # values of rd spanning the mean distance between delay vectors +/- nd standard deviations.
    # Returns the ntv x 2 array of the standardized distance and the target variance.
    n = x.size
    n_dv = n - m * tau  # number of delay vectors (each with a target)
    dvs = x[np.arange(n_dv)[:, None] + tau * np.arange(m)[None, :]]
    d = np.sqrt(np.sum((dvs[:, None, :] - dvs[ref][None, :, :]) ** 2, axis=2))  # n_dv x nsub

    # Mean and standard deviation of the distances over all (reference, vector) pairs except
    # each reference with itself
    count = nsub * n_dv - nsub
    avg = np.sum(d) / count
    sd = np.sqrt((np.sum((d - avg) ** 2) - nsub * avg ** 2) / (count - 1))
    rd = avg - nd * sd + (2 * nd * sd * np.arange(ntv)) / (ntv - 1)

    # For each reference, the variance of the targets of the delay vectors within each rd (the set
    # within rd is a leading run of the targets sorted by distance, whatever the order among equal
    # distances; the reference vector itself is excluded); only sets of at least 30 count. Each
    # variance is computed in two passes, the mean and then the mean squared deviation from it
    # (Chan, Golub and LeVeque, Am. Stat. 37, 242 (1983)), so it is never negative and is exactly 0
    # for equal targets; one below tol^2, with tol = 1e-12 of the range of x, is that of targets
    # equal up to rounding, and is 0.
    tol = 1e-12 * (np.max(x) - np.min(x))
    tot = np.zeros(ntv)
    num = np.zeros(ntv)
    for k in range(nsub):
        order = np.argsort(d[:, k], kind='stable')
        order = order[order != ref[k]]  # (the reference vector itself)
        ds = d[order, k]
        xs = x[order + m * tau]
        c = np.searchsorted(ds, rd, side='right')  # number of vectors within each rd
        for j in np.flatnonzero((c >= 30) & (rd > 0)):
            v = np.var(xs[:c[j]], ddof=1)
            tot[j] += 0.0 if v <= tol ** 2 else v
            num[j] += 1
    y = np.full(ntv, np.nan)
    has = num > 0
    y[has] = tot[has] / (num[has] * np.var(x, ddof=1))
    return np.column_stack([(rd - avg) / sd, y])


def _dvv_iaaft(x: np.ndarray, perm: np.ndarray, max_it: int = 100, tol: float = 1e-5) -> np.ndarray:
    # Iterated amplitude-adjusted Fourier transform surrogate (the DVV toolbox's surrogate.m,
    # real-valued branch as modified in hctsa): from the permutation ``perm`` of x, alternately
    # impose the amplitude spectrum and rank-order back onto the sorted values of x, returning the
    # iterate (of at most max_it) that matches the amplitude spectrum best.
    x_amp = np.abs(np.fft.fft(x))
    x_sorted = np.sort(x)
    r_prev = x[perm]
    mse, mse_prev = np.inf, 0.0
    best_mse, best_r = np.inf, r_prev
    it = 0
    while abs(mse - mse_prev) > tol and it < max_it:
        mse_prev = mse
        s = np.fft.ifft(x_amp * np.exp(1j * np.angle(np.fft.fft(r_prev)))).real
        r = np.empty_like(x)
        r[np.argsort(s, kind='stable')] = x_sorted
        mse = np.mean(np.abs(x_amp - np.abs(np.fft.fft(r))))
        if mse < best_mse:
            best_mse, best_r = mse, r
        r_prev = r
        it += 1
    return best_r
