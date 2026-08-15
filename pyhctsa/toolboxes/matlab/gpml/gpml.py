import numpy as np
from scipy.linalg import cho_solve, cholesky

from ..optimizers import brentmin

_EPS = np.finfo(float).eps

class CovSEisoNoise:
    """
    ``{'covSum', {'covSEiso', 'covNoise'}}`` for scalar (D=1) inputs.

    k(x, z) = sf^2 * exp(-maha(x, z) / 2) + sn^2 * delta(x, z)

    with ``maha(x, z) = (x - z)^2 / ell^2`` and hyperparameters
    ``hyp = [log(ell), log(sf), log(sn)]``. Two points count as equal in the
    cross-covariance term when their squared distance falls below ``eps^2``,
    matching ``covEye``.
    """

    n_hyp = 3

    @staticmethod
    def _maha(hyp, x, z=None):
        """
        gpml ``covMaha`` in 'iso' mode: squared Mahalanobis distance.

        The mean subtraction is not cosmetic -- gpml removes it to keep
        ``a^2 - 2ab + b^2`` stable, and which mean gets removed depends on
        whether this is the symmetric or the cross-covariance case.
        """
        ell2_inv = np.exp(-2.0 * hyp[0])
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        n = x.shape[0]
        if z is None:                                    # symmetric matrix D2xx
            mu = x.mean(axis=0)
            xc = x - mu
            Ax = xc * ell2_inv
            sax = np.sum(xc * Ax, axis=1)
            saz = sax
            Az = Ax
        else:                                                 # cross terms D2xz
            z = np.asarray(z, dtype=float).reshape(-1, 1)
            m = z.shape[0]
            mu = (m / (n + m)) * z.mean(axis=0) + (n / (n + m)) * x.mean(axis=0)
            xc = x - mu
            zc = z - mu
            Ax = xc * ell2_inv
            Az = zc * ell2_inv
            sax = np.sum(xc * Ax, axis=1)
            saz = np.sum(zc * Az, axis=1)
        # remove numerical noise at the end and ensure that D2 >= 0
        D2 = sax[:, None] + (saz[None, :] - 2.0 * (xc @ Az.T))
        return np.maximum(D2, 0.0)

    @classmethod
    def parts(cls, hyp, x, z=None):
        """
        Return ``(K_se, K_noise, D2)``, the two summands and the squared
        distances backing the SE term. ``z='diag'`` gives the diagonals.
        """
        hyp = np.asarray(hyp, dtype=float).ravel()
        sf2 = np.exp(2.0 * hyp[1])
        sn2 = np.exp(2.0 * hyp[2])
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        n = x.shape[0]

        if isinstance(z, str) and z == 'diag':
            D2 = np.zeros(n)
            return sf2 * np.ones(n), sn2 * np.ones(n), D2

        D2 = cls._maha(hyp, x, z)
        K_se = sf2 * np.exp(-D2 / 2.0)
        if z is None:
            K_noise = sn2 * np.eye(n)
        else:
            zz = np.asarray(z, dtype=float).reshape(-1, 1)
            d2 = (x - zz.T) ** 2
            K_noise = sn2 * (d2 < _EPS * _EPS).astype(float)
        return K_se, K_noise, D2

    @classmethod
    def K(cls, hyp, x, z=None):
        K_se, K_noise, _ = cls.parts(hyp, x, z)
        return K_se + K_noise

    @classmethod
    def dK(cls, hyp, x, Q, z=None):
        """
        Directional hyperparameter derivative ``dhyp(i) = tr(Q' dK/dhyp_i)``.

        Closed form for this covariance, equivalent to the chain of
        ``covSum`` -> ``covScale`` -> ``covMaha`` closures in gpml:

            d/dlog(ell) : sum(Q .* K_se .* D2)
            d/dlog(sf)  : 2 * sum(Q .* K_se)
            d/dlog(sn)  : 2 * sum(Q .* K_noise)
        """
        K_se, K_noise, D2 = cls.parts(hyp, x, z)
        return np.array([
            np.sum(Q * K_se * D2),
            2.0 * np.sum(Q * K_se),
            2.0 * np.sum(Q * K_noise),
        ])

def _lik_gauss_laplace(hyp_lik, y, mu):
    """gpml ``likGauss(hyp, y, mu, [], 'infLaplace')`` -> lp, dlp, d2lp, d3lp."""
    sn2 = np.exp(2.0 * hyp_lik)
    ymmu = y - mu
    lp = -ymmu ** 2 / (2 * sn2) - np.log(2 * np.pi * sn2) / 2
    dlp = ymmu / sn2
    d2lp = -np.ones_like(ymmu) / sn2
    d3lp = np.zeros_like(ymmu)
    return lp, dlp, d2lp, d3lp


def _lik_gauss_laplace_dhyp(hyp_lik, y, mu):
    """gpml ``likGauss(..., 'infLaplace', i)`` -> lp_dhyp, dlp_dhyp, d2lp_dhyp."""
    sn2 = np.exp(2.0 * hyp_lik)
    lp_dhyp = (y - mu) ** 2 / sn2 - 1
    dlp_dhyp = 2 * (mu - y) / sn2
    d2lp_dhyp = 2 * np.ones_like(mu) / sn2
    return lp_dhyp, dlp_dhyp, d2lp_dhyp

def _solve_chol(L, B):
    """gpml ``solve_chol``: ``L \\ (L' \\ B)`` for an upper-triangular L."""
    return cho_solve((L, False), B, check_finite=False)


def _ldB2_exact(W, K):
    """
    gpml ``apx/ldB2_exact`` for strictly positive W.

    A Gaussian likelihood gives ``W = 1/sn2 > 0`` everywhere, so only the
    symmetric Cholesky branch ``B = I + sW*K*sW`` is ever reachable; the
    asymmetric LU branch for negative W is not ported.
    """
    n = W.size
    sW = np.sqrt(W)
    L = cholesky(np.eye(n) + (sW[:, None] * sW[None, :]) * K,
                 lower=False, check_finite=False)
    ldB2 = np.sum(np.log(np.diag(L)))

    def solveKiW(r):
        # bsxfun(@times, ., sW) scales rows, for both vector and matrix r
        s = sW if r.ndim == 1 else sW[:, None]
        return s * _solve_chol(L, s * r)

    return ldB2, solveKiW, L, sW


def _ldB2_exact_cached(W, K, cache):
    """
    ``_ldB2_exact`` memoised on ``W``.

    Within a single ``inf_laplace`` call ``K`` is fixed and, for a Gaussian
    likelihood, ``W = 1/sn2`` is the same array on every Newton step, so the
    Cholesky is recomputed to the bit each time. Keying the cache on ``W`` alone
    is therefore safe (the cache lives no longer than a fixed ``K``); a changed
    ``W`` simply misses and recomputes, so the general contract is preserved.
    """
    key = W.tobytes()
    hit = cache.get(key)
    if hit is None:
        hit = _ldB2_exact(W, K)
        cache[key] = hit
    return hit


def _psi(alpha, m, K, hyp_lik, y):
    """gpml ``infLaplace/Psi``: alpha'*K*alpha/2 + neg. log lik at f = K*alpha+m."""
    f = K @ alpha + m
    lp, dlp, d2lp, _ = _lik_gauss_laplace(hyp_lik, y, f)
    W = -d2lp
    psi = alpha.dot(f - m) / 2 - np.sum(lp)
    return psi, f, alpha, dlp, W


def _irls(alpha, m, K, hyp_lik, y, maxit=20, Wmin=0.0, tol=1e-6, cache=None):
    """gpml ``infLaplace/irls``: Newton steps on Psi with a Brent line search."""
    smin_line, smax_line = 0, 2      # min/max line search steps size range
    nmax_line = 10                   # maximum number of line search steps
    thr_line = 1e-4                  # line search threshold

    if cache is None:
        cache = {}

    def psi_line(s, a, da):
        psi, f, a_out, dlp, W = _psi(a + s * da, m, K, hyp_lik, y)
        # gpml's Psi returns [psi, dpsi, f, alpha, dlp, W]; dpsi is unused by the
        # caller, so a placeholder keeps the output positions aligned.
        return psi, None, f, a_out, dlp, W

    f = K @ alpha + m
    _, dlp, d2lp, _ = _lik_gauss_laplace(hyp_lik, y, f)
    W = -d2lp
    Psi_new = _psi(alpha, m, K, hyp_lik, y)[0]
    Psi_old = np.inf
    it = 0
    while Psi_old - Psi_new > tol and it < maxit:                # begin Newton
        Psi_old = Psi_new
        it += 1
        W = np.maximum(W, Wmin)
        _, solveKiW, _, _ = _ldB2_exact_cached(W, K, cache)
        b = W * (f - m) + dlp
        dalpha = b - solveKiW(K @ b) - alpha       # Newton direction + line search
        _, Psi_new, _, extras = brentmin(
            smin_line, smax_line, nmax_line, thr_line, psi_line, 5, alpha, dalpha)
        _, f, alpha, dlp, W = extras
    return alpha


def inf_laplace(hyp, cov, x, y, want_dnlZ=True):
    """
    gpml ``infLaplace`` for a zero mean and a Gaussian likelihood.

    Returns ``(post, nlZ, dnlZ)`` where ``post`` carries ``alpha``, ``sW`` and a
    callable ``L``, matching gpml's posterior representation.
    """
    hyp_cov = np.asarray(hyp['cov'], dtype=float).ravel()
    hyp_lik = float(np.ravel(hyp['lik'])[0])
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size

    K = cov.K(hyp_cov, x)
    m = np.zeros(n)                                                 # meanZero

    alpha = np.zeros(n)                        # start at the mean, as gpml does
    ldB2_cache = {}                     # memoises the (W-keyed) Cholesky factor
    alpha = _irls(alpha, m, K, hyp_lik, y, cache=ldB2_cache)

    f = K @ alpha + m
    lp, dlp, d2lp, d3lp = _lik_gauss_laplace(hyp_lik, y, f)
    W = -d2lp
    ldB2, solveKiW, L, sW = _ldB2_exact_cached(W, K, ldB2_cache)

    post = {
        'alpha': alpha,
        'sW': np.sqrt(np.abs(W)) * np.sign(W),
        'L': lambda r: -solveKiW(r),
    }

    nlZ = alpha.dot(f - m) / 2 - np.sum(lp) + ldB2
    if not want_dnlZ:
        return post, nlZ, None

    # Q = diag(1/sW) * solve_chol(L, diag(sW));  dW = diag(inv(inv(K)+W))
    Q = (1.0 / sW)[:, None] * _solve_chol(L, np.diag(sW))
    dW = np.sum(Q * K, axis=1) / 2

    dfhat = dW * d3lp                          # zero for a Gaussian likelihood
    dahat = dfhat - solveKiW(K @ dfhat)
    R = np.outer(alpha, alpha) + 2 * np.outer(dfhat, dahat)
    dnlZ_cov = cov.dK(hyp_cov, x, (Q * W[:, None]) - R) / 2

    lp_dhyp, dlp_dhyp, d2lp_dhyp = _lik_gauss_laplace_dhyp(hyp_lik, y, f)
    dnlZ_lik = -dW.dot(d2lp_dhyp) - np.sum(lp_dhyp)              # explicit part
    b = K @ dlp_dhyp
    dnlZ_lik = dnlZ_lik - dfhat.dot(b - K @ solveKiW(b))         # implicit part

    dnlZ = {'cov': dnlZ_cov, 'lik': np.array([dnlZ_lik]), 'mean': np.zeros(0)}
    return post, nlZ, dnlZ

def gp_train(hyp, cov, x, y, want_dnlZ=True):
    """
    gpml ``gp(hyp, inf, mean, cov, lik, x, y)`` -- training mode.

    Returns ``(nlZ, dnlZ)``: the negative log marginal likelihood and its
    partial derivatives w.r.t. the hyperparameters. ``nlZ`` is computed
    identically either way; pass ``want_dnlZ=False`` when only the marginal
    likelihood is needed to skip the (dominant) derivative computation, in
    which case ``dnlZ`` is ``None``.
    """
    _, nlZ, dnlZ = inf_laplace(hyp, cov, x, y, want_dnlZ=want_dnlZ)
    return nlZ, dnlZ


def gp_predict(hyp, cov, x, y, xs, nperbatch=1000):
    """
    gpml ``gp(hyp, inf, mean, cov, lik, x, y, xs)`` -- prediction mode.

    Returns ``(ymu, ys2, fmu, fs2)``. ``ys2`` is the predictive *output*
    variance, i.e. ``fs2 + sn2``.

    ``nperbatch`` is not tunable in practice: gpml fixes it at 1000, and because
    the covariance re-centres on the current batch the value affects the result
    in the last few digits.
    """
    hyp_cov = np.asarray(hyp['cov'], dtype=float).ravel()
    hyp_lik = float(np.ravel(hyp['lik'])[0])
    sn2 = np.exp(2.0 * hyp_lik)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    xs = np.asarray(xs, dtype=float).ravel()

    post, _, _ = inf_laplace(hyp, cov, x, y, want_dnlZ=False)
    alpha = post['alpha']
    L = post['L']

    ns = xs.size
    fmu = np.zeros(ns)
    fs2 = np.zeros(ns)
    nact = 0
    while nact < ns:              # process minibatches of test cases, as gpml does
        idx = slice(nact, min(nact + nperbatch, ns))
        Kse_s, Knoise_s, _ = cov.parts(hyp_cov, xs[idx], 'diag')
        kss = Kse_s + Knoise_s                                   # self-variance
        Ks = cov.K(hyp_cov, x, xs[idx])                       # cross-covariances
        ms = np.zeros(xs[idx].size)                                   # meanZero

        fmu[idx] = ms + Ks.T @ alpha                       # conditional mean fs|f
        # post.L is a callback, not a Cholesky factor, so gpml takes the
        # alternative parametrisation here.
        LKs = L(Ks)
        fs2[idx] = kss + np.sum(Ks * LKs, axis=0)
        fs2[idx] = np.maximum(fs2[idx], 0)     # remove numerical noise, i.e. < 0
        nact = min(nact + nperbatch, ns)

    ymu = fmu                                                 # likGauss moments
    ys2 = fs2 + sn2
    return ymu, ys2, fmu, fs2
