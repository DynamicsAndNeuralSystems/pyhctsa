"""
gpml covariance functions for scalar (1-d) inputs: covSEiso, covPeriodic, covMaterniso(d),
covRQiso, covNoise and their sum, covSum (gpml v4.2).

Each covariance object has ``n_hyp`` (the number of hyperparameters), ``K(hyp, x, z=None)``
(the covariance matrix between ``x`` and ``z``, or ``x`` and itself), ``diag(hyp, x)`` (its
diagonal, gpml's ``z = 'diag'``) and ``dK(hyp, x, Q)`` (gpml's directional hyperparameter
derivative for the symmetric matrix: ``dhyp(i) = sum(Q .* dK/dhyp_i)``), so that they can be
used with ``gp_train`` and ``gp_predict`` in :mod:`gpml`. The hyperparameters are in gpml's
(log) order:

- ``covSEiso``: ``[log(ell), log(sf)]``,
- ``covPeriodic``: ``[log(ell), log(p), log(sf)]`` (length scale, period, magnitude),
- ``covMaterniso(d)``: ``[log(ell), log(sf)]``,
- ``covRQiso``: ``[log(ell), log(sf), log(alpha)]`` (gpml's "old" hyperparameter order),
- ``covNoise``: ``[log(sn)]``.
"""
import re

import numpy as np

from .gpml import CovSEisoNoise

_EPS = np.finfo(float).eps


def _maha(hyp, x, z=None):
    """gpml ``covMaha`` in 'iso' mode with length scale ``exp(hyp[0])`` (as in ``CovSEisoNoise``)."""
    return CovSEisoNoise._maha(hyp, x, z)


class _ScaledMahaCov:
    """
    ``covScale`` of an isotropic function of the Mahalanobis distance, ``K = sf^2 k(D2)``,
    with hyperparameters ``[log(ell), log(sf)]`` (gpml's ``covScale({'covSE'|'covMatern'...,
    'iso', []})``). Subclasses provide ``k(d2)`` and ``dk(d2, k)``, the derivative of ``k``
    with respect to ``d2``.
    """

    n_hyp = 2

    def k(self, d2):
        raise NotImplementedError

    def dk(self, d2, k):
        raise NotImplementedError

    def K(self, hyp, x, z=None):
        hyp = np.asarray(hyp, dtype=float).ravel()
        return np.exp(2.0 * hyp[1]) * self.k(_maha(hyp, x, z))

    def diag(self, hyp, x):
        hyp = np.asarray(hyp, dtype=float).ravel()
        return np.exp(2.0 * hyp[1]) * np.ones(np.size(x))

    def dK(self, hyp, x, Q):
        hyp = np.asarray(hyp, dtype=float).ravel()
        sf2 = np.exp(2.0 * hyp[1])
        D2 = _maha(hyp, x)
        K0 = self.k(D2)
        R = self.dk(D2, K0) * (Q * sf2)
        return np.array([-2.0 * np.sum(R * D2), 2.0 * sf2 * np.sum(Q * K0)])


class CovSEiso(_ScaledMahaCov):
    """gpml ``covSEiso``: ``k(d2) = exp(-d2 / 2)``."""

    def k(self, d2):
        return np.exp(-d2 / 2.0)

    def dk(self, d2, k):
        return -0.5 * k


class CovMaterniso(_ScaledMahaCov):
    """gpml ``covMaterniso(d)`` for ``d`` in 1, 3, 5, 7."""

    def __init__(self, d: int = 3):
        if d not in (1, 3, 5, 7):
            raise ValueError('covMaterniso is implemented for d = 1, 3, 5 or 7')
        self.d = d

    def _f(self, t):
        return {1: lambda t: np.ones_like(t),
                3: lambda t: 1 + t,
                5: lambda t: 1 + t * (1 + t / 3),
                7: lambda t: 1 + t * (1 + t * (6 + t) / 15)}[self.d](t)

    def k(self, d2):
        t = np.sqrt(self.d * d2)
        return self._f(t) * np.exp(-t)

    def dk(self, d2, k):
        d = self.d
        if d == 1:
            t = np.sqrt(d2)
            with np.errstate(divide='ignore', invalid='ignore'):
                dk = -(np.exp(-t) / t) / 2.0
            dk[d2 == 0] = 0.0  # fix the limit d2 -> 0
            return dk
        t = np.sqrt(d * d2)
        df = {3: lambda t: np.ones_like(t),
              5: lambda t: (1 + t) / 3,
              7: lambda t: (1 + t + t ** 2 / 3) / 5}[d](t)
        return -df * np.exp(-t) * d / 2.0


class CovRQiso:
    """gpml ``covRQiso``: ``K = sf^2 (1 + D2 / (2 alpha))^-alpha``, ``hyp = [log(ell), log(sf), log(alpha)]``."""

    n_hyp = 3

    @staticmethod
    def _k(d2, alpha):
        return (1 + 0.5 * d2 / alpha) ** (-alpha)

    def K(self, hyp, x, z=None):
        hyp = np.asarray(hyp, dtype=float).ravel()
        return np.exp(2.0 * hyp[1]) * self._k(_maha(hyp, x, z), np.exp(hyp[2]))

    def diag(self, hyp, x):
        hyp = np.asarray(hyp, dtype=float).ravel()
        return np.exp(2.0 * hyp[1]) * np.ones(np.size(x))

    def dK(self, hyp, x, Q):
        hyp = np.asarray(hyp, dtype=float).ravel()
        sf2 = np.exp(2.0 * hyp[1])
        alpha = np.exp(hyp[2])
        D2 = _maha(hyp, x)
        K0 = self._k(D2, alpha)
        Qs = Q * sf2
        R = (-K0 / (2 + D2 / alpha)) * Qs
        B = 1 + 0.5 * D2 / alpha
        d_alpha = np.sum(Qs * K0 * (0.5 * D2 / B - alpha * np.log(B)))
        return np.array([-2.0 * np.sum(R * D2), 2.0 * sf2 * np.sum(Q * K0), d_alpha])


class CovPeriodic:
    """
    gpml ``covPeriodic``: ``K = sf^2 exp(-2 sin^2(pi (x - z) / p) / ell^2)``, 1-d inputs,
    ``hyp = [log(ell), log(p), log(sf)]``.
    """

    n_hyp = 3

    @staticmethod
    def _T(p, x, z=None):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        zz = x if z is None else np.asarray(z, dtype=float).reshape(-1, 1)
        return np.pi / p * (x - zz.T)

    def K(self, hyp, x, z=None):
        hyp = np.asarray(hyp, dtype=float).ravel()
        ell, p, sf2 = np.exp(hyp[0]), np.exp(hyp[1]), np.exp(2.0 * hyp[2])
        S2 = (np.sin(self._T(p, x, z)) / ell) ** 2
        return sf2 * np.exp(-2.0 * S2)

    def diag(self, hyp, x):
        hyp = np.asarray(hyp, dtype=float).ravel()
        return np.exp(2.0 * hyp[2]) * np.ones(np.size(x))

    def dK(self, hyp, x, Q):
        hyp = np.asarray(hyp, dtype=float).ravel()
        ell, p, sf2 = np.exp(hyp[0]), np.exp(hyp[1]), np.exp(2.0 * hyp[2])
        T = self._T(p, x)
        S2 = (np.sin(T) / ell) ** 2
        K = sf2 * np.exp(-2.0 * S2)
        Qk = K * Q
        P = np.sin(2 * T) * Qk
        return np.array([4.0 * np.sum(S2 * Qk),
                         2.0 / ell ** 2 * np.sum(P * T),
                         2.0 * np.sum(Qk)])


class CovNoise:
    """gpml ``covNoise``: ``K = sn^2 I`` (for ``z`` not ``None``, 1 where points coincide), ``hyp = [log(sn)]``."""

    n_hyp = 1

    def K(self, hyp, x, z=None):
        sn2 = np.exp(2.0 * np.asarray(hyp, dtype=float).ravel()[0])
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        if z is None:
            return sn2 * np.eye(x.shape[0])
        zz = np.asarray(z, dtype=float).reshape(-1, 1)
        return sn2 * ((x - zz.T) ** 2 < _EPS * _EPS).astype(float)

    def diag(self, hyp, x):
        return np.exp(2.0 * np.asarray(hyp, dtype=float).ravel()[0]) * np.ones(np.size(x))

    def dK(self, hyp, x, Q):
        sn2 = np.exp(2.0 * np.asarray(hyp, dtype=float).ravel()[0])
        return np.array([2.0 * sn2 * np.sum(np.diag(Q))])


class CovSum:
    """gpml ``covSum``: the sum of covariance functions, with their hyperparameters concatenated."""

    def __init__(self, terms):
        self.terms = list(terms)
        self.n_hyp = int(sum(t.n_hyp for t in self.terms))

    def _split(self, hyp):
        hyp = np.asarray(hyp, dtype=float).ravel()
        if hyp.size != self.n_hyp:
            raise ValueError('Wrong number of hyperparameters')
        pos = np.cumsum([0] + [t.n_hyp for t in self.terms])
        return [hyp[pos[i]:pos[i + 1]] for i in range(len(self.terms))]

    def K(self, hyp, x, z=None):
        return sum(t.K(h, x, z) for t, h in zip(self.terms, self._split(hyp)))

    def diag(self, hyp, x):
        return sum(t.diag(h, x) for t, h in zip(self.terms, self._split(hyp)))

    def dK(self, hyp, x, Q):
        return np.concatenate([t.dK(h, x, Q) for t, h in zip(self.terms, self._split(hyp))])


def parse_cov(cov_func):
    """
    Build a covariance function from hctsa's / gpml's specification.

    ``cov_func`` is either the gpml form ``['covSum', ['covSEiso', 'covPeriodic', 'covNoise']]``
    (a parametrized component is a pair, ``['covMaterniso', 3]``), or a string of the component
    names joined with underscores, as in hctsa's operation names, with the Matern degree
    appended: ``'covSEiso_covPeriodic_covNoise'``, ``'covMaterniso3_covNoise'``,
    ``'covRQiso_covNoise'``.

    Returns ``(cov, components)``: the :class:`CovSum` and the list of ``(name, degree)`` of its
    components (the degree is None for components without one).
    """
    if isinstance(cov_func, str):
        comps = []
        for name in cov_func.split('_'):
            m = re.fullmatch(r'(covMaterniso)(\d+)', name)
            comps.append((m.group(1), int(m.group(2))) if m else (name, None))
    else:
        if not (len(cov_func) == 2 and cov_func[0] == 'covSum'):
            raise ValueError("Only {'covSum', {components...}} covariance functions are supported")
        comps = []
        for c in cov_func[1]:
            if isinstance(c, str):
                comps.append((c, None))
            else:
                comps.append((c[0], int(c[1])))
    terms = []
    for name, d in comps:
        if name == 'covSEiso':
            terms.append(CovSEiso())
        elif name == 'covPeriodic':
            terms.append(CovPeriodic())
        elif name == 'covMaterniso':
            terms.append(CovMaterniso(3 if d is None else d))
        elif name == 'covRQiso':
            terms.append(CovRQiso())
        elif name == 'covNoise':
            terms.append(CovNoise())
        else:
            raise ValueError(f"Unsupported covariance function '{name}'")
    return CovSum(terms), comps
