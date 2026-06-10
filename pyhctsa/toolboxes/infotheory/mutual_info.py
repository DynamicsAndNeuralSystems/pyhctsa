"""
Faithful Python port of the Kraskov-Stoegbauer-Grassberger (KSG) mutual
information estimators (algorithms 1 and 2) as implemented in the Java
Information Dynamics Toolkit (JIDT) by J. T. Lizier.

Ported from:
  infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov
  infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1
  infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov2
  infodynamics.measures.continuous.MutualInfoMultiVariateCommon
  infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian

Reference:
  A. Kraskov, H. Stoegbauer, P. Grassberger, "Estimating mutual information",
  Phys. Rev. E 69, 066138 (2004).  https://doi.org/10.1103/PhysRevE.69.066138

  T. M. Cover & J. A. Thomas, "Elements of Information Theory", Wiley (1991).
  H. Kantz & T. Schreiber, "Nonlinear Time Series Analysis", CUP (1997).
  J. T. Lizier, "JIDT: An information-theoretic toolkit ...", Front. Robot. AI (2014).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2
from scipy.special import digamma

def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a

_NORM_TO_P = {
    "max": np.inf, "maximum": np.inf, "chebyshev": np.inf, "inf": np.inf,
    "euclidean": 2.0, "euclidean_squared": 2.0, "l2": 2.0,
}

def _normalise_columns(a: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per column, using the sample std (ddof=1).

    Divides the sum of squares by (N-1). Note the KSG estimate is actually invariant to the ddof choice: it
    rescales every variable by the same factor sqrt((N-1)/N), which leaves the
    (max-norm) neighbour sets unchanged.
    """
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (a - mu) / sd

class KraskovMI:
    """KSG mutual information estimator.

    Parameters
    ----------
    k : int
        Number of nearest neighbours in the joint space (default 4).
    algorithm : {1, 2}
        KSG algorithm 1 or 2 (default 1).
    norm : {'max', 'euclidean', 'euclidean_squared'}
        Norm used within each variable / the joint space (default 'max',
        i.e. the maximum / Chebyshev norm used in the original KSG paper).
    normalise : bool
        If True, z-score each variable (per dimension) before estimation.
        Default is True.
    add_noise : bool
        If True, add tiny Gaussian noise to break ties / degenerate distances.
        Default is True for the Kraskov estimators.
    noise_level : float
        Standard deviation of the added noise (default 1e-8).
        Applied AFTER normalisation.
    theiler : int
        Dynamic correlation exclusion (Theiler) window. Neighbours j with
        ``|j - i| <= theiler`` are excluded for query point i. Default 0
        (no exclusion). The input is treated as a single contiguous
        observation set.
    seed : int or None
        Seed for the noise RNG for reproducibility.
    """

    def __init__(self, k: int = 4, algorithm: int = 1, norm: str = "max",
                 normalise: bool = True, add_noise: bool = True,
                 noise_level: float = 1e-8, theiler: int = 0, seed=None):
        if algorithm not in (1, 2):
            raise ValueError("algorithm must be 1 or 2")
        nlow = str(norm).lower()
        if nlow not in _NORM_TO_P:
            raise ValueError(f"norm must be one of {sorted(set(_NORM_TO_P))}")
        if k < 1:
            raise ValueError("k must be >= 1")
        if theiler < 0:
            raise ValueError("theiler must be >= 0")

        self.k = int(k)
        self.algorithm = int(algorithm)
        self.norm = nlow
        self.p = _NORM_TO_P[nlow]
        self.normalise = bool(normalise)
        self.add_noise = bool(add_noise)
        self.noise_level = float(noise_level)
        self.theiler = int(theiler)
        self.rng = np.random.default_rng(seed)
    
    def compute(self, X, Y, local: bool = False):
        """Compute the KSG MI between X and Y (in nats).

        Returns the average MI (float) unless ``local=True``, in which case the
        per-sample local MI values (np.ndarray of length N) are returned; their
        mean equals the average MI.
        """
        X = _as_2d(X)
        Y = _as_2d(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        N = X.shape[0]
        if N <= self.k + 2 * self.theiler:
            raise ValueError(
                f"Too few samples ({N}) for k={self.k} and theiler={self.theiler}")

        if self.normalise:
            X = _normalise_columns(X)
            Y = _normalise_columns(Y)
        if self.add_noise and self.noise_level > 0:
            X = X + self.rng.normal(0.0, self.noise_level, X.shape)
            Y = Y + self.rng.normal(0.0, self.noise_level, Y.shape)

        joint = np.hstack([X, Y])
        joint_tree = cKDTree(joint)
        x_tree = cKDTree(X)
        y_tree = cKDTree(Y)

        if self.algorithm == 1:
            eps = self._joint_radius_alg1(joint_tree, joint, N)
            n_x = self._count_marginal(x_tree, X, eps, strict=True)
            n_y = self._count_marginal(y_tree, Y, eps, strict=True)
            local_mi = (digamma(self.k)
                        - digamma(n_x + 1) - digamma(n_y + 1)
                        + digamma(N))
        else:  # algorithm 2
            eps_x, eps_y = self._joint_radii_alg2(joint_tree, joint, X, Y, N)
            n_x = self._count_marginal(x_tree, X, eps_x, strict=False)
            n_y = self._count_marginal(y_tree, Y, eps_y, strict=False)
            local_mi = (digamma(self.k) - 1.0 / self.k
                        - digamma(n_x) - digamma(n_y)
                        + digamma(N))

        return local_mi if local else float(np.mean(local_mi))

    def significance(self, X, Y, n_permutations: int = 100):
        """Permutation significance test (shuffles Y to break X-Y dependence).

        Returns a dict with the measured MI, the surrogate mean/std, a p-value
        (fraction of surrogates >= measured, Laplace-corrected) and the raw
        surrogate values.
        """
        measured = self.compute(X, Y)
        Y2 = _as_2d(Y)
        surrogates = np.empty(n_permutations)
        for s in range(n_permutations):
            perm = self.rng.permutation(Y2.shape[0])
            surrogates[s] = self.compute(X, Y2[perm])
        p_value = (np.sum(surrogates >= measured) + 1) / (n_permutations + 1)
        return {
            "mi": measured,
            "surrogate_mean": float(surrogates.mean()),
            "surrogate_std": float(surrogates.std()),
            "p_value": float(p_value),
            "surrogates": surrogates,
        }
    
    def _joint_radius_alg1(self, joint_tree, joint, N):
        """Distance to the kth nearest neighbour in the joint space (self excluded)."""
        if self.theiler == 0:
            # query k+1 because the nearest returned point is the point itself.
            d, _ = joint_tree.query(joint, k=self.k + 1, p=self.p)
            return np.ascontiguousarray(d[:, -1])
        return self._joint_radius_alg1_theiler(joint_tree, joint, N)

    def _joint_radius_alg1_theiler(self, joint_tree, joint, N):
        kq = min(N, self.k + 2 * self.theiler + 1)  # guarantees >= k survivors
        d, idxs = joint_tree.query(joint, k=kq, p=self.p)
        th = self.theiler
        eps = np.empty(N)
        for t in range(N):
            count = 0
            chosen = None
            for dist_t, j in zip(d[t], idxs[t]):
                if abs(int(j) - t) > th:
                    count += 1
                    if count == self.k:
                        chosen = dist_t
                        break
            if chosen is None:  # safety net (shouldn't trigger given kq)
                chosen = self._kth_joint_distance_fallback(joint_tree, joint, t, N)
            eps[t] = chosen
        return eps

    def _joint_radii_alg2(self, joint_tree, joint, X, Y, N):
        """Per-axis radii eps_x, eps_y = max marginal norm over the k joint NN."""
        if self.theiler == 0:
            _, idxs = joint_tree.query(joint, k=self.k + 1, p=self.p)
            nb = idxs[:, 1:]  # k neighbours, excluding the point itself (col 0)
            eps_x = self._marginal_norm(X[nb] - X[:, None, :]).max(axis=1)
            eps_y = self._marginal_norm(Y[nb] - Y[:, None, :]).max(axis=1)
            return eps_x, eps_y
        return self._joint_radii_alg2_theiler(joint_tree, joint, X, Y, N)

    def _joint_radii_alg2_theiler(self, joint_tree, joint, X, Y, N):
        kq = min(N, self.k + 2 * self.theiler + 1)
        _, idxs = joint_tree.query(joint, k=kq, p=self.p)
        th = self.theiler
        eps_x = np.empty(N)
        eps_y = np.empty(N)
        for t in range(N):
            survivors = [int(j) for j in idxs[t] if abs(int(j) - t) > th][:self.k]
            surv = np.asarray(survivors, dtype=int)
            eps_x[t] = self._marginal_norm(X[surv] - X[t]).max()
            eps_y[t] = self._marginal_norm(Y[surv] - Y[t]).max()
        return eps_x, eps_y
    
    def _count_marginal(self, tree, data, R, strict):
        """Count points within radius R[i] of point i in `data`.

        strict=True  -> distance <  R[i]  (algorithm 1)
        strict=False -> distance <= R[i]  (algorithm 2)
        The query point itself and any point inside the Theiler window are
        excluded (handled by the window-correction loop; the d=0 term removes
        the point itself).
        """
        N = data.shape[0]
        if strict:
            # dist <= nextafter(R, -inf)  is equivalent to  dist < R
            r_query = np.nextafter(R, -np.inf)
        else:
            r_query = R
        full = np.asarray(
            tree.query_ball_point(data, r_query, p=self.p, return_length=True),
            dtype=np.int64,
        )

        # Subtract contributions from indices within the Theiler window
        # (range includes 0, which removes the point itself).
        excluded = np.zeros(N, dtype=np.int64)
        idx = np.arange(N)
        for d in range(-self.theiler, self.theiler + 1):
            j = idx + d
            valid = (j >= 0) & (j < N)
            ii = idx[valid]
            jj = j[valid]
            dist = self._marginal_norm(data[ii] - data[jj])
            cond = (dist < R[ii]) if strict else (dist <= R[ii])
            np.add.at(excluded, ii[cond], 1)
        return full - excluded
    
    def _marginal_norm(self, diff):
        """Norm along the last axis under the configured Minkowski p."""
        ad = np.abs(diff)
        if np.isinf(self.p):
            return ad.max(axis=-1)
        return (ad ** self.p).sum(axis=-1) ** (1.0 / self.p)

    def _kth_joint_distance_fallback(self, joint_tree, joint, t, N):
        d, idxs = joint_tree.query(joint[t], k=N, p=self.p)
        count = 0
        for dist_t, j in zip(np.atleast_1d(d), np.atleast_1d(idxs)):
            if abs(int(j) - t) > self.theiler:
                count += 1
                if count == self.k:
                    return dist_t
        raise RuntimeError("Not enough valid neighbours after Theiler exclusion")
    
    
def ksg_mi(X, Y, k: int = 4, algorithm: int = 1, norm: str = "max",
           normalise: bool = True, add_noise: bool = True,
           noise_level: float = 1e-8, theiler: int = 0, local: bool = False,
           seed=None):
    """Functional wrapper around :class:`KraskovMI`. Returns MI in nats.

    Example
    -------
    >>> ksg_mi(x, y, algorithm=1)
    >>> ksg_mi(x, y, algorithm=2, normalise=False, add_noise=False)
    """
    est = KraskovMI(k=k, algorithm=algorithm, norm=norm, normalise=normalise,
                    add_noise=add_noise, noise_level=noise_level,
                    theiler=theiler, seed=seed)
    return est.compute(X, Y, local=local)


class GaussianMI:
    r"""Mutual information assuming a (multivariate) Gaussian model.

    Uses the closed form  MI = 0.5 * ln( det(C_x) det(C_y) / det(C_xy) )  in nats,
    where C_x, C_y, C_xy are the sample covariances of X, Y and the stacked
    [X, Y]. This is the maximum-likelihood plug-in estimate; it is the right tool
    when the dependence is (close to) linear-Gaussian and is far lower variance
    than KSG/kernel in that regime, but it only captures *linear* dependence.

    Parameters
    ----------
    bias_correction : bool
        If True, subtract the analytic finite-sample bias (the mean of the
        chi-square null distribution, d_x * d_y / (2N)) from the estimate,
        matching JIDT's ``BIAS_CORRECTION`` property. Default False.

    Notes
    -----
    The result is exactly invariant to per-variable rescaling (and hence to
    JIDT's ``NORMALISE`` flag), so no ``normalise`` option is exposed.
    Full linear-redundancy pruning (JIDT's Cholesky-of-independent-components)
    is not reproduced; for full-rank inputs the estimate is exact. Add a hair of
    noise if you have exactly collinear columns.
    """

    def __init__(self, bias_correction: bool = False):
        self.bias_correction = bool(bias_correction)

    def _moments(self, X, Y):
        X = _as_2d(X)
        Y = _as_2d(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        N, ds, dd = X.shape[0], X.shape[1], Y.shape[1]
        Z = np.hstack([X, Y])
        # rowvar=False -> variables in columns; ddof=1 (sample covariance, as JIDT)
        Cx = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
        Cy = np.atleast_2d(np.cov(Y, rowvar=False, ddof=1))
        Cz = np.atleast_2d(np.cov(Z, rowvar=False, ddof=1))
        return X, Y, Z, N, ds, dd, Cx, Cy, Cz

    def compute(self, X, Y, local: bool = False):
        X, Y, Z, N, ds, dd, Cx, Cy, Cz = self._moments(X, Y)

        s_x = np.linalg.slogdet(Cx)
        s_y = np.linalg.slogdet(Cy)
        s_z = np.linalg.slogdet(Cz)
        if s_x.sign <= 0 or s_y.sign <= 0:
            raise np.linalg.LinAlgError(
                "Marginal covariance is singular (collinear columns within X or Y)")
        if s_z.sign <= 0:
            return np.inf if not local else np.full(N, np.inf)  # X,Y jointly collinear

        avg_mi = 0.5 * (s_x.logabsdet + s_y.logabsdet - s_z.logabsdet)
        bias = (ds * dd) / (2.0 * N)  # mean of chi-square null, in nats

        if not local:
            return float(avg_mi - bias) if self.bias_correction else float(avg_mi)

        # Local values: log[ p_joint / (p_x p_y) ] per sample, Gaussian PDFs
        mu = Z.mean(axis=0)
        mu_x, mu_y = mu[:ds], mu[ds:]
        inv_x = np.linalg.inv(Cx)
        inv_y = np.linalg.inv(Cy)
        inv_z = np.linalg.inv(Cz)
        dev_x = X - mu_x
        dev_y = Y - mu_y
        dev_z = Z - mu
        maha = lambda dev, inv: np.einsum("ij,jk,ik->i", dev, inv, dev)
        arg_x = maha(dev_x, inv_x)
        arg_y = maha(dev_y, inv_y)
        arg_z = maha(dev_z, inv_z)
        const = 0.5 * (s_x.logabsdet + s_y.logabsdet - s_z.logabsdet)
        local_mi = 0.5 * (arg_x + arg_y - arg_z) + const  # (2*pi)^d factors cancel
        if self.bias_correction:
            local_mi = local_mi - bias
        return local_mi

    def significance(self, X, Y):
        """Analytic significance test (chi-square null), as in JIDT.

        2*N*MI is chi-square distributed with d_x*d_y degrees of freedom under the
        null of independence. Returns the measured MI, the null mean/std and a
        p-value (P(null >= measured)).
        """
        _, _, _, N, ds, dd, *_ = self._moments(X, Y)
        mi = self.compute(X, Y)
        dof = ds * dd
        stat = 2.0 * N * mi
        return {
            "mi": mi,
            "p_value": float(chi2.sf(stat, dof)),
            "null_mean": dof / (2.0 * N),
            "null_std": float(np.sqrt(2.0 * dof) / (2.0 * N)),
            "dof": dof,
        }
