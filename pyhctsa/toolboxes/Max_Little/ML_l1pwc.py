"""Discrete total variation denoising (TVD) via a primal-dual interior-point solver.

Python port of ``ML_l1pwc.m`` from Max Little's steps_bumps_toolkit.

Minimizes

    E = (1/2)||y - x||_2^2 + lambda*||Dx||_1

over x, given input signal y, for each value of the regularization parameter
lambda > 0.  D is the first difference matrix.  Hot-restarts from each value of
lambda speed up convergence for subsequent values; best use of this feature is
made by ensuring the chosen lambda values are close to each other.

(c) Max Little, 2010.  Based around code originally written by S.J. Kim, K. Koh,
S. Boyd and D. Gorinevsky.  If you use this code for your research, please cite:

    M.A. Little, Nick S. Jones (2010) "Sparse Bayesian Step-Filtering for
    High-Throughput Analysis of Molecular Machine Dynamics", in 2010 IEEE
    International Conference on Acoustics, Speech and Signal Processing,
    ICASSP 2010 Proceedings.

Released under the terms of the GNU General Public License as published by the
Free Software Foundation; version 2 or later.
"""

import numpy as np
from scipy.linalg import solveh_banded, solve_banded
from scipy.linalg.blas import ddot, dnrm2

__all__ = ["l1pwc"]


def _fmt(spec, *vals):
    """
    printf with MATLAB's spelling of the non-finite values.
    """
    return (spec % vals).replace("inf", "Inf").replace("nan", "NaN")


def _d_mul(v):
    """D*v for v of length n, returns length m = n-1.

    Column order for row i is (col i, col i+1): 0 + v[i] - v[i+1].
    """
    return v[:-1] - v[1:]


def _dt_mul(z):
    """D'*z for z of length m, returns n.

    Column j of D has entries at rows j-1 (-1) and j (+1), ascending, so
    out[j] = -z[j-1] + z[j] with the appropriate end truncations.
    """
    m = z.shape[0]
    out = np.empty(m + 1, dtype=np.float64)
    out[0] = z[0]
    out[1:m] = -z[:-1] + z[1:]
    out[m] = -z[m - 1]
    return out


def _ddt_mul(z):
    """DDT*z for z of length m.

    Row i draws from columns i-1 (-1), i (+2), i+1 (-1) in that order:
    out[i] = (-z[i-1] + 2*z[i]) - z[i+1].
    """
    m = z.shape[0]
    out = np.empty(m, dtype=np.float64)
    out[0] = 0.0
    out[1:] = -z[:-1]
    out += 2.0 * z
    out[:-1] -= z[1:]
    return out


def _solve_tridiag(diag, b):
    """Solve T*x = b where T is symmetric tridiagonal with the given diagonal
    and -1 on both off-diagonals.
    """
    m = diag.shape[0]
    if m == 1:
        return b / diag[0]
    # Lower-form banded storage for solveh_banded: row 0 = diagonal,
    # row 1 = first subdiagonal (last entry unused).
    ab = np.empty((2, m), dtype=np.float64)
    ab[0] = diag
    ab[1, : m - 1] = -1.0
    ab[1, m - 1] = 0.0
    try:
        return solveh_banded(ab, b, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        ab_gen = np.zeros((3, m), dtype=np.float64)
        ab_gen[0, 1:] = -1.0
        ab_gen[1] = diag
        ab_gen[2, : m - 1] = -1.0
        return solve_banded((1, 1), ab_gen, b, check_finite=False)


def l1pwc(y: np.ndarray, lam: float | np.ndarray, display: bool = True, 
          stoptol: float = 1e-3, maxiter: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Discrete total variation denoising.

    Python port of ``ML_l1pwc.m`` from Max Little's steps_bumps_toolkit. Max Little, 2010.  
    Based around code originally written by S.J. Kim, K. Koh, S. Boyd and D. Gorinevsky [1].

    References
    ----------
    .. [1] S.J. Kim, K. Koh, S. Boyd and D. Gorinevsky (2007) "An Interior-Point Method for Large-Scale 
        ℓ1 -Regularized Least Squares," in IEEE Journal of Selected Topics in Signal Processing, 
        vol. 1, no. 4, pp. 606-617, Dec. 2007, doi: 10.1109/JSTSP.2007.910971. 
    .. [2] M.A. Little, Nick S. Jones (2010) "Sparse Bayesian Step-Filtering for
        High-Throughput Analysis of Molecular Machine Dynamics", in 2010 IEEE
        International Conference on Acoustics, Speech and Signal Processing,
        ICASSP 2010 Proceedings.

    Parameters
    ----------
    y : array_like, shape (n,)
        Original signal to denoise.
    lam : float or array_like, shape (num_lambda,)
        Positive regularization parameter(s).  TVD is applied to each value.
    display : bool, optional
        Progress display.  Defaults to True.
    stoptol : float, optional
        Precision, as a duality gap tolerance.  Defaults to 1e-3.
    maxiter : int, optional
        Maximum interior-point iterations.  Defaults to 60.

    Returns
    -------
    x : ndarray, shape (n, num_lambda)
        Denoised output signal for each value of lambda.
    energy : ndarray, shape (num_lambda,)
        Objective functional at the minimum for each lambda.
    status : ndarray, shape (num_lambda,)
        Optimization result, 1 = solved, 0 = maximum iterations exceeded before
        reaching the duality gap tolerance.
    lambda_max : float
        Maximum value of lambda for the given y.  If lambda >= lambda_max, the
        output is the trivial constant solution x = mean(y).
    """
    # Search tuning parameters
    ALPHA = 0.01       # Backtracking linesearch parameter (0,0.5]
    BETA = 0.5         # Backtracking linesearch parameter (0,1)
    MAX_LS_ITER = 20   # Max iterations of backtracking linesearch
    MU = 2.0           # t update

    y = np.asarray(y, dtype=np.float64).reshape(-1, order="F")
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64)).ravel(order="F")

    n = y.shape[0]     # Length of input signal y
    m = n - 1          # Size of Dx
    if m < 1:
        raise ValueError("y must have at least 2 elements")

    # DDT is tridiagonal: 2 on the diagonal, -1 off-diagonal.
    ddt_diag = np.full(m, 2.0)
    dy = _d_mul(y)

    # Find max value of lambda
    lambda_max = np.max(np.abs(_solve_tridiag(ddt_diag, dy)))

    if display:
        print(_fmt("lambda_max=%5.2e", lambda_max))

    num_lambda = lam.shape[0]
    x = np.zeros((n, num_lambda), dtype=np.float64)
    status = np.zeros(num_lambda, dtype=np.float64)
    energy = np.zeros(num_lambda, dtype=np.float64)

    # Optimization variables set up once at the start
    z = np.zeros(m)          # Dual variable
    mu1 = np.ones(m)         # Dual of dual variable
    mu2 = np.ones(m)         # Dual of dual variable

    # Work through each value of lambda, with hot-restart on optimization
    # variables
    for li in range(num_lambda):

        lam_i = lam[li]

        t = 1e-10
        step = np.inf
        f1 = z - lam_i
        f2 = -z - lam_i

        # Main optimization loop
        status[li] = 1

        if display:
            print(_fmt(
                "Solving for lambda=%5.2e, lambda/lambda_max=%5.2e\n"
                "Iter# Primal    Dual      Gap", lam_i, lam_i / lambda_max
            ))

        gap = np.nan
        new_z = new_mu1 = new_mu2 = new_f1 = new_f2 = None

        iters = 0
        for iters in range(maxiter + 1):

            dtz = _dt_mul(z)
            ddtz = _d_mul(dtz)
            w = dy - (mu1 - mu2)

            # Calculate objectives and primal-dual gap
            pobj1 = ddot(0.5 * w, _solve_tridiag(ddt_diag, w)) \
                + lam_i * np.sum(mu1 + mu2)
            pobj2 = ddot(0.5 * dtz, dtz) + lam_i * np.sum(np.abs(dy - ddtz))
            pobj = min(pobj1, pobj2)
            dobj = ddot(-0.5 * dtz, dtz) + ddot(dy, z)
            gap = pobj - dobj

            if display:
                print(_fmt("%5d %7.2e %7.2e %7.2e", iters, pobj, dobj, gap))

            # Test duality gap stopping criterion
            if gap <= stoptol:
                status[li] = 1
                break

            if step >= 0.2:
                t = max(2 * m * MU / gap, 1.2 * t)

            # Do Newton step
            rz = ddtz - w
            s_diag = ddt_diag - (mu1 / f1 + mu2 / f2)
            r = -ddtz + dy + (1.0 / t) / f1 - (1.0 / t) / f2
            dz = _solve_tridiag(s_diag, r)
            dmu1 = -(mu1 + ((1.0 / t) + dz * mu1) / f1)
            dmu2 = -(mu2 + ((1.0 / t) - dz * mu2) / f2)

            res_dual = rz
            res_cent = np.concatenate((-mu1 * f1 - 1.0 / t, -mu2 * f2 - 1.0 / t))
            residual = np.concatenate((res_dual, res_cent))

            # Perform backtracking linesearch
            neg_idx1 = dmu1 < 0
            neg_idx2 = dmu2 < 0
            step = 1.0
            if neg_idx1.any():
                step = min(step, 0.99 * np.min(-mu1[neg_idx1] / dmu1[neg_idx1]))
            if neg_idx2.any():
                step = min(step, 0.99 * np.min(-mu2[neg_idx2] / dmu2[neg_idx2]))

            for _ls_iter in range(MAX_LS_ITER):
                new_z = z + step * dz
                new_mu1 = mu1 + step * dmu1
                new_mu2 = mu2 + step * dmu2
                new_f1 = new_z - lam_i
                new_f2 = -new_z - lam_i

                # Update residuals
                new_res_dual = _ddt_mul(new_z) - dy + new_mu1 - new_mu2
                new_res_cent = np.concatenate(
                    (-new_mu1 * new_f1 - 1.0 / t, -new_mu2 * new_f2 - 1.0 / t)
                )
                new_residual = np.concatenate((new_res_dual, new_res_cent))

                if (max(np.max(new_f1), np.max(new_f2)) < 0) and (
                    dnrm2(new_residual) <= (1 - ALPHA * step) * dnrm2(residual)
                ):
                    break
                step = BETA * step

            # Update primal and dual optimization parameters
            z = new_z
            mu1 = new_mu1
            mu2 = new_mu2
            f1 = new_f1
            f2 = new_f2

        x[:, li] = y - _dt_mul(z)
        energy[li] = 0.5 * np.sum((y - x[:, li]) ** 2) \
            + lam_i * np.sum(np.abs(_d_mul(x[:, li])))

        # We may have a close solution that does not satisfy the duality gap
        if iters >= maxiter:
            status[li] = 0

        if display:
            if status[li]:
                print(_fmt("Solved to precision of duality gap %5.2e", gap))
            else:
                print("Max iterations exceeded - solution may be inaccurate")

    return x, energy, status, lambda_max

def l1pwc_lmax(y: np.ndarray) -> float:
    """Maximum useful regularisation parameter for L1-PWC denoising.

    Port of ML_l1pwclmax.m. Above this value the solution collapses to a
    single constant segment.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = y.size
    M = N - 1
    if M < 1:
        raise ValueError("y must contain at least two samples")

    # D @ y  where D = [I 0] - [0 I]  ->  y[:-1] - y[1:]  ==  -diff(y)
    Dy = -np.diff(y)

    # D @ D.T is symmetric tridiagonal: 2 on the diagonal, -1 off-diagonal.
    ab = np.empty((2, M))
    ab[0, 0] = 0.0          # unused corner of the upper-diagonal band
    ab[0, 1:] = -1.0
    ab[1, :] = 2.0

    return np.max(np.abs(solveh_banded(ab, Dy, lower=False)))