# Log: cr-B3 (short statistics), 2026-10-09

Worktree SP/cr-B3, branch cr-B3. Tests: tests/native/test_thirdparty_B3.py (15 tests, pass).

## Functions and sources
- `correlation.theiler_q`, `crinkle_statistic`, `time_rev_kaplan`: formulas in specs/thirdparty/kaplan.md; time-reversal
  statistic credited to Ramsey & Rothman (1996). numpy only. Zero-variance / too-short input returns NaN (orchestrator
  decision 2); `time_rev_kaplan` with fewer than 2L+1 samples returns NaN with a warning.
- `nonlinearity._increment_pnorm_curve / _windowed_mean_increment_pnorm_curve`: Ozkurt et al. (2020) definition as in specs/thirdparty/nsamdf.md
  (p-norm of lag-d increments, summed, mean over sliding windows). Increments scaled by their maximum before the
  p-th power (overflow-safe); numpy.
- `nonlinearity._fractal_dim_error`: van de Water & Schram (1988) moment scaling Gamma(k+g/d)/Gamma(k) (digamma form
  for g = 0), Lorentzian loss log(1+e^2/2) (Black & Anandan 1996), best scale from scipy `minimize_scalar` bounded
  on [0, 1e6], xatol 1e-5 (spec). scipy.special gammaln/digamma.
- `entropy._app_samp_entropy` replaced by `_apen_phis(x, m, r, tau)`: Pincus (1991) Phi^m - Phi^(m+1) with
  Chebyshev counts from `scipy.spatial.cKDTree.query_ball_point(..., p=inf, return_length=True)` (<= r, self included).
  `_xlogx` removed; `permutation_entropy` uses `-sum(scipy.special.xlogy(p, p)) / ln 2`.
- `utils.x_corr`: `scipy.signal.correlate(mode='full')` sliced to +-max_lags, optional normalization by
  sqrt(sum x^2 sum y^2); errors as in specs/thirdparty/xcorr.md.
- antropy removed from pyproject.toml dependencies and from hctsa.yaml `permutation_entropy.dependencies`.

## Agreement with hctsa fixtures (thirdparty_blackbox.json)
- Kaplan (10 cases incl. hand vectors, lags 1-4, NaN pattern): max abs diff 1.1e-13.
- nsAMDF L (both settings, 5 series): 9.7e-17.
- NL_FractalDimensions fd_a (9 outputs, 5 series): within the 1e-7 gate (fd_b: <= 4.6e-10).
- EN_ApEn (6 variants x 7 series, incl. ties and ac1e): 6.2e-15.
- xcorr (coeff/raw, maxlag 1 and 5): 2.1e-14.
Also: brute-force Pincus check on tied data (boundary matches), x_corr definition check, antropy public
`app_entropy` cross-check (test, skipped if absent).

## Statement
Only the specs, the fixture, the cited papers and numpy/scipy documentation were used. No third-party source code
(Kaplan, nsamdf, TSTOOL, antropy), no quarantined material, no other worktree and no Claude transcript was consulted.
