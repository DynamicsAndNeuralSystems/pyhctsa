# Clean-room log: work package B4 (library swaps: HMM via hmmlearn, exponential smoothing via statsmodels)
Worktree SP/cr-B4, branch cr-B4. Date 2026-10-09.

## Sources used
- Specs: SP/cleanroom/specs/thirdparty/{hmm.md, exp_smoothing.md, README.md, thirdparty_blackbox.json, gen/thirdparty_inputs.json}.
- Published: Rabiner (1989) Proc. IEEE 77:257-286 (scaled forward-backward, Baum-Welch; test oracle written from
  these formulas); Brown (1963); Hyndman & Athanasopoulos, FPP3 sec. 8.1 (SES recursion).
- Library documentation / installed library code of hmmlearn 0.3.3 (BSD-3) and statsmodels 0.15 (BSD-3), only to
  check how the tied-covariance M step applies `covars_prior`, `covars_weight`, `min_covar` and what `fittedvalues` holds.
- Not consulted: any third-party (Ghahramani / Arora) source, the quarantine, other worktrees, earlier commits of
  pyhctsa, hctsa Toolboxes/. No `git log -p`/`git show`.

## HMM (`operations/model_fit.py`)
Renamed `_zg_hmm_em` -> `_gaussian_hmm_em`, `_zg_hmm_loglik` -> `_gaussian_hmm_loglik`, Ben's wrapper `_zg_hmm_fit` ->
`_gaussian_hmm_fit` (logic unchanged); callers `hmm_fit`, `hmm_compare_n_states` and the two existing tests updated.
Helper `_gaussian_hmm(mu, cov, P, pi)` builds a `GaussianHMM(covariance_type='tied', n_iter=1, tol=0,
init_params='', params='stmc', implementation='log', startprob_prior=1, transmat_prior=1, means_prior=0,
means_weight=0, covars_prior=0, covars_weight=1, random_state=0)` carrying the given parameters.
- hmmlearn 0.3.3 tied M step (verified in installed hmm.py): covars = (covars_prior + sum_c c_n[c]) /
  (max(covars_weight - n_features, 0) + sum gamma). With covars_prior=0, covars_weight=1, n_features=1 that is
  sum_t sum_k gamma_t(k)(x_t - mu_k)^2 / T, the plain Baum-Welch update; `min_covar` is used only when
  hmmlearn itself initializes the covariance (init_params contains 'c'), never in the M step, so it is not set.
  Startprob/transmat priors of 1 give no pseudo-counts. One `fit` call = one E + one M step; the monitor's last
  history entry is the log-likelihood of the parameters before the M step (= the spec's LL[c]).
- Note: hmmlearn freezes entries that are exactly 0 in startprob/transmat (`np.where(x == 0, 0, ...)`); the starts
  used by `_gaussian_hmm_fit` are strictly positive. A transition row of zeros (state never visited) is replaced by a
  uniform row before parameters are set (spec sec. 4).
- Convergence rule as spec: stop after cycle c >= 2 when LL is not finite or 0 <= LL[c]-LL[c-1] < tol |LL[c-1]|;
  variance floored after every M step. A cycle that hmmlearn rejects (ValueError, degenerate parameters) ends EM with
  NaN appended to LL (the multi-start wrapper then discards that start).
- Determinism / initialisation: nothing is initialised by hmmlearn (init_params=''), so no random numbers are used;
  random_state=0 is fixed regardless. The initialisation is entirely Ben's six fixed starts in `_gaussian_hmm_fit`
  (shared variance = var(x), pi uniform, stay probability rho in {0.9, 0.5, 0.99}, means at the (k-1/2)/K data quantiles
  or evenly spaced mean +- std). `test_fit_is_deterministic` checks bitwise equality across runs.
- Loglik: `GaussianHMM.score` (forward algorithm); NaN for non-finite parameters or if hmmlearn rejects them.

## Exponential smoothing
`_fit_exp_smooth(x, a)`: `SimpleExpSmoothing(x, initialization_method='known', initial_level=mean(x[:10]))
.fit(smoothing_level=a, optimized=False).fittedvalues` (entry t = forecast of x[t]; matches a numpy recursion exactly).
`_exp_smooth_best_alpha(xtrain)`: as spec sec. 2: RMSE of one-step errors over t >= 1; coarse grid 0.1:0.2:0.9,
`np.polyfit` quadratic through the 3 lowest -> alphamin_1 = -c1/(2 c2), p1_1 = |c2|, cup_1 = sign(c2); final alpha =
`minimize_scalar(bounded, (0.01, 1), xatol=1e-5)` compared with a = 1. NaN alpha and diagnostics if the errors are
not finite. Choice documented: fixed 10-sample initial level (spec recommendation), not statsmodels' 'estimated'
initialisation (that would make alpha and l0 interact; the wrapper's contract is alpha-only). The "Code adapted from
Siddharth Arora" line was removed from the `exp_smoothing` references; Brown (1963) added.

## Validation (tests/native/test_thirdparty_B4.py: 31 pass, 1 skipped)
- HMM vs direct numpy Baum-Welch oracle (T=200, K=2,3, 4 seeds each): one cycle matches means, cov, P, pi, LL[0] to
  1e-8 (observed far below); five cycles match to 1e-7; loglik matches the oracle forward algorithm to 1e-9.
  LL monotone without floor; relative-change stop; K=1 gives mean/variance; floor respected.
- Simulated known-parameter HMMs (T=6000): 2-state (mu -2,2; sigma 1; stay .95/.90) and 3-state (mu -3,0,3; sigma .7;
  stay .9): means within 0.1, variance within 6%, P within 0.04, fitted LL >= LL under the true parameters - 5.
- ETS: local-level model x_t = l_{t-1}+e_t, l_t = l_{t-1}+alpha e_t, N=1000: alpha_hat within 0.08 of 0.3, 0.6, 0.85;
  AR(1) phi=.9 -> alpha in (0.5,1]; random walk -> alpha within 0.05 of 1; white noise alpha < 0.1; NaN input -> NaN.
  The end-to-end `exp_smoothing` test skips while `_arfit` (needed by `residual_analysis('full')`) is unimplemented.

## Agreement with hctsa (thirdparty_blackbox.json; evidence, not a gate)
HMM, 5 series x (fit_07_3, fit_08_2, cmp): largest per-output absolute difference per case 7e-5 .. 9.4e-2 (the latter
`noise`/fit_08_2 Mu_2, a poorly identified fit); LL-per-sample outputs differ by <= 1.3e-2 (mostly <= 3e-3); Cov by <=
3.7e-3. `noise`/fit_07_3 agrees to 1e-14. Expected given the different stopping rule.
ETS (alpha-search part; the 15 residual statistics need `_arfit`, not run): ar2 1.0000/1.0000, noise 0.0171/0.0100,
logistic 0.0100/0.0100, sine_noise 0.8494/0.8548, ar1_expnoise 0.8902/0.8937 (ours/hctsa); alphamin_1 within 0.12
(noise) and <= 0.005 elsewhere, identical to the spec writer's table.

## Existing tests changed
`tests/test_p2_c.py`: renamed calls; `h073`, `h082`, `cmp` tolerance 1e-8 -> 2e-2 (they pinned hctsa to the old EM's
numerics); series s0 (nearly flat likelihood over the state means; EM stops after 8 cycles with Mu_3 0.49 vs 0.38)
is excluded from the h073/h082 parameter comparison and checked on LLtrain/LLtest per sample to 5e-3 instead.
`tests/test_model_fit_ports.py`: rename only.

## Not done (for L3)
`hmmlearn` must be added to pyproject dependencies (not edited here, per README packaging note). statsmodels
and scipy are already dependencies. numba is still imported in model_fit.py for other functions.
