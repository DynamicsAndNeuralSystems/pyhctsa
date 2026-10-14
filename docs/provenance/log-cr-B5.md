# Clean-room log: WP B5 (rpde + close returns), worktree SP/cr-B5, branch cr-B5

## entropy.rpde, entropy._close_returns_hist (pyhctsa/operations/entropy.py)
Sources used: SP/cleanroom/specs/thirdparty/rpde.md (+ README.md, thirdparty_blackbox.json["rpde"], gen/thirdparty_inputs.json);
Little et al. (2007) BioMedical Engineering OnLine 6:23 (method); Mindlin & Gilmore (1992) Physica D 58:229 (close
returns); numpy and numba documentation. Implementation: forward delay embedding, an @njit scan per point
(inside -> outside -> first re-entry, Euclidean distance computed on the fly, no n x n matrix), histogram of
recurrence times, normalized, then the five entropy summaries as in spec section 3/4.

Choices: the series is used as given (no z-scoring inside; epsilon is absolute and the pipeline z-scores, as in
hctsa.yaml). NaN delay -> logger warning + NaN (dict_output expands to all-NaN dict); n < 2 or no returns -> NaN.
H_norm is NaN if there is only one bin (ln 1 = 0). `rpd` key not returned.

Agreement with the hctsa/MATLAB fixture: the 3 close-returns histograms are identical; rpde(y,3,1), rpde(y,2,1,0.12)
and rpde(y,3,'ac') for 5 series agree to max abs difference 1.1e-14 on all 5 outputs ('ac' tau matches hctsa's on
these series). Spec acceptance: periodic H_norm < 0.1, white noise H_norm > 0.6 (both tested).
Tests: tests/native/test_thirdparty_B5.py (10 tests; hand example, brute-force n x n cross-check, t_max, NaN cases,
fixture).

Runtime (m=3, tau=1, this Mac, after JIT warm-up): N=1000 about 1 ms; N=5000 about 20-24 ms per series.

## Statement
No third-party source (Max Little's rpde / ML_close_ret code, hctsa Toolboxes/), no quarantined material, no other
worktree or older commit, and no Claude transcript was consulted. EN_rpde.m's body was not opened.

## Not done (for the packaging agent)
`close_returns` Extension in pyhctsa/_build/ext_modules.py (and its entry in the returned list) and
toolboxes/Max_Little/ML_close_ret.c are to be removed; I left packaging untouched.
