# Provenance log: group signal-a (clean-room implementer)

Branch `impl-signal-a`. Specs used: CR/specs/signal/{bin_picker, linspace_colon, quantile, rng, buffer, autocorr_fft,
medfilt1, findpeaks, ksdensity_bandwidth}.md and their JSON fixtures (+ signalA_extra.json), plus RULES.md,
DEFAULT_DECISIONS.md and the brief. Libraries: numpy only in the implementations (scipy `peak_prominences`,
statsmodels `acf` only in tests).

**Statement:** no MathWorks source (nothing under /Applications/MATLAB_R2026a.app/toolbox, no `type`/`edit`/`which`
on built-ins) and no removed or derived pyhctsa code (no other worktree, no git history, no PyPI/GitHub copy, no
SP/fix, SP/prov or quarantine) was consulted. I ran no MATLAB myself; all MATLAB evidence is the spec fixtures. The code
was written from the specs' stated rules and public documentation, in my own structure.

Agreement below is measured on the FULL spec fixtures (tests ship subsampled copies in tests/cleanroom/fixtures; set
`PYHCTSA_FIXTURES` to the full spec folder to rerun everything).

| function (file) | sources | agreement on fixtures |
|---|---|---|
| `utils._ml_rng` | spec rng sec. 4 (MT19937 via numpy legacy RandomState; seed 0 -> 5489); Matsumoto & Nishimura 1998 | `rand` streams bitwise for all 7 seeds, randi mapping exact |
| `utils._ml_randperm` | spec rng (rank order of n uniform draws) | 32/32 perms and following stream exact |
| `nonlinearity._ml_randsample` | spec rng (4k>n branch rule, draw-and-dedupe branch) | all 233 cases (subsample 78 in tests) exact incl. stream state; NL_DVV call sequence exact |
| `utils._linspace` | spec linspace sec. 4.1 (multiply-then-divide; overflow fallback) | 420/420 bitwise |
| `scaling._colon` | spec colon sec. 4.2 (tolerance 2 eps max(abs a, abs b), two-ended generation, midpoint formula) | element count 2811/2811; values bitwise 2804/2811; the 7 others differ by <= 1.085 eps*max(abs a, abs b) |
| `utils.matlab_quantile`, `extreme_events._hazen`, `model_fit._nan_quantile` | spec quantile (Hazen, Hyndman & Fan 1996 def. 5; percent round trip; (1-f)a+fb) | 8377/8377 values bitwise; special cases exact |
| `utils.histc` | spec bin_picker sec. 3 + histc doc | all 4 inclusion cases exact |
| `utils.bin_picker` | spec bin_picker rules A, B, C (black-box derived) | nbins 4640/4640 bitwise; auto (Scott) 600/600, sqrt/sturges 600/600; nearly-constant 178/178 |
| `correlation._histcounts` (new helper, spec tree (a) only) | spec bin_picker sec. 3 (np.histogram + division by len(x)) | inclusion cases abs 1e-15 |
| `utils.make_mat_buffer` | spec buffer sec. 4 (column count rules) | 134/134 exact; the short nodelay cases incl. the error case |
| `correlation._acf_fft` | spec autocorr_fft (Box-Jenkins-Reinsel, Wiener-Khinchin, nfft = 2^(ceil(log2 N)+1)) | max abs error <= 1e-13 (offset case 1e-10) as specified; matches statsmodels `acf(fft=False)` to 1e-12 |
| `pre_process._med_filt_1d` | spec medfilt1 (tree (b) zero-pad contract, window alignment) | 168/170 exact, rest <= 1.1e-16 |
| `utils._local_maxima` (shared helper), `stationarity._matlab_local_maxima`, `spectral._findpeaks` | spec findpeaks sec. 4 rules 1-3 | 333/333 non-error cases exact (locs and heights, order included); 42 error cases raise ValueError; prominence cases: locs exact, prominences from scipy within 1e-12 |
| `distribution._ksdensity_bandwidth` (new helper, spec tree (a) only) | spec ksdensity_bandwidth (Bowman & Azzalini 1997 p. 31; 0.6745; degenerate fallbacks) | max rel error 2.0e-16 over cases + 8 degenerate cases |

## Deviations from the spec / notes
- `bin_picker` for very large constant data (|x| above about 4.5e15/nbins): not specified (spec sec. 10); my output
  differs from MATLAB there (the `histcounts_huge_constant` cases outside the stated regime are skipped in tests;
  the 12 inside it are exact).
- `_findpeaks` raises ValueError for `min_pk_dist >= L - 1` only when `min_pk_dist > 0` (spec only says d >= L-1 for a
  supplied distance).
- `_histcounts` and `_ksdensity_bandwidth` are tree-(a)-only helpers; the stubbed (b) tree has no callers for them. I added
  them as specified so that they are available at the merge.
- `_med_filt_1d` implements the tree (b) zero-pad contract only (no `pad='edge'` keyword).
- `utils._round_half_away`, `_ml_std`, `_zscore_matlab`, `scaling._round/_polyfit/_std` etc. are not in my specs and
  were left as stubs.

## Environment note
The venv's numpy 2.0.2 breaks the anaconda pandas (built for numpy 1.x), so `PYTHONPATH=<worktree>` alone cannot import
pyhctsa. I ran with `PYTHONPATH=<worktree>:/opt/homebrew/anaconda3/lib/python3.12/site-packages` (numpy 1.26.4
first); nothing was installed. Under numpy 1.26 `np.trapezoid` is missing, which breaks some unrelated existing tests.
