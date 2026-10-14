<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyhctsa/main/img/pyhctsa_logo_dark.png" media="(prefers-color-scheme: dark)">
    <img src="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyhctsa/main/img/pyhctsa_logo.png" alt="pyhctsa logo" height="320"/>
  </picture>
</p>

<h1 align="center"><em>pyhctsa</em></h1>
<p align="center"><strong>Highly comparative time-series analysis in Python</strong></p>

<p align="center">
  <a href="https://pepy.tech/projects/pyhctsa"><img src="https://static.pepy.tech/personalized-badge/pyhctsa?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads" alt="PyPI Downloads"></a>
  <a href="https://pypi.org/project/pyhctsa/"><img src="https://img.shields.io/pypi/v/pyhctsa.svg" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/pyhctsa/"><img src="https://img.shields.io/pypi/pyversions/pyhctsa.svg" alt="Python Version"></a>
  <a href="https://github.com/DynamicsAndNeuralSystems/pyhctsa/actions/workflows/run_unit_tests.yaml"><img src="https://img.shields.io/github/actions/workflow/status/DynamicsAndNeuralSystems/pyhctsa/run_unit_tests.yaml?branch=main&label=CI" alt="CI"></a>
  <a href="https://github.com/DynamicsAndNeuralSystems/pyhctsa/actions/workflows/run_unit_tests.yaml"><img src="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyhctsa/coverage-badge/coverage.svg" alt="Coverage"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL_v3-orange.svg" alt="License"></a>
  <br>
  <a href="https://doi.org/10.21105/joss.10581"><img src="https://joss.theoj.org/papers/10.21105/joss.10581/status.svg" alt="JOSS"></a>
  <a href="https://github.com/pyOpenSci/software-submission/issues/282"><img src="https://pyopensci.org/badges/peer-reviewed.svg" alt="pyOpenSci Peer-Reviewed"></a>
  <a href="https://doi.org/10.5281/zenodo.20820138"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20820138-blue.svg" alt="Zenodo DOI"></a>
</p>

<p align="center">
  <a href="https://dynamicsandneuralsystems.github.io/pyhctsa/">Documentation</a> &nbsp;·&nbsp;
  <a href="https://dynamicsandneuralsystems.github.io/pyhctsa/usage/getting_started.html">Getting started</a> &nbsp;·&nbsp;
  <a href="https://dynamicsandneuralsystems.github.io/pyhctsa/methods/index.html">Method list</a> &nbsp;·&nbsp;
  <a href="https://dynamicsandneuralsystems.github.io/pyhctsa/api.html">API reference</a> &nbsp;·&nbsp;
  <a href="https://doi.org/10.21105/joss.10581">Paper</a>
</p>

---

The **PY**thon toolkit for **H**ighly **C**omparative **T**ime-**S**eries **A**nalysis (_pyhctsa_) is a living library of
time-series analysis methods. It computes **nearly 6800 interpretable time-series features** (6791 in the default feature set, matching `hctsa`) from a single univariate
series — spanning distributional shape, autocorrelation, entropy and information theory, scaling, stationarity,
nonlinear dynamics, spectral and wavelet properties, model fits, and more — making it the most comprehensive feature
set available in native Python.

<p align="center">
  <img src="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyhctsa/main/paper/pyhctsa_graphical_abstract.png" alt="Overview of the pyhctsa workflow" width="750"/>
</p>

## Installation

_pyhctsa_ requires Python 3.10 or newer:

```bash
pip install pyhctsa
```

We strongly recommend installing into a fresh virtual environment to prevent dependency clashes:

```bash
conda create -n pyhctsa python=3.12 -y && conda activate pyhctsa && pip install pyhctsa
```

## Quickstart

Instantiate a `FeatureCalculator` and call `extract` on your data:

```python
from pyhctsa.calculator import FeatureCalculator
from pyhctsa.utils import get_dataset

calc = FeatureCalculator()          # Loaded 939 master operations.

e1000 = get_dataset()               # Empirical1000 dataset (downloaded once, 33 MB)
data = e1000[0]                     # a list, array, or pandas Series

res = calc.extract(data)            # pandas DataFrame, one row per series
```

`extract` accepts either a single time series or a list of series, which do **not** need to be the same length:

```python
res = calc.extract(e1000[:10], verbose=True)
print(res.shape)                    # (10, F) -> N series x F features
```

Results are returned as a `pandas.DataFrame` of shape *N* × *F*, where *N* is the number of time-series
instances and *F* is the number of time-series features.

New to _pyhctsa_? Step-by-step notebooks and example workflows live in [`/tutorials`](https://github.com/DynamicsAndNeuralSystems/pyhctsa/tree/main/tutorials), and a walkthrough
is available in the [getting started guide](https://dynamicsandneuralsystems.github.io/pyhctsa/usage/getting_started.html).

## Usage

### Custom feature sets

By default, `FeatureCalculator` initializes the full feature set. To compute a subset, pass the corresponding
configuration `.yaml` file:

```python
custom_calc = FeatureCalculator(config_path="subset.yaml")
```

The number of master operations (callable functions) specified by the `.yaml` is displayed for verification, e.g.
`Loaded 939 master operations.` for the default feature set.

### Calling individual operations

Individual operations can be imported directly from their module. For example, `raw_hrv_meas` lives in the `medical`
module:

```python
from pyhctsa.operations.medical import raw_hrv_meas

data = ...              # your ArrayLike data
res = raw_hrv_meas(data)  # a dictionary or scalar value
```

> [!NOTE]
> Individual operations can only be called directly on individual time-series instances.

Operations are grouped into the following modules — see the
[method list](https://dynamicsandneuralsystems.github.io/pyhctsa/methods/index.html) for the full catalogue:

| | | | |
|---|---|---|---|
| `changepoint` | `correlation` | `criticality` | `distribution` |
| `entropy` | `extreme_events` | `graph` | `hypothesis_tests` |
| `information` | `medical` | `model_fit` | `nonlinearity` |
| `physics` | `pre_process` | `scaling` | `spectral` |
| `stationarity` | `surrogates` | `symbolic` | `wavelet` |

> [!NOTE]
> These conceptual groupings are a convenience only, and are not intended as definitive classifications.

### Parallel computing

Time-series feature extraction is computationally intensive. To speed up processing, _pyhctsa_ can distribute the
workload across multiple CPU cores on your local machine using the `LocalDistributor`:

```python
from pyhctsa.calculator import FeatureCalculator
from pyhctsa.distribute import LocalDistributor

calc = FeatureCalculator()

# it is generally recommended to set n_workers to the number of physical CPU cores
dist = LocalDistributor(n_workers=4)

res = calc.extract(data, distributor=dist)
```

Coming from MATLAB _hctsa_? A mapping between legacy operation names and their _pyhctsa_ equivalents is available in
the [name mappings table](https://dynamicsandneuralsystems.github.io/pyhctsa/mappings/index.html).

## Version 2 and this repository's history

_pyhctsa_ 2.0 was published in October 2026 in a re-created repository with a fresh history. An audit found that
parts of earlier versions were derived from code we did not have the right to distribute. Every affected function
was re-implemented clean-room from the published methods and public documentation, and third-party code is now
bundled only under GPL-compatible licenses (see [Licenses](#licenses)). Earlier releases (1.x) have been withdrawn;
please upgrade.

Feature values from 2.0 agree closely with _hctsa_ but can differ from those of 1.x, and some features were renamed,
redefined or removed to match _hctsa_ v3.0. See the release notes for details.

## Citation

If you use _pyhctsa_ in your work, please cite the accompanying JOSS paper:

> Moore, J. B., & Fulcher, B. D. (2026). _pyhctsa_: A Python package for highly comparative time-series analysis.
> _Journal of Open Source Software_, 11(123), 10581. https://doi.org/10.21105/joss.10581

```bibtex
@article{Moore2026pyhctsa,
  author  = {Moore, Joshua B. and Fulcher, Ben D.},
  title   = {pyhctsa: A Python package for highly comparative time-series analysis},
  journal = {Journal of Open Source Software},
  year    = {2026},
  volume  = {11},
  number  = {123},
  pages   = {10581},
  doi     = {10.21105/joss.10581}
}
```

Machine-readable metadata is provided in [`CITATION.cff`](https://github.com/DynamicsAndNeuralSystems/pyhctsa/blob/main/CITATION.cff).

## Licenses

### pyhctsa

_pyhctsa_ is licensed under the [GNU General Public License v3.0 or later](LICENSE). It is a Python implementation of
the feature library of [_hctsa_](https://github.com/benfulcher/hctsa) (Ben D. Fulcher). The _hctsa_ feature code is
GPL-3.0-or-later; the few _hctsa_ helper functions that were released under CC BY-NC-SA 4.0 were written solely by
Ben D. Fulcher, who has relicensed their Python translations in _pyhctsa_ under GPL-3.0-or-later.

### Acknowledgements and third-party code

Some features use code written by others. Third-party code is kept in
[`pyhctsa/toolboxes/<name>/`](pyhctsa/toolboxes), each directory with its original copyright notice and license; full
texts are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). We thank the authors.

| Component | Authors | License | Location in pyhctsa | Reference |
|---|---|---|---|---|
| TISEAN 3.0.1 (d2, c2g/c2t, poincare, false_nearest, boxcount, lyap_r, lyap_spec, c1/d1/c2d, nrlazy, nstat_z) | R. Hegger, H. Kantz, T. Schreiber | GPL-2.0-or-later | `toolboxes/Tisean_3_0_1/` | Hegger, Kantz & Schreiber, Chaos 9, 413 (1999) |
| GPML (Gaussian process inference, covariance functions, `minimize`) | C. E. Rasmussen, H. Nickisch | BSD-2-Clause (FreeBSD) | `toolboxes/gpml/` | Rasmussen & Nickisch, JMLR 11, 3011 (2010) |
| catch22 (PD_PeriodicityWang, splinefit) | C. H. Lubba, B. D. Fulcher et al.; splinefit after J. Lundgren | GPL-3.0; splinefit BSD-2-Clause | `toolboxes/c22/` | Lubba et al., Data Min. Knowl. Disc. 33, 1821 (2019) |
| Delay vector variance (DVV) toolbox | D. P. Mandic, T. Gautama | GPL-2.0-or-later | `toolboxes/DVV/` | Gautama, Mandic & Van Hulle, Physica D 190, 167 (2004) |
| Sample entropy (sampen.c) | D. Lake, J. R. Moorman, G. Moody (PhysioNet) | GPL-2.0-or-later | `toolboxes/physionet/` | Richman & Moorman, Am. J. Physiol. 278, H2039 (2000) |
| Multiscale multifractal analysis (MMA) | J. Gieraltowski (PhysioNet) | GPL-2.0-or-later | `toolboxes/physionet/` | Gieraltowski, Zebrowski & Baranowski, Phys. Rev. E 85, 021915 (2012) |
| Steps and bumps toolkit (l1pwc) | M. A. Little; after S.-J. Kim, K. Koh, S. Boyd, D. Gorinevsky (l1_tf) | GPL-2.0-or-later | `toolboxes/Max_Little/` | Little & Jones, ICASSP (2010) |
| Rescaled auto-density (RAD) | B. Harris | MIT | `toolboxes/RAD/` | Harris et al., Phys. Rev. X 14, 031021 (2024) |
| Continuous wavelet transform (cut-down `pywt.cwt`) | The PyWavelets developers | MIT | `toolboxes/pywavelets/` | Lee et al., JOSS 4, 1237 (2019) |

Further credits:

- The falling sticks statistic (`CO_FallingSticks`) is adapted from the Python FALLstick implementation written by
  Eugene Chon, a student of B. D. Fulcher, for _hctsa_.
- The heart-rate-variability measures (`MD_*`) are Copyright (C) 2016 Max A. Little and Ben D. Fulcher.
- The permutation-entropy variant is that of B. Land and D. Elias (Cornell University).
- The visibility-graph measures follow Lacasa et al., PNAS 105, 4972 (2008) and Luque et al., Phys. Rev. E 80, 046103
  (2009).
- The Kraskov-Stoegbauer-Grassberger and Gaussian mutual-information estimators (`pyhctsa/ksg.py`) are written from the
  published method (Kraskov et al., Phys. Rev. E 69, 066138, 2004); the Java Information Dynamics Toolkit (J. T. Lizier,
  Front. Robot. AI 1, 11, 2014) is credited as the method reference used by _hctsa_. No JIDT code is included.
- Several methods follow TSTOOL (C. Merkwirth, U. Parlitz, I. Wedekind, D. Engster, W. Lauterborn): the tc3, trev,
  delay-time, fractal-dimension and return-time measures.

#### Clean-room re-implementations (implemented from the published method)

The following methods are implemented in _pyhctsa_ from their published descriptions, without use of the original
source code, and we credit the authors whose software defined them in _hctsa_: ARFIT stepwise autoregressive fitting
(T. Schneider, A. Neumaier; Neumaier & Schneider, ACM TOMS 27, 27 (2001)); Lempel-Ziv complexity, Shannon block
entropy, nonlinear prediction error as in M. Small's toolbox (Kaspar & Schuster, Phys. Rev. A
36, 842 (1987); Small, _Applied Nonlinear Time Series Analysis_, 2005); the Theiler, crinkle and time-reversal
statistics (D. Kaplan; Theiler et al., Physica D 58, 77 (1992); Kaplan & Glass, _Understanding Nonlinear Dynamics_,
1995); nsAMDF (T. E. Özkurt; NeuroImage 223, 117356 (2020)); RPDE and close returns (M. A. Little; Little et al.,
BioMed. Eng. OnLine 6, 23 (2007)); and the fractal-dimension error function (van de Water & Schram, Phys. Rev. A 37,
3118 (1988)). Approximate entropy is implemented from Pincus, PNAS 88, 2297 (1991). Clean-room provenance logs are in
[`docs/provenance/`](docs/provenance).

Daniel Kaplan's statistics are re-implemented from his published methods; his `DK_*` files in _hctsa_ are GPL-3 and none of
that source is included. The exponential-smoothing feature (`MF_ExpSmoothing`) follows the design of the code provided by
S. Arora for _hctsa_ (training/held-out split, a smoothing parameter chosen on the training part, residual analysis); the
smoothing itself is computed with statsmodels.

#### Dependencies

Runtime dependencies, called through their public interfaces (no code copied; all GPL-compatible):
NumPy, SciPy, pandas, statsmodels and scikit-learn (BSD-3-Clause), numba (BSD-2-Clause), PyYAML, PyWavelets and rich (MIT),
pathos, lmfit and [hmmlearn](https://github.com/hmmlearn/hmmlearn) (BSD-3-Clause; hmmlearn replaces Z. Ghahramani's HMM code
used in _hctsa_), [arch](https://github.com/bashtage/arch) (NCSA), and, optionally,
[ripser](https://github.com/scikit-tda/ripser.py) (MIT). See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), section 4.2.

## AI usage disclosure

Portions of this codebase (including tests and function documentation) were refactored and generated with the
assistance of Large Language Models (LLMs). All AI-generated contributions have been reviewed and verified by the
human maintainers.
