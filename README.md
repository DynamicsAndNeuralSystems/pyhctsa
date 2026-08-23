<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyhctsa/main/img/pyhctsa_logo_dark.png" media="(prefers-color-scheme: dark)">
    <img src="https://raw.githubusercontent.com/DynamicsAndNeuralSystems/pyhctsa/main/img/pyhctsa_logo.png" alt="pyhctsa logo" height="320"/>
  </picture>
</p>

<h1 align="center"><em>pyhctsa</em></h1>
<p align="center"><strong>Highly comparative time-series analysis in Python</strong></p>

<p align="center">
  <a href="https://pypi.org/project/pyhctsa/"><img src="https://img.shields.io/pypi/v/pyhctsa.svg" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/pyhctsa/"><img src="https://img.shields.io/pypi/pyversions/pyhctsa.svg" alt="Python Version"></a>
  <a href="https://pepy.tech/projects/pyhctsa"><img src="https://static.pepy.tech/personalized-badge/pyhctsa?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads" alt="PyPI Downloads"></a>
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
time-series analysis methods. It computes **over 4500 interpretable time-series features** from a single univariate
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

calc = FeatureCalculator()          # Loaded 791 master operations.

e1000 = get_dataset()               # bundled Empirical 1000 dataset
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
`Loaded 700 master operations.`.

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

### Internal licenses

Code for computing features from time-series data is licensed under the
[GNU General Public License version 3](http://www.gnu.org/licenses/gpl-3.0.en.html).

### External packages and dependencies

While the majority of features in _pyhctsa_ rely on standard Python libraries, a small subset of features require
external toolboxes. The following external time-series analysis code packages are bundled with the software (in
[`pyhctsa/toolboxes`](https://github.com/DynamicsAndNeuralSystems/pyhctsa/tree/main/pyhctsa/toolboxes)) and are used by the feature-extraction calculator:

| Package | Author | License |
|---|---|---|
| [Time-series analysis code](https://github.com/m-small) | Michael Small | unlicensed |
| [Time-series analysis code](http://www.maxlittle.net/software/index.php) | Max Little | GPL |
| [TISEAN](http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html) (v3.0.1) | Hegger, Kantz & Schreiber | GPL |

The following codebases have been adapted directly into Python code within _pyhctsa_, rather than being included as
external toolboxes:

| Code | Author | License |
|---|---|---|
| Embedding statistics | Danny Kaplan | GPL |
| Histogram code | Rudy Moddemeijer | unlicensed |

## AI usage disclosure

Portions of this codebase (including tests and function documentation) were refactored and generated with the
assistance of Large Language Models (LLMs). All AI-generated contributions have been reviewed and verified by the
human maintainers.
