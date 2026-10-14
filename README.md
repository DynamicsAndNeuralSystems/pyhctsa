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
  <a href="https://doi.org/10.5281/zenodo.18529934"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18529934-blue.svg" alt="Zenodo DOI"></a>
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

## Next steps

- [Getting started](https://dynamicsandneuralsystems.github.io/pyhctsa/usage/getting_started.html): computing features, calling individual operations, and
  parallel computing with `LocalDistributor`.
- [Advanced usage](https://dynamicsandneuralsystems.github.io/pyhctsa/usage/advanced_usage.html): custom feature sets from your own `.yaml` configuration,
  and feature filtering.
- [Method list](https://dynamicsandneuralsystems.github.io/pyhctsa/methods/index.html) and [API reference](https://dynamicsandneuralsystems.github.io/pyhctsa/api.html): every operation and its outputs.
- [Tutorials](https://github.com/DynamicsAndNeuralSystems/pyhctsa/tree/main/tutorials): step-by-step notebooks.

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

_pyhctsa_ is licensed under the [GNU General Public License v3.0 or later](LICENSE). It is a Python implementation of
the feature library of [_hctsa_](https://github.com/benfulcher/hctsa). Some features use third-party code, bundled in
[`pyhctsa/toolboxes/`](pyhctsa/toolboxes) under its own GPL-compatible license (full texts in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)); others are clean-room re-implementations of published methods.
Full credits are on the [Acknowledgements](https://dynamicsandneuralsystems.github.io/pyhctsa/acknowledgements.html) page of the documentation.

## AI usage disclosure

Much of _pyhctsa_ 2.0, including the clean-room re-implementations, tests and documentation, was written with AI coding
assistants (Anthropic's Claude) under the direction of the maintainers. Not every AI-generated contribution has been
reviewed line by line by a human; each is instead verified against the outputs of _hctsa_ or by appropriate tests.
