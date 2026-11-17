"""Smoke test of an installed wheel (run by cibuildwheel after each wheel is built).

Checks that the compiled extensions import and that the default feature set runs end to end on
one series without a feature function raising.
"""
import importlib
import sys
import warnings

import numpy as np

import pyhctsa
from pyhctsa.calculator import FeatureCalculator
from pyhctsa.quality import Quality

EXTENSIONS = [
    "pyhctsa.toolboxes.physionet.sampen",
    "pyhctsa.toolboxes.c22.PD_PeriodicityWang",
    "pyhctsa.toolboxes.Tisean_3_0_1.d2",
    "pyhctsa.toolboxes.Tisean_3_0_1.poincare",
    "pyhctsa.toolboxes.Tisean_3_0_1.false_nearest",
]


def main() -> int:
    print(f"pyhctsa {pyhctsa.__version__} from {pyhctsa.__file__}, Python {sys.version.split()[0]}")
    for name in EXTENSIONS:
        importlib.import_module(name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # (features skipped without the optional ripser)
        calc = FeatureCalculator()
    rng = np.random.default_rng(0)
    e = rng.standard_normal(400)
    y = np.empty(400)
    y[0] = e[0]
    for t in range(1, 400):  # AR(1)
        y[t] = 0.7 * y[t - 1] + e[t]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = calc.extract(y)
    n_error = int((calc.quality.to_numpy() == int(Quality.ERROR)).sum())
    n_good = int((calc.quality.to_numpy() == int(Quality.GOOD)).sum())
    print(f"{len(calc.feature_funcs)} feature functions, {values.shape[1]} features: "
          f"{n_good} good, {n_error} errors")
    if n_error:
        print(calc.errors.to_string())
        return 1
    if values.shape[1] < 6000 or n_good < 0.9 * values.shape[1]:
        print("unexpectedly few features or good values")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
