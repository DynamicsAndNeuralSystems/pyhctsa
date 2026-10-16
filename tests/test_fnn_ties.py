import numpy as np
import pytest

from pyhctsa.toolboxes.Tisean_3_0_1 import tisean


@pytest.mark.parametrize("delay", [1, 3])
def test_false_nearest_ties_do_not_depend_on_rounding(delay):
    # an integer-valued series has many exactly tied distances; its affine transforms and
    # its negation embed identically up to rounding, so the false-neighbor curve and the
    # point counts must be identical
    rng = np.random.default_rng(1)
    e = rng.standard_normal(2000)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.8 * x[t - 1] + e[t]
    y = np.round(5 * x)
    r = tisean.false_nearest(y, delay, 1, 6, 10, 5.0)
    for z in (3.7 * y - 12, -y):
        rz = tisean.false_nearest(z, delay, 1, 6, 10, 5.0)
        np.testing.assert_array_equal(rz['dim'], r['dim'])
        np.testing.assert_array_equal(rz['pfnn'], r['pfnn'])
        np.testing.assert_array_equal(rz['count'], r['count'])
