# SPDX-License-Identifier: GPL-2.0-or-later
"""
Multiscale multifractal analysis (MMA), ported from the PhysioNet MMA code.

``MMA.m`` is Copyright (C) 2014 Jan Gierałtowski (Warsaw University of Technology, Faculty of
Physics) and is distributed on PhysioNet under the GNU General Public License, version 2 or (at
your option) any later version. This module is a derived work: a port to Python of the algorithm of
``MMA.m``, via hctsa's ``SC_MMA.m`` (modified by B. D. Fulcher, 2015), and modified for pyhctsa. It
is distributed with pyhctsa under GPL-3.0-or-later (permitted by the "or later" clause).

References
----------
.. [1] J. Gierałtowski, J. J. Żebrowski, and R. Baranowski, "Multiscale multifractal analysis of
    heart rate variability recordings with a large number of occurrences of arrhythmia",
    Phys. Rev. E 85, 021915 (2012). http://dx.doi.org/10.1103/PhysRevE.85.021915
.. [2] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
    Research Resource for Complex Physiologic Signals", Circulation 101(23):e215-e220 (2000).
"""
import logging

import numpy as np

from ...fitting import polyfit
from ...utils import _arange_inclusive, _round_half_away_array

logger = logging.getLogger('pyhctsa')


def mma_hurst(y, do_overlap: bool = False, scale_range=None, q_range=None):
    """The generalized Hurst exponents of the multiscale multifractal analysis of a time series.

    Parameters
    ----------
    y : array_like
        Input time series (a 1-D vector).
    do_overlap : bool, optional
        False (default): partition into non-overlapping windows of analysis.
        True: overlapping windows with a step of 1 (much longer calculations).
    scale_range : sequence of 2 numbers, optional
        [min_scale, max_scale]. Defaults to [10, max(100, round(N/40))].
    q_range : sequence of 2 numbers, optional
        [q_min, q_max] multifractal parameter range. Defaults to [-5, 5].

    Returns
    -------
    tuple or None
        ``(hqs, s_list_scaled, q_list)``: the exponents (``q_list.size`` x ``s_list_scaled.size``),
        the scales (times 3, as recorded in the original algorithm), and the values of q. None if
        the series is too short for the scale range.
    """
    y = np.asarray(y, dtype=float).ravel()

    # Time-series length:
    n = y.size

    # --------------------------------------------------------------------------
    # Check inputs:
    # --------------------------------------------------------------------------
    if scale_range is None:
        scale_range = [10, float(max(100, _round_half_away_array(n / 40)))]
    min_scale = scale_range[0]
    max_scale = scale_range[1]

    if max_scale > n:
        logger.warning(
            "Time-series (N=%u) too short for multiscale multifractal analysis "
            "(max_scale=%u exceeds N)" % (n, max_scale)
        )
        return None
    elif (max_scale / 5) < min_scale:
        logger.warning(
            "Time-series (N=%u) too short for multiscale multifractal analysis" % n
        )
        return None
    elif max_scale % 5 != 0:
        max_scale = float(_round_half_away_array(max_scale / 5)) * 5
        logger.warning("adjusted max_scale to %u" % max_scale)

    if q_range is None:
        q_range = [-5, 5]
    q_min = q_range[0]
    q_max = q_range[1]

    q_list = _arange_inclusive(q_min, 0.1, q_max)
    q_list[q_list == 0] = 0.0001

    # --------------------------------------------------------------------------

    prof = np.cumsum(y)
    slength = prof.size

    num_increments = 20
    s_list_full = np.unique(_round_half_away_array(np.linspace(min_scale, max_scale, num_increments)))

    # Preallocate fqs (one row per scale x q combination):
    fqs = np.zeros((s_list_full.size * q_list.size, 3))
    row_idx = 0

    for s in s_list_full:
        s = int(s)

        if do_overlap:
            # Sliding windows of length s, step 1:
            num_segments = slength - s + 1
            segments = np.lib.stride_tricks.sliding_window_view(prof, s)
        else:
            num_segments = slength // s
            segments = prof[: num_segments * s].reshape(num_segments, s)

        x_base = _arange_inclusive(1, 1, s)
        f2_nis = np.zeros(num_segments)

        for ni in range(num_segments):
            seg = segments[ni, :]
            fit = polyfit(x_base, seg, 2)
            f2_nis[ni] = np.mean((seg - np.polyval(fit, x_base)) ** 2)
            if f2_nis[ni] <= 1e-20 * np.mean(seg ** 2):
                f2_nis[ni] = 0.0  # the quadratic fits the segment exactly, to within rounding

        for q in q_list:
            # (a segment with no fluctuation has an infinite moment for q < 0, and for q = 0,
            # the geometric mean, which q = 1e-4 stands in for; so it is left out there, and
            # contributes 0 for q > 0)
            f2_q = f2_nis if q > 0.001 else f2_nis[f2_nis > 0]
            fq = np.mean(f2_q ** (q / 2)) ** (1 / q) if f2_q.size else np.nan
            fqs[row_idx, :] = [q, s, fq]
            row_idx += 1

    fqs_ll = np.column_stack(
        (fqs[:, 0], fqs[:, 1], np.log(fqs[:, 1]), np.log(fqs[:, 2]))
    )

    # --------------------------------------------------------------------------
    # Now compute Hurst exponents as the gradients of F(q) curves
    # --------------------------------------------------------------------------
    if np.sum(s_list_full <= max_scale / 5) >= 10:
        s_list = s_list_full[s_list_full <= max_scale / 5]
    elif min_scale == max_scale / 5:
        # Single-point range: a zero increment would give an empty range, so take the
        # one point directly:
        s_list = np.array([float(min_scale)])
    else:
        # Sample higher in the scale dimension:
        s_spacing = ((max_scale / 5) - min_scale) / 10
        s_list = _arange_inclusive(min_scale, s_spacing, max_scale / 5)

    # Coarser sampling of q space:
    q_list = _arange_inclusive(q_min, 0.5, q_max)
    q_list[q_list == 0] = 0.0001

    hqs = np.zeros((q_list.size, s_list.size))
    for si, s_val in enumerate(s_list):
        for qi, q_val in enumerate(q_list):
            mask = (
                (np.abs(fqs_ll[:, 0] - q_val) < 1e-8)
                & (fqs_ll[:, 1] >= s_val)
                & (fqs_ll[:, 1] <= 5 * s_val)
            )
            fit_temp = fqs_ll[mask, :]
            hqs[qi, si] = polyfit(fit_temp[:, 2], fit_temp[:, 3], 1)[0]

    # Not completely on top of the algorithm, but for some reason this was
    # recorded as a multiple of 3 in the original algorithm:
    s_list_scaled = s_list * 3
    return hqs, s_list_scaled, q_list
