# SPDX-License-Identifier: MIT
"""
The rescaled auto-density (RAD), after B. Harris's ``RAD.m`` (the Criticality package).

Derived from RAD.m of https://github.com/brendanjohnharris/Criticality, which is
Copyright (c) 2021 Brendan Harris and contributors, and is distributed under the MIT licence
(reproduced below). The RAD method was devised and implemented by Brendan Harris; it is described in
Harris et al., "Tracking the Distance to Criticality in Systems with Unknown Noise", Phys. Rev. X 14,
031021 (2024). This module is a port to Python (via hctsa's ``CR_RAD.m``) and is modified for pyhctsa.

MIT License

Copyright (c) 2021 Brendan Harris and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np


def rad_centre(x):
    """The series centered at its median, in absolute value (RAD's ``centre`` option)."""
    x = np.asarray(x)
    x = x - np.median(x)
    return np.abs(x)


def rad_core(x, tau: int) -> float:
    """The RAD of the series ``x`` (already centered, if wanted) at the integer delay ``tau``."""
    # Delay embed at interval tau
    y = x[tau:]
    x = x[:-tau]

    # Median split
    sub_medians = x < np.median(x)
    super_median_sd = np.std(x[~sub_medians], ddof=1)
    sub_median_sd = np.std(x[sub_medians], ddof=1)

    # Properties of the auto-density
    sigma_dx = np.std(y - x, ddof=1)
    density_difference = (1/super_median_sd) - (1/sub_median_sd)

    # return RAD
    return sigma_dx * density_difference
