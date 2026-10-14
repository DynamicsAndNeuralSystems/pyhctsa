# SPDX-License-Identifier: MIT
"""
A cut-down version of ``pywt.cwt`` from PyWavelets, adapted to handle complex-type casting.

Derived from PyWavelets (https://github.com/PyWavelets/pywt), ``pywt/_cwt.py``, which is distributed
under the MIT licence, reproduced below. Modified for pyhctsa: only the direct convolution method on
1-D input is kept, the pseudo-frequencies are not computed, complex-type casting is handled
differently, and the difference of the convolution is taken over the retained window only.

Reference: G. R. Lee, R. Gommers, F. Wasilewski, K. Wohlfahrt and A. O'Leary, "PyWavelets: A Python
package for wavelet analysis", Journal of Open Source Software 4(36), 1237 (2019).

Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
Copyright (c) 2012-     The PyWavelets Developers <https://github.com/PyWavelets/pywt>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from pywt._extensions._pywt import (
    ContinuousWavelet,
    DiscreteContinuousWavelet,
    Wavelet,
    _check_dtype,
)
from pywt._functions import integrate_wavelet


def _custom_cwt(data, scales, wavelet, *, precision=12):
    """
    One dimensional Continuous Wavelet Transform.

    Parameters
    ----------
    data : array-like
        Input signal (1-D).
    scales : array-like
        The wavelet scales to use.
    wavelet : Wavelet object or name
        Wavelet to use.
    precision: int, optional
        Length of wavelet (``2 ** precision``) used to compute the CWT. Greater
        will increase resolution, especially for higher scales, but will
        compute a bit slower. Too low will distort coefficients and their
        norms, with a zipper-like effect. The default is 12, it's recommended
        to use >=12.

    Returns
    -------
    coefs : ndarray
        Continuous wavelet transform of the input signal for the given scales
        and wavelet, shape ``(len(scales), len(data))``.

    Note
    ----
    This is a cut-down version of `pywt.cwt` from the PyWavelets package,
    adapted to handle complex-type casting. It keeps only the direct
    convolution method on 1-D input, and does not compute the pseudo-
    frequencies (which callers here do not use).
    """
    dt = _check_dtype(data)
    data = np.asarray(data, dtype=dt)
    if data.ndim != 1:
        raise ValueError("`data` must be one dimensional")
    dt_cplx = np.result_type(dt, np.complex64)
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)

    scales = np.atleast_1d(scales)
    if np.any(scales <= 0):
        raise ValueError("`scales` must only include positive values")

    is_complex = isinstance(wavelet, ContinuousWavelet) and wavelet.complex_cwt
    dt_out = dt_cplx if is_complex else dt
    n = data.size
    out = np.empty((scales.size, n), dtype=dt_out)

    int_psi, x = integrate_wavelet(wavelet, precision=precision)
    if is_complex:
        int_psi = np.conj(int_psi)

    # convert int_psi, x to the same precision as the data
    dt_psi = dt_cplx if int_psi.dtype.kind == 'c' else dt
    int_psi = np.asarray(int_psi, dtype=dt_psi)
    x = np.asarray(x, dtype=data.real.dtype)

    # loop invariants: the sampling grid `x` does not depend on the scale
    step = x[1] - x[0]
    span = x[-1] - x[0]
    int_psi_size = int_psi.size

    for i, scale in enumerate(scales):
        j = (np.arange(scale * span + 1) / (scale * step)).astype(int)  # floor
        if j[-1] >= int_psi_size:
            j = np.extract(j < int_psi_size, j)
        int_psi_scale = int_psi[j][::-1]

        m = int_psi_scale.size
        if m < 2:
            raise ValueError(f"Selected scale of {scale} too small.")
        conv = np.convolve(data, int_psi_scale)

        # np.diff() of the full convolution followed by a symmetric trim to
        # len(data); differencing only the retained window gives the same values
        # and avoids materialising the full-length temporary.
        start = (m - 2) // 2
        coef = -np.sqrt(scale) * (conv[start + 1:start + 1 + n] - conv[start:start + n])
        out[i, :] = coef.real if out.dtype.kind != 'c' else coef

    return out
