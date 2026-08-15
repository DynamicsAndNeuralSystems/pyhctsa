"""
Python front-end to the TISEAN 3.0.1 routines used by pyhctsa.

hctsa drives TISEAN through the shell: it writes the time series to a temporary
file, runs ``d2``, and then pipes the resulting ``.c2`` file through ``c2g`` and
``c2t``.  Here the same three steps happen in-process:

* :func:`d2` calls the C kernel in ``TS_d2.c`` (a re-entrant transcription of
  TISEAN's ``d2.c``) and assembles the ``.c2``/``.d2``/``.h2`` tables from the
  raw pair counts, using the expressions ``d2.c`` prints.
* :func:`c2g` and :func:`c2t` are ports of ``source_f/c2g.f`` and
  ``source_f/c2t.f``.  Those are Fortran, so they are reimplemented rather than
  wrapped -- both are short, and this keeps the package free of a Fortran
  toolchain.

TISEAN is Copyright (c) 1998-2007 Rainer Hegger, Holger Kantz, Thomas
Schreiber, and is distributed under the GNU General Public License v2 or later.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from . import d2 as _d2_c
from . import poincare as _poincare_c

__all__ = ["d2", "c2g", "c2t", "poincare"]


# 15-point Gauss-Kronrod rule, as tabulated in SLATEC's dqk15.f (which is what
# c2g.f integrates with).  Only the positive abscissae are listed there.
_XGK = np.array([
    0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
    0.7415311855993945, 0.5860872354676911, 0.4058451513773972,
    0.2077849550078985, 0.0,
])
_WGK = np.array([
    0.02293532201052922, 0.06309209262997855, 0.10479001032225018,
    0.14065325971552592, 0.16900472663926790, 0.19035057806478540,
    0.20443294007529889, 0.20948214108472782,
])
# Mirrored to the full 15 nodes/weights on (-1, 1).
_GK_X = np.concatenate([-_XGK[:7], [0.0], _XGK[6::-1]])
_GK_W = np.concatenate([_WGK[:7], [_WGK[7]], _WGK[6::-1]])

# c2g.f / c2t.f hold at most this many length scales per embedding dimension.
_MEPS = 1000


def _e(x: float) -> float:
    """Round through C's ``%e`` -- what TISEAN writes and hctsa reads back."""
    return float("%e" % x)


def _round_significant(y: np.ndarray, digits: int) -> np.ndarray:
    """Round to `digits` significant figures, as ``dlmwrite('precision', n)``."""
    fmt = "%%.%dg" % digits
    return np.array([float(fmt % v) for v in y], dtype=float)


def d2(
    y: ArrayLike,
    delay: int = 1,
    embed: int = 10,
    theiler: int = 0,
    howoften: int = 100,
    maxfound: int = 1000,
    epsmax: Optional[float] = None,
    epsmin: Optional[float] = None,
    write_precision: Optional[int] = 7,
) -> dict:
    """
    Estimate correlation sums, dimensions and entropies (TISEAN's ``d2``).

    Parameters
    ----------
    y : array-like
        Input time series.
    delay : int, optional
        Time delay of the embedding (``d2 -d``). Default is 1.
    embed : int, optional
        Maximum embedding dimension (``d2 -M1,<embed>``). Default is 10.
    theiler : int, optional
        Theiler window in samples (``d2 -t``). Default is 0.
    howoften : int, optional
        Number of length scales to scan (``d2 -#``). Default is 100.
    maxfound : int, optional
        Maximum number of pairs; 0 means all (``d2 -N``). Default is 1000.
    epsmax, epsmin : float, optional
        Upper/lower length scale (``d2 -R`` / ``d2 -r``). ``None`` (the default)
        reproduces TISEAN's own defaults: the data interval, and a thousandth
        of it.
    write_precision : int or None, optional
        Round the series to this many significant digits before running, which
        is what hctsa's ``BF_WriteTempFile`` does on its way through a text
        file. Pass ``None`` to use the series as given. Default is 7.

    Returns
    -------
    dict
        ``'c2'``, ``'d2'`` and ``'h2'``, each a list of ``embed`` arrays of
        shape ``(n_i, 2)``: length scale in the first column, and respectively
        the correlation sum, its local slope, and the correlation entropy in
        the second. These are exactly the contents of the ``.c2``, ``.d2`` and
        ``.h2`` files TISEAN would have written.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    if write_precision is not None:
        y = _round_significant(y, write_precision)

    found, norm, epsmax1, epsfactor = _d2_c.correlation_sums(
        y, int(delay), int(embed), int(theiler), int(maxfound), int(howoften),
        None if epsmax is None else float(epsmax),
        None if epsmin is None else float(epsmin),
    )
    lnfac = math.log(epsfactor)

    # d2.c walks its length scales by repeated division, so we do too: starting
    # from a different value (as the .d2 file does) gives a different last ulp.
    eps_c2 = np.empty(howoften)
    eps = epsmax1 * epsfactor
    for j in range(howoften):
        eps /= epsfactor
        eps_c2[j] = eps

    eps_d2 = np.empty(howoften - 1)
    eps = epsmax1
    for j in range(howoften - 1):
        eps /= epsfactor
        eps_d2[j] = eps

    c2_blocks: List[np.ndarray] = []
    d2_blocks: List[np.ndarray] = []
    h2_blocks: List[np.ndarray] = []

    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(embed):
            rows = [(_e(eps_c2[j]), _e(found[i, j] / norm[j]))
                    for j in range(howoften) if norm[j] > 0.0]
            c2_blocks.append(np.array(rows, dtype=float).reshape(-1, 2))

            rows = [(_e(eps_d2[j - 1]),
                     _e(math.log(found[i, j - 1] / found[i, j]
                                 / norm[j - 1] * norm[j]) / lnfac))
                    for j in range(1, howoften)
                    if found[i, j] > 0.0 and found[i, j - 1] > 0.0]
            d2_blocks.append(np.array(rows, dtype=float).reshape(-1, 2))

            if i == 0:
                rows = [(_e(eps_c2[j]), _e(-math.log(found[0, j] / norm[j])))
                        for j in range(howoften) if found[0, j] > 0.0]
            else:
                rows = [(_e(eps_c2[j]), _e(math.log(found[i - 1, j] / found[i, j])))
                        for j in range(howoften)
                        if found[i - 1, j] > 0.0 and found[i, j] > 0.0]
            h2_blocks.append(np.array(rows, dtype=float).reshape(-1, 2))

    return {"c2": c2_blocks, "d2": d2_blocks, "h2": h2_blocks}


def poincare(
    y: ArrayLike,
    dim: int = 2,
    delay: int = 1,
    comp: Optional[int] = None,
    direction: int = 0,
    where: Optional[float] = None,
    write_precision: Optional[int] = 7,
    as_written: bool = False,
) -> np.ndarray:
    """
    Make a Poincare section of a scalar time series (TISEAN's ``poincare``).

    The series is delay-embedded, and the section is taken where one component
    of the delay vector crosses a given level. Each crossing yields the
    remaining ``dim - 1`` coordinates of the delay vector, linearly interpolated
    to the crossing, and the time since the previous crossing.

    Parameters
    ----------
    y : array-like
        Input time series.
    dim : int, optional
        Embedding dimension (``poincare -m``). Default is 2.
    delay : int, optional
        Time delay of the embedding (``poincare -d``). Default is 1.
    comp : int, optional
        Which component of the delay vector to cut, 1-based (``poincare -q``);
        must not exceed ``dim``. ``None`` (the default) cuts the last one, as
        TISEAN does.
    direction : int, optional
        Direction of the cut: 0 crosses from below, 1 from above
        (``poincare -C``). Default is 0.
    where : float, optional
        Level to cut at (``poincare -a``). ``None`` (the default) reproduces
        TISEAN's own default, the mean of the series. Must lie within the range
        of the data.
    write_precision : int or None, optional
        Round the series to this many significant digits before running, which
        is what hctsa's ``BF_WriteTempFile`` does on its way through a text
        file. Pass ``None`` to use the series as given. Default is 7.
    as_written : bool, optional
        Round each value through ``%e``, the format ``poincare.c`` prints with.
        Set this when reproducing a pipeline that read the ``.poin`` file back,
        as hctsa does. Default is False, i.e. keep full precision.

    Returns
    -------
    ndarray
        One row per crossing, of shape ``(n_cuts, dim)``: the ``dim - 1``
        coordinates at the crossing, then the time since the previous crossing.
        These are the lines TISEAN would have written to its ``.poin`` file, at
        full double precision unless ``as_written`` asks for the printed values.
        The first crossing only starts the clock, so it has no row.

    Raises
    ------
    ValueError
        If the series is constant, or if ``where`` lies outside the data.
        ``poincare.c`` exits in both cases.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    if write_precision is not None:
        y = _round_significant(y, write_precision)

    # poincare.c defaults -q to the dimension, i.e. cuts the last component.
    if comp is None:
        comp = int(dim)

    cuts, _ = _poincare_c.section(
        y, int(dim), int(delay), int(comp), int(direction),
        None if where is None else float(where),
    )
    if as_written and cuts.size:
        cuts = np.array([_e(v) for v in cuts.ravel()]).reshape(cuts.shape)
    return cuts


def _tisean_argsort(x: np.ndarray) -> np.ndarray:
    """
    Ascending sort order with TISEAN's tie-breaking.

    ``indexx`` (source_f/rank.f) builds a per-bucket linked list and inserts a
    new point *before* the first element it does not exceed, so equal keys come
    out in reverse order of appearance -- the opposite of a stable sort. Ties do
    occur: ``c2g``'s off-by-one can duplicate a length scale, and which of the
    two correlation sums then leads changes the interpolation.
    """
    n = x.size
    return (n - 1) - np.argsort(x[::-1], kind="stable")


def _gk15(f, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised 15-point Gauss-Kronrod estimate of int_a^b f, per interval."""
    centr = 0.5 * (a + b)
    hlgth = 0.5 * (b - a)
    u = centr[:, None] + hlgth[:, None] * _GK_X[None, :]
    return (f(u) * _GK_W[None, :]).sum(axis=1) * hlgth


def c2t(c2: List[np.ndarray]) -> List[np.ndarray]:
    """
    Takens' maximum likelihood estimator from correlation sums (``c2t``).

    The integral is computed from the discrete values of ``C(r)`` by assuming an
    exact power law between the available points.

    Parameters
    ----------
    c2 : list of ndarray
        Correlation sums, one ``(n_i, 2)`` array per embedding dimension --
        i.e. the ``'c2'`` entry of :func:`d2`.

    Returns
    -------
    list of ndarray
        One ``(n_i - 1, 2)`` array per embedding dimension: the upper length
        scale, and Takens' estimator at that scale.
    """
    out: List[np.ndarray] = []
    for block in c2:
        e_vals, c_vals = [], []
        for ee, cc in block:
            if cc <= 0.0:  # c2t.f stops the block at the first non-positive C
                break
            e_vals.append(math.log(ee))
            c_vals.append(math.log(cc))

        me = len(e_vals)
        e_vals = np.asarray(e_vals, dtype=float)
        c_vals = np.asarray(c_vals, dtype=float)
        order = _tisean_argsort(e_vals)
        e = e_vals[order]
        c = c_vals[order]

        cint = 0.0
        rows = []
        for i in range(1, me):
            de = e[i] - e[i - 1]
            b = (e[i] * c[i - 1] - e[i - 1] * c[i]) / de
            a = (c[i] - c[i - 1]) / de
            if a != 0:
                cint += (math.exp(b) / a) * (math.exp(a * e[i]) - math.exp(a * e[i - 1]))
            else:
                cint += math.exp(b) * de
            rows.append((math.exp(e[i]), math.exp(c[i]) / cint))
        out.append(np.array(rows, dtype=float).reshape(-1, 2))
    return out


def c2g(c2: List[np.ndarray]) -> List[np.ndarray]:
    """
    Gaussian kernel correlation integral from correlation sums (``c2g``).

    Parameters
    ----------
    c2 : list of ndarray
        Correlation sums, one ``(n_i, 2)`` array per embedding dimension --
        i.e. the ``'c2'`` entry of :func:`d2`.

    Returns
    -------
    list of ndarray
        One ``(n_i, 3)`` array per embedding dimension: the kernel bandwidth
        ``r``, the Gaussian kernel correlation integral, and its logarithmic
        derivative with respect to ``r``.

    Notes
    -----
    ``c2g.f`` increments its point counter *before* testing whether the
    correlation sum is positive, so a block that is cut short by a zero keeps
    one trailing slot holding whatever the previous block left in the (static,
    zero-initialised) arrays. That off-by-one is reproduced here, since hctsa's
    feature values were computed with it.
    """
    out: List[np.ndarray] = []
    # Stand-ins for c2g.f's static REAL arrays, which persist between blocks.
    e_buf = np.zeros(_MEPS)
    c_buf = np.zeros(_MEPS)

    for block in c2:
        me = 0
        for ee, cc in block:
            me += 1
            if cc <= 0.0:
                break
            e_buf[me - 1] = math.log(ee)
            c_buf[me - 1] = math.log(cc)

        if me == 0:
            out.append(np.empty((0, 3)))
            continue

        # c2g.f sorts in place, so the leftover slot above picks up a value from
        # the *sorted* previous block, not from it in file order.
        order = _tisean_argsort(e_buf[:me])
        e_buf[:me] = e_buf[:me][order]
        c_buf[:me] = c_buf[:me][order]
        e = e_buf[:me].copy()
        c = c_buf[:me].copy()

        # Piecewise power-law interpolation between successive points: on
        # [e_k, e_k+1] the correlation sum is f * exp(d * u).
        de = e[1:] - e[:-1]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            f = np.exp((e[1:] * c[:-1] - e[:-1] * c[1:]) / de)
            d = (c[1:] - c[:-1]) / de
        # c2g.f only integrates over intervals of non-zero width.
        keep = e[1:] != e[:-1]
        a, b = e[:-1][keep], e[1:][keep]
        f, d = f[keep], d[keep]

        e_last, rows = e[me - 1], []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            for j in range(me):
                h = math.exp(e[j])
                g = _gk15(lambda u: f[:, None] * np.exp((2 + d[:, None]) * u
                                                        - np.exp(2 * u) / (2 * h ** 2)),
                          a, b).sum()
                gd = _gk15(lambda u: f[:, None] * np.exp((4 + d[:, None]) * u
                                                         - np.exp(2 * u) / (2 * h ** 2)),
                           a, b).sum()
                tail = math.exp(-math.exp(2 * e_last) / (2 * h ** 2))
                cgauss = g / h ** 2 + tail
                cgd = gd / h ** 4 + (2 + math.exp(2 * e_last) / h ** 2) * tail
                rows.append((h, cgauss, -2 + cgd / cgauss))
        out.append(np.array(rows, dtype=float).reshape(-1, 3))
    return out
