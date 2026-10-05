"""
Python front-end to the TISEAN 3.0.1 routines used by pyhctsa.

hctsa drives TISEAN through the shell: it writes the time series to a temporary
file, runs ``d2``, and then pipes the resulting ``.c2`` file through ``c2g`` and
``c2t``.  Here the same three steps happen in-process:

* :func:`d2` calls the C kernel in ``TS_d2.c`` (a re-entrant transcription of
  TISEAN's ``d2.c``) and assembles the ``.c2``/``.d2``/``.h2`` tables from the
  raw pair counts, using the expressions ``d2.c`` prints.
* :func:`false_nearest` calls the C kernel in ``TS_false_nearest.c`` (a re-entrant
  transcription of TISEAN's ``false_nearest.c``), and :func:`fnn_embedding_dimension`
  turns its output into hctsa's choice of embedding dimension.
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
from . import false_nearest as _fnn_c
from . import poincare as _poincare_c

__all__ = ["d2", "c2g", "c2t", "poincare", "false_nearest", "fnn_first_under",
           "fnn_embedding_dimension"]


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


_FNN_FAILURES = {
    1: "the data are constant",
    2: "the maximal embedding dimension times the delay is too large for the data length",
    3: "not enough points found (no neighbour within the search radius)",
}


def false_nearest(
    y: ArrayLike,
    delay: int = 1,
    minemb: int = 1,
    maxemb: int = 10,
    theiler: int = 0,
    escape_factor: float = 2.0,
    write_precision: Optional[int] = 7,
) -> dict:
    """
    Fraction of false nearest neighbors as a function of embedding dimension
    (TISEAN's ``false_nearest``).

    For each embedding dimension ``m`` the nearest neighbor of every point is
    found in ``m`` dimensions; it is *false* if the extra coordinate of the
    ``m + 1``-dimensional embedding separates the two points by more than
    ``escape_factor`` times their distance. Neighbors closer in time than
    ``theiler`` samples are not considered, and only points whose neighbor is
    within a fraction ``1/escape_factor`` of the standard deviation of the
    (range-scaled) series count.

    Parameters
    ----------
    y : array-like
        Input time series.
    delay : int, optional
        Time delay (``false_nearest -d``): the lag in samples between successive
        embedding coordinates. Default is 1. NOTE: TISEAN 3.0.1's ``false_nearest``
        ignores it for a scalar series (the coordinates are consecutive samples, and
        the delay only limits the number of embedded points used); hctsa's copy and
        this port use it as the lag.
    minemb, maxemb : int, optional
        Smallest and largest embedding dimension tested (``-m``, ``-M1,<maxemb>``).
        Defaults are 1 and 10.
    theiler : int, optional
        Theiler window in samples (``-t``). Default is 0.
    escape_factor : float, optional
        Ratio of the extra-coordinate separation to the distance above which a
        neighbor is false (``-f``). TISEAN's own default, 2.0, is the default here;
        hctsa's embedding-dimension choice uses 5.
    write_precision : int or None, optional
        Round the series to this many significant digits first, as hctsa's
        ``BF_WriteTempFile`` does on its way through a text file. Pass ``None`` to use the
        series as given. Default is 7.

    Returns
    -------
    dict
        ``'dim'``, ``'pfnn'``, ``'aveps'`` and ``'sdeps'``: arrays with one entry per
        embedding dimension completed, which are the four columns TISEAN prints:
        the dimension, the fraction of false nearest neighbors, and the mean and the
        standard deviation of the distance to the nearest neighbor (in the units of
        ``y``). If no neighbor can be found at some dimension, TISEAN stops there and
        the arrays hold the dimensions before it.

    Raises
    ------
    ValueError
        If not even the first dimension can be computed (constant series, delay times
        dimension too large for the series, no neighbors found), where TISEAN exits
        without output.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    if write_precision is not None:
        y = _round_significant(y, write_precision)
    rows, status = _fnn_c.run(y, int(delay), int(minemb), int(maxemb), int(theiler),
                              float(escape_factor))
    if rows.shape[0] == 0:
        raise ValueError("false_nearest: " + _FNN_FAILURES.get(status, "failed"))
    return {"dim": rows[:, 0].astype(int), "pfnn": rows[:, 1],
            "aveps": rows[:, 2], "sdeps": rows[:, 3]}


def fnn_first_under(dim: ArrayLike, pfnn: ArrayLike, threshold: float) -> int:
    """
    First embedding dimension at which the fraction of false nearest neighbors
    drops below ``threshold`` (``firstunderf`` in hctsa's ``NL_FNN``); one more
    than the largest dimension if it never does.
    """
    dim = np.asarray(dim)
    below = np.flatnonzero(np.asarray(pfnn) < threshold)
    return int(dim[below[0]]) if below.size else int(dim[-1]) + 1


def fnn_embedding_dimension(
    y: ArrayLike,
    delay: int = 1,
    theiler: int = 0,
    threshold: float = 0.4,
    maxemb: int = 10,
    escape_factor: float = 5.0,
    write_precision: Optional[int] = 7,
) -> float:
    """
    Embedding dimension chosen by false nearest neighbors, as hctsa's
    ``BF_Embed(y, tau, {'fnn', threshold})`` does.

    This is the first dimension (1 to ``maxemb``) at which the fraction of false
    nearest neighbors, :func:`false_nearest` with the given ``theiler`` window and
    ``escape_factor``, falls below ``threshold``; ``maxemb + 1`` if it never does.
    hctsa's defaults are reproduced: threshold 0.4, 10 dimensions, escape factor 5
    (NL_nlpe uses a threshold of 0.05). NaN where TISEAN produces no output.

    Parameters
    ----------
    y : array-like
        Input time series.
    delay : int, optional
        Time delay of the embedding, passed to :func:`false_nearest`. Default is 1.
    theiler : int, optional
        Theiler window in samples; hctsa uses one autocorrelation time. Default is 0.
    threshold : float, optional
        Fraction of false nearest neighbors to get under. Default is 0.4.
    maxemb : int, optional
        Largest dimension tested. Default is 10.
    escape_factor : float, optional
        See :func:`false_nearest`. Default is 5.
    write_precision : int or None, optional
        See :func:`false_nearest`. Default is 7.

    Returns
    -------
    float
        The embedding dimension (an integer value), or NaN.
    """
    try:
        t = false_nearest(y, delay, 1, maxemb, theiler, escape_factor, write_precision)
    except ValueError:
        return float("nan")
    return float(fnn_first_under(t["dim"], t["pfnn"], threshold))


def _gk15(f, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised 15-point Gauss-Kronrod estimate of int_a^b f, per interval."""
    centr = 0.5 * (a + b)
    hlgth = 0.5 * (b - a)
    u = centr[:, None] + hlgth[:, None] * _GK_X[None, :]
    return (f(u) * _GK_W[None, :]).sum(axis=1) * hlgth


def _read_c2_block(block: np.ndarray) -> tuple:
    """Log length scales and log correlation sums of one block of a ``.c2`` table, as
    ``c2g.f`` and ``c2t.f`` read them: the block ends at the first non-positive correlation
    sum, and the points are then sorted by increasing length scale (an insertion sort, so
    equal length scales keep their order)."""
    block = np.asarray(block, dtype=np.float64).reshape(-1, 2)
    nonpos = np.flatnonzero(block[:, 1] <= 0.0)
    if nonpos.size:
        block = block[:nonpos[0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        e, c = np.log(block[:, 0]), np.log(block[:, 1])
    order = np.argsort(e, kind="stable")
    return e[order], c[order]


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

    Notes
    -----
    As in hctsa's modified ``c2t.f``, all quantities are double precision (the original
    keeps the logarithms and sums in single precision, and relies on a sort that
    reverses equal keys).
    """
    out: List[np.ndarray] = []
    for block in c2:
        e, c = _read_c2_block(block)
        me = e.size
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            de = e[1:] - e[:-1]
            b_all = (e[1:] * c[:-1] - e[:-1] * c[1:]) / de
            a_all = (c[1:] - c[:-1]) / de

            cint = 0.0
            rows = []
            for i in range(1, me):
                a, b = a_all[i - 1], b_all[i - 1]
                if a != 0:
                    cint = cint + (np.exp(b) / a) * (np.exp(a * e[i]) - np.exp(a * e[i - 1]))
                else:
                    cint = cint + np.exp(b) * de[i - 1]
                rows.append((np.exp(e[i]), np.exp(c[i]) / cint))
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
        One ``(m_i, 3)`` array per embedding dimension (``m_i`` the number of
        points with a positive correlation sum): the kernel bandwidth ``r``, the
        Gaussian kernel correlation integral, and its logarithmic derivative with
        respect to ``r``.

    Notes
    -----
    As in hctsa's modified ``c2g.f``, all quantities are double precision (in the original
    the single-precision logarithms overflowed in the interpolation prefactor
    ``exp((e_{k+1} c_k - e_k c_{k+1}) / (e_{k+1} - e_k))`` for series with steep local slopes,
    which gave Inf/NaN output), and the points of an embedding dimension are only those with
    a positive correlation sum: the original counted the point that ended the list, so that
    a stale point of the previous embedding dimension entered the integral.
    """
    out: List[np.ndarray] = []
    for block in c2:
        e, c = _read_c2_block(block)
        me = e.size
        if me == 0:
            out.append(np.empty((0, 3)))
            continue

        # Piecewise power-law interpolation between successive points: on
        # [e_k, e_k+1] the correlation sum is f * exp(d * u).
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            de = e[1:] - e[:-1]
            f = np.exp((e[1:] * c[:-1] - e[:-1] * c[1:]) / de)
            d = (c[1:] - c[:-1]) / de
        # c2g.f only integrates over intervals of non-zero width.
        keep = e[1:] != e[:-1]
        a, b = e[:-1][keep], e[1:][keep]
        f, d = f[keep], d[keep]

        e_last, rows = float(e[me - 1]), []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            for j in range(me):
                h = math.exp(float(e[j]))
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
