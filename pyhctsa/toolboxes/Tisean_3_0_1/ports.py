# SPDX-License-Identifier: GPL-2.0-or-later
"""
Python ports of routines from the TISEAN 3.0.1 nonlinear time-series package.

TISEAN is Copyright (c) 1998-2007 Rainer Hegger, Holger Kantz and Thomas Schreiber, and is
distributed under the GNU General Public License, version 2 or (at your option) any later version
(see ``COPYING`` in this directory). The copyright to each TISEAN program is held by the person
named in its source code. This module is a derived work: each routine below is a port to Python
(NumPy / numba) of the TISEAN routine named in its section, restructured to run in process
(no files, no exit()), and in places modified as noted for use in pyhctsa (and, before it, in hctsa
[B. D. Fulcher, GPL-3.0-or-later]). It is distributed with pyhctsa under GPL-3.0-or-later, which is
permitted by the "or later" clause of the TISEAN licence.

Routine, TISEAN source file, and author named in that file:

- ``tisean_boxcount``: ``source_c/boxcount.c`` (Rainer Hegger).
- ``tisean_lyap_r``: ``source_c/lyap_r.c`` (Rainer Hegger), as modified in hctsa (repeated points
  are not neighbors; equally near neighbors are averaged; merging pairs are left out).
- ``c1_curves`` and helpers: ``source_f/c1.f`` (Thomas Schreiber, 1999) and ``source_f/d1.f`` (the
  digamma table ``_C1_PSI`` is d1.f's), as modified in hctsa (a fixed order of reference points;
  a mass at which a reference point is repeated is skipped).
- ``c2d_slopes``: ``source_f/c2d.f`` (Thomas Schreiber, 1998).
- ``tisean_rnd_deltas``: ``source_c/routines/rand.c`` (Rainer Hegger).
- ``gram_schmidt``, ``solve_in_place``, ``lyap_neighbors``, ``lyap_spec_sums``:
  ``source_c/lyap_spec.c`` (Rainer Hegger) with ``source_c/routines/solvele.c`` and
  ``source_c/routines/invert_matrix.c`` (Rainer Hegger).
- ``nrlazy`` and ``nrlazy_numba``: ``source_c/nrlazy.c`` (Rainer Hegger).
- ``nstat_z_std`` and ``nstat_z_error``: ``source_c/nstat_z.c`` (Rainer Hegger).

Modified for pyhctsa: ported from C / Fortran to Python; single-precision arithmetic is emulated
where the original is single precision (``c1``, ``c2d``); all behaviour is otherwise as described
in each docstring.

Reference: R. Hegger, H. Kantz and T. Schreiber, "Practical implementation of nonlinear time series
methods: The TISEAN package", Chaos 9, 413 (1999). https://www.pks.mpg.de/tisean
"""
import math
from typing import Union

import numba
import numpy as np
from numba import njit


# ------------------------------------------------------------------------------
# Data-dependent failures
# ------------------------------------------------------------------------------

class TiseanDataError(ValueError):
    """A data-dependent failure of a TISEAN routine (d2, boxcount, lyap_r, ...), for which hctsa returns NaN."""


# ------------------------------------------------------------------------------
# boxcount (source_c/boxcount.c) and lyap_r (source_c/lyap_r.c)
# ------------------------------------------------------------------------------

def _tisean_boxcount(y: np.ndarray, delay: int, maxembed: int, epscount: int, q: float = 0.0) -> tuple:
    # TISEAN's ``boxcount -M1,<maxembed> -d<delay> -Q<q> -#<epscount>`` (source_c/boxcount.c), in
    # process: the Renyi entropy of order q, H_q = log(sum_i p_i^q) / (1 - q) (Shannon, -sum p log p,
    # for q = 1), of the partition of the delay embedding into cubes of side eps -- p_i being the
    # fraction of embedded points in cube i -- for embedding dimensions 1..maxembed. For q = 0
    # that is ln N(eps), the log of the number of occupied cubes.
    # Returns (eps, H, dH): eps of shape (epscount,), and H and dH, of shape (epscount, maxembed),
    # the entropy and its increment with the embedding dimension (H itself for dimension 1).
    y = np.asarray(y, dtype=float).ravel()
    ymin = np.min(y)
    maxinterval = np.max(y) - ymin
    if maxinterval == 0:
        raise TiseanDataError('boxcount: the data are constant')
    epsmin, epsmax = 1e-3, 1.0  # relative to the data interval
    x = (y - ymin) / maxinterval
    x[x >= 1.0] -= epsmin / 2.0
    length = y.size - (maxembed - 1) * delay
    if length < 1 or epscount < 2:
        raise TiseanDataError('boxcount: time series too short for this embedding')
    epsfaktor = (epsmax / epsmin) ** (1.0 / (epscount - 1))

    eps = np.empty(epscount)
    h = np.empty((epscount, maxembed))
    heps = epsmax * epsfaktor
    epsi_old = 0
    for k in range(epscount):
        while True:  # the number of boxes per axis is an integer that must increase
            heps /= epsfaktor
            epsi = int(1.0 / heps)
            if epsi > epsi_old:
                break
        epsi_old = epsi
        eps[k] = heps * maxinterval
        labels = np.zeros(length, dtype=np.int64)
        for d in range(maxembed):  # nested partition: cells are distinguished by coordinates 1..d+1
            box = (x[d * delay:d * delay + length] * epsi).astype(np.int64)
            _, labels, counts = np.unique(labels * epsi + box, return_inverse=True, return_counts=True)
            p = counts / length
            h[k, d] = -np.sum(p * np.log(p)) if q == 1.0 else np.log(np.sum(p ** q)) / (1.0 - q)
    dh = np.diff(h, axis=1, prepend=0.0)
    return eps, h, dh


@njit(cache=True)
def _lyap_r_neighbors(emb, n_refs, n_cand, mindist, tol2):
    # For each reference vector (the first ``n_refs`` rows of ``emb``) the candidate neighbors
    # (the first ``n_cand`` rows) outside its Theiler window and not repeats of it, that are within
    # 1e-12 of the nearest one's distance (only a neighbor closer than 1 is accepted). Returns the
    # pairs (reference, neighbor) in reference order and the number of neighbors of each
    # reference, or None if a reference has no neighbor.
    dim = emb.shape[1]
    d2 = np.empty(n_cand)
    cap = 4 * n_refs
    pr = np.empty(cap, dtype=np.int64)
    pn = np.empty(cap, dtype=np.int64)
    n_near = np.zeros(n_refs)
    tot = 0
    for r in range(n_refs):
        dmin = np.inf
        for j in range(n_cand):
            acc = 0.0
            for k in range(dim):
                dd = emb[r, k] - emb[j, k]
                acc += dd * dd
            if abs(r - j) <= mindist or acc <= tol2:
                acc = np.inf
            d2[j] = acc
            if acc < 1.0 and acc < dmin:
                dmin = acc
        if np.isinf(dmin):
            return None
        root = np.sqrt(dmin) + 1e-12
        lim = root * root
        for j in range(n_cand):
            if d2[j] <= lim:
                if tot == cap:
                    cap *= 2
                    pr2 = np.empty(cap, dtype=np.int64)
                    pn2 = np.empty(cap, dtype=np.int64)
                    pr2[:tot] = pr[:tot]
                    pn2[:tot] = pn[:tot]
                    pr = pr2
                    pn = pn2
                pr[tot] = r
                pn[tot] = j
                tot += 1
                n_near[r] += 1.0
    return pr[:tot].copy(), pn[:tot].copy(), n_near


@njit(cache=True)
def _lyap_r_divergence(emb, pair_ref, pair_nb, weight, steps, tol2, chunk):
    # The weighted sum over the pairs that never merge (distance within tol of zero at some step
    # after the first) of the log squared distance at each step, and which pairs those are. The
    # pairs are summed in order within chunks of ``chunk`` pairs, and the chunk sums added in
    # order, so that the result does not depend on how the work is done.
    dim = emb.shape[1]
    lyap = np.zeros(steps + 1)
    csum = np.zeros(steps + 1)
    kept = np.zeros(pair_ref.size, dtype=np.bool_)
    for c in range(0, pair_ref.size, chunk):
        csum[:] = 0.0
        for p in range(c, min(c + chunk, pair_ref.size)):
            a = pair_ref[p]
            b = pair_nb[p]
            merged = False
            for t in range(1, steps + 1):
                acc = 0.0
                for k in range(dim):
                    dd = emb[a + t, k] - emb[b + t, k]
                    acc += dd * dd
                if acc <= tol2:
                    merged = True
                    break
            if merged:
                continue
            kept[p] = True
            w = weight[p]
            for t in range(steps + 1):
                acc = 0.0
                for k in range(dim):
                    dd = emb[a + t, k] - emb[b + t, k]
                    acc += dd * dd
                csum[t] += w * np.log(acc)
        for t in range(steps + 1):
            lyap[t] += csum[t]
    return lyap, kept


def _tisean_lyap_r(y: np.ndarray, delay: int, dim: int, mindist: int, steps: int) -> Union[tuple, None]:
    # TISEAN's ``lyap_r -d<delay> -m<dim> -t<mindist> -s<steps>`` (Rosenstein et al.; source_c/lyap_r.c
    # as modified in hctsa), in process. For every reference vector, the nearest neighbors (outside
    # the Theiler window ``mindist``) are found: all the vectors whose distance to it is within
    # 1e-12 (of the data interval) of the nearest distance below 1, vectors closer than 1e-12 to it
    # being repeats of it and not neighbors. Each pair is followed for ``steps`` steps; a pair
    # that merges (comes within 1e-12) is left out at every step, and each reference contributes
    # the mean of the log distance over its nearest neighbors, weighted by the fraction of them
    # kept (the expected curve if one of them were chosen at random, so the result does not depend
    # on the order of a neighbor search). Returns the (time, mean ln distance) rows as in the .ros
    # file, or None where TISEAN gives no output (a reference point has no neighbor within the
    # search radius: exit status 54).
    y = np.asarray(y, dtype=float).ravel()
    length = y.size
    ymin = np.min(y)
    interval = np.max(y) - ymin
    if interval == 0:
        raise TiseanDataError('lyap_r: the data are constant')
    x = (y - ymin) / interval  # rescaled to [0, 1]
    delay_span = delay * (dim - 1)
    maxlength = length - delay_span - steps - 1 - mindist
    if maxlength < 0:
        raise TiseanDataError('lyap_r: time series too short')
    n_cand = length - delay_span - steps  # reference vectors that can be neighbors
    emb = x[np.arange(length - delay_span)[:, None] + delay * np.arange(dim)[None, :]]
    refs = np.arange(maxlength + 1)
    tol2 = 1e-12 ** 2  # (squared) distances closer than 1e-12 are equal; DIST_TOL in lyap_r.c

    # Nearest neighbors, then the divergence of each neighbor pair over time
    nbrs = _lyap_r_neighbors(emb, refs.size, n_cand, mindist, tol2)
    if nbrs is None:
        return None  # some reference point never gets a neighbor
    pair_ref, pair_nb, n_near = nbrs
    weight = 1.0 / n_near[pair_ref]
    chunk = max(1, int(2e6 // ((steps + 1) * dim)))
    lyap, kept = _lyap_r_divergence(emb, pair_ref, pair_nb, weight, steps, tol2, chunk)
    found = 0.0
    for c in range(0, pair_ref.size, chunk):
        found += np.sum(weight[c:c + chunk][kept[c:c + chunk]])
    if found == 0:
        return np.empty((0, 2))
    return np.column_stack([np.arange(steps + 1), lyap / found / 2.0])


# ------------------------------------------------------------------------------
# c1 (source_f/c1.f, d1.f) and c2d (source_f/c2d.f): information dimension, fixed mass
# ------------------------------------------------------------------------------

F32 = np.float32


# digamma at the integers 0..20 (as tabulated in TISEAN's d1.f)
_C1_PSI = np.array([0., -0.57721566490, 0.42278433509, 0.92278433509, 1.25611766843,
                    1.50611766843, 1.70611766843, 1.87278433509, 2.01564147795, 2.14064147795,
                    2.25175258906, 2.35175258906, 2.44266167997, 2.52599501330, 2.60291809023,
                    2.67434666166, 2.74101332832, 2.80351332832, 2.86233685773, 2.91789241329,
                    2.97052399224], dtype=np.float32)


def _expf(x) -> np.float32:
    with np.errstate(over='ignore', under='ignore'):
        return F32(np.exp(np.float64(x)))


def _logf(x) -> np.float32:
    with np.errstate(divide='ignore', invalid='ignore'):
        return F32(np.log(np.float64(x)))


@njit(cache=True)
def _c1_centers(nmax, m, delay, ju):
    """
    The order of the reference points in hctsa's d1.f: the embedded points ``(m-1) * delay + 1 ..
    nmax`` (``nvalid`` of them) taken as ``i -> (i * s mod nvalid)``, with ``s`` the golden-ratio
    stride (computed in single precision, as in the Fortran) raised until it is coprime to
    ``nvalid``. Every prefix is spread evenly over the series and the order is fixed (stock TISEAN
    used the first entries of a random permutation, so the estimate depended on the draw).
    """
    nvalid = nmax - (m - 1) * delay
    istr = int(np.float32(np.float32(nvalid) * np.float32(0.6180339887)))
    if istr < 1:
        istr = 1
    while True:
        a, b = istr, nvalid
        while b != 0:
            a, b = b, a % b
        if a == 1 or nvalid <= 1:
            break
        istr += 1
    acc = 0
    for i in range(nvalid):
        ju[i] = acc + 1 + (m - 1) * delay
        acc = (acc + istr) % nvalid


@njit(cache=True)
def _c1_kth_distance(ye, off, refs, delay, m, ncomp, nmin, k):
    """
    For each reference point, the k-th smallest max-norm distance (float32) between its
    m-dimensional delay vector and those of the points outside its (cyclic) Theiler window; NaN
    where there are fewer than k such points. (TISEAN finds these by growing a box search until
    k neighbors are inside it, then taking the k-th distance, which is this.)
    """
    lo = (m - 1) * delay + 1
    hi = ncomp + (m - 1) * delay
    out = np.empty(refs.size, dtype=np.float32)
    buf = np.empty(max(hi - lo + 1, 1), dtype=np.float32)
    for r in range(refs.size):
        n = refs[r]
        cnt = 0
        for q in range(lo, hi + 1):
            nmd = abs(q - n) % ncomp
            if nmd <= nmin or nmd >= ncomp - nmin:
                continue
            dis = np.float32(0.0)
            for i in range(m):
                dx = abs(ye[off + n - i * delay] - ye[off + q - i * delay])
                if dx > dis:
                    dis = dx
            buf[cnt] = dis
            cnt += 1
        if cnt < k:
            out[r] = np.nan
        else:
            out[r] = np.partition(buf[:cnt], k - 1)[k - 1]
    return out


@njit(cache=True)
def _c1_kth_distances(ye, refs, delay, m, nmin, ncomps, ks):
    """
    :func:`_c1_kth_distance` for several ``(ncomp, k)`` at once: ``ncomps`` (G,) are distinct
    numbers of candidate points and ``ks`` (G, K) the neighbor ranks asked for at each (0 pads).
    Returns an array of shape (G, K, refs.size). The distances from a reference point to all the
    candidates are computed once; the Theiler window around it, cyclic over the ``ncomp``
    candidates, is a single cyclic interval of ``2 * nmin + 1`` of them, and the ranks asked for
    at one ``ncomp`` are read off one partial sort.
    """
    lo = (m - 1) * delay + 1
    G = ncomps.size
    K = ks.shape[1]
    ncmax = 0
    for g in range(G):
        if ncomps[g] > ncmax:
            ncmax = ncomps[g]
    out = np.full((G, K, refs.size), np.nan, dtype=np.float32)
    dfull = np.empty(ncmax, dtype=np.float32)
    buf = np.empty(ncmax, dtype=np.float32)
    for r in range(refs.size):
        n = refs[r]
        for idx in range(ncmax):
            q = lo + idx
            dis = np.float32(0.0)
            for i in range(m):
                dx = abs(ye[n - i * delay] - ye[q - i * delay])
                if dx > dis:
                    dis = dx
            dfull[idx] = dis
        for g in range(G):
            nc = ncomps[g]
            if 2 * nmin + 1 >= nc:
                continue  # every candidate is in the Theiler window
            cidx = (n - lo) % nc
            a = cidx - nmin
            b = cidx + nmin
            cnt = 0
            if a >= 0 and b < nc:
                for idx in range(0, a):
                    buf[cnt] = dfull[idx]
                    cnt += 1
                for idx in range(b + 1, nc):
                    buf[cnt] = dfull[idx]
                    cnt += 1
            elif a < 0:
                for idx in range(b + 1, a + nc):
                    buf[cnt] = dfull[idx]
                    cnt += 1
            else:
                for idx in range(b - nc + 1, a):
                    buf[cnt] = dfull[idx]
                    cnt += 1
            kmx = 0
            for j in range(K):
                if ks[g, j] > kmx:
                    kmx = ks[g, j]
            if kmx < 1 or cnt < 1:
                continue
            mm = min(kmx, cnt)
            srt = np.sort(np.partition(buf[:cnt], mm - 1)[:mm])
            for j in range(K):
                k = ks[g, j]
                if k >= 1 and k <= cnt:
                    out[g, j, r] = srt[k - 1]
    return out


@njit(cache=True)
def _f32_cumsum_last(v):
    s = np.float32(0.0)
    for i in range(v.size):
        s = np.float32(s + v[i])
    return s


def _c1_curves(y: np.ndarray, delay: int, m_from: int, m_to: int, nmin: int, n_ref: int,
               kmax: int = 100, res: float = 2.0) -> list:
    """
    TISEAN's ``c1 -d<delay> -m<m_from> -M<m_to> -t<nmin> -n<n_ref>`` (as patched in hctsa): for
    each embedding dimension, the table of [mean log distance to the k-th neighbor -> exp, mass]
    pairs (as float32, as TISEAN writes them) for masses doubling every ``1/res`` octaves.
    Everything that is single precision in the Fortran is single precision here. A mass for which
    no point has a neighbor outside the Theiler window is skipped, the number of neighbors asked
    for never exceeds the number that exist, and at most as many reference points are used as there
    are embedded points. A reference point that coincides with its k-th neighbor (to within 1e-12
    of the range of the data) is a repeated point (its log distance is minus infinity), so a mass
    at which any reference point is one has no finite mean log distance and is skipped.
    """
    y32 = np.asarray(y, dtype=np.float32)
    nmax = y32.size
    ye = np.zeros(nmax + 1, dtype=np.float32)  # the data at 1..nmax, as in the Fortran
    ye[1:] = y32
    off = 0
    sd = F32(math.sqrt(float(F32(_f32_cumsum_last(((y32 - F32(_f32_cumsum_last(y32) / F32(nmax)))
                                                   ** 2).astype(np.float32)) / F32(nmax)))))
    ju = np.zeros(nmax + 1, np.int64)
    tol = F32(F32(1e-12) * F32(np.max(y32) - np.min(y32)))  # closer than this: a repeated point
    resl = F32(_logf(F32(2.0)) / F32(res))
    sqrt2 = F32(math.sqrt(2.0))
    curves = []
    rln = F32(0.0)
    for m in range(m_from, m_to + 1):
        rows = []
        pr = F32(0.0)
        nvalid = nmax - (m - 1) * delay
        if nvalid < 1:  # the delay vectors are longer than the series: nothing for this m
            curves.append(np.empty((0, 2), dtype=np.float32))
            continue
        pl = _logf(F32(F32(1.0) / F32(nvalid)))
        trips = int(F32(F32(0.0) - pl) / resl) + 1
        # the (ncomp, k) of every mass step, so that the k-th neighbor distances of all of them
        # come from one pass over the candidates of each reference point
        _c1_centers(nmax, m, delay, ju)
        refs = ju[:min(n_ref, nvalid)].copy()  # at most as many centers as embedded points
        wanted = {}
        pl_d = pl
        for _ in range(trips):
            n_eff_d = nvalid - 2 * nmin - 1
            if n_eff_d >= 1:
                k_d = min(int(F32(_expf(pl_d) * F32(n_eff_d))) + 1, n_eff_d)
                nc_d = nvalid
                if k_d > kmax:
                    nc_d = int(F32(F32(F32(n_eff_d) * F32(kmax)) / F32(k_d)) + F32(2 * nmin + 1))
                    k_d = kmax
                wanted.setdefault(nc_d, set()).add(k_d)
            pl_d = F32(pl_d + resl)
        nc_arr = np.array(sorted(wanted), dtype=np.int64)
        k_cols = max((len(v) for v in wanted.values()), default=1)
        k_arr = np.zeros((nc_arr.size, k_cols), dtype=np.int64)
        for g, nc_g in enumerate(nc_arr):
            for j, k_g in enumerate(sorted(wanted[int(nc_g)])):
                k_arr[g, j] = k_g
        e_all = _c1_kth_distances(ye, refs, delay, m, nmin, nc_arr, k_arr) if nc_arr.size else None
        # (a Fortran DO loop over a real: the trip count int((0 - pl) / resl) + 1 is fixed first, the
        # variable then accumulates; the last mass can be 1 or a hair above, for instance when
        # nvalid is a power of two)
        for _ in range(trips):
            ncomp = nvalid
            n_eff = ncomp - 2 * nmin - 1  # neighbors a point can have outside its Theiler window
            if n_eff < 1:  # every point lies in every other's Theiler window: no mass to compute
                pl = F32(pl + resl)
                continue
            kpr = int(F32(_expf(pr) * F32(n_eff))) + 1
            k = min(int(F32(_expf(pl) * F32(n_eff))) + 1, n_eff)
            if k > kmax:  # fixed mass: use fewer points rather than more neighbors
                ncomp = int(F32(F32(F32(n_eff) * F32(kmax)) / F32(k)) + F32(2 * nmin + 1))
                k = kmax
            psi = _C1_PSI[k] if k <= 20 else F32(_logf(F32(k)) - F32(F32(1.0) / F32(2.0 * k)))
            pln = F32(psi - _logf(F32(ncomp - 2 * nmin - 1)))
            if k != kpr:
                g = int(np.searchsorted(nc_arr, ncomp))
                j = int(np.flatnonzero(k_arr[g] == k)[0])
                e = e_all[g, j]
                # TISEAN sweeps the reference points with a neighborhood size that grows by
                # sqrt(2) until each has k neighbors; the sum of logs is in that order
                eps = F32(_expf(F32(pln / F32(m))) * sd)
                eps_seq = [eps]
                while eps <= e.max() and len(eps_seq) < 500:
                    eps = F32(eps * sqrt2)
                    eps_seq.append(eps)
                sweep = np.searchsorted(np.array(eps_seq, dtype=np.float32), e, side='right')
                if int(sweep.max()) + 1 > 201:  # the search is abandoned after 200 sweeps
                    pl = F32(pl + resl)
                    continue
                order = np.argsort(sweep, kind='stable')
                # a center whose k-th neighbor is within tol of it is a repeated point: its log
                # distance is minus infinity, so the mean log distance at this mass is not
                # finite, and the mass is skipped
                if np.any(e <= tol):
                    pl = F32(pl + resl)
                    continue
                log_e = np.log(e[order].astype(np.float64)).astype(np.float32)
                rln = F32(_f32_cumsum_last(log_e) / F32(refs.size))  # mean over the centers used
            if pln != pr:
                pr = pln
                rows.append((_expf(rln), _expf(pln)))
            pl = F32(pl + resl)
        curves.append(np.array(rows, dtype=np.float32).reshape(-1, 2))
    return curves


def _c2d_slopes(curves: list, iav: int = 2) -> list:
    """TISEAN's ``c2d -a<iav>``: local slopes of log mass against log distance over windows of
    ``2 * iav + 1`` points (float32), keeping the positive ones."""
    n = 2 * iav + 1
    out = []
    for rows in curves:
        e = np.array([_logf(F32(r[0])) for r in rows], dtype=np.float32)
        c = np.array([_logf(F32(r[1])) for r in rows], dtype=np.float32)
        block = []
        for j in range(iav, len(rows) - iav):
            x, v = e[j - iav:j + iav + 1], c[j - iav:j + iav + 1]
            sx = F32(0.0)
            for xi in x:
                sx = F32(sx + xi)
            sa, a = F32(0.0), F32(0.0)
            for i in range(n):
                dev = F32(x[i] - F32(sx / F32(n)))
                sa = F32(sa + F32(dev * dev))
                a = F32(a + F32(v[i] * dev))
            slope = F32(a / sa)
            if slope > 0:
                block.append((_expf(F32(F32(0.5) * F32(e[j + iav] + e[j - iav]))), slope))
        out.append(np.array([(float(a), float(b)) for a, b in block]).reshape(-1, 2))
    return out


# ------------------------------------------------------------------------------
# lyap_spec (source_c/lyap_spec.c): Lyapunov spectrum
# ------------------------------------------------------------------------------

@njit(cache=True)
def _tisean_rnd_deltas(n, seed):
    """TISEAN's ``rnd_init(seed)``, 10000 warm-up calls of ``rnd_long``, then ``n * n`` values
    ``rnd_long() / ULONG_MAX`` (the random starting vectors of ``lyap_spec``)."""
    factor = np.uint64(13) ** np.uint64(13)
    arr = np.empty(9689, dtype=np.uint64)
    arr[0] = np.uint64(seed)
    for i in range(1, 9689):
        arr[i] = factor * arr[i - 1] + np.uint64(1)
    t, t1, t2, t3 = 9688, 9688 - 157, 9688 - 314, 9688 - 471
    out = np.empty((n, n))
    for c in range(10000 + n * n):
        t = (t + 1) % 9689
        t1 = (t1 + 1) % 9689
        t2 = (t2 + 1) % 9689
        t3 = (t3 + 1) % 9689
        arr[t] = arr[t] ^ arr[t1] ^ arr[t2] ^ arr[t3]
        if c >= 10000:
            out[(c - 10000) // n, (c - 10000) % n] = np.float64(arr[t]) / 18446744073709551615.0
    return out


@njit(cache=True)
def _gram_schmidt(delta):
    """Orthonormalize the rows of ``delta`` in place (classical Gram-Schmidt, as TISEAN); returns
    their lengths before normalization."""
    n = delta.shape[0]
    dnew = np.zeros((n, n))
    stretch = np.zeros(n)
    for i in range(n):
        diff = np.zeros(n)
        for j in range(i):
            nrm = 0.0
            for k in range(n):
                nrm += delta[i, k] * dnew[j, k]
            for k in range(n):
                diff[k] -= nrm * dnew[j, k]
        nrm = 0.0
        for j in range(n):
            nrm += (delta[i, j] + diff[j]) ** 2
        nrm = np.sqrt(nrm)
        stretch[i] = nrm
        for j in range(n):
            dnew[i, j] = (delta[i, j] + diff[j]) / nrm
    delta[:, :] = dnew
    return stretch


@njit(cache=True)
def _solve_in_place(mat, vec):
    """TISEAN's ``solvele`` (Gaussian elimination with row pivoting, in place); False if singular."""
    n = vec.size
    for i in range(n - 1):
        mx = abs(mat[i, i])
        maxi = i
        for j in range(i + 1, n):
            if abs(mat[j, i]) > mx:
                mx = abs(mat[j, i])
                maxi = j
        if maxi != i:
            for c in range(n):
                tmp = mat[i, c]
                mat[i, c] = mat[maxi, c]
                mat[maxi, c] = tmp
            tmp = vec[i]
            vec[i] = vec[maxi]
            vec[maxi] = tmp
        pivot = mat[i, i]
        if pivot == 0.0:
            return False
        for j in range(i + 1, n):
            q = -mat[j, i] / pivot
            mat[j, i] = 0.0
            for k in range(i + 1, n):
                mat[j, k] += q * mat[i, k]
            vec[j] += q * vec[i]
    vec[n - 1] /= mat[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        for j in range(n - 1, i, -1):
            vec[i] -= mat[i, j] * vec[j]
        vec[i] /= mat[i, i]
    return True


_LYAP_BOX = 512  # cells per axis of the neighbor search grid, as in lyap_spec.c


@njit(cache=True)
def _lyap_neighbors(S, act, theiler, k_nn, epsmin, box, link, found, dist):
    """
    The neighbors of point ``act`` as TISEAN's ``lyap_spec`` finds them, in its order: a grid of cells
    (on the first and last coordinates) of side epsilon, grown by 1.2 until more than ``k_nn`` points
    lie within epsilon in every coordinate (the point itself, and the points inside the Theiler
    window dropped); the ``k_nn`` nearest of them, by the selection sort in ``sort()`` (so ties are
    resolved as there). Returns (number found, average distance of the last neighbor).
    """
    L, m = S.shape
    nl = L - 1  # points that have a successor
    ib = _LYAP_BOX - 1
    epsilon = epsmin / 1.2
    nfound = 0
    foundeps = 0.0
    while True:
        epsilon *= 1.2
        if epsilon > 1.0:
            epsilon = 1.0
        for x in range(_LYAP_BOX):
            for y in range(_LYAP_BOX):
                box[x, y] = -1
        for i in range(nl):
            x = int(S[i, 0] / epsilon) & ib
            y = int(S[i, m - 1] / epsilon) & ib
            link[i] = box[x, y]
            box[x, y] = i
        nfound = 0
        i0 = int(S[act, 0] / epsilon) & ib
        j0 = int(S[act, m - 1] / epsilon) & ib
        for i1 in range(i0 - 1, i0 + 2):
            for j1 in range(j0 - 1, j0 + 2):
                element = box[i1 & ib, j1 & ib]
                while element != -1:
                    inside = True
                    for c in range(m):
                        if abs(S[act, c] - S[element, c]) > epsilon:
                            inside = False
                            break
                    if inside:
                        found[nfound] = element
                        nfound += 1
                    element = link[element]
        if theiler > 0:  # drop neighbors within the Theiler window (the point itself is kept)
            kept = 0
            for q in range(nfound):
                if found[q] == act or abs(found[q] - act) > theiler:
                    found[kept] = found[q]
                    kept += 1
            nfound = kept
        if nfound > k_nn:
            # sort(): distances to all but the point itself, which is swapped to the end
            self_at = 0
            for q in range(nfound):
                if found[q] != act:
                    d = 0.0
                    for c in range(m):
                        dx = abs(S[act, c] - S[found[q], c])
                        if dx > d:
                            d = dx
                    dist[q] = d
                else:
                    self_at = q
            if self_at != nfound - 1:
                dist[self_at] = dist[nfound - 1]
                found[self_at] = found[nfound - 1]
            for i in range(k_nn):
                for j in range(i + 1, nfound - 1):
                    if dist[j] < dist[i]:
                        tmp = dist[i]
                        dist[i] = dist[j]
                        dist[j] = tmp
                        tmpi = found[i]
                        found[i] = found[j]
                        found[j] = tmpi
            return k_nn, dist[k_nn - 1]
        if epsilon >= 1.0:
            break
    return nfound, foundeps


@njit(cache=True)
def _lyap_spec_sums(S, k_nn, theiler, seed):
    """
    Sano and Sawada's method as in TISEAN's ``lyap_spec`` for the m columns of ``S`` (each
    rescaled to [0, 1]) as the state, mapped forward one sample: at each point a local affine
    map is fitted to the next values of its ``k_nn`` nearest neighbors (max norm, outside the
    Theiler window), and a set of m tangent vectors is carried along it and re-orthonormalized.
    Returns the status (0 ok, 50 too few neighbors, 1 singular fit), the sums of the log
    stretching rates and the number of steps.
    """
    L, m = S.shape
    delta = _tisean_rnd_deltas(m, seed)
    _gram_schmidt(delta)
    factor = np.zeros(m)
    nl = L - 1
    dyn = np.zeros((m, m))
    mat = np.zeros((m + 1, m + 1))
    hmat = np.zeros((m + 1, m + 1))
    imat = np.zeros((m + 1, m + 1))
    vec = np.zeros(m + 1)
    dist = np.empty(nl)
    found = np.empty(nl, dtype=np.int64)
    link = np.empty(nl, dtype=np.int64)
    box = np.empty((_LYAP_BOX, _LYAP_BOX), dtype=np.int64)
    epsmin = 1e-3
    aveps = 0.0
    count = 0
    for act in range(nl):
        count += 1
        nf, foundeps = _lyap_neighbors(S, act, theiler, k_nn, epsmin, box, link, found, dist)
        aveps += foundeps
        epsmin = aveps / count  # the neighborhood size to start the next search with
        if nf < k_nn:
            return 50, factor, count
        # normal equations of the local affine fit
        mat[:, :] = 0.0
        for a in range(nf):
            p = found[a]
            mat[0, 0] += 1.0
            for j in range(m):
                mat[0, j + 1] += S[p, j]
            for j in range(m):
                for k in range(j, m):
                    mat[j + 1, k + 1] += S[p, k] * S[p, j]
        for i in range(m + 1):
            for j in range(i, m + 1):
                mat[i, j] = mat[i, j] / nf
                mat[j, i] = mat[i, j]
        for i in range(m + 1):  # invert column by column
            for j in range(m + 1):
                vec[j] = 1.0 if i == j else 0.0
                hmat[j, :] = mat[j, :]
            if not _solve_in_place(hmat, vec):
                return 1, factor, count
            imat[:, i] = vec
        for d in range(m):
            vec[:] = 0.0
            for a in range(nf):
                p = found[a]
                hv = S[p + 1, d]
                vec[0] += hv
                for j in range(m):
                    vec[j + 1] += hv * S[p, j]
            vec /= nf
            for i in range(1, m + 1):
                dyn[d, i - 1] = 0.0
                for j in range(m + 1):
                    dyn[d, i - 1] += imat[i, j] * vec[j]
        dnew = np.zeros((m, m))  # carry the tangent vectors along the local map
        for i in range(m):
            for j in range(m):
                dnew[i, j] = dyn[j, 0] * delta[i, 0]
                for k in range(1, m):
                    dnew[i, j] += dyn[j, k] * delta[i, k]
        delta[:, :] = dnew
        stretch = _gram_schmidt(delta)
        for j in range(m):
            factor[j] += np.log(stretch[j])
    return 0, factor, count


# ------------------------------------------------------------------------------
# nrlazy (source_c/nrlazy.c): simple nonlinear noise reduction
# ------------------------------------------------------------------------------

_NRLAZY_BOX = 512  # side of the hash grid used by TISEAN's nrlazy to find neighbors


@numba.njit(cache=True)
def _nrlazy_numba(x, m, d, num_iter, eps):
    """Schreiber's simple nonlinear noise reduction, a port of TISEAN's ``nrlazy`` (one component).

    ``x`` is the series rescaled to [0, 1] and ``eps`` the neighborhood radius in these units
    (maximum norm). Each iteration corrects every embedding vector to the mean of its neighbors
    (all vectors within ``eps`` in every coordinate, including itself), found as in the C code by
    hashing the first and last coordinates into a ``_NRLAZY_BOX``-squared grid. Each sample
    becomes the average of its corrections from the (up to m) vectors that contain it.
    Returns the new series and, for the last iteration, the number of neighbors of each vector
    (1 for the first (m-1)*d samples, which start no vector).
    """
    n = len(x)
    back = (m - 1) * d
    ibox = _NRLAZY_BOX - 1
    epsinv = 1.0 / eps
    nmf = np.ones(n, dtype=np.int64)
    for _ in range(num_iter):
        box = -np.ones((_NRLAZY_BOX, _NRLAZY_BOX), dtype=np.int64)
        nxt = np.zeros(n, dtype=np.int64)
        for i in range(back, n):
            bx = int(x[i] / eps) & ibox
            by = int(x[i - back] / eps) & ibox
            nxt[i] = box[bx, by]
            box[bx, by] = i
        corr = np.zeros(n)
        nf = np.zeros(n, dtype=np.int64)
        nmf[:] = 1
        hcor = np.zeros(m)
        for k in range(back, n):
            for q in range(m):
                hcor[q] = 0.0
            i = int(x[k] * epsinv) & ibox
            j = int(x[k - back] * epsinv) & ibox
            nfound = 0
            for i1 in range(i - 1, i + 2):
                i2 = i1 & ibox
                for j1 in range(j - 1, j + 2):
                    element = box[i2, j1 & ibox]
                    while element != -1:
                        q = 0
                        while q < m:
                            if abs(x[k - q * d] - x[element - q * d]) > eps:
                                break
                            q += 1
                        if q == m:
                            nfound += 1
                            for q in range(m):
                                hcor[q] += x[element - q * d]
                        element = nxt[element]
            for q in range(m):
                corr[k - q * d] += hcor[q] / nfound
                nf[k - q * d] += 1
            nmf[k] = nfound
        for k in range(n):
            if nf[k] > 0:
                x[k] = corr[k] / nf[k]
    return x, nmf


def _nrlazy(y: np.ndarray, m: int, d: int, num_iter: int, neighborhood_std: float):
    """Run ``_nrlazy_numba`` as TISEAN's ``nrlazy -m1,m -d -i -v`` does on the series ``y``.

    Returns ``(y_denoised, num_neighbors)``, or ``None`` where TISEAN would exit with an error
    (a constant series).
    """
    y = np.asarray(y, dtype=float)
    lo = y.min()
    interval = y.max() - lo
    if not interval > 0:
        return None
    x = (y - lo) / interval
    dvar = np.sqrt(abs(np.mean(x ** 2) - np.mean(x) ** 2))  # TISEAN 'variance': the standard deviation
    if not dvar > 0:
        return None
    x, nmf = _nrlazy_numba(x.copy(), int(m), int(d), int(num_iter), neighborhood_std * dvar)
    return x * interval + lo, nmf


# ------------------------------------------------------------------------------
# nstat_z (source_c/nstat_z.c): cross-forecast errors between segments
# ------------------------------------------------------------------------------

@njit(cache=True)
def _nstat_z_std(seg):
    # TISEAN's variance(): sqrt(|<x^2> - <x>^2|), accumulated sequentially (as in C)
    av = 0.0
    var = 0.0
    for h in seg:
        av += h
        var += h * h
    av /= len(seg)
    return np.sqrt(abs(var / len(seg) - av * av))


@njit(cache=True)
def _nstat_z_error(ser, base1, base2, clength, m, tau, minn, step, causal, center, eps0, epsf):
    """
    The summed squared one-step forecast error of the zeroth-order model of TISEAN's nstat_z:
    each point (an embedding vector) of the segment starting at ``base2`` is forecast by the mean of
    what followed its neighbors (embedding vectors within the maximum-norm distance eps, with
    eps grown from ``eps0`` by the factor ``epsf`` until there are at least ``minn``) in the
    segment starting at ``base1``, excluding neighbors within the causality window of the point.
    The window is applied to the index within each segment, so it also applies when the two
    segments differ.
    Returns NaN if some point can never find ``minn`` neighbors.
    """
    pstart = (m - 1) * tau
    dists = np.empty(clength)
    idxs = np.empty(clength, dtype=np.int64)
    err = 0.0
    for i in range(pstart, pstart + center):
        cnt = 0
        for j in range(pstart, clength - step):
            if j >= i - causal + 1 and j <= i + causal + pstart - 1:
                continue # exclude_interval
            d = 0.0
            for k in range(m):
                dd = abs(ser[base2 + i - k * tau] - ser[base1 + j - k * tau])
                if dd > d:
                    d = dd
            dists[cnt] = d
            idxs[cnt] = j
            cnt += 1
        if cnt < minn:
            return np.nan
        # the first eps in the sequence eps0, eps0*epsf, ... with at least minn neighbors
        dmin = np.partition(dists[:cnt], minn - 1)[minn - 1]
        eps = eps0 / epsf
        while True:
            eps *= epsf
            if eps >= dmin:
                break
        casted = 0.0
        nf = 0
        for q in range(cnt):
            if dists[q] <= eps:
                casted += ser[base1 + step + idxs[q]]
                nf += 1
        casted /= nf
        err += (casted - ser[base2 + i + step]) ** 2
    return err
