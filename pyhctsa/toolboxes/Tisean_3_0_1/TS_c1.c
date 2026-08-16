/*
 *   Python C-extension wrapper around TISEAN's `c1` program
 *   (TISEAN 3.0.1, source_f/c1.f + source_f/d1.f, author Thomas Schreiber).
 *
 *   `c1` computes the fixed-mass curves used to estimate the information
 *   dimension D1: for a geometrically spaced sequence of neighbour masses it
 *   reports the mean log radius that encloses that mass, one curve per
 *   embedding dimension.
 *
 *   The original is Fortran 77 and a command-line program: it parses argv into
 *   locals, reads the series from a text file, and writes `#m=` separated
 *   blocks to a `.c1` file or to stdout. None of that survives repeated
 *   in-process calls, so the algorithm is transcribed here: the driver loop of
 *   c1.f, the fixed-mass estimator d1.f, the box-assisted neighbour search of
 *   neigh.f (`base`/`neigh`, i.e. the scalar branch of `mbase`/`mneigh`), the
 *   naive mean/rms of normal.f and the k-th smallest of rank.f's `which` all
 *   live in one `c1_ctx`, the series arrives as a numpy array, and the blocks
 *   are handed back as arrays.
 *
 *   Two things are transcribed rather than translated, because the feature
 *   values depend on them:
 *
 *   * Everything is single precision, as Fortran's default REAL is. The box
 *     index is a truncated float quotient and the sum of log radii runs to
 *     thousands of terms, so a double-precision port drifts away from what the
 *     Fortran produces.
 *   * d1.f draws its reference points with the compiler's `rand`, so the curves
 *     depend on that generator. `ts_rand` below reproduces gfortran's, the only
 *     one TISEAN still builds against: the Park-Miller minimal standard
 *     sequence, truncated to a 22-bit fixed-point fraction. Its state is
 *     per-call here rather than a process-wide static.
 *
 *   One deviation is deliberate. d1.f's shuffle subtracts the embedding offset
 *   in the wrong place,
 *
 *       iperm=min(int(rand(0.0)*nmax-(mt-1)*id)+1,nmax-(mt-1)*id)
 *
 *   which for m > 1 lands below 1 whenever the draw is small -- about once per
 *   thousand draws -- and Fortran then swaps against memory in front of `ju`,
 *   leaving that entry pointing at whatever happened to be there. The
 *   expression is kept as written, so the random stream and the bias it induces
 *   are unchanged, but a draw that lands off-array is passed over rather than
 *   pulled back to the first slot: the Fortran does not touch `ju(1)` in that
 *   case either, and skipping keeps every in-range entry agreeing with it. What
 *   cannot be reproduced is the one entry per bad draw that the Fortran fills
 *   with whatever lies in front of the array; it keeps its own value here.
 *
 *   Checked against the Fortran built with gfortran, over several series and
 *   settings: the blocks come out the same shape, every mass agrees to the last
 *   bit, and so does every radius computed from draws that all stayed in range.
 *   The rest sit within a few percent, which carries through to the features as
 *   a median relative difference of 3e-5 -- the dimension estimates are
 *   unaffected and only the spreads taken across a scaling range move at the
 *   percent level.
 *
 *   TISEAN is Copyright (c) 1998-2007 Rainer Hegger, Holger Kantz,
 *   Thomas Schreiber, and is distributed under the GNU General Public License
 *   version 2 or later.
 */

#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* neigh.f's box-assisted search: IM boxes per coordinate, offset by II so that
   a negative quotient still lands on a non-negative box index. */
#define TS_IM 100
#define TS_II 100000000LL

/* d1.f: the floor it puts under a radius before taking its logarithm. */
#define TS_TINY 1e-20f

/* Enough sweeps that a run which cannot converge stops rather than hangs; each
   sweep grows the search radius by sqrt(2), so a genuine run needs a handful. */
#define TS_MAX_SWEEPS 4096

typedef struct {
    const float *y;            /* series, 1-based via Y(c, n)                */
    int nmax;                  /* number of samples                          */

    int id;                    /* delay (c1 -d)                              */
    int nmin;                  /* minimal time separation (c1 -t)            */
    int ncmin;                 /* minimal number of centre points (c1 -n)    */
    int kmax;                  /* maximal number of neighbours (c1 -K)       */

    uint32_t seed;             /* state of the Park-Miller stream            */

    int *jh;                   /* box histogram, 0..TS_IM*TS_IM              */
    int *jpntr;                /* points ordered by box, 1-based             */
    int *box;                  /* box index of each point, 1-based scratch   */
    int *ju;                   /* centre points to visit, 1-based            */
    int *nlist;                /* candidates from the box scan               */
    float *nd;                 /* their max-norm distances, parallel to nlist*/
    float *d;                  /* compacted distances fed to the selection   */
    float *yref;               /* reference delay vector, hoisted per scan   */

    double *out;               /* emitted (radius, mass) pairs, row-major    */
    int nrows, cap;
} c1_ctx;

#define Y(c, n) ((c)->y[(n) - 1])

static void ctx_free(c1_ctx *c)
{
    free(c->jh); free(c->jpntr); free(c->box); free(c->ju); free(c->nlist);
    free(c->nd); free(c->d); free(c->yref);
    free(c->out);
    memset(c, 0, sizeof(*c));
}

static int ctx_alloc(c1_ctx *c)
{
    c->jh = (int *)malloc(sizeof(int) * (size_t)(TS_IM * TS_IM + 1));
    c->jpntr = (int *)malloc(sizeof(int) * (size_t)(c->nmax + 1));
    c->box = (int *)malloc(sizeof(int) * (size_t)(c->nmax + 1));
    c->ju = (int *)malloc(sizeof(int) * (size_t)(c->nmax + 1));
    c->nlist = (int *)malloc(sizeof(int) * (size_t)(c->nmax + 1));
    c->nd = (float *)malloc(sizeof(float) * (size_t)(c->nmax + 1));
    c->d = (float *)malloc(sizeof(float) * (size_t)(c->nmax + 1));
    c->yref = (float *)malloc(sizeof(float) * (size_t)(c->nmax + 1));
    return (c->jh && c->jpntr && c->box && c->ju && c->nlist && c->nd && c->d
            && c->yref) ? 0 : -1;
}

static int emit(c1_ctx *c, float radius, float mass)
{
    if (c->nrows == c->cap) {
        int cap = (c->cap == 0) ? 256 : c->cap * 2;
        double *out = (double *)realloc(c->out, sizeof(double) * (size_t)cap * 2);
        if (out == NULL)
            return -1;
        c->out = out;
        c->cap = cap;
    }
    c->out[c->nrows * 2] = (double)radius;
    c->out[c->nrows * 2 + 1] = (double)mass;
    c->nrows++;
    return 0;
}

/* ---------------------------------------------------------------------------
 * gfortran's RAND(0): the next value of the Park-Miller minimal standard
 * sequence, mapped to [0,1) by keeping its top 22 bits.
 * ------------------------------------------------------------------------ */
static float ts_rand(c1_ctx *c)
{
    c->seed = (uint32_t)(((uint64_t)16807 * (uint64_t)c->seed) % 2147483647ULL);
    return (float)(c->seed >> 9) / 4194304.0f;
}

/* normal.f's psi(): the digamma function, tabulated for small arguments. */
static float ts_psi(int i)
{
    static const float p[21] = {
        0.0f,
        -0.57721566490f,  0.42278433509f,  0.92278433509f,  1.25611766843f,
         1.50611766843f,  1.70611766843f,  1.87278433509f,  2.01564147795f,
         2.14064147795f,  2.25175258906f,  2.35175258906f,  2.44266167997f,
         2.52599501330f,  2.60291809023f,  2.67434666166f,  2.74101332832f,
         2.80351332832f,  2.86233685773f,  2.91789241329f,  2.97052399224f
    };
    if (i <= 20)
        return p[i];
    return logf((float)i) - 1.0f / (2.0f * (float)i);
}

/* normal.f's rms(): mean and rms amplitude, accumulated naively. */
static void ts_rms(const c1_ctx *c, float *sc_out, float *sd_out)
{
    float sc = 0.0f, sd = 0.0f;
    int n;

    for (n = 1; n <= c->nmax; n++)
        sc = sc + Y(c, n);
    sc = sc / (float)c->nmax;
    for (n = 1; n <= c->nmax; n++)
        sd = sd + (Y(c, n) - sc) * (Y(c, n) - sc);
    sd = sqrtf(sd / (float)c->nmax);

    *sc_out = sc;
    *sd_out = sd;
}

/* The box holding delay vector n: its first and last coordinates, each
   truncated to a multiple of eps and folded onto TS_IM boxes. */
static int ts_box(const c1_ctx *c, int n, int m, float eps)
{
    long long i = (long long)(Y(c, n) / eps);
    int b = (int)((i + TS_II) % TS_IM);

    if (m > 1) {
        long long k = (long long)(Y(c, n - (m - 1) * c->id) / eps);
        b = TS_IM * b + (int)((k + TS_II) % TS_IM);
    }
    return b;
}

/* neigh.f's base(): sort the points 1..nlim into boxes of side eps. */
static void ts_base(c1_ctx *c, int nlim, int m, float eps)
{
    const int off = (m - 1) * c->id;
    int i, n;

    for (i = 0; i <= TS_IM * TS_IM; i++)
        c->jh[i] = 0;
    for (n = off + 1; n <= nlim; n++) {
        int b = ts_box(c, n, m, eps);
        c->box[n] = b;                  /* reused below; ts_box is expensive */
        c->jh[b]++;
    }
    for (i = 1; i <= TS_IM * TS_IM; i++)
        c->jh[i] += c->jh[i - 1];
    for (n = off + 1; n <= nlim; n++) {
        int b = c->box[n];
        c->jpntr[c->jh[b]] = n;
        c->jh[b]--;
    }
}

/* neigh.f's neigh(): every point within eps (maximum norm) of delay vector n,
   looked up in the neighbouring boxes. Returns how many were written to
   c->nlist. */
static int ts_neigh(c1_ctx *c, int n, int nlast, int m, float eps)
{
    const int id = c->id;
    const int kloop = (m == 1) ? 0 : 1;
    long long jj = (long long)(Y(c, n) / eps);
    long long kk = (long long)(Y(c, n - (m - 1) * id) / eps);
    long long j, k;
    int nfound = 0;
    int i;
    /* The reference delay vector is invariant across every candidate below;
       gather it once instead of re-addressing Y(c, n - i*id) per candidate. */
    float *const yref = c->yref;
    for (i = 0; i < m; i++)
        yref[i] = Y(c, n - i * id);

    for (j = jj - 1; j <= jj + 1; j++) {
        for (k = kk - kloop; k <= kk + kloop; k++) {
            int jk = (int)((j + TS_II) % TS_IM);
            int lo, hi, ip;
            if (m > 1)
                jk = TS_IM * jk + (int)((k + TS_II) % TS_IM);
            /* Descending through the box, which is ascending in time. */
            lo = c->jh[jk] + 1;
            hi = c->jh[jk + 1];
            for (ip = hi; ip >= lo; ip--) {
                int np = c->jpntr[ip], near = 1;
                float dis = 0.0f;
                if (np > nlast)
                    break;
                /* A kept point clears every coordinate, so this loop already
                   visits all m of them: accumulate the max-norm distance here
                   rather than have d1 walk the same coordinates again. */
                for (i = 0; i < m; i++) {
                    float a = fabsf(yref[i] - Y(c, np - i * id));
                    if (a >= eps) {
                        near = 0;
                        break;
                    }
                    if (a > dis)
                        dis = a;
                }
                if (near) {
                    c->nlist[nfound] = np;
                    c->nd[nfound] = dis;
                    nfound++;
                }
            }
        }
    }
    return nfound;
}

/* rank.f's which(): the k-th smallest of x[0..n-1], 1-based k. The Fortran
   ranks the whole array; only the value is used, so selection suffices (it
   agrees even on ties, which share a value). x is scratch and is reordered. */
static float ts_which(float *x, int n, int k)
{
    int lo = 0, hi = n - 1, target = k - 1;

    while (lo < hi) {
        float pivot = x[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (x[i] < pivot) i++;
            while (x[j] > pivot) j--;
            if (i <= j) {
                float t = x[i]; x[i] = x[j]; x[j] = t;
                i++; j--;
            }
        }
        if (target <= j)      hi = j;
        else if (target >= i) lo = i;
        else                  return x[target];
    }
    return x[target];
}

/* ---------------------------------------------------------------------------
 * d1.f: the mean log radius enclosing the mass exp(*pln), with Grassberger's
 * finite-sample correction. *pln is snapped to the mass actually realisable
 * with an integer number of neighbours; the run is skipped (returning 0) when
 * that lands on the same neighbour count as the previous level, *pr.
 * Returns 1 when a radius was computed, 0 when skipped, -1 on failure.
 * ------------------------------------------------------------------------ */
static int ts_d1(c1_ctx *c, int m, float pr, float *pln_io, float *eln_io)
{
    const int id = c->id, nmax = c->nmax, nmin = c->nmin, ncmin = c->ncmin;
    const int off = (m - 1) * id;      /* mt - 1 == m - 1, the series is scalar */
    const int ncomp0 = nmax - off;
    int ncomp = ncomp0;
    float pln = *pln_io, eln, eps, sc, sd;
    int k, kpr, i, iu, sweeps;

    kpr = (int)(expf(pr) * (float)(ncomp - 2 * nmin - 1)) + 1;
    k = (int)(expf(pln) * (float)(ncomp - 2 * nmin - 1)) + 1;
    if (k > c->kmax) {
        /* Too many neighbours for the mass asked for: shorten the series
           instead, so that kmax neighbours carry it. */
        float t = (float)(ncomp - 2 * nmin - 1) * (float)c->kmax;
        t = t / (float)k;
        t = t + (float)(2 * nmin);
        t = t + 1.0f;
        ncomp = (int)t;
        k = c->kmax;
    }
    pln = ts_psi(k) - logf((float)(ncomp - 2 * nmin - 1));
    *pln_io = pln;
    if (k == kpr)
        return 0;

    ts_rms(c, &sc, &sd);
    eps = expf(pln / (float)m) * sd;

    for (i = 1; i <= ncomp0; i++)
        c->ju[i] = i + off;
    for (i = 1; i <= ncomp0; i++) {
        /* d1.f's shuffle, parenthesised as written; see the header note. */
        int iperm = (int)(ts_rand(c) * (float)nmax - (float)off) + 1;
        int ih;
        if (iperm > ncomp0) iperm = ncomp0;
        if (iperm < 1)
            continue;                   /* the swap the Fortran makes off-array */
        ih = c->ju[i];
        c->ju[i] = c->ju[iperm];
        c->ju[iperm] = ih;
    }

    iu = ncmin;
    eln = 0.0f;
    for (sweeps = 0; ; sweeps++) {
        int iunp = 0, nn;

        if (sweeps >= TS_MAX_SWEEPS)
            return -1;

        ts_base(c, ncomp + off, m, eps);
        for (nn = 1; nn <= iu; nn++) {
            int n = c->ju[nn];
            int nfound = ts_neigh(c, n, nmax, m, eps);
            int nf = 0, ip;

            for (ip = 0; ip < nfound; ip++) {
                int np = c->nlist[ip];
                int nmd = abs(np - n) % ncomp;
                if (nmd <= nmin || nmd >= ncomp - nmin)
                    continue;                       /* temporal neighbours */
                c->d[nf++] = c->nd[ip];             /* max norm, from ts_neigh */
            }
            if (nf < k) {
                c->ju[++iunp] = n;                  /* mark for the next sweep */
            } else {
                float e = ts_which(c->d, nf, k);
                eln = eln + logf(e > TS_TINY ? e : TS_TINY);
            }
        }
        iu = iunp;
        eps = eps * sqrtf(2.0f);
        if (iunp == 0)
            break;
    }
    *eln_io = eln / (float)(ncmin - off);
    return 1;
}

/* ---------------------------------------------------------------------------
 * c1.f's main loop: for each embedding dimension, walk the mass from 1/N up to
 * 1 in steps of one resolution unit, and emit the (radius, mass) pairs whose
 * mass differs from the previous one.
 * ------------------------------------------------------------------------ */
static int c1_run(c1_ctx *c, int mfrom, int mto, float res, int *counts)
{
    const float resl = logf(2.0f) / res;
    /* c1.f leaves rln from the previous level in place when d1 returns early,
       so it is deliberately not reset per level or per dimension. */
    float rln = 0.0f;
    int m;

    /* c1.f seeds the generator with its -I option before anything else, which
       for the default seed of 0 means "keep going", i.e. discards one draw. */
    (void)ts_rand(c);

    for (m = mfrom; m <= mto; m++) {
        const int ncomp0 = c->nmax - (m - 1) * c->id;
        const float start = logf(1.0f / (float)ncomp0);
        /* Fortran's trip count for a REAL do-loop, in the same precision. */
        const int trip = (int)((0.0f - start + resl) / resl);
        float pr = 0.0f, pl = start;
        int t, before = c->nrows;

        for (t = 0; t < trip; t++) {
            float pln = pl;
            int rc = ts_d1(c, m, pr, &pln, &rln);
            if (rc < 0)
                return -1;
            if (pln != pr) {
                pr = pln;
                if (emit(c, expf(rln), expf(pln)) != 0)
                    return -1;
            }
            pl = pl + resl;
        }
        counts[m - mfrom] = c->nrows - before;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */

PyDoc_STRVAR(fixed_mass_doc,
"fixed_mass(series, delay, mmin, mmax, tsep, nref, resolution, kmax, seed)\n"
"\n"
"Fixed-mass curves for the information dimension (TISEAN's ``c1``).\n"
"\n"
"Parameters\n"
"----------\n"
"series : ndarray of float64\n"
"    1-D, C-contiguous input time series. Rounded to single precision, as\n"
"    Fortran's default REAL is.\n"
"delay : int\n"
"    Time delay of the embedding (``c1 -d``).\n"
"mmin, mmax : int\n"
"    Minimal and maximal embedding dimension (``c1 -m`` / ``c1 -M``).\n"
"tsep : int\n"
"    Minimal time separation between a centre point and its neighbours,\n"
"    in samples (``c1 -t``).\n"
"nref : int\n"
"    Number of centre points (``c1 -n``).\n"
"resolution : float\n"
"    Mass levels per octave (``c1 -#``).\n"
"kmax : int\n"
"    Maximal number of neighbours (``c1 -K``).\n"
"seed : int\n"
"    Initial state of the Park-Miller stream the reference points are drawn\n"
"    with. TISEAN starts from 1.\n"
"\n"
"Returns\n"
"-------\n"
"blocks : list of ndarray, shape (n_i, 2)\n"
"    One array per embedding dimension from ``mmin`` to ``mmax``: the radius\n"
"    in the first column and the neighbour mass it encloses in the second.\n"
"    These are the ``#m=`` blocks TISEAN would have written.\n");

static PyObject *py_fixed_mass(PyObject *self, PyObject *args)
{
    PyObject *series_obj;
    PyArrayObject *series_arr = NULL;
    PyObject *result = NULL, *blocks = NULL;
    float *series = NULL;
    int *counts = NULL;
    c1_ctx c;
    long delay, mmin, mmax, tsep, nref, kmax, seed;
    double resolution;
    npy_intp i, n;
    int rc, m, nblocks, row = 0;

    if (!PyArg_ParseTuple(args, "Ollllldll", &series_obj, &delay, &mmin, &mmax,
                          &tsep, &nref, &resolution, &kmax, &seed))
        return NULL;

    series_arr = (PyArrayObject *)PyArray_FROMANY(series_obj, NPY_DOUBLE, 1, 1,
                                                  NPY_ARRAY_C_CONTIGUOUS |
                                                  NPY_ARRAY_ALIGNED);
    if (series_arr == NULL)
        return NULL;

    memset(&c, 0, sizeof(c));
    n = PyArray_SIZE(series_arr);
    c.nmax = (int)n;
    c.id = (int)delay;
    c.nmin = (int)tsep;
    c.ncmin = (int)nref;
    c.kmax = (int)kmax;
    c.seed = (uint32_t)seed;

    if (delay < 1 || mmin < 1 || mmax < mmin) {
        PyErr_SetString(PyExc_ValueError,
                        "delay must be >= 1 and 1 <= mmin <= mmax");
        goto fail;
    }
    if (tsep < 0 || nref < 1 || kmax < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "tsep must be >= 0, and nref and kmax >= 1");
        goto fail;
    }
    if (resolution <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "resolution must be positive");
        goto fail;
    }
    if (seed < 1 || seed > 2147483646L) {
        PyErr_SetString(PyExc_ValueError, "seed must be in [1, 2147483646]");
        goto fail;
    }
    /* d1.f needs at least one non-temporal neighbour and a delay vector that
       fits, at the largest dimension asked for. */
    if (n < 2 || (npy_intp)((mmax - 1) * delay) >= n
        || (npy_intp)(n - (mmax - 1) * delay) - 2 * tsep - 1 < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "the time series is too short for these settings");
        goto fail;
    }
    if (nref > n - (mmax - 1) * delay) {
        PyErr_SetString(PyExc_ValueError,
                        "more reference points than embedding vectors");
        goto fail;
    }

    /* The Fortran reads the series into REAL, i.e. single precision. */
    series = (float *)malloc(sizeof(float) * (size_t)n);
    nblocks = (int)(mmax - mmin + 1);
    counts = (int *)calloc((size_t)nblocks, sizeof(int));
    if (series == NULL || counts == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    {
        const double *src = (const double *)PyArray_DATA(series_arr);
        for (i = 0; i < n; i++)
            series[i] = (float)src[i];
    }
    c.y = series;

    if (ctx_alloc(&c) != 0) {
        PyErr_NoMemory();
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    rc = c1_run(&c, (int)mmin, (int)mmax, (float)resolution, counts);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "c1 failed to converge on a neighbourhood radius");
        goto fail;
    }

    blocks = PyList_New(nblocks);
    if (blocks == NULL)
        goto fail;
    for (m = 0; m < nblocks; m++) {
        npy_intp dims[2];
        PyArrayObject *block;
        dims[0] = counts[m];
        dims[1] = 2;
        block = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (block == NULL)
            goto fail;
        if (counts[m] > 0)
            memcpy(PyArray_DATA(block), c.out + (size_t)row * 2,
                   sizeof(double) * (size_t)counts[m] * 2);
        row += counts[m];
        PyList_SET_ITEM(blocks, m, (PyObject *)block);
    }
    result = blocks;
    blocks = NULL;

fail:
    Py_XDECREF(blocks);
    free(counts);
    ctx_free(&c);
    free(series);
    Py_XDECREF(series_arr);
    return result;
}

static PyMethodDef C1Methods[] = {
    {"fixed_mass", py_fixed_mass, METH_VARARGS, fixed_mass_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef c1_module = {
    PyModuleDef_HEAD_INIT,
    "c1",
    "Fixed-mass kernel of TISEAN 3.0.1's c1 program.",
    -1,
    C1Methods
};

PyMODINIT_FUNC PyInit_c1(void)
{
    PyObject *module = PyModule_Create(&c1_module);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}
