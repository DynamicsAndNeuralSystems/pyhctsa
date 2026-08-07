/*
 *   Python C-extension wrapper around the correlation-sum kernel of TISEAN's
 *   `d2` program (TISEAN 3.0.1, source_c/d2.c, author Rainer Hegger).
 *
 *   The original is a command-line program: it parses argv into file-scope
 *   globals, reads the series from a text file, and writes the .c2/.d2/.h2/.stat
 *   files itself (calling exit() when the smallest length scale is exhausted).
 *   None of that survives repeated in-process calls, so the algorithm has been
 *   made re-entrant here: every global lives in a `d2_ctx`, the series arrives
 *   as a numpy array, and the raw pair counts are handed back to Python, which
 *   assembles the c2/d2/h2 tables using the same expressions d2.c prints.
 *
 *   The numerics are a line-by-line transcription of d2.c -- including the
 *   integer truncation in the length-scale index and the hand-rolled PRNG in
 *   scramble() -- so the counts are identical to those of a stock TISEAN build.
 *   The one deliberate change is that scramble() always uses the 64-bit
 *   constants (d2.c picks them via sizeof(long)), so results no longer depend
 *   on the platform's `long` width.
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

/* Size of the field for box assisted neighbour searching (power of 2) */
#define NMAX 256
/* Size of the box for the scramble routine */
#define SCBOX 4096

typedef struct {
    const double *series;      /* single-column time series (DIM == 1)       */
    int64_t length;

    unsigned int DIM;          /* kept for fidelity with d2.c; always 1 here */
    unsigned int EMBED;
    unsigned int HOWOFTEN;
    unsigned int DELAY;
    int64_t MINDIST;           /* Theiler window                             */
    double MAXFOUND;           /* compared against the (double) pair counts  */

    double EPSMAX, EPSMIN, EPSMAX1;
    double epsinv, epsfactor, lneps, lnfac;

    int imax, howoften1, imin;

    int64_t *scr, *oscr, *list, *listc1;
    int64_t *box;              /* NMAX * NMAX, row-major                     */
    int64_t *boxc1;
    double *hs;                /* scratch delay vector, EMBED * DIM          */
    double **found;            /* EMBED*DIM rows of HOWOFTEN counters        */
    double *found_flat;
    double *norm;
    double *epsm;
} d2_ctx;

static void ctx_free(d2_ctx *c)
{
    free(c->scr);
    free(c->oscr);
    free(c->list);
    free(c->listc1);
    free(c->box);
    free(c->boxc1);
    free(c->hs);
    free(c->found);
    free(c->found_flat);
    free(c->norm);
    free(c->epsm);
    memset(c, 0, sizeof(*c));
}

/* ---------------------------------------------------------------------------
 * scramble(): d2.c's deterministic shuffle of the centre-point order.
 * ------------------------------------------------------------------------ */
static int scramble(d2_ctx *c)
{
    int64_t i, j, k, m;
    uint64_t rnd, rndf;
    int64_t hlength, allscr = 0;
    int64_t *scfound, *scnhelp, scnfound;
    int64_t *scbox, lswap, element;
    const int64_t scbox1 = SCBOX - 1;
    double *rz, *schelp, swap;
    const double sceps = (double)SCBOX - 0.001;

    hlength = c->length - (int64_t)(c->EMBED - 1) * c->DELAY;

    /* d2.c selects these constants when sizeof(long) == 8; pinned here so the
       shuffle -- and therefore every pair count -- is platform independent. */
    rndf = 13ULL * 13ULL * 13ULL * 13ULL;
    rndf = rndf * rndf * rndf * 13ULL;
    rnd = 0x849178ULL;
    for (i = 0; i < 1000; i++)
        rnd = rnd * rndf + 1;

    rz = (double *)malloc(sizeof(double) * (size_t)hlength);
    scfound = (int64_t *)malloc(sizeof(int64_t) * (size_t)hlength);
    scnhelp = (int64_t *)malloc(sizeof(int64_t) * (size_t)hlength);
    schelp = (double *)malloc(sizeof(double) * (size_t)hlength);
    scbox = (int64_t *)malloc(sizeof(int64_t) * SCBOX);
    if (!rz || !scfound || !scnhelp || !schelp || !scbox) {
        free(rz); free(scfound); free(scnhelp); free(schelp); free(scbox);
        return -1;
    }

    for (i = 0; i < hlength; i++)
        rz[i] = (double)(rnd = rnd * rndf + 1) / (double)UINT64_MAX;

    for (i = 0; i < SCBOX; i++)
        scbox[i] = -1;
    for (i = 0; i < hlength; i++) {
        m = (int)(rz[i] * sceps) & scbox1;
        scfound[i] = scbox[m];
        scbox[m] = i;
    }
    for (i = 0; i < SCBOX; i++) {
        scnfound = 0;
        element = scbox[i];
        while (element != -1) {
            scnhelp[scnfound] = element;
            schelp[scnfound++] = rz[element];
            element = scfound[element];
        }

        for (j = 0; j < scnfound - 1; j++)
            for (k = j + 1; k < scnfound; k++)
                if (schelp[k] < schelp[j]) {
                    swap = schelp[k];
                    schelp[k] = schelp[j];
                    schelp[j] = swap;
                    lswap = scnhelp[k];
                    scnhelp[k] = scnhelp[j];
                    scnhelp[j] = lswap;
                }
        for (j = 0; j < scnfound; j++)
            c->scr[allscr + j] = scnhelp[j];
        allscr += scnfound;
    }

    free(rz);
    free(scfound);
    free(scnhelp);
    free(schelp);
    free(scbox);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Pair counting for embedding dimensions 2 .. EMBED*DIM (make_c2_dim in d2.c).
 * ------------------------------------------------------------------------ */
static void make_c2_dim(d2_ctx *c, int64_t n)
{
    char small;
    int64_t i, j, k, x, y, i1, i2, j1, element, n1, maxi, count, hi;
    double *hs = c->hs;
    double max, dx;

    n1 = c->scr[n];

    count = 0;
    for (i1 = 0; i1 < (int64_t)c->EMBED; i1++) {
        i2 = i1 * (int64_t)c->DELAY;
        for (j = 0; j < (int64_t)c->DIM; j++)
            hs[count++] = c->series[n1 + i2];
    }

    x = (int)(hs[0] * c->epsinv) & c->imax;
    y = (int)(hs[1] * c->epsinv) & c->imax;

    for (i1 = x - 1; i1 <= x + 1; i1++) {
        i2 = i1 & c->imax;
        for (j1 = y - 1; j1 <= y + 1; j1++) {
            element = c->box[i2 * NMAX + (j1 & c->imax)];
            while (element != -1) {
                if (llabs((long long)(element - n1)) > c->MINDIST) {
                    count = 0;
                    max = 0.0;
                    maxi = c->howoften1;
                    small = 0;
                    for (i = 0; i < (int64_t)c->EMBED; i++) {
                        hi = i * (int64_t)c->DELAY;
                        for (j = 0; j < (int64_t)c->DIM; j++) {
                            dx = fabs(hs[count] - c->series[element + hi]);
                            if (dx <= c->EPSMAX) {
                                if (dx > max) {
                                    max = dx;
                                    if (max < c->EPSMIN)
                                        maxi = c->howoften1;
                                    else
                                        maxi = (int64_t)((c->lneps - log(max)) / c->lnfac);
                                }
                                if (count > 0)
                                    for (k = c->imin; k <= maxi; k++)
                                        c->found[count][k] += 1.0;
                            } else {
                                small = 1;
                                break;
                            }
                            count++;
                        }
                        if (small)
                            break;
                    }
                }
                element = c->list[element];
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Pair counting for embedding dimension 1 (make_c2_1 in d2.c).
 * ------------------------------------------------------------------------ */
static void make_c2_1(d2_ctx *c, int64_t n)
{
    int i, x, i1, maxi;
    int64_t element, n1;
    double hs, max;

    n1 = c->scr[n];
    hs = c->series[n1];

    x = (int)(hs * c->epsinv) & c->imax;

    for (i1 = x - 1; i1 <= x + 1; i1++) {
        element = c->boxc1[i1 & c->imax];
        while (element != -1) {
            if (llabs((long long)(element - n1)) > c->MINDIST) {
                max = fabs(hs - c->series[element]);
                if (max <= c->EPSMAX) {
                    if (max < c->EPSMIN)
                        maxi = c->howoften1;
                    else
                        maxi = (int)((c->lneps - log(max)) / c->lnfac);
                    for (i = c->imin; i <= maxi; i++)
                        c->found[0][i] += 1.0;
                }
            }
            element = c->listc1[element];
        }
    }
}

/* ---------------------------------------------------------------------------
 * The body of d2.c's main(), minus argument parsing and file output.
 * Returns 0 on success, -1 on allocation failure.
 * ------------------------------------------------------------------------ */
static int d2_run(d2_ctx *c)
{
    char smaller;
    int maxembed;
    int64_t i1, j1, x, y, sn, n, i, n1, n2, nmax, lnorm;

    if (scramble(c) != 0)
        return -1;
    for (i = 0; i < (c->length - (int64_t)(c->EMBED - 1) * c->DELAY); i++)
        c->oscr[c->scr[i]] = i;

    maxembed = (int)(c->DIM * c->EMBED) - 1;
    nmax = c->length - (int64_t)c->DELAY * (c->EMBED - 1);

    for (i1 = 0; i1 < NMAX; i1++) {
        c->boxc1[i1] = -1;
        for (j1 = 0; j1 < NMAX; j1++)
            c->box[i1 * NMAX + j1] = -1;
    }
    lnorm = 0;

    for (n = 1; n < nmax; n++) {
        smaller = 0;
        sn = c->scr[n - 1];
        /* d2.c boxes on components 0 and 1 when DIM > 1; with a single column
           (all this wrapper supports) it boxes on x(t) and x(t+DELAY). */
        x = (int64_t)(c->series[sn] * c->epsinv) & c->imax;
        y = (int64_t)(c->series[sn + c->DELAY] * c->epsinv) & c->imax;
        c->list[sn] = c->box[x * NMAX + y];
        c->box[x * NMAX + y] = sn;
        c->listc1[sn] = c->boxc1[x];
        c->boxc1[x] = sn;

        i = c->imin;
        while (c->found[maxembed][i] >= c->MAXFOUND) {
            smaller = 1;
            if (++i > c->howoften1)
                break;
        }
        if (smaller) {
            c->imin = (int)i;
            if (c->imin <= c->howoften1) {
                c->EPSMAX = c->epsm[c->imin];
                c->epsinv = 1.0 / c->EPSMAX;
                for (i1 = 0; i1 < NMAX; i1++) {
                    c->boxc1[i1] = -1;
                    for (j1 = 0; j1 < NMAX; j1++)
                        c->box[i1 * NMAX + j1] = -1;
                }
                for (i1 = 0; i1 < n; i1++) {
                    sn = c->scr[i1];
                    x = (int64_t)(c->series[sn] * c->epsinv) & c->imax;
                    y = (int64_t)(c->series[sn + c->DELAY] * c->epsinv) & c->imax;
                    c->list[sn] = c->box[x * NMAX + y];
                    c->box[x * NMAX + y] = sn;
                    c->listc1[sn] = c->boxc1[x];
                    c->boxc1[x] = sn;
                }
            }
        }

        if (c->imin <= c->howoften1) {
            lnorm = n;
            if (c->MINDIST > 0) {
                sn = c->scr[n];
                n1 = (sn - c->MINDIST >= 0) ? sn - c->MINDIST : 0;
                n2 = (sn + c->MINDIST < c->length - (int64_t)(c->EMBED - 1) * c->DELAY)
                         ? sn + c->MINDIST
                         : c->length - (int64_t)(c->EMBED - 1) * c->DELAY - 1;
                for (i1 = n1; i1 <= n2; i1++)
                    if (c->oscr[i1] < n)
                        lnorm--;
            }

            if (c->EMBED * c->DIM > 1)
                make_c2_dim(c, n);
            make_c2_1(c, n);
            for (i = c->imin; i < (int64_t)c->HOWOFTEN; i++)
                c->norm[i] += (double)lnorm;
        }

        /* d2.c writes its output files and exit(0)s here; the counters are
           already final at this point, so we simply stop accumulating. */
        if (c->imin > c->howoften1)
            break;
    }

    return 0;
}

/* ------------------------------------------------------------------------ */

PyDoc_STRVAR(correlation_sums_doc,
"correlation_sums(series, delay, embed, theiler, maxfound, howoften, epsmax, epsmin)\n"
"\n"
"Run the pair-counting kernel of TISEAN's ``d2`` on a single time series.\n"
"\n"
"Parameters\n"
"----------\n"
"series : ndarray of float64\n"
"    1-D, C-contiguous input time series.\n"
"delay : int\n"
"    Time delay of the embedding (``d2 -d``).\n"
"embed : int\n"
"    Maximum embedding dimension (``d2 -M1,<embed>``).\n"
"theiler : int\n"
"    Theiler window, in samples (``d2 -t``).\n"
"howoften : int\n"
"    Number of length scales (``d2 -#``).\n"
"maxfound : int\n"
"    Maximum number of pairs; 0 means all (``d2 -N``).\n"
"epsmax, epsmin : float or None\n"
"    Upper/lower length scale (``d2 -R`` / ``d2 -r``). ``None`` reproduces the\n"
"    defaults, i.e. the data interval and one thousandth of it.\n"
"\n"
"Returns\n"
"-------\n"
"found : ndarray, shape (embed, howoften)\n"
"    Pair counts per embedding dimension and length scale.\n"
"norm : ndarray, shape (howoften,)\n"
"    Normalisation (number of centre points) per length scale.\n"
"epsmax1 : float\n"
"    The largest length scale actually used.\n"
"epsfactor : float\n"
"    Ratio between successive length scales.\n");

static PyObject *py_correlation_sums(PyObject *self, PyObject *args)
{
    PyObject *series_obj, *epsmax_obj, *epsmin_obj;
    PyArrayObject *series_arr = NULL;
    d2_ctx c;
    npy_intp dims[2];
    PyArrayObject *found_out = NULL, *norm_out = NULL;
    PyObject *result = NULL;
    long delay, embed, theiler, howoften;
    unsigned long long maxfound;
    double maxinterval, mn, interval;
    int64_t i, j;
    char eps_max_set, eps_min_set;
    double EPSMAX, EPSMIN;
    int rc;

    if (!PyArg_ParseTuple(args, "OlllKlOO", &series_obj, &delay, &embed, &theiler,
                          &maxfound, &howoften, &epsmax_obj, &epsmin_obj))
        return NULL;

    series_arr = (PyArrayObject *)PyArray_FROMANY(series_obj, NPY_DOUBLE, 1, 1,
                                                  NPY_ARRAY_C_CONTIGUOUS |
                                                  NPY_ARRAY_ALIGNED);
    if (series_arr == NULL)
        return NULL;

    memset(&c, 0, sizeof(c));
    c.series = (const double *)PyArray_DATA(series_arr);
    c.length = (int64_t)PyArray_SIZE(series_arr);
    c.DIM = 1;
    c.EMBED = (unsigned int)embed;
    c.DELAY = (unsigned int)delay;
    c.HOWOFTEN = (unsigned int)howoften;
    c.MINDIST = (int64_t)theiler;
    c.MAXFOUND = (maxfound == 0) ? (double)UINT64_MAX : (double)maxfound;
    c.imax = NMAX - 1;
    c.howoften1 = (int)howoften - 1;
    c.imin = 0;

    /* embed >= 2: d2.c boxes on x(t+DELAY) without shortening its loop for a
       one-dimensional embedding, so embed == 1 reads past the end of the data. */
    if (delay < 1 || embed < 2 || howoften < 2 || theiler < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "delay must be >= 1, embed >= 2, howoften >= 2, theiler >= 0");
        goto fail;
    }

    /* Default length scales are relative to the data interval (d2.c main()). */
    EPSMAX = 1.0;
    EPSMIN = 1.e-3;
    eps_max_set = 0;
    eps_min_set = 0;
    if (epsmax_obj != Py_None) {
        EPSMAX = PyFloat_AsDouble(epsmax_obj);
        if (PyErr_Occurred()) goto fail;
        eps_max_set = 1;
    }
    if (epsmin_obj != Py_None) {
        EPSMIN = PyFloat_AsDouble(epsmin_obj);
        if (PyErr_Occurred()) goto fail;
        eps_min_set = 1;
    }

    maxinterval = 0.0;
    mn = interval = c.series[0];
    for (j = 1; j < c.length; j++) {
        if (mn > c.series[j]) mn = c.series[j];
        if (interval < c.series[j]) interval = c.series[j];
    }
    interval -= mn;
    if (interval > maxinterval)
        maxinterval = interval;

    if (!eps_max_set) EPSMAX *= maxinterval;
    if (!eps_min_set) EPSMIN *= maxinterval;
    EPSMAX = (fabs(EPSMAX) < maxinterval) ? fabs(EPSMAX) : maxinterval;
    EPSMIN = (fabs(EPSMIN) < EPSMAX) ? fabs(EPSMIN) : EPSMAX / 2.;
    c.EPSMAX = EPSMAX;
    c.EPSMIN = EPSMIN;
    c.EPSMAX1 = EPSMAX;

    if (maxinterval <= 0.0) {
        PyErr_SetString(PyExc_ValueError,
                        "the time series is constant: no length scales to scan");
        goto fail;
    }
    if ((c.length - (int64_t)(c.EMBED - 1) * c.DELAY) <= 1) {
        PyErr_SetString(PyExc_ValueError,
                        "embedding dimension and delay are too large: the delay "
                        "vector would be longer than the whole series");
        goto fail;
    }

    c.epsinv = 1.0 / c.EPSMAX;
    c.epsfactor = pow(c.EPSMAX / c.EPSMIN, 1.0 / (double)c.howoften1);
    c.lneps = log(c.EPSMAX);
    c.lnfac = log(c.epsfactor);

    /* Allocations ------------------------------------------------------- */
    {
        int64_t hlength = c.length - (int64_t)(c.EMBED - 1) * c.DELAY;
        int64_t nrows = (int64_t)c.DIM * c.EMBED;

        c.list = (int64_t *)malloc(sizeof(int64_t) * (size_t)c.length);
        c.listc1 = (int64_t *)malloc(sizeof(int64_t) * (size_t)c.length);
        c.scr = (int64_t *)malloc(sizeof(int64_t) * (size_t)hlength);
        c.oscr = (int64_t *)malloc(sizeof(int64_t) * (size_t)hlength);
        c.box = (int64_t *)malloc(sizeof(int64_t) * NMAX * NMAX);
        c.boxc1 = (int64_t *)malloc(sizeof(int64_t) * NMAX);
        c.hs = (double *)malloc(sizeof(double) * (size_t)nrows);
        c.found = (double **)malloc(sizeof(double *) * (size_t)nrows);
        c.found_flat = (double *)calloc((size_t)(nrows * howoften), sizeof(double));
        c.norm = (double *)calloc((size_t)howoften, sizeof(double));
        c.epsm = (double *)malloc(sizeof(double) * (size_t)howoften);

        if (!c.list || !c.listc1 || !c.scr || !c.oscr || !c.box || !c.boxc1 ||
            !c.hs || !c.found || !c.found_flat || !c.norm || !c.epsm) {
            PyErr_NoMemory();
            goto fail;
        }
        for (i = 0; i < nrows; i++)
            c.found[i] = c.found_flat + i * howoften;
    }

    c.epsm[0] = c.EPSMAX;
    for (i = 1; i < howoften; i++)
        c.epsm[i] = c.epsm[i - 1] / c.epsfactor;

    Py_BEGIN_ALLOW_THREADS
    rc = d2_run(&c);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_NoMemory();
        goto fail;
    }

    dims[0] = (npy_intp)(c.DIM * c.EMBED);
    dims[1] = (npy_intp)howoften;
    found_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (found_out == NULL) goto fail;
    memcpy(PyArray_DATA(found_out), c.found_flat,
           sizeof(double) * (size_t)(dims[0] * dims[1]));

    norm_out = (PyArrayObject *)PyArray_SimpleNew(1, &dims[1], NPY_DOUBLE);
    if (norm_out == NULL) goto fail;
    memcpy(PyArray_DATA(norm_out), c.norm, sizeof(double) * (size_t)howoften);

    result = Py_BuildValue("OOdd", (PyObject *)found_out, (PyObject *)norm_out,
                           c.EPSMAX1, c.epsfactor);

fail:
    Py_XDECREF(found_out);
    Py_XDECREF(norm_out);
    ctx_free(&c);
    Py_XDECREF(series_arr);
    return result;
}

static PyMethodDef D2Methods[] = {
    {"correlation_sums", py_correlation_sums, METH_VARARGS, correlation_sums_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef d2_module = {
    PyModuleDef_HEAD_INIT,
    "d2",
    "Correlation-sum kernel of TISEAN 3.0.1's d2 program.",
    -1,
    D2Methods
};

PyMODINIT_FUNC PyInit_d2(void)
{
    PyObject *module = PyModule_Create(&d2_module);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}
