/*
 *   Python C-extension wrapper around TISEAN's `false_nearest` program
 *   (TISEAN 3.0.1, source_c/false_nearest.c, author Rainer Hegger).
 *
 *   The original is a command-line program: it parses argv into file-scope
 *   globals, reads the series from a text file, and prints one line per
 *   embedding dimension to stdout, calling exit() on trouble. None of that
 *   survives repeated in-process calls, so the algorithm has been made
 *   re-entrant here: every global lives in an `fnn_ctx`, the series arrives
 *   as a numpy array, and the rows the program would have printed are handed
 *   back as an (n_rows, 4) array.
 *
 *   The numerics are a transcription of false_nearest.c (single
 *   component case; the embedding coordinates are lagged by `delay` samples,
 *   which the original, using lag 1 whatever -d, does not do), including the box-assisted neighbour search with its
 *   1024 x 1024 hashed grid, the growth of the search radius by sqrt(2) until
 *   every point has a neighbour, routines/rescale_data.c and the naive
 *   summation of routines/variance.c. A condition under which the program
 *   would exit() is reported as a status code; rows already completed at that
 *   point are returned (the program had already printed them).
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

#define BOX 1024
#define IBOX (BOX - 1)

/* status codes (0 = ran to completion) */
#define FNN_OK 0
#define FNN_CONSTANT 1      /* rescale_data / variance: constant series      */
#define FNN_TOO_LARGE 2     /* (maxemb+1)*delay >= length                     */
#define FNN_NOT_ENOUGH 3    /* "Not enough points found!"                     */
#define FNN_NOMEM 4

typedef struct {
    double *series;         /* rescaled copy of the data                      */
    int64_t length;
    int64_t theiler;
    double rt;              /* escape factor (false_nearest -f)               */
    double varianz;
    double aveps, vareps;
    int64_t toolarge;
    int64_t *box;           /* BOX * BOX                                      */
    int64_t *list;
} fnn_ctx;

static void mmb(fnn_ctx *c, int64_t maxemb, int64_t delay, int64_t hemb, double eps)
{
    int64_t i, x, y;
    int64_t n = c->length - (maxemb + 1) * delay;

    for (i = 0; i < (int64_t)BOX * BOX; i++)
        c->box[i] = -1;

    for (i = 0; i < n; i++) {
        x = (int64_t)(c->series[i] / eps) & IBOX;
        y = (int64_t)(c->series[i + hemb] / eps) & IBOX;
        c->list[i] = c->box[x * BOX + y];
        c->box[x * BOX + y] = i;
    }
}

/* component 0 only: embedding coordinate i is the sample at lag vemb[i] = i * delay
   (vcomp[i] = 0). */
static char find_nearest(fnn_ctx *c, int64_t n, int64_t dim, int64_t delay, double eps)
{
    int64_t x, y, x1, x2, y1, i, element, which = -1;
    double dx, maxdx, mindx = 1.1, hfactor, factor;
    const double *s = c->series;

    x = (int64_t)(s[n] / eps) & IBOX;
    y = (int64_t)(s[n + dim * delay] / eps) & IBOX;

    for (x1 = x - 1; x1 <= x + 1; x1++) {
        x2 = x1 & IBOX;
        for (y1 = y - 1; y1 <= y + 1; y1++) {
            element = c->box[x2 * BOX + (y1 & IBOX)];
            while (element != -1) {
                int64_t d = element - n;
                if (d < 0) d = -d;
                if (d > c->theiler) {
                    maxdx = fabs(s[n] - s[element]);
                    for (i = 1; i <= dim; i++) {
                        dx = fabs(s[n + i * delay] - s[element + i * delay]);
                        if (dx > maxdx)
                            maxdx = dx;
                    }
                    if ((maxdx < mindx) && (maxdx > 0.0)) {
                        which = element;
                        mindx = maxdx;
                    }
                }
                element = c->list[element];
            }
        }
    }

    if ((which != -1) && (mindx <= eps) && (mindx <= c->varianz / c->rt)) {
        c->aveps += mindx;
        c->vareps += mindx * mindx;
        /* comp == 1: the next component is the next delay coordinate */
        hfactor = fabs(s[n + (dim + 1) * delay] - s[which + (dim + 1) * delay]) / mindx;
        factor = 0.0;
        if (hfactor > factor)
            factor = hfactor;
        if (factor > c->rt)
            c->toolarge++;
        return 1;
    }
    return 0;
}

PyDoc_STRVAR(run_doc,
"run(series, delay, minemb, maxemb, theiler, escape_factor)\n"
"\n"
"Kernel of TISEAN's false_nearest for a scalar series.\n"
"\n"
"Returns\n"
"-------\n"
"rows : ndarray, shape (n, 4)\n"
"    One row per embedding dimension completed: the dimension, the fraction\n"
"    of false nearest neighbours, the mean and the standard deviation of the\n"
"    neighbour distance (in the original units).\n"
"status : int\n"
"    0 if every dimension was done, 1 constant series, 2 delay * (maxemb+1)\n"
"    not smaller than the length, 3 no neighbour found at some dimension.\n");

static PyObject *py_run(PyObject *self, PyObject *args)
{
    PyObject *series_obj;
    PyArrayObject *series_arr = NULL, *out = NULL;
    PyObject *result = NULL;
    long delay, minemb, maxemb, theiler;
    double rt, mn, mx, interval, av, var, h;
    fnn_ctx c;
    char *nearest = NULL;
    double *rows = NULL;
    int64_t i, nrows = 0, emb, dim, donesofar;
    double epsilon, eps0 = 1.0e-5;
    int status = FNN_OK, alldone;
    npy_intp dims[2];

    if (!PyArg_ParseTuple(args, "Olllld", &series_obj, &delay, &minemb,
                          &maxemb, &theiler, &rt))
        return NULL;

    series_arr = (PyArrayObject *)PyArray_FROMANY(series_obj, NPY_DOUBLE, 1, 1,
                                                  NPY_ARRAY_C_CONTIGUOUS |
                                                  NPY_ARRAY_ALIGNED);
    if (series_arr == NULL)
        return NULL;

    memset(&c, 0, sizeof(c));
    c.length = (int64_t)PyArray_SIZE(series_arr);
    c.theiler = theiler;
    c.rt = rt;

    if (delay < 1 || minemb < 1 || maxemb < minemb || theiler < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "delay and minemb must be >= 1, maxemb >= minemb, theiler >= 0");
        goto done;
    }
    if (c.length < 1) {
        PyErr_SetString(PyExc_ValueError, "the time series is empty");
        goto done;
    }

    rows = (double *)malloc(sizeof(double) * 4 * (size_t)(maxemb - minemb + 1));
    if (rows == NULL) { PyErr_NoMemory(); goto done; }

    if ((int64_t)(maxemb + 1) * delay >= c.length) {
        status = FNN_TOO_LARGE;
        goto build;
    }

    c.series = (double *)malloc(sizeof(double) * (size_t)c.length);
    c.list = (int64_t *)malloc(sizeof(int64_t) * (size_t)c.length);
    nearest = (char *)malloc((size_t)c.length);
    c.box = (int64_t *)malloc(sizeof(int64_t) * (size_t)BOX * BOX);
    if (!c.series || !c.list || !nearest || !c.box) { PyErr_NoMemory(); goto done; }
    memcpy(c.series, PyArray_DATA(series_arr), sizeof(double) * (size_t)c.length);

    /* routines/rescale_data.c */
    mn = interval = c.series[0];
    for (i = 1; i < c.length; i++) {
        if (c.series[i] < mn) mn = c.series[i];
        if (c.series[i] > interval) interval = c.series[i];
    }
    mx = interval;
    interval -= mn;
    (void)mx;
    if (interval == 0.0) { status = FNN_CONSTANT; goto build; }
    for (i = 0; i < c.length; i++)
        c.series[i] = (c.series[i] - mn) / interval;

    /* routines/variance.c: naive running sums */
    av = var = 0.0;
    for (i = 0; i < c.length; i++) {
        h = c.series[i];
        av += h;
        var += h * h;
    }
    av /= (double)c.length;
    var = sqrt(fabs(var / (double)c.length - av * av));
    if (var == 0.0) { status = FNN_CONSTANT; goto build; }
    c.varianz = var;

    Py_BEGIN_ALLOW_THREADS
    for (emb = minemb; emb <= maxemb; emb++) {
        dim = emb - 1;
        epsilon = eps0;
        c.toolarge = 0;
        alldone = 0;
        donesofar = 0;
        c.aveps = 0.0;
        c.vareps = 0.0;
        memset(nearest, 0, (size_t)c.length);
        while (!alldone && (epsilon < 2. * c.varianz / c.rt)) {
            alldone = 1;
            mmb(&c, maxemb, delay, dim * delay, epsilon);
            for (i = 0; i < c.length - maxemb * delay; i++)
                if (!nearest[i]) {
                    nearest[i] = find_nearest(&c, i, dim, delay, epsilon);
                    alldone &= nearest[i];
                    donesofar += (int64_t)nearest[i];
                }
            epsilon *= sqrt(2.0);
            if (!donesofar)
                eps0 = epsilon;
        }
        if (donesofar == 0) { status = FNN_NOT_ENOUGH; break; }
        c.aveps *= (1. / (double)donesofar);
        c.vareps *= (1. / (double)donesofar);
        rows[4 * nrows + 0] = (double)(dim + 1);
        rows[4 * nrows + 1] = (double)c.toolarge / (double)donesofar;
        rows[4 * nrows + 2] = c.aveps * interval;
        rows[4 * nrows + 3] = sqrt(c.vareps) * interval;
        nrows++;
    }
    Py_END_ALLOW_THREADS

build:
    dims[0] = (npy_intp)nrows;
    dims[1] = 4;
    out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == NULL) goto done;
    if (nrows > 0)
        memcpy(PyArray_DATA(out), rows, sizeof(double) * 4 * (size_t)nrows);
    result = Py_BuildValue("Oi", (PyObject *)out, status);

done:
    Py_XDECREF(out);
    free(rows);
    free(c.series);
    free(c.list);
    free(nearest);
    free(c.box);
    Py_XDECREF(series_arr);
    return result;
}

static PyMethodDef FnnMethods[] = {
    {"run", py_run, METH_VARARGS, run_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fnn_module = {
    PyModuleDef_HEAD_INIT,
    "false_nearest",
    "False-nearest-neighbour kernel of TISEAN 3.0.1's false_nearest program.",
    -1,
    FnnMethods
};

PyMODINIT_FUNC PyInit_false_nearest(void)
{
    PyObject *module = PyModule_Create(&fnn_module);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}
