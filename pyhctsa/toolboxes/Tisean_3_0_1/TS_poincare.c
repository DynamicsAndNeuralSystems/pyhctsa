/*
 *   Python C-extension wrapper around TISEAN's `poincare` program
 *   (TISEAN 3.0.1, source_c/poincare.c, author Rainer Hegger).
 *
 *   The original is a command-line program: it parses argv into file-scope
 *   globals, reads the series from a text file, and prints one line per cut to
 *   stdout or to a `.poin` file (calling exit() when the crossing level lies
 *   outside the data). None of that survives repeated in-process calls, so the
 *   algorithm has been made re-entrant here: every global lives in a
 *   `poin_ctx`, the series arrives as a numpy array, and the cuts are handed
 *   back as a single (n_cuts, dim) array -- the rows the program would have
 *   written, at full double precision rather than through `%e`.
 *
 *   The numerics are a line-by-line transcription of poincare.c, including the
 *   `lasttime > 0.0` gate that drops the first crossing and the naive summation
 *   in routines/variance.c, which fixes the default crossing level (and hence,
 *   at the last ulp, which samples count as crossings).
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

typedef struct {
    const double *series;
    int64_t length;

    int dim;                   /* embedding dimension (poincare -m)          */
    int delay;                 /* embedding delay (poincare -d)              */
    int comp;                  /* component to cut (poincare -q)             */
    int dir;                   /* 0: crossing from below, 1: from above      */
    double where;              /* crossing level (poincare -a)               */

    double *rows;              /* nrows * dim, row-major                     */
    int64_t nrows, cap;
} poin_ctx;

static void ctx_free(poin_ctx *c)
{
    free(c->rows);
    memset(c, 0, sizeof(*c));
}

/* Hand out the next row of the output table, growing it as needed. */
static double *next_row(poin_ctx *c)
{
    if (c->nrows == c->cap) {
        int64_t cap = (c->cap == 0) ? 256 : c->cap * 2;
        double *rows = (double *)realloc(c->rows,
                                         sizeof(double) * (size_t)(cap * c->dim));
        if (rows == NULL)
            return NULL;
        c->rows = rows;
        c->cap = cap;
    }
    return c->rows + c->nrows++ * c->dim;
}

/* ---------------------------------------------------------------------------
 * poincare(): the body of poincare.c's poincare(), writing into `rows` instead
 * of to a file. Returns 0 on success, -1 on allocation failure.
 * ------------------------------------------------------------------------ */
static int poincare_run(poin_ctx *c)
{
    const double *series = c->series;
    const double where = c->where;
    const int dim = c->dim, delay = c->delay, comp = c->comp;
    int64_t i, jd, start, stop;
    long j;
    int k, cut;
    double delta, xcut, *row;
    double time = 0.0, lasttime = 0.0;

    start = (int64_t)(comp - 1) * delay;
    /* poincare.c computes this bound in unsigned long, so a series shorter than
       the delay vector wraps it around and walks off the data; signed here, the
       loop simply does not run. */
    stop = c->length - (int64_t)(dim - comp) * delay - 1;

    for (i = start; i < stop; i++) {
        /* poincare.c writes the two directions out as separate loops. */
        cut = (c->dir == 0) ? (series[i] < where && series[i + 1] >= where)
                            : (series[i] > where && series[i + 1] <= where);
        if (!cut)
            continue;

        delta = (series[i] - where) / (series[i] - series[i + 1]);
        time = (double)i + delta;
        /* The first crossing only sets the clock: there is no interval yet. */
        if (lasttime > 0.0) {
            if ((row = next_row(c)) == NULL)
                return -1;
            k = 0;
            for (j = -(long)(comp - 1); j <= (long)(dim - comp); j++) {
                if (j != 0) {
                    jd = i + (int64_t)j * delay;
                    xcut = series[jd] + delta * (series[jd + 1] - series[jd]);
                    row[k++] = xcut;
                }
            }
            row[k] = time - lasttime;
        }
        lasttime = time;
    }

    return 0;
}

/* ------------------------------------------------------------------------ */

PyDoc_STRVAR(section_doc,
"section(series, dim, delay, comp, direction, where)\n"
"\n"
"Make a Poincare section of a single time series (TISEAN's ``poincare``).\n"
"\n"
"Parameters\n"
"----------\n"
"series : ndarray of float64\n"
"    1-D, C-contiguous input time series.\n"
"dim : int\n"
"    Embedding dimension (``poincare -m``).\n"
"delay : int\n"
"    Time delay of the embedding (``poincare -d``).\n"
"comp : int\n"
"    Component to cut, 1-based (``poincare -q``); must not exceed ``dim``.\n"
"direction : int\n"
"    Direction of the cut: 0 from below, 1 from above (``poincare -C``).\n"
"where : float or None\n"
"    Crossing level (``poincare -a``). ``None`` reproduces TISEAN's default,\n"
"    the mean of the series.\n"
"\n"
"Returns\n"
"-------\n"
"cuts : ndarray, shape (n_cuts, dim)\n"
"    One row per crossing after the first: the ``dim - 1`` coordinates of the\n"
"    delay vector at the crossing, then the time since the previous crossing.\n"
"where : float\n"
"    The crossing level actually used.\n");

static PyObject *py_section(PyObject *self, PyObject *args)
{
    PyObject *series_obj, *where_obj;
    PyArrayObject *series_arr = NULL;
    PyArrayObject *cuts_out = NULL;
    PyObject *result = NULL;
    poin_ctx c;
    npy_intp dims[2];
    long dim, delay, comp, direction;
    double av, var, h, mn, mx;
    int64_t i;
    int rc;

    if (!PyArg_ParseTuple(args, "OllllO", &series_obj, &dim, &delay, &comp,
                          &direction, &where_obj))
        return NULL;

    series_arr = (PyArrayObject *)PyArray_FROMANY(series_obj, NPY_DOUBLE, 1, 1,
                                                  NPY_ARRAY_C_CONTIGUOUS |
                                                  NPY_ARRAY_ALIGNED);
    if (series_arr == NULL)
        return NULL;

    memset(&c, 0, sizeof(c));
    c.series = (const double *)PyArray_DATA(series_arr);
    c.length = (int64_t)PyArray_SIZE(series_arr);
    c.dim = (int)dim;
    c.delay = (int)delay;
    c.comp = (int)comp;
    c.dir = (int)direction;

    if (dim < 2 || delay < 1 || comp < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "dim must be >= 2, delay >= 1, comp >= 1");
        goto fail;
    }
    if (comp > dim) {
        PyErr_SetString(PyExc_ValueError,
                        "component to cut is larger than the dimension");
        goto fail;
    }
    if (direction != 0 && direction != 1) {
        PyErr_SetString(PyExc_ValueError, "direction must be 0 or 1");
        goto fail;
    }
    if (c.length < 2) {
        PyErr_SetString(PyExc_ValueError, "the time series is too short");
        goto fail;
    }

    /* routines/variance.c: naive running sums, and a hard stop on a constant
       series. The mean it produces is poincare.c's default crossing level. */
    av = var = 0.0;
    for (i = 0; i < c.length; i++) {
        h = c.series[i];
        av += h;
        var += h * h;
    }
    av /= (double)c.length;
    var = sqrt(fabs(var / (double)c.length - av * av));
    if (var == 0.0) {
        PyErr_SetString(PyExc_ValueError, "the variance of the data is zero");
        goto fail;
    }

    mn = mx = c.series[0];
    for (i = 1; i < c.length; i++) {
        if (c.series[i] < mn) mn = c.series[i];
        if (c.series[i] > mx) mx = c.series[i];
    }

    if (where_obj != Py_None) {
        c.where = PyFloat_AsDouble(where_obj);
        if (PyErr_Occurred()) goto fail;
    } else {
        c.where = av;
    }
    if (c.where < mn || c.where > mx) {
        char msg[128];
        /* PyErr_Format has no %e, so build the message the way poincare.c does. */
        snprintf(msg, sizeof(msg),
                 "the crossing is outside the data interval [%e,%e]", mn, mx);
        PyErr_SetString(PyExc_ValueError, msg);
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    rc = poincare_run(&c);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_NoMemory();
        goto fail;
    }

    dims[0] = (npy_intp)c.nrows;
    dims[1] = (npy_intp)c.dim;
    cuts_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (cuts_out == NULL) goto fail;
    if (c.nrows > 0)
        memcpy(PyArray_DATA(cuts_out), c.rows,
               sizeof(double) * (size_t)(dims[0] * dims[1]));

    result = Py_BuildValue("Od", (PyObject *)cuts_out, c.where);

fail:
    Py_XDECREF(cuts_out);
    ctx_free(&c);
    Py_XDECREF(series_arr);
    return result;
}

static PyMethodDef PoincareMethods[] = {
    {"section", py_section, METH_VARARGS, section_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef poincare_module = {
    PyModuleDef_HEAD_INIT,
    "poincare",
    "Poincare section kernel of TISEAN 3.0.1's poincare program.",
    -1,
    PoincareMethods
};

PyMODINIT_FUNC PyInit_poincare(void)
{
    PyObject *module = PyModule_Create(&poincare_module);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}
