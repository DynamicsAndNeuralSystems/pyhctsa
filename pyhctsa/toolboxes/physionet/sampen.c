/* file: sampen.c	Doug Lake	2 August 2002
			Last revised:	1 November 2004 (by george@mit.edu) 1.2
-------------------------------------------------------------------------------
sampen: calculate Sample Entropy
Copyright (C) 2002-2004 Doug Lake

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place - Suite 330, Boston, MA 02111-1307, USA.  You may also view the agreement
at http://www.fsf.org/copyleft/gpl.html.

You may contact the author via electronic mail (dlake@virginia.edu).  For
updates to this software, please visit PhysioNet (http://www.physionet.org/).

_______________________________________________________________________________

Revision history:
  1.0 (2 August 2002, Doug Lake)	Original version
  1.1 (6 January 2004, George Moody)	Removed limits on input series length
  1.2 (1 November 2004, George Moody)	Merged bug fixes from DL (normalize
					by standard deviation, detect and
					avoid divide by zero); changed code to
					use double precision, to avoid loss of
					precision for small m and large N

Compile this program using any standard C compiler, linking with the standard C
math library.  For example, if your compiler is gcc, use:
    gcc -o sampen -O sampen.c -lm

For brief instructions, use the '-h' option:
    sampen -h

Additional information is available at:
    http://www.physionet.org/physiotools/sampen/.

*/
// Computes Sample Entropy for m = 0..M-1 and writes results to result[0..M-1]
// y: input array (length n)
// M: maximum embedding dimension (calculates for m=0..M-1)
// r: similarity threshold
// n: length of y
// result: output array of length M (must be allocated by caller)
// countA, countB: optional (may be NULL) output arrays of length M for the
//   template-match counts, SampEn(m) = -log(countA[m]/countB[m])
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* The extension is built with -ffast-math, which lets the compiler assume that
   NaN and Inf never occur; values that must be NaN/Inf are therefore written
   from their IEEE bit patterns. */
static double ieee_value(uint64_t bits)
{
    double d;
    memcpy(&d, &bits, sizeof d);
    return d;
}

/* Core sampen calculation function */
static void sampen_core(double *y, int M, double r, int n, double *sampEnt,
                        double *countA, double *countB)
{
    double *p = NULL;
    long *run = NULL, *lastrun = NULL;
    double *A = NULL, *B = NULL, nA, nB;
    int M1, j, nj, jj, m;
    int i;
    double y1;
    
    /* Allocate memory */
    M++;
    if ((run = (long *) calloc(n, sizeof(long))) == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for run array");
        return;
    }
    if ((lastrun = (long *) calloc(n, sizeof(long))) == NULL) {
        free(run);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for lastrun array");
        return;
    }
    if ((A = (double *) calloc(M, sizeof(double))) == NULL) {
        free(run);
        free(lastrun);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for A array");
        return;
    }
    if ((B = (double *) calloc(M, sizeof(double))) == NULL) {
        free(run);
        free(lastrun);
        free(A);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for B array");
        return;
    }
    if ((p = (double *) calloc(M, sizeof(double))) == NULL) {
        free(run);
        free(lastrun);
        free(A);
        free(B);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for p array");
        return;
    }
    
    /* Main calculation loop */
    for (i = 0; i < n - 1; i++) {
        nj = n - i - 1;
        y1 = y[i];
        for (jj = 0; jj < nj; jj++) {
            j = jj + i + 1;
            if (((y[j] - y1) < r) && ((y1 - y[j]) < r)) {
                run[jj] = lastrun[jj] + 1;
                M1 = M < run[jj] ? M : run[jj];
                for (m = 0; m < M1; m++) {
                    A[m]++;
                    if (j < n - 1)
                        B[m]++;
                }
            }
            else
                run[jj] = 0;
        }
        for (j = 0; j < nj; j++)
            lastrun[j] = run[j];
    }
    
    /* Calculate sample entropy, as hctsa's sampen_mex.c: SampEn(m) = -log(A/B),
       from the count B of pairs of templates of length m that match (all
       n(n-1)/2 pairs for m = 0) and the count A of those that still match at
       length m+1. It is NaN when B = 0 (no matches to form the ratio of) and
       when A = 0, B > 0: the estimate is then infinite, i.e., the data cannot
       support it at this m, r and n. */
    for (m = 0; m < M-1; m++) {
        nA = A[m];
        nB = (m == 0) ? (double) n * (n - 1) / 2 : B[m - 1];
        if (countA != NULL) {
            countA[m] = nA;
            countB[m] = nB;
        }
        if ((nA > 0) && (nB > 0)) {
            p[m] = nA / nB;
            sampEnt[m] = -log(p[m]);
        } else {
            sampEnt[m] = ieee_value(0x7ff8000000000000ULL); /* NaN */
        }
    }
    
    /* Clean up */
    free(A);
    free(B);
    free(p);
    free(run);
    free(lastrun);
}

/* Python wrapper function */
static PyObject *sampen_calculate(PyObject *self, PyObject *args)
{
    PyArrayObject *input_array, *output_array;
    PyArrayObject *count_a = NULL, *count_b = NULL;
    double *data, *result;
    int M, n;
    double r;
    int return_counts = 0;
    
    /* Parse arguments: (array, M, r[, return_counts]) */
    if (!PyArg_ParseTuple(args, "O!id|p", &PyArray_Type, &input_array, &M, &r,
                          &return_counts)) {
        return NULL;
    }
    
    /* Validate input array */
    if (PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional");
        return NULL;
    }
    
    if (PyArray_TYPE(input_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Input array must be of type float64");
        return NULL;
    }

    if (!PyArray_IS_C_CONTIGUOUS(input_array)) {
        PyErr_SetString(PyExc_ValueError, "Input array must be C-contiguous");
        return NULL;
    }
    
    if (M < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    
    if (r <= 0) {
        PyErr_SetString(PyExc_ValueError, "r must be positive");
        return NULL;
    }
    
    n = PyArray_DIM(input_array, 0);
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "Input array must have at least 2 elements");
        return NULL;
    }
    
    /* Get data pointer */
    data = (double*)PyArray_DATA(input_array);
    
    /* Create output array */
    npy_intp dims[1] = {M + 1};
    output_array = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output_array == NULL) {
        return NULL;
    }
    
    result = (double*)PyArray_DATA(output_array);

    /* Template-match counts (orders 0..M-1), if requested */
    if (return_counts) {
        npy_intp cdims[1] = {M};
        count_a = (PyArrayObject*)PyArray_ZEROS(1, cdims, NPY_DOUBLE, 0);
        count_b = (PyArrayObject*)PyArray_ZEROS(1, cdims, NPY_DOUBLE, 0);
        if (count_a == NULL || count_b == NULL) {
            Py_XDECREF(count_a);
            Py_XDECREF(count_b);
            Py_DECREF(output_array);
            return NULL;
        }
    }
    
    /* Call core function */
    sampen_core(data, M, r, n, result,
                return_counts ? (double*)PyArray_DATA(count_a) : NULL,
                return_counts ? (double*)PyArray_DATA(count_b) : NULL);
    
    /* Check for errors */
    if (PyErr_Occurred()) {
        Py_DECREF(output_array);
        Py_XDECREF(count_a);
        Py_XDECREF(count_b);
        return NULL;
    }

    if (return_counts) {
        /* Steal the references */
        return Py_BuildValue("NNN", (PyObject*)output_array, (PyObject*)count_a,
                             (PyObject*)count_b);
    }
    
    return (PyObject*)output_array;
}

/* Method definitions */
static PyMethodDef SampenMethods[] = {
    {"calculate", sampen_calculate, METH_VARARGS,
     "calculate(y, M, r[, return_counts]): sample entropy SampEn(m) for m = 0..M-1 "
     "(in the first M entries of a length-(M+1) array); with return_counts, also the "
     "template-match counts A and B (length M), SampEn(m) = -log(A[m]/B[m])"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef sampenmodule = {
    PyModuleDef_HEAD_INIT,
    "sampen",
    "Sample Entropy calculation module",
    -1,
    SampenMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_sampen(void)
{
    PyObject *module;
    
    /* Import numpy */
    import_array();
    
    module = PyModule_Create(&sampenmodule);
    if (module == NULL) {
        return NULL;
    }
    
    return module;
}
