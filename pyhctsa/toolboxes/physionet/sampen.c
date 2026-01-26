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
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>

/* Core sampen calculation function - unchanged from original */
static void sampen_core(double *y, int M, double r, int n, double *sampEnt)
{
    double *p = NULL;
    long *run = NULL, *lastrun = NULL, N;
    double *A = NULL, *B = NULL;
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
    
    /* Calculate sample entropy */
    N = (long) (n * (n - 1) / 2);
    if (N > 0) {
        p[0] = A[0] / N;
        sampEnt[0] = (p[0] > 0) ? -log(p[0]) : 0;
        for (m = 1; m < M-1; m++) {
            if (B[m - 1] > 0) {
                p[m] = A[m] / B[m - 1];
                sampEnt[m] = (p[m] > 0) ? -log(p[m]) : 0;
            } else {
                sampEnt[m] = 0;
            }
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
    double *data, *result;
    int M, n;
    double r;
    
    /* Parse arguments: (array, M, r) */
    if (!PyArg_ParseTuple(args, "O!id", &PyArray_Type, &input_array, &M, &r)) {
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
    
    /* Call core function */
    sampen_core(data, M, r, n, result);
    
    /* Check for errors */
    if (PyErr_Occurred()) {
        Py_DECREF(output_array);
        return NULL;
    }
    
    return (PyObject*)output_array;
}

/* Method definitions */
static PyMethodDef SampenMethods[] = {
    {"calculate", sampen_calculate, METH_VARARGS,
     "Calculate sample entropy for given data, M, and r parameters"},
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
