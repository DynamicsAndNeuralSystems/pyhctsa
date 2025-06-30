/* Performs fast detrended fluctuation analysis on a nonstationary input signal.

   Useage:
   Inputs
    x          - input signal: must be a row vector
   Optional inputs:
    intervals  - List of sample interval widths at each scale
                 (If not specified, then a binary subdivision is constructed)

   Outputs:
    intervals  - List of sample interval widths at each scale
    flucts     - List of fluctuation amplitudes at each scale

   (c) 2006 Max Little. If you use this code, please cite:
   M. Little, P. McSharry, I. Moroz, S. Roberts (2006),
   Nonlinear, biophysically-informed speech pathology detection
   in Proceedings of ICASSP 2006, IEEE Publishers: Toulouse, France.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define REAL double

/* Calculate accumulated sum signal */
static void cumulativeSum(
   unsigned long elements,
   REAL *x,
   REAL *y
)
{
   unsigned int i;
   REAL accum = 0.0f;
   for (i = 0; i < elements; i++)
   {
      accum += x[i];
      y[i] = accum;
   }
}

/* Calculate intervals if not specified */
static void calculateIntervals(
   unsigned long elements,
   unsigned long *N_scales,
   unsigned long **intervals
)
{
   unsigned long scales, subdivs;
   REAL idx_inc;
   long scale;

   scales = (unsigned long)(log10(elements) / log10(2.0));
   if (((REAL)(1 << (scales - 1))) > ((REAL)elements / 2.5f))
   {
      scales--;
   }
   *N_scales = scales;
   *intervals = (unsigned long *)calloc(scales, sizeof(unsigned long));
   for (scale = scales - 1; scale >= 0; scale--)
   {
      subdivs = 1 << scale;
      idx_inc = (REAL)elements / (REAL)subdivs;
      (*intervals)[scale] = (unsigned long)(idx_inc + 0.5f);
   }
}

/* Measure the fluctuations at each scale */
static void dfa(
   REAL *x,
   unsigned long elements,
   unsigned long *intervals,
   REAL *flucts,
   unsigned long N_scales
)
{
   unsigned long idx, i, start, end, iwidth, accum_idx;
   long scale;

   REAL Sy, Sxy;                   /* y and x-y components of normal equations */
   REAL Sx, Sxx;                   /* x-component of normal equations */
   REAL a, b;                      /* Straight-line fit parameters */
   REAL *trend;                    /* Trend vector */
   REAL diff, accum, delta;

   trend = (REAL *)calloc(elements, sizeof(REAL));

   for (scale = N_scales - 1; scale >= 0; scale--)
   {
      for (accum_idx = 0, idx = 0; idx < elements; idx += intervals[scale], accum_idx++)
      {
         start = idx;
         end = idx + intervals[scale] - 1;

         if (end >= elements)
         {
            for (i = start; i < elements; i++)
            {
               trend[i] = x[i];
            }
            break;
         }
         iwidth = end - start + 1;

         Sy = 0.0f;
         Sxy = 0.0f;
         for (i = start; i <= end; i++)
         {
            Sy += x[i];
            Sxy += x[i] * (REAL)i;
         }

         Sx = ((REAL)end + (REAL)start) * (REAL)iwidth / 2.0;
         Sxx = (REAL)iwidth * (2 * (REAL)end * (REAL)end + 2 * (REAL)start * (REAL)start +
                               2 * (REAL)start * (REAL)end + (REAL)end - (REAL)start) / 6.0;
         delta = (REAL)iwidth * Sxx - (Sx * Sx);

         b = (Sy * Sxx - Sx * Sxy) / delta;
         a = ((REAL)iwidth * Sxy - Sx * Sy) / delta;

         for (i = start; i <= end; i++)
         {
            trend[i] = a * (REAL)i + b;
         }
      }

      accum = 0.0f;
      for (i = 0; i < elements; i++)
      {
         diff = x[i] - trend[i];
         accum += diff * diff;
      }
      flucts[scale] = sqrt(accum / (REAL)elements);
   }

   free(trend);
}

void fastdfa_core(
    const double *x,
    unsigned long elements,
    unsigned long **intervals, // pointer-to-pointer: will be allocated and filled
    double *flucts,
    unsigned long *N_scales
)
{
    double *y_in;
    unsigned long *intervals_local = NULL;
    unsigned long n_scales_local, i;

    y_in = (double *)calloc(elements, sizeof(double));
    if (!y_in) return;
    cumulativeSum(elements, (double *)x, y_in);

    if (*intervals == NULL) {
        calculateIntervals(elements, &n_scales_local, &intervals_local);
        *N_scales = n_scales_local;
        *intervals = (unsigned long *)calloc(n_scales_local, sizeof(unsigned long));
        if (!*intervals) {
            free(intervals_local);
            free(y_in);
            return;
        }
        for (i = 0; i < n_scales_local; i++) {
            (*intervals)[i] = intervals_local[i];
        }
        free(intervals_local);
    } else {
        n_scales_local = *N_scales;
    }

    for (i = 0; i < n_scales_local; i++) {
        if (((*intervals)[i] > elements) || ((*intervals)[i] < 3)) {
            free(y_in);
            return;
        }
    }

    dfa(y_in, elements, *intervals, flucts, n_scales_local);

    free(y_in);
}

/* Python wrapper function */
static PyObject* py_fastdfa(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject *input_array = NULL;
    PyObject *intervals_obj = NULL;
    
    static char *kwlist[] = {"x", "intervals", NULL};
    
    // Parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", kwlist,
                                     &PyArray_Type, &input_array,
                                     &intervals_obj)) {
        return NULL;
    }
    
    // Convert input to contiguous double array
    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)input_array, 
                                                              NPY_DOUBLE, 
                                                              NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        return NULL;
    }
    
    // Check input array
    if (PyArray_NDIM(x_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional");
        Py_DECREF(x_array);
        return NULL;
    }
    
    unsigned long elements = (unsigned long)PyArray_DIM(x_array, 0);
    if (elements < 10) {
        PyErr_SetString(PyExc_ValueError, "Input signal must have at least 10 elements");
        Py_DECREF(x_array);
        return NULL;
    }
    
    double *x_data = (double*)PyArray_DATA(x_array);
    
    // Estimate maximum possible scales
    unsigned long max_scales = (unsigned long)(log10((double)elements) / log10(2.0)) + 2;
    
    // Allocate flucts array
    double *flucts = (double*)calloc(max_scales, sizeof(double));
    if (!flucts) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for fluctuations");
        Py_DECREF(x_array);
        return NULL;
    }
    
    // Initialize for automatic interval calculation
    unsigned long *intervals_ptr = NULL;
    unsigned long N_scales = 0;
    
    // Call the core function - it will allocate intervals internally
    fastdfa_core(x_data, elements, &intervals_ptr, flucts, &N_scales);
    
    // Clean up input array
    Py_DECREF(x_array);
    
    // Check if the function succeeded
    if (intervals_ptr == NULL || N_scales == 0) {
        free(flucts);
        PyErr_SetString(PyExc_RuntimeError, "FastDFA computation failed");
        return NULL;
    }
    
    // Create output numpy arrays
    npy_intp dims[1] = {(npy_intp)N_scales};
    
    PyObject *intervals_out = PyArray_SimpleNew(1, dims, NPY_ULONG);
    PyObject *flucts_out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    
    if (!intervals_out || !flucts_out) {
        Py_XDECREF(intervals_out);
        Py_XDECREF(flucts_out);
        free(intervals_ptr);
        free(flucts);
        return NULL;
    }
    
    // Copy data to output arrays
    unsigned long *intervals_out_data = (unsigned long*)PyArray_DATA((PyArrayObject*)intervals_out);
    double *flucts_out_data = (double*)PyArray_DATA((PyArrayObject*)flucts_out);
    
    memcpy(intervals_out_data, intervals_ptr, N_scales * sizeof(unsigned long));
    memcpy(flucts_out_data, flucts, N_scales * sizeof(double));
    
    // Clean up C arrays - the function allocated intervals_ptr for us
    free(intervals_ptr);
    free(flucts);
    
    // Return tuple (intervals, flucts)
    return PyTuple_Pack(2, intervals_out, flucts_out);
}

/* Method definitions */
static PyMethodDef FastDFAMethods[] = {
    {"fastdfa", (PyCFunction)py_fastdfa, METH_VARARGS | METH_KEYWORDS,
     "Perform fast detrended fluctuation analysis\n\n"
     "Parameters:\n"
     "  x : array_like\n"
     "      Input signal (1D array)\n"
     "  intervals : array_like, optional\n"
     "      List of sample interval widths at each scale (not yet implemented)\n"
     "      If not specified, binary subdivision is used\n\n"
     "Returns:\n"
     "  intervals : ndarray\n"
     "      Sample interval widths at each scale\n"
     "  flucts : ndarray\n"
     "      Fluctuation amplitudes at each scale\n"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef fastdfa_module = {
    PyModuleDef_HEAD_INIT,
    "fastdfa",
    "Fast Detrended Fluctuation Analysis module",
    -1,
    FastDFAMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_fastdfa(void) {
    PyObject *module;
    
    module = PyModule_Create(&fastdfa_module);
    if (module == NULL)
        return NULL;
    
    // Import numpy
    import_array();
    if (PyErr_Occurred())
        return NULL;
    
    return module;
}
