/*=================================================================
 *
 * SHANNON.C
 * returns the approximate Shannon Entropy, for a n-bin
 * encoding of the time series x at depth d
 * I.e. -\sum Plog(P)
 * where P=(x_i,x_{i+1},....x_{i+d}) and the sum is
 * over all trajectories P
 *
 * The calling syntax is:
 *
 * ent= shannon_entropy(x,n,d)
 *
 * Original MATLAB MEX implementation - Michael Small
 * Modified to standalone C function
 *
 *=================================================================*/
/* $Revision: 1.5 $ */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int compare(const void *arg1, const void *arg2)
     /* compare two doubles and return arg1-arg2 */
{
  return( *(int *)arg1 - *(int *)arg2 );
}

/**
 * Calculate Shannon entropy of a time series using n-bin encoding at depth d
 * 
 * @param data: Input time series data
 * @param length: Length of the data array
 * @param bins: Number of bins for encoding (typically 2-10)
 * @param depth: Embedding depth (typically 1-5)
 * @return: Shannon entropy value, or -1.0 on error
 */
double shannon_entropy(double *data, int length, int bins, int depth) {
    if (!data || length <= 0 || bins <= 1 || depth <= 0 || depth >= length) {
        return -1.0;
    }
    
    // Allocate memory
    unsigned short *symbols = calloc(length, sizeof(unsigned short));
    double *sorted = calloc(length, sizeof(double));
    float *thresholds = calloc(bins - 1, sizeof(float));
    unsigned long *tally = calloc((unsigned long)pow(bins, depth), sizeof(unsigned long));
    
    if (!symbols || !sorted || !thresholds || !tally) {
        free(symbols);
        free(sorted);
        free(thresholds);
        free(tally);
        return -1.0;
    }
    
    // Copy and sort data to determine thresholds
    for (int i = 0; i < length; i++) {
        sorted[i] = data[i];
    }
    qsort(sorted, length, sizeof(double), compare);
    
    // Extract percentile thresholds for n-bin encoding
    for (int i = 1; i < bins; i++) {
        thresholds[i-1] = (float)sorted[i * length / bins];  // Cast to float like original
    }
    
    // Encode the data into symbols
    for (int i = 0; i < length; i++) {
        symbols[i] = 0;
        for (int j = 0; j < bins - 1; j++) {
            if (data[i] >= thresholds[j]) {
                symbols[i]++;
            }
        }
    }
    
    // Count symbol sequences of given depth
    int valid_length = length - depth + 1;
    for (int k = 0; k < valid_length; k++) {
        unsigned long pattern = 0;
        for (int i = 0; i < depth; i++) {
            pattern += (unsigned long)pow(bins, i) * symbols[k + i];
        }
        tally[pattern]++;
    }
    
    // Calculate Shannon entropy: -sum(P * log(P))
    double entropy = 0.0;
    unsigned long total_patterns = (unsigned long)pow(bins, depth);
    
    for (unsigned long i = 0; i < total_patterns; i++) {
        if (tally[i] > 0) {
            double prob = (double)tally[i] / valid_length;
            entropy -= prob * log(prob);
        }
    }
    
    // Cleanup
    free(symbols);
    free(sorted);
    free(thresholds);
    free(tally);
    
    return entropy;
}

/* Python wrapper function */
static PyObject* py_shannon_entropy(PyObject* self, PyObject* args) {
    PyArrayObject *data_array;
    int bins = 2;     // default values
    int depth = 3;
    
    // Parse arguments: data array is required, bins and depth are optional
    if (!PyArg_ParseTuple(args, "O!|ii", &PyArray_Type, &data_array, &bins, &depth)) {
        return NULL;
    }
    
    // Check that input is a 1D array of doubles
    if (PyArray_NDIM(data_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input must be a 1D array");
        return NULL;
    }
    
    if (PyArray_TYPE(data_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Input array must be of type float64 (double)");
        return NULL;
    }
    
    // Get data pointer and length
    double *data = (double*)PyArray_DATA(data_array);
    int length = (int)PyArray_SIZE(data_array);
    
    // Validate parameters
    if (length <= 0) {
        PyErr_SetString(PyExc_ValueError, "Input array must not be empty");
        return NULL;
    }
    
    if (bins <= 1) {
        PyErr_SetString(PyExc_ValueError, "Number of bins must be greater than 1");
        return NULL;
    }
    
    if (depth <= 0 || depth >= length) {
        PyErr_SetString(PyExc_ValueError, "Depth must be positive and less than array length");
        return NULL;
    }
    
    // Calculate entropy
    double result = shannon_entropy(data, length, bins, depth);
    
    if (result < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error calculating Shannon entropy");
        return NULL;
    }
    
    return PyFloat_FromDouble(result);
}

/* Method definition */
static PyMethodDef ShannonMethods[] = {
    {"entropy", py_shannon_entropy, METH_VARARGS, 
     "Calculate Shannon entropy of a time series.\n\n"
     "Parameters:\n"
     "  data (array): 1D numpy array of float64 values\n"
     "  bins (int, optional): Number of bins for encoding (default: 2)\n"
     "  depth (int, optional): Embedding depth (default: 3)\n\n"
     "Returns:\n"
     "  float: Shannon entropy value\n\n"
     "Example:\n"
     "  >>> import numpy as np\n"
     "  >>> import shannon\n"
     "  >>> data = np.random.randn(1000)\n"
     "  >>> ent = shannon.entropy(data, bins=4, depth=3)\n"
    },
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef shannon_module = {
    PyModuleDef_HEAD_INIT,
    "shannon",
    "Shannon entropy calculation for time series data",
    -1,
    ShannonMethods
};

/* Module initialisation */
PyMODINIT_FUNC PyInit_shannon(void) {
    PyObject *module = PyModule_Create(&shannon_module);
    if (module == NULL) {
        return NULL;
    }
    
    // Initialize numpy
    import_array();
    
    return module;
}
