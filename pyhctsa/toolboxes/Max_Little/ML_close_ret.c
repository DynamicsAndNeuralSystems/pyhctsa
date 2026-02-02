/* Close returns code by M. Little (c) 2006 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define REAL double

/* Create embedded version of given sequence */
static void embedSeries(
    unsigned long embedDims,
    unsigned long embedDelay,
    unsigned long embedElements,
    const REAL *x,
    REAL *y
)
{
    unsigned int i, d, inputDelay;
    for (d = 0; d < embedDims; d++) {
        inputDelay = (embedDims - d - 1) * embedDelay;
        for (i = 0; i < embedElements; i++) {
            y[i * embedDims + d] = x[i + inputDelay];
        }
    }
}

/* Search for first close returns in the embedded sequence */
static void findCloseReturns(
    const REAL *x,
    REAL eta,
    unsigned long embedElements,
    unsigned long embedDims,
    unsigned long *closeRets
)
{
    REAL eta2 = eta * eta;
    REAL diff, dist2;
    unsigned long i, j, d, timeDiff, etaFlag;
    
    for (i = 0; i < embedElements; i++) {
        closeRets[i] = 0;
    }
    
    for (i = 0; i < embedElements; i++) {
        j = i + 1;
        etaFlag = 0;
        
        while ((j < embedElements) && !etaFlag) {
            dist2 = 0.0f;
            for (d = 0; d < embedDims; d++) {
                diff = x[i * embedDims + d] - x[j * embedDims + d];
                dist2 += diff * diff;
            }
            if (dist2 > eta2) {
                etaFlag = 1;
            }
            j++;
        }
        
        etaFlag = 0;
        while ((j < embedElements) && !etaFlag) {
            dist2 = 0.0f;
            for (d = 0; d < embedDims; d++) {
                diff = x[i * embedDims + d] - x[j * embedDims + d];
                dist2 += diff * diff;
            }
            if (dist2 <= eta2) {
                timeDiff = j - i;
                closeRets[timeDiff]++;
                etaFlag = 1;
            }
            j++;
        }
    }
}

/* Python wrapper function */
static PyObject *close_returns_analysis(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *input_array = NULL;
    unsigned long embedDims;
    unsigned long embedDelay;
    double eta;
    
    static char *kwlist[] = {"x", "embed_dims", "embed_delay", "eta", NULL};
    
    // Parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!kkd", kwlist,
                                     &PyArray_Type, &input_array,
                                     &embedDims, &embedDelay, &eta)) {
        return NULL;
    }
    
    // Check input array
    if (PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional");
        return NULL;
    }
    
    if (PyArray_TYPE(input_array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Input array must be float64");
        return NULL;
    }
    
    // Get input data
    npy_intp vectorElements = PyArray_SIZE(input_array);
    double *x_data = (double *)PyArray_DATA(input_array);
    
    // Validate parameters
    if (embedDims < 1) {
        PyErr_SetString(PyExc_ValueError, "embed_dims must be >= 1");
        return NULL;
    }
    
    if (embedDelay < 1) {
        PyErr_SetString(PyExc_ValueError, "embed_delay must be >= 1");
        return NULL;
    }
    
    if (eta <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "eta must be > 0");
        return NULL;
    }
    
    if (vectorElements < embedDims * embedDelay) {
        PyErr_SetString(PyExc_ValueError, 
                       "Input too short for given embedding parameters");
        return NULL;
    }
    
    // Calculate embedding parameters
    unsigned long embedElements = vectorElements - ((embedDims - 1) * embedDelay);
    
    // Allocate memory for embedded sequence
    REAL *embedSequence = (REAL *)calloc(embedElements * embedDims, sizeof(REAL));
    if (!embedSequence) {
        return PyErr_NoMemory();
    }
    
    // Allocate memory for close returns histogram
    unsigned long *closeRets = (unsigned long *)calloc(embedElements, sizeof(unsigned long));
    if (!closeRets) {
        free(embedSequence);
        return PyErr_NoMemory();
    }
    
    // Perform the analysis
    embedSeries(embedDims, embedDelay, embedElements, x_data, embedSequence);
    findCloseReturns(embedSequence, eta, embedElements, embedDims, closeRets);
    
    // Create output NumPy array
    npy_intp dims[1] = {embedElements};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    
    if (!result) {
        free(embedSequence);
        free(closeRets);
        return PyErr_NoMemory();
    }
    
    // Copy data to output array
    unsigned long *result_data = (unsigned long *)PyArray_DATA(result);
    for (unsigned long i = 0; i < embedElements; i++) {
        result_data[i] = closeRets[i];
    }
    
    // Clean up
    free(embedSequence);
    free(closeRets);
    
    return (PyObject *)result;
}

/* Method definitions */
static PyMethodDef CloseReturnsMethods[] = {
    {"close_returns", (PyCFunction)close_returns_analysis, METH_VARARGS | METH_KEYWORDS,
     "Compute close returns analysis.\n\n"
     "Parameters:\n"
     "  x : array_like\n"
     "      Input time series (1D array of float64)\n"
     "  embed_dims : int\n"
     "      Embedding dimension (must be >= 1)\n"
     "  embed_delay : int\n"
     "      Embedding delay (must be >= 1)\n"
     "  eta : float\n"
     "      Close return distance threshold (must be > 0)\n\n"
     "Returns:\n"
     "  close_returns : ndarray\n"
     "      Close return time histogram (length = embed_elements)\n"
     "      where embed_elements = len(x) - (embed_dims - 1) * embed_delay"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef close_returns_module = {
    PyModuleDef_HEAD_INIT,
    "close_returns",
    "Close Returns Analysis module by M. Little (c) 2006",
    -1,
    CloseReturnsMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_close_returns(void)
{
    PyObject *m;
    
    // Initialize NumPy
    import_array();
    
    m = PyModule_Create(&close_returns_module);
    if (m == NULL)
        return NULL;
    
    return m;
}
