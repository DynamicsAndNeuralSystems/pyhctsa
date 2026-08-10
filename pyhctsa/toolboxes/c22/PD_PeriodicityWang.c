//
//  PD_PeriodicityWang.c
//  C_polished
//
//  Created by Carl Henning Lubba on 28/09/2018.
//  Copyright © 2018 Carl Henning Lubba. All rights reserved.
//

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#include "PD_PeriodicityWang.h"
#include "splinefit.h"
#include "stats.h"

#ifndef ACF_B
#define ACF_B 4
#endif

static void PD_acf_block(const double * restrict y, const int n, const int tLo,
                         const int tHi, double * restrict acf)
{
    int t = tLo;

    for (; t + (ACF_B-1) <= tHi; t += ACF_B) {
        const int tmax = t + (ACF_B-1);

        int iEnd = n - tmax;
        if (iEnd < 0) iEnd = 0;

        double a[ACF_B];
        for (int k = 0; k < ACF_B; k++) a[k] = 0.0;

        for (int i = 0; i < iEnd; i++) {
            const double v = y[i];
            const double * restrict w = y + i + t;
            for (int k = 0; k < ACF_B; k++) a[k] += v * w[k];
        }

        for (int k = 0; k < ACF_B; k++) {
            const int tau = t + k;
            const int m   = n - tau;
            for (int i = iEnd; i < m; i++) a[k] += y[i] * y[i+tau];
            acf[tau-1] = a[k] / m;
        }
    }

    for (; t <= tHi; t++) {
        const int m = n - t;
        double acc = 0.0;
        for (int i = 0; i < m; i++) acc += y[i] * y[i+t];
        acf[t-1] = acc / m;
    }
}

PD_PeriodicityWang_Results PD_PeriodicityWang(const double * y, const int size){
    
    // Initialize results structure with default value 1 (matching MATLAB)
    PD_PeriodicityWang_Results results = {1, 1, 1, 1, 1, 1, 1};
    
    // NaN check
    for(int i = 0; i < size; i++) {
        if(isnan(y[i])) {
            return results;
        }
    }
    
    // Define thresholds (matching MATLAB)
    const int numThresholds = 7;
    double ths[7];
    ths[0] = 0.0;
    ths[1] = 0.01;
    ths[2] = 0.1;
    ths[3] = 0.2;
    ths[4] = 1.0/sqrt((double)size);
    ths[5] = 5.0/sqrt((double)size);
    ths[6] = 10.0/sqrt((double)size);
    
    double * ySpline = malloc(size * sizeof(double));
    // fit a spline with 3 nodes to the data
    splinefit(y, size, ySpline);

    // subtract spline from data to remove trend
    double * ySub = malloc(size * sizeof(double));
    for(int i = 0; i < size; i++){
        ySub[i] = y[i] - ySpline[i];
    }
    free(ySpline);

    const int acmax = (int)ceil((double)size/3);
    double * acf = malloc(acmax*sizeof(double));
    int * troughs = malloc(acmax * sizeof(int));

    // Array to store results for each threshold
    int theFreqs[7];
    int resolved[7];
    for(int k = 0; k < numThresholds; k++) {
        theFreqs[k] = 1; // default value matching MATLAB
        resolved[k] = 0;
    }
    int nResolved = 0;

    int nTroughs = 0;
    int jT    = -1;
    int have  = 0;
    int nextI = 1;

    const int CHUNK = 256;

    while(nResolved < numThresholds && have < acmax){

        int upto = have + CHUNK;
        if(upto > acmax) upto = acmax;

        PD_acf_block(ySub, size, have+1, upto, acf);
        have = upto;

        for(int i = nextI; i <= have-2; i++){

            const double slopeIn  = acf[i] - acf[i-1];
            const double slopeOut = acf[i+1] - acf[i];

            if(slopeIn < 0 && slopeOut > 0) {
                troughs[nTroughs] = i + 1; // +1 to match MATLAB 1-based indexing
                nTroughs += 1;
            }
            else if(slopeIn > 0 && slopeOut < 0) {
                const int iPeak = i + 1; // +1 to match MATLAB 1-based indexing
                const double thePeak = acf[iPeak-1];

                // (c) peak corresponds to positive correlation
                if(thePeak < 0) {
                    continue;
                }

                // find trough before this peak
                while(jT+1 < nTroughs && troughs[jT+1] < iPeak) jT++;
                if(jT == -1) {
                    continue; // no trough before this peak
                }

                const double theTrough = acf[troughs[jT]-1];
                const double diff = thePeak - theTrough;

                // (a) should be implicit - there's a trough before it
                // (b) difference between peak and trough is at least threshold
                for(int k = 0; k < numThresholds; k++){
                    if(!resolved[k] && diff >= ths[k]){
                        // use this frequency that first fulfils all conditions
                        theFreqs[k] = iPeak;
                        resolved[k] = 1;
                        nResolved += 1;
                    }
                }

                if(nResolved == numThresholds) break;
            }
        }

        if(have-1 > nextI) nextI = have-1;
    }

    // Assign results to structure
    results.th1 = theFreqs[0];
    results.th2 = theFreqs[1];
    results.th3 = theFreqs[2];
    results.th4 = theFreqs[3];
    results.th5 = theFreqs[4];
    results.th6 = theFreqs[5];
    results.th7 = theFreqs[6];
    
    free(ySub);
    free(acf);
    free(troughs);

    return results;
}

// Python wrapper function
static PyObject* periodicity_wang_wrapper(PyObject* self, PyObject* args) {
    PyArrayObject* input_array;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array)) {
        return NULL;
    }
    
    // Check array properties
    if (PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input must be 1-dimensional");
        return NULL;
    }
    
    if (PyArray_TYPE(input_array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Input must be float64 array");
        return NULL;
    }
    
    // Get array data
    double* data = (double*)PyArray_DATA(input_array);
    npy_intp size_npy = PyArray_SIZE(input_array);
    
    // Check size constraints
    if (size_npy <= 0) {
        PyErr_SetString(PyExc_ValueError, "Array cannot be empty");
        return NULL;
    }
    
    if (size_npy > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "Array too large");
        return NULL;
    }
    
    int size = (int)size_npy;
    
    // Call your existing C function
    PD_PeriodicityWang_Results result = PD_PeriodicityWang(data, size);
    
    // Create Python dictionary to return
    PyObject* dict = PyDict_New();
    if (!dict) {
        return PyErr_NoMemory();
    }
    
    // Helper macro for adding items to dictionary
    #define ADD_TO_DICT(key, value) do { \
        PyObject* py_val = PyLong_FromLong(value); \
        if (!py_val) { \
            Py_DECREF(dict); \
            return NULL; \
        } \
        if (PyDict_SetItemString(dict, key, py_val) < 0) { \
            Py_DECREF(py_val); \
            Py_DECREF(dict); \
            return NULL; \
        } \
        Py_DECREF(py_val); \
    } while(0)
    
    // Add results to dictionary
    ADD_TO_DICT("th1", result.th1);
    ADD_TO_DICT("th2", result.th2);
    ADD_TO_DICT("th3", result.th3);
    ADD_TO_DICT("th4", result.th4);
    ADD_TO_DICT("th5", result.th5);
    ADD_TO_DICT("th6", result.th6);
    ADD_TO_DICT("th7", result.th7);
    
    #undef ADD_TO_DICT
    
    return dict;
}

// Method definitions
static PyMethodDef module_methods[] = {
    {"periodicity_wang_wrapper", periodicity_wang_wrapper, METH_VARARGS, 
     "Compute periodicity using Wang method for multiple thresholds"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "PD_PeriodicityWang",
    "Periodicity Wang C extension for highly comparative time-series analysis",
    -1,
    module_methods
};

// Module initialization
PyMODINIT_FUNC PyInit_PD_PeriodicityWang(void) {
    PyObject* module;
    
    // Initialize NumPy
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    // Create module
    module = PyModule_Create(&module_definition);
    if (!module) {
        return NULL;
    }
    
    return module;
}
