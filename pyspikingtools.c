#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "computeSpiking.h"
#include <math.h>


/* module definition */
static char module_name[] = "spikingtools";
static char module_docstring[] = "This module provides an interface for calculating spiking block operations";

/* function definitions */
static char func1_name[] = "SCIF";
static char func1_docstring[] = "Computes SCIF(Spiking Continuous Independent Feedback) Spiking block operations";
static char func2_name[] = "SCIF_get_h";
static char func2_docstring[] = "Computes SCIF and get h, which is the membrane potential added with feedback";
static char func3_name[] = "SCIF_gain";
static char func3_docstring[] = "Computes the instantaneous gain of the SCIF model";
static char func4_name[] = "SCIEF";
static char func4_docstring[] = "Computes the SCIEF model";
static char func5_name[] = "SCIEF_get_h";
static char func5_docstring[] = "Computes the SCIEF model and get h";
static char func6_name[] = "SCIEF_gain";
static char func6_docstring[] = "Computes the instantaneous gain of the SCIEF model";
static char func7_name[] = "SDF";
static char func7_docstring[] = "Computes the SDF model";
static char func8_name[] = "SCIF2";
static char func8_docstring[] = "Computes SCIF(Spiking Continuous Independent Feedback) Spiking block operations";
static char func9_name[] = "SCIF2_get_h";
static char func9_docstring[] = "Computes SCIF and get h, which is the membrane potential added with feedback";
static char func10_name[] = "SCIF2_gain";
static char func10_docstring[] = "Computes the instantaneous gain of the SCIF2 model";
static char func11_name[] = "SC1DF";
static char func11_docstring[] = "Computes SC1DF(Spiking Continuous 1 Dimension Feedback) Spiking block operations";
static char func12_name[] = "SC1DF_get_h";
static char func12_docstring[] = "Computes SC1DF and get h, which is the membrane potential added with feedback";
static char func13_name[] = "SC1DF_gain";
static char func13_docstring[] = "Computes the instantaneous gain of the SC1DF model";
static char func14_name[] = "SC1DF_get_m";
static char func14_docstring[] = "Computes SC1DF and get m, which is the weighted sum of membrane potential and its derivative, added with feedback";

/* PyObject functions declarations */
static PyObject *py_SCIF(PyObject *self, PyObject *args);
static PyObject *py_SCIF_get_h(PyObject *self, PyObject *args);
static PyObject *py_SCIF_gain(PyObject *self, PyObject *args);
static PyObject *py_SCIEF(PyObject *self, PyObject *args);
static PyObject *py_SCIEF_get_h(PyObject *self, PyObject *args);
static PyObject *py_SCIEF_gain(PyObject *self, PyObject *args);
static PyObject *py_SDF(PyObject *self, PyObject *args);
static PyObject *py_SCIF2(PyObject *self, PyObject *args);
static PyObject *py_SCIF2_get_h(PyObject *self, PyObject *args);
static PyObject *py_SCIF2_gain(PyObject *self, PyObject *args);
static PyObject *py_SC1DF(PyObject *self, PyObject *args);
static PyObject *py_SC1DF_get_h(PyObject *self, PyObject *args);
static PyObject *py_SC1DF_gain(PyObject *self, PyObject *args);
static PyObject *py_SC1DF_get_m(PyObject *self, PyObject *args);

double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin); 
double **ptrvector(long n); 
void free_Carrayptrs(double **v); 
int  not_doublematrix(PyArrayObject *mat); 
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin); 


/* module method table */
static PyMethodDef spikingtoolsMethods[] = {
    {func1_name, py_SCIF, METH_VARARGS, func1_docstring},
    {func2_name, py_SCIF_get_h, METH_VARARGS, func2_docstring},
    {func3_name, py_SCIF_gain, METH_VARARGS, func3_docstring},
    {func4_name, py_SCIEF, METH_VARARGS, func4_docstring},
    {func5_name, py_SCIEF_get_h, METH_VARARGS, func5_docstring},
    {func6_name, py_SCIEF_gain, METH_VARARGS, func6_docstring},
    {func7_name, py_SDF, METH_VARARGS, func7_docstring},
    {func8_name, py_SCIF2, METH_VARARGS, func8_docstring},
    {func9_name, py_SCIF2_get_h, METH_VARARGS, func9_docstring},
    {func10_name, py_SCIF2_gain, METH_VARARGS, func10_docstring},
    {func11_name, py_SC1DF, METH_VARARGS, func11_docstring},
    {func12_name, py_SC1DF_get_h, METH_VARARGS, func12_docstring},
    {func13_name, py_SC1DF_gain, METH_VARARGS, func13_docstring},
    {func14_name, py_SC1DF_get_m, METH_VARARGS, func14_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spikingtoolsModule = {
    PyModuleDef_HEAD_INIT,
    module_name,    /* name of module */
    module_docstring,//"A sample module",  /* doc string, may be NULL */
    -1, /* size of per-interpreter state of the module, 
           or -1 if the module keeps state in global variables */
    spikingtoolsMethods   /* methods table */
};

PyMODINIT_FUNC PyInit_spikingtools(void){

    PyObject *m;
    m = PyModule_Create(&spikingtoolsModule);

    if (m == NULL)
        return NULL;
    return m;
}


/*
 * SCIF
 */
static PyObject *py_SCIF(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r;
    double *cp, *cv, *cdv, *cr;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    
    SCIF(cp, cv, cdv, cr, N);


    return PyArray_Return(r);

}

static PyObject *py_SCIF_get_h(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r, *h;
    double *cp, *cv, *cdv, *cr, *ch;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    ch = pyvector_to_Carrayptrs(h);
    
    SCIF_get_h(cp, cv, cdv, cr, ch, N);


    return PyArray_Return(h);

}


static PyObject *py_SCIF_gain(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r, *gain;
    double *cp, *cv, *cdv, *cr, *cgain;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    gain = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    cgain = pyvector_to_Carrayptrs(gain);
    
    SCIF_gain(cp, cv, cdv, cr, cgain, N);


    return PyArray_Return(gain);

}


/*
 * SCIEF
 */
static PyObject *py_SCIEF(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    PyArrayObject *p, *v, *dv, *r;
    double *cp, *cv, *cdv, *cr;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    
    SCIEF(cp, cv, cdv, cr, N);


    return PyArray_Return(r);

}


static PyObject *py_SCIEF_get_h(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    PyArrayObject *p, *v, *dv, *r, *h;
    double *cp, *cv, *cdv, *cr, *ch;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    cr = pyvector_to_Carrayptrs(r);
    ch = pyvector_to_Carrayptrs(h);
    
    SCIEF_get_h(cp, cv, cdv, cr, ch, N);


    return PyArray_Return(h);

}


static PyObject *py_SCIEF_gain(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    PyArrayObject *p, *v, *dv, *r, *gain;
    double *cp, *cv, *cdv, *cr, *cgain;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    gain = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    if (!gain) {
        PyErr_SetString(PyExc_MemoryError, "Could not create gain array");
        return NULL;
    }

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    cr = pyvector_to_Carrayptrs(r);
    cgain = pyvector_to_Carrayptrs(gain);
    
    SCIEF_gain(cp, cv, cdv, cr, cgain, N);


    return PyArray_Return(gain);

}


/*
 * SDF
 */
static PyObject *py_SDF(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *r;
    double *cp, *cv, *cr;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &p, &PyArray_Type, &v))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cr = pyvector_to_Carrayptrs(r);
    
    SDF(cp, cv, cr, N);


    return PyArray_Return(r);

}

/*
 * SCIF2
 */
static PyObject *py_SCIF2(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r;
    double *cp, *cv, *cdv, *cr;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    
    SCIF2(cp, cv, cdv, cr, N);


    return PyArray_Return(r);

}

static PyObject *py_SCIF2_get_h(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r, *h;
    double *cp, *cv, *cdv, *cr, *ch;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    ch = pyvector_to_Carrayptrs(h);
    
    SCIF2_get_h(cp, cv, cdv, cr, ch, N);


    return PyArray_Return(h);

}


static PyObject *py_SCIF2_gain(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r, *gain;
    double *cp, *cv, *cdv, *cr, *cgain;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    gain = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    cgain = pyvector_to_Carrayptrs(gain);
    
    SCIF2_gain(cp, cv, cdv, cr, cgain, N);


    return PyArray_Return(gain);

}



/*
 * SC1DF
 */
static PyObject *py_SC1DF(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r;
    double *cp, *cv, *cdv, *cr;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    
    SC1DF(cp, cv, cdv, cr, N);


    return PyArray_Return(r);

}

static PyObject *py_SC1DF_get_h(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r, *h;
    double *cp, *cv, *cdv, *cr, *ch;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    ch = pyvector_to_Carrayptrs(h);
    
    SC1DF_get_h(cp, cv, cdv, cr, ch, N);


    return PyArray_Return(h);

}


static PyObject *py_SC1DF_get_m(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    PyArrayObject *p, *v, *dv, *r, *m;
    double *cp, *cv, *cdv, *cr, *cm;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    m = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    if (!m) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    cm = pyvector_to_Carrayptrs(m);
    
    SC1DF_get_m(cp, cv, cdv, cr, cm, N);


    return PyArray_Return(m);

}



static PyObject *py_SC1DF_gain(PyObject *self, PyObject *args) {

    import_array();

    // numpy array 
    //PyArrayObject *p, *v, *dv, *r, *h;
    PyArrayObject *p, *v, *dv, *r, *gain;
    double *cp, *cv, *cdv, *cr, *cgain;

    /* parse single numpy array argument
     */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &p, &PyArray_Type, &v, &PyArray_Type, &dv))
        return NULL;

    npy_int N = PyArray_DIM(v, 0);
    npy_intp dims[1] = {N};
    r = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    gain = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    //h = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (!r) {
        PyErr_SetString(PyExc_MemoryError, "Could not create r array");
        return NULL;
    }
    /*
    if (!h) {
        PyErr_SetString(PyExc_MemoryError, "Could not create h array");
        return NULL;
    }
    */

    cp = pyvector_to_Carrayptrs(p);
    cv = pyvector_to_Carrayptrs(v);
    cdv = pyvector_to_Carrayptrs(dv);
    //ch = pyvector_to_Carrayptrs(h);
    cr = pyvector_to_Carrayptrs(r);
    cgain = pyvector_to_Carrayptrs(gain);
    
    SC1DF_gain(cp, cv, cdv, cr, cgain, N);


    return PyArray_Return(gain);

}



double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
	double **c, *a;
	int i;
	
	//n=arrayin->dimensions[0];
	//m=arrayin->dimensions[1];
	npy_int n = PyArray_DIM(arrayin, 0);
	npy_int m = PyArray_DIM(arrayin, 1);
	c=ptrvector(n);
	//a=(double *) arrayin->data;  /* pointer to arrayin data as double */
	a=(double *) PyArray_DATA(arrayin);  /* pointer to arrayin data as double */
	for ( i=0; i<n; i++)  {
		c[i]=a+i*m;  }
	return c;
}
/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  }
	return v;
}
/* ==== Free a double *vector (vec of pointers) ========================== */ 
void free_Carrayptrs(double **v)  {
	free((char*) v);
}


/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
	//int i,n;
        //npy_int n = PyArray_NDIM(arrayin);	
	//n=arrayin->dimensions[0];
	//return (double *) arrayin->data;  /* pointer to arrayin data as double */
	return (double *) PyArray_DATA(arrayin);
}

/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
/*
int  not_doublematrix(PyArrayObject *mat)  {
	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
		return 1;  }
	return 0;
}
*/
