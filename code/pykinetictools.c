#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "computeKinetics.h"
#include <math.h>


/* module definition */
static char module_name[] = "kinetictools";
static char module_docstring[] = "This module provides an interface for calculating kinetics block operations";

/* function definitions */
static char func1_name[] = "K4S";
static char func1_docstring[] = "Computes 4 State Kinetics block operations";

/* PyObject functions declarations */
static PyObject *py_K4S(PyObject *self, PyObject *args);

double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin); 
double **ptrvector(long n); 
void free_Carrayptrs(double **v); 
int  not_doublematrix(PyArrayObject *mat); 
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin); 


/* module method table */
static PyMethodDef kinetictoolsMethods[] = {
    {func1_name, py_K4S, METH_VARARGS, func1_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kinetictoolsModule = {
    PyModuleDef_HEAD_INIT,
    module_name,    /* name of module */
    module_docstring,//"A sample module",  /* doc string, may be NULL */
    -1, /* size of per-interpreter state of the module, 
           or -1 if the module keeps state in global variables */
    kinetictoolsMethods   /* methods table */
};

PyMODINIT_FUNC PyInit_kinetictools(void){

    PyObject *m;
    m = PyModule_Create(&kinetictoolsModule);

    if (m == NULL)
        return NULL;
    return m;
}


static PyObject *py_K4S(PyObject *self, PyObject *args) {

    import_array();

    // numpy array p, u, Xin
    PyArrayObject *p, *u, *Xin, *Xout;
    double **cXin, **cXout, dt, *cp, *cu;
    int M, N, dims[2];
    int i, j;

    /* parse single numpy array argument
     * (double p[], size_t M, size_t N, double X[M][N], double u[], double sampleTime) 
     */
    if (!PyArg_ParseTuple(args, "O!O!O!iid", &PyArray_Type, &p,
                &PyArray_Type, &Xin, &PyArray_Type, &u, &M, &N, &dt))
        return NULL;

    
    dims[0] = M;
    dims[1] = N;
    Xout = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_DOUBLE);

    cXin = pymatrix_to_Carrayptrs(Xin);
    cXout = pymatrix_to_Carrayptrs(Xout);

    cp = pyvector_to_Carrayptrs(p);
    cu = pyvector_to_Carrayptrs(u);

    K4S(cp, cXin, cu, M, N, dt);

    for (i=0; i<M; i++) {
        for (j=0; j<N; j++) {
            cXout[i][j] = cXin[i][j];
        }
    }

    free_Carrayptrs(cXin);
    free_Carrayptrs(cXout);

    //return Py_BuildValue("O", out_array);
    return PyArray_Return(Xout);

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
