#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_
#include "numpy/ufuncobject.h"

/* private ufunct object data, attached to public PyUfuncObject->internal */
typedef struct {
    /* generalized ufunc parameters */

    /* 0 for scalar ufunc; 1 for generalized ufunc */
    int core_enabled;
    /* number of distinct dimension names in signature */
    int core_num_dim_ix;

    /*
     * dimension indices of input/output argument k are stored in
     * core_dim_ixs[core_offsets[k]..core_offsets[k]+core_num_dims[k]-1]
     */

    /* numbers of core dimensions of each argument */
    int *core_num_dims;
    /*
     * dimension indices in a flatted form; indices
     * are in the range of [0,core_num_dim_ix)
     */
    int *core_dim_ixs;
    /*
     * positions of 1st core dimensions of each
     * argument in core_dim_ixs
     */
    int *core_offsets;
    /* signature string for printing purpose */
    char *core_signature;

    /*
     * A function which returns an inner loop for the new mechanism
     * in NumPy 1.7 and later. If provided, this is used, otherwise
     * if NULL the legacy_inner_loop_selector is used instead.
     */
    PyUFunc_InnerLoopSelectionFunc *inner_loop_selector;
    /*
     * A function which returns a masked inner loop for the ufunc.
     */
    PyUFunc_MaskedInnerLoopSelectionFunc *masked_inner_loop_selector;

    /*
     * List of flags for each operand when ufunc is called by nditer object.
     * These flags will be used in addition to the default flags for each
     * operand set by nditer object.
     */
    npy_uint32 *op_flags;

    /*
     * List of global flags used when ufunc is called by nditer object.
     * These flags will be used in addition to the default global flags
     * set by nditer object.
     */
    npy_uint32 iter_flags;
} PyUfuncObjectPrivate;

/* get private part of ufunc, may return NULL */
static NPY_INLINE PyUfuncObjectPrivate *
get_private_ufunc(PyUFuncObject * ufunc)
{
    return (PyUfuncObjectPrivate *)ufunc->internal;
}

/* initializes private part and adds it to ufunc object */
static NPY_INLINE PyUfuncObjectPrivate *
init_private_ufunc(PyUFuncObject * ufunc)
{
    PyUfuncObjectPrivate * self =
        PyArray_malloc(sizeof(PyUfuncObjectPrivate));
    ufunc->internal = self;
    /* generalized ufunc */
    self->core_enabled = 0;
    self->core_num_dim_ix = 0;
    self->core_num_dims = NULL;
    self->core_dim_ixs = NULL;
    self->core_offsets = NULL;
    self->core_signature = NULL;
    self->op_flags = PyArray_malloc(sizeof(npy_uint32)*ufunc->nargs);
    if (self->op_flags == NULL) {
        return (PyUfuncObjectPrivate *)PyErr_NoMemory();
    }
    memset(self->op_flags, 0, sizeof(npy_uint32)*ufunc->nargs);
    self->iter_flags = 0;
    return self;
}

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

/* interned strings (on umath import) */
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_out;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_subok;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_array_prepare;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_array_finalize;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_ufunc;
NPY_VISIBILITY_HIDDEN extern PyObject * npy_um_str_pyvals_name;

#endif
