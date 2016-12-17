#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "common.h"
#include "number.h"

/*************************************************************************
 ****************   Implement Number Protocol ****************************
 *************************************************************************/

NPY_NO_EXPORT NumericOps n_ops; /* NB: static objects initialized to zero */

/*
 * Dictionary can contain any of the numeric operations, by name.
 * Those not present will not be changed
 */

/* FIXME - macro contains a return */
#define SET(op)   temp = PyDict_GetItemString(dict, #op); \
    if (temp != NULL) { \
        if (!(PyCallable_Check(temp))) { \
            return -1; \
        } \
        Py_INCREF(temp); \
        Py_XDECREF(n_ops.op); \
        n_ops.op = temp; \
    }

#define NPY_NUMBER_MAX(a, b) ((a) > (b) ? (a) : (b))


/*NUMPY_API
 *Set internal structure with number functions that all arrays will use
 */
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict)
{
    PyObject *temp = NULL;
    SET(add);
    SET(subtract);
    SET(multiply);
    SET(divide);
    SET(remainder);
    SET(power);
    SET(square);
    SET(reciprocal);
    SET(_ones_like);
    SET(sqrt);
    SET(cbrt);
    SET(negative);
    SET(absolute);
    SET(invert);
    SET(left_shift);
    SET(right_shift);
    SET(bitwise_and);
    SET(bitwise_or);
    SET(bitwise_xor);
    SET(less);
    SET(less_equal);
    SET(equal);
    SET(not_equal);
    SET(greater);
    SET(greater_equal);
    SET(floor_divide);
    SET(true_divide);
    SET(logical_or);
    SET(logical_and);
    SET(floor);
    SET(ceil);
    SET(maximum);
    SET(minimum);
    SET(rint);
    SET(conjugate);
    return 0;
}

/* FIXME - macro contains goto */
#define GET(op) if (n_ops.op &&                                         \
                    (PyDict_SetItemString(dict, #op, n_ops.op)==-1))    \
        goto fail;

static int
has_ufunc_attr(PyObject * obj) {
    /* attribute check is expensive for scalar operations, avoid if possible */
    if (PyArray_CheckExact(obj) || PyArray_CheckAnyScalarExact(obj) ||
        _is_basic_python_type(obj)) {
        return 0;
    }
    else {
        return PyObject_HasAttrString(obj, "__numpy_ufunc__");
    }
}

/*
 * Check whether the operation needs to be forwarded to the right-hand binary
 * operation.
 *
 * This is the case when all of the following conditions apply:
 *
 * (i) the other object defines __numpy_ufunc__
 * (ii) the other object defines the right-hand operation __r*__
 * (iii) Python hasn't already called the right-hand operation
 *       [occurs if the other object is a strict subclass provided
 *       the operation is not in-place]
 *
 * An additional check is made in GIVE_UP_IF_HAS_RIGHT_BINOP macro below:
 *
 * (iv) other.__class__.__r*__ is not self.__class__.__r*__
 *
 *      This is needed, because CPython does not call __rmul__ if
 *      the tp_number slots of the two objects are the same.
 *
 * This always prioritizes the __r*__ routines over __numpy_ufunc__, independent
 * of whether the other object is an ndarray subclass or not.
 */

NPY_NO_EXPORT int
needs_right_binop_forward(PyObject *self, PyObject *other,
                          const char *right_name, int inplace_op)
{
    if (other == NULL ||
        self == NULL ||
        Py_TYPE(self) == Py_TYPE(other) ||
        PyArray_CheckExact(other) ||
        PyArray_CheckAnyScalar(other)) {
        /*
         * Quick cases
         */
        return 0;
    }
    if ((!inplace_op && PyType_IsSubtype(Py_TYPE(other), Py_TYPE(self))) ||
        !PyArray_Check(self)) {
        /*
         * Bail out if Python would already have called the right-hand
         * operation.
         */
        return 0;
    }
    if (has_ufunc_attr(other) &&
        PyObject_HasAttrString(other, right_name)) {
        return 1;
    }
    else {
        return 0;
    }
}

/* In pure-Python, SAME_SLOTS can be replaced by
   getattr(m1, op_name) is getattr(m2, op_name) */
#define SAME_SLOTS(m1, m2, slot_name)                                   \
    (Py_TYPE(m1)->tp_as_number != NULL && Py_TYPE(m2)->tp_as_number != NULL && \
     Py_TYPE(m1)->tp_as_number->slot_name == Py_TYPE(m2)->tp_as_number->slot_name)

#define GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, left_name, right_name, inplace, slot_name) \
    do {                                                                          \
        if (needs_right_binop_forward((PyObject *)m1, m2, right_name, inplace) && \
                (inplace || !SAME_SLOTS(m1, m2, slot_name))) {                    \
            Py_INCREF(Py_NotImplemented);                                         \
            return Py_NotImplemented;                                             \
        }                                                                         \
    } while (0)


/*NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void)
{
    PyObject *dict;
    if ((dict = PyDict_New())==NULL)
        return NULL;
    GET(add);
    GET(subtract);
    GET(multiply);
    GET(divide);
    GET(remainder);
    GET(power);
    GET(square);
    GET(reciprocal);
    GET(_ones_like);
    GET(sqrt);
    GET(negative);
    GET(absolute);
    GET(invert);
    GET(left_shift);
    GET(right_shift);
    GET(bitwise_and);
    GET(bitwise_or);
    GET(bitwise_xor);
    GET(less);
    GET(less_equal);
    GET(equal);
    GET(not_equal);
    GET(greater);
    GET(greater_equal);
    GET(floor_divide);
    GET(true_divide);
    GET(logical_or);
    GET(logical_and);
    GET(floor);
    GET(ceil);
    GET(maximum);
    GET(minimum);
    GET(rint);
    GET(conjugate);
    return dict;

 fail:
    Py_DECREF(dict);
    return NULL;
}

static PyObject *
_get_keywords(int rtype, PyArrayObject *out)
{
    PyObject *kwds = NULL;
    if (rtype != NPY_NOTYPE || out != NULL) {
        kwds = PyDict_New();
        if (rtype != NPY_NOTYPE) {
            PyArray_Descr *descr;
            descr = PyArray_DescrFromType(rtype);
            if (descr) {
                PyDict_SetItemString(kwds, "dtype", (PyObject *)descr);
                Py_DECREF(descr);
            }
        }
        if (out != NULL) {
            PyDict_SetItemString(kwds, "out", (PyObject *)out);
        }
    }
    return kwds;
}

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "reduce");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "accumulate");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    if (!PyArray_Check(m2) && !has_ufunc_attr(m2)) {
          /*
           * Catch priority inversion and punt, but only if it's guaranteed
           * that we were called through m1 and the other guy is not an array
           * at all. Note that some arrays need to pass through here even
           * with priorities inverted, for example: float(17) * np.matrix(...)
           *
           * See also:
           * - https://github.com/numpy/numpy/issues/3502
           * - https://github.com/numpy/numpy/issues/3503
           *
           * NB: there's another copy of this code in
           *    numpy.ma.core.MaskedArray._delegate_binop
           * which should possibly be updated when this is.
           */
          double m1_prio = PyArray_GetPriority((PyObject *)m1,
                                               NPY_SCALAR_PRIORITY);
          double m2_prio = PyArray_GetPriority((PyObject *)m2,
                                               NPY_SCALAR_PRIORITY);
          if (m1_prio < m2_prio) {
              Py_INCREF(Py_NotImplemented);
              return Py_NotImplemented;
          }
    }

    return PyObject_CallFunctionObjArgs(op, m1, m2, NULL);
}

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunctionObjArgs(op, m1, NULL);
}

static PyObject *
PyArray_GenericInplaceBinaryFunction(PyArrayObject *m1,
                                     PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunctionObjArgs(op, m1, m2, m1, NULL);
}

static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunctionObjArgs(op, m1, m1, NULL);
}

static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2);
#if !defined(NPY_PY3K)
static PyObject *
array_inplace_divide(PyArrayObject *m1, PyObject *m2);
#endif
static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2);


/*
 * helper functions to convert operations to inplace operations when refcount
 * is 1 and we are called from python
 */
#if defined HAVE_BACKTRACE && defined HAVE_DLFCN_H && ! defined PYPY_VERSION
/* 1 prints elided operations, 2 prints stacktraces */
#define NPY_ELIDE_DEBUG 0
#define NPY_MAX_STACKSIZE 10

#if PY_VERSION_HEX >= 0x03060000
/* TODO can pep523 be used to somehow? */
#define PYFRAMEEVAL_FUNC "_PyEval_EvalFrameDefault"
#else
#define PYFRAMEEVAL_FUNC "PyEval_EvalFrameEx"
#endif
/*
 * Heuristic number at which backtrace overhead generation becomes less than
 * speed gain. Depends on stack depth being checked.
 * Measurements with 10 stacks show it getting worthwhile around 100KiB but to
 * be conservative put it higher around where the L2 cache spills.
 */
#ifndef Py_DEBUG
#define NPY_MIN_ELIDE_BYTES (256 * 1024)
#else
/*
 * in debug mode always elide but skip scalars as these can convert to 0d array
 * during in place operations
 */
#define NPY_MIN_ELIDE_BYTES (32)
#endif
#include <dlfcn.h>
#include <execinfo.h>

/*
 * linear search pointer in table
 * number of pointers is usually quite small but if a performance impact can be
 * measured this could be converted to a binary search
 */
static int
find_addr(void * addresses[], npy_intp naddr, void * addr)
{
    npy_intp j;
    for (j = 0; j < naddr; j++) {
        if (addr == addresses[j]) {
            return 1;
        }
    }
    return 0;
}

static int
check_callers(int * cannot)
{
    /*
     * get base addresses of multiarray, python, check if
     * backtrace is in these libraries only calling dladdr if a new max address
     * is found, if after the initial multiarray stack everything is inside
     * python or libc we can elide as no C-API user could have messed up the
     * reference counts
     * Only check until the python frame evaluation function is found
     * approx 10us overhead for stack size of 10
     *
     * TODO some calls go over scalarmath in umath but we cannot get the base
     * address of it from multiarraymodule as it is not linked against it
     */
    static int init = 0;
    /* measured DSO object memory start and end */
    static void * pos_python_start;
    static void * pos_python_end;
    static void * pos_ma_start;
    static void * pos_ma_end;
    /* known address storage to save dladdr calls */
    static void * py_addr[64];
    static void * pyeval_addr[64];
    static npy_intp npy_addr = 0;
    static npy_intp npyeval = 0;
    void *buffer[NPY_MAX_STACKSIZE];
    int i, nptrs;
    int ok = 0;
    /* cannot determine callers */
    if (init == -1) {
        *cannot = 1;
        return 0;
    }

    nptrs = backtrace(buffer, NPY_MAX_STACKSIZE);
    if (nptrs == 0) {
        /* complete failure, disable elision */
        init = -1;
        *cannot = 1;
        return 0;
    }

    /* setup DSO base addresses, ends updated later */
    if (NPY_UNLIKELY(init == 0)) {
        Dl_info info;
        /* get python base address */
        if (dladdr(&PyNumber_Or, &info)) {
            pos_python_start = info.dli_fbase;
            pos_python_end = info.dli_fbase;
        }
        else {
            init = -1;
            return 0;
        }
        /* get multiarray base address */
        if (dladdr(&PyArray_SetNumericOps, &info)) {
            pos_ma_start = info.dli_fbase;
            pos_ma_end = info.dli_fbase;
        }
        else {
            init = -1;
            return 0;
        }
        init = 1;
    }

    for (i = 0; i < nptrs; i++) {
        Dl_info info;
        int in_python = 0;
        int in_multiarray = 0;
#if NPY_ELIDE_DEBUG >= 2
        dladdr(buffer[i], &info);
        printf("%s(%p) %s(%p)\n", info.dli_fname, info.dli_fbase,
               info.dli_sname, info.dli_saddr);
#endif

        /* check stored DSO boundaries first */
        if (buffer[i] >= pos_python_start && buffer[i] <= pos_python_end) {
            in_python = 1;
        }
        else if (buffer[i] >= pos_ma_start && buffer[i] <= pos_ma_end) {
            in_multiarray = 1;
        }

        /* update DSO boundaries via dladdr if necessary */
        if (!in_python && !in_multiarray) {
            if (dladdr(buffer[i], &info) == 0) {
                init = -1;
                ok = 0;
                break;
            }
            /* update DSO end */
            if (info.dli_fbase == pos_python_start) {
                pos_python_end = NPY_NUMBER_MAX(buffer[i], pos_python_end);
                in_python = 1;
            }
            else if (info.dli_fbase == pos_ma_start) {
                pos_ma_end = NPY_NUMBER_MAX(buffer[i], pos_ma_end);
                in_multiarray = 1;
            }
        }

        /* left ok libraries and not reached PyEval -> no elide */
        if (!in_python && !in_multiarray) {
            ok = 0;
            break;
        }

        /* in python check if the frame eval function was reached */
        if (in_python) {
            /* if reached eval we are done */
            if (find_addr(pyeval_addr, npyeval, buffer[i])) {
                ok = 1;
                break;
            }
            /*
             * check if its some other function, use pointer lookup table to
             * save expensive dladdr calls
             */
            if (find_addr(py_addr, npy_addr, buffer[i])) {
                continue;
            }

            /* new python address, check for PyEvalFrame */
            if (dladdr(buffer[i], &info) == 0) {
                init = -1;
                ok = 0;
                break;
            }
            if (info.dli_sname &&
                    strcmp(info.dli_sname, PYFRAMEEVAL_FUNC) == 0) {
                if (npyeval < sizeof(pyeval_addr) / sizeof(pyeval_addr[0])) {
                    /* store address to not have to dladdr it again */
                    pyeval_addr[npyeval++] = buffer[i];
                }
                ok = 1;
                break;
            }
            else if (npy_addr < sizeof(py_addr) / sizeof(py_addr[0])) {
                /* store other py function to not have to dladdr it again */
                py_addr[npy_addr++] = buffer[i];
            }
        }
    }

    /* all stacks after numpy are from python, we can elide */
    if (ok) {
        *cannot = 0;
        return 1;
    }
    else {
#if NPY_ELIDE_DEBUG != 0
        puts("cannot elide due to c-api usage");
#endif
        *cannot = 1;
        return 0;
    }
}

/*
 * check if in "alhs @op@ orhs" that alhs is a temporary (refcnt == 1) so we
 * can do inplace operations instead of creating a new temporary
 * "cannot" is set to true if it cannot be done even with swapped arguments
 */
static int
can_elide_temp(PyArrayObject * alhs, PyObject * orhs, int * cannot)
{
    if (Py_REFCNT(alhs) != 1 || !PyArray_CheckExact(alhs) ||
            PyArray_DESCR(alhs)->type_num == NPY_VOID ||
            !(PyArray_FLAGS(alhs) & NPY_ARRAY_OWNDATA) ||
            PyArray_NBYTES(alhs) < NPY_MIN_ELIDE_BYTES) {
        return 0;
    }
    if (PyArray_CheckExact(orhs) ||
        PyArray_CheckAnyScalar(orhs)) {
        PyArrayObject * arhs;

        /* create array from right hand side */
        Py_INCREF(orhs);
        arhs = (PyArrayObject *)PyArray_EnsureArray(orhs);
        if (arhs == NULL) {
            return 0;
        }

        /*
         * if rhs is not a scalar dimensions must match
         * TODO: one could allow broadcasting on equal types
         */
        if (!(PyArray_NDIM(arhs) == 0 ||
              (PyArray_NDIM(arhs) == PyArray_NDIM(alhs) &&
               PyArray_CompareLists(PyArray_DIMS(alhs), PyArray_DIMS(arhs),
                                    PyArray_NDIM(arhs))))) {
                Py_DECREF(arhs);
                return 0;
        }

        /* must be safe to cast (checks values for scalar in rhs) */
        if (PyArray_CanCastArrayTo(arhs, PyArray_DESCR(alhs),
                                   NPY_SAFE_CASTING)) {
            Py_DECREF(arhs);
            return check_callers(cannot);
        }
        Py_DECREF(arhs);
    }

    return 0;
}

/*
 * try eliding a binary op, if commutative is true also try swapped arguments
 */
static int
try_binary_elide(PyArrayObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative)
{
    /* set when no elision can be done independent of argument order */
    int cannot = 0;
    if (can_elide_temp(m1, m2, &cannot)) {
        *res = inplace_op(m1, m2);
#if NPY_ELIDE_DEBUG != 0
        puts("elided temporary in binary op");
#endif
        return 1;
    }
    else if (commutative && !cannot) {
        if (can_elide_temp((PyArrayObject *)m2, (PyObject *)m1, &cannot)) {
            *res = inplace_op((PyArrayObject *)m2, (PyObject *)m1);
#if NPY_ELIDE_DEBUG != 0
        puts("elided temporary in commutative binary op");
#endif
            return 1;
        }
    }
    *res = NULL;
    return 0;
}

/* try elide unary temporary */
static int can_elide_temp_unary(PyArrayObject * m1)
{
    int cannot;
    if (Py_REFCNT(m1) != 1 || !PyArray_CheckExact(m1) ||
            PyArray_DESCR(m1)->type_num == NPY_VOID ||
            !(PyArray_FLAGS(m1) & NPY_ARRAY_OWNDATA) ||
            PyArray_NBYTES(m1) < NPY_MIN_ELIDE_BYTES) {
        return 0;
    }
    if (check_callers(&cannot)) {
#if NPY_ELIDE_DEBUG != 0
        puts("elided temporary in unary op");
#endif
        return 1;
    }
    else {
        return 0;
    }
}
#else
static int can_elide_temp_unary(PyArrayObject * m1)
{
    return 0;
}

static int
try_binary_elide(PyArrayObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative)
{
    return 0;
}
#endif


static PyObject *
array_add(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__add__", "__radd__", 0, nb_add);
    if (try_binary_elide(m1, m2, &array_inplace_add, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_subtract(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__sub__", "__rsub__", 0, nb_subtract);
    if (try_binary_elide(m1, m2, &array_inplace_subtract, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_multiply(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__mul__", "__rmul__", 0, nb_multiply);
    if (try_binary_elide(m1, m2, &array_inplace_multiply, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.multiply);
}

#if !defined(NPY_PY3K)
static PyObject *
array_divide(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__div__", "__rdiv__", 0, nb_divide);
    if (try_binary_elide(m1, m2, &array_inplace_divide, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.divide);
}
#endif

static PyObject *
array_remainder(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__mod__", "__rmod__", 0, nb_remainder);
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.remainder);
}


#if PY_VERSION_HEX >= 0x03050000
/* Need this to be version dependent on account of the slot check */
static PyObject *
array_matrix_multiply(PyArrayObject *m1, PyObject *m2)
{
    static PyObject *matmul = NULL;

    npy_cache_import("numpy.core.multiarray", "matmul", &matmul);
    if (matmul == NULL) {
        return NULL;
    }
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__matmul__", "__rmatmul__",
                               0, nb_matrix_multiply);
    return PyArray_GenericBinaryFunction(m1, m2, matmul);
}

static PyObject *
array_inplace_matrix_multiply(PyArrayObject *m1, PyObject *m2)
{
    PyErr_SetString(PyExc_TypeError,
                    "In-place matrix multiplication is not (yet) supported. "
                    "Use 'a = a @ b' instead of 'a @= b'.");
    return NULL;
}
#endif

/* Determine if object is a scalar and if so, convert the object
 *   to a double and place it in the out_exponent argument
 *   and return the "scalar kind" as a result.   If the object is
 *   not a scalar (or if there are other error conditions)
 *   return NPY_NOSCALAR, and out_exponent is undefined.
 */
static NPY_SCALARKIND
is_scalar_with_conversion(PyObject *o2, double* out_exponent)
{
    PyObject *temp;
    const int optimize_fpexps = 1;

    if (PyInt_Check(o2)) {
        *out_exponent = (double)PyInt_AsLong(o2);
        return NPY_INTPOS_SCALAR;
    }
    if (optimize_fpexps && PyFloat_Check(o2)) {
        *out_exponent = PyFloat_AsDouble(o2);
        return NPY_FLOAT_SCALAR;
    }
    if (PyArray_Check(o2)) {
        if ((PyArray_NDIM((PyArrayObject *)o2) == 0) &&
                ((PyArray_ISINTEGER((PyArrayObject *)o2) ||
                 (optimize_fpexps && PyArray_ISFLOAT((PyArrayObject *)o2))))) {
            temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
            if (temp == NULL) {
                return NPY_NOSCALAR;
            }
            *out_exponent = PyFloat_AsDouble(o2);
            Py_DECREF(temp);
            if (PyArray_ISINTEGER((PyArrayObject *)o2)) {
                return NPY_INTPOS_SCALAR;
            }
            else { /* ISFLOAT */
                return NPY_FLOAT_SCALAR;
            }
        }
    }
    else if (PyArray_IsScalar(o2, Integer) ||
                (optimize_fpexps && PyArray_IsScalar(o2, Floating))) {
        temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
        if (temp == NULL) {
            return NPY_NOSCALAR;
        }
        *out_exponent = PyFloat_AsDouble(o2);
        Py_DECREF(temp);

        if (PyArray_IsScalar(o2, Integer)) {
                return NPY_INTPOS_SCALAR;
        }
        else { /* IsScalar(o2, Floating) */
            return NPY_FLOAT_SCALAR;
        }
    }
    else if (PyIndex_Check(o2)) {
        PyObject* value = PyNumber_Index(o2);
        Py_ssize_t val;
        if (value==NULL) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return NPY_NOSCALAR;
        }
        val = PyInt_AsSsize_t(value);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return NPY_NOSCALAR;
        }
        *out_exponent = (double) val;
        return NPY_INTPOS_SCALAR;
    }
    return NPY_NOSCALAR;
}

/* optimize float array or complex array to a scalar power */
static PyObject *
fast_scalar_power(PyArrayObject *a1, PyObject *o2, int inplace)
{
    double exponent;
    NPY_SCALARKIND kind;   /* NPY_NOSCALAR is not scalar */

    if (PyArray_Check(a1) && ((kind=is_scalar_with_conversion(o2, &exponent))>0)) {
        PyObject *fastop = NULL;
        if (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) {
            if (exponent == 1.0) {
                /* we have to do this one special, as the
                   "copy" method of array objects isn't set
                   up early enough to be added
                   by PyArray_SetNumericOps.
                */
                if (inplace) {
                    Py_INCREF(a1);
                    return (PyObject *)a1;
                } else {
                    return PyArray_Copy(a1);
                }
            }
            else if (exponent == -1.0) {
                fastop = n_ops.reciprocal;
            }
            else if (exponent ==  0.0) {
                fastop = n_ops._ones_like;
            }
            else if (exponent ==  0.5) {
                fastop = n_ops.sqrt;
            }
            else if (exponent ==  2.0) {
                fastop = n_ops.square;
            }
            else {
                return NULL;
            }

            if (inplace || can_elide_temp_unary(a1)) {
                return PyArray_GenericInplaceUnaryFunction(a1, fastop);
            } else {
                return PyArray_GenericUnaryFunction(a1, fastop);
            }
        }
        /* Because this is called with all arrays, we need to
         *  change the output if the kind of the scalar is different
         *  than that of the input and inplace is not on ---
         *  (thus, the input should be up-cast)
         */
        else if (exponent == 2.0) {
            fastop = n_ops.multiply;
            if (inplace) {
                return PyArray_GenericInplaceBinaryFunction
                    (a1, (PyObject *)a1, fastop);
            }
            else {
                PyArray_Descr *dtype = NULL;
                PyObject *res;

                /* We only special-case the FLOAT_SCALAR and integer types */
                if (kind == NPY_FLOAT_SCALAR && PyArray_ISINTEGER(a1)) {
                    dtype = PyArray_DescrFromType(NPY_DOUBLE);
                    a1 = (PyArrayObject *)PyArray_CastToType(a1, dtype,
                            PyArray_ISFORTRAN(a1));
                    if (a1 == NULL) {
                        return NULL;
                    }
                }
                else {
                    Py_INCREF(a1);
                }
                res = PyArray_GenericBinaryFunction(a1, (PyObject *)a1, fastop);
                Py_DECREF(a1);
                return res;
            }
        }
    }
    return NULL;
}

static PyObject *
array_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    GIVE_UP_IF_HAS_RIGHT_BINOP(a1, o2, "__pow__", "__rpow__", 0, nb_power);
    value = fast_scalar_power(a1, o2, 0);
    if (!value) {
        value = PyArray_GenericBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}


static PyObject *
array_negative(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.negative);
    }
    return PyArray_GenericUnaryFunction(m1, n_ops.negative);
}

static PyObject *
array_absolute(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.absolute);
    }
    return PyArray_GenericUnaryFunction(m1, n_ops.absolute);
}

static PyObject *
array_invert(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.invert);
    }
    return PyArray_GenericUnaryFunction(m1, n_ops.invert);
}

static PyObject *
array_left_shift(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__lshift__", "__rlshift__", 0, nb_lshift);
    if (try_binary_elide(m1, m2, &array_inplace_left_shift, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_right_shift(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__rshift__", "__rrshift__", 0, nb_rshift);
    if (try_binary_elide(m1, m2, &array_inplace_right_shift, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__and__", "__rand__", 0, nb_and);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_and, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__or__", "__ror__", 0, nb_or);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_or, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__xor__", "__rxor__", 0, nb_xor);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_xor, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__iadd__", "__radd__", 1, nb_inplace_add);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__isub__", "__rsub__", 1, nb_inplace_subtract);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__imul__", "__rmul__", 1, nb_inplace_multiply);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.multiply);
}

#if !defined(NPY_PY3K)
static PyObject *
array_inplace_divide(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__idiv__", "__rdiv__", 1, nb_inplace_divide);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.divide);
}
#endif

static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__imod__", "__rmod__", 1, nb_inplace_remainder);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.remainder);
}

static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    GIVE_UP_IF_HAS_RIGHT_BINOP(a1, o2, "__ipow__", "__rpow__", 1, nb_inplace_power);
    value = fast_scalar_power(a1, o2, 1);
    if (!value) {
        value = PyArray_GenericInplaceBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}

static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__ilshift__", "__rlshift__", 1, nb_inplace_lshift);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__irshift__", "__rrshift__", 1, nb_inplace_rshift);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__iand__", "__rand__", 1, nb_inplace_and);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__ior__", "__ror__", 1, nb_inplace_or);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__ixor__", "__rxor__", 1, nb_inplace_xor);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__floordiv__", "__rfloordiv__", 0, nb_floor_divide);
    if (try_binary_elide(m1, m2, &array_inplace_floor_divide, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.floor_divide);
}

static PyObject *
array_true_divide(PyArrayObject *m1, PyObject *m2)
{
    PyObject * res;
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__truediv__", "__rtruediv__", 0, nb_true_divide);
    if (PyArray_CheckExact(m1) &&
            (PyArray_ISFLOAT(m1) || PyArray_ISCOMPLEX(m1)) &&
            try_binary_elide(m1, m2, &array_inplace_true_divide, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.true_divide);
}

static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__ifloordiv__", "__rfloordiv__", 1, nb_inplace_floor_divide);
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.floor_divide);
}

static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2)
{
    GIVE_UP_IF_HAS_RIGHT_BINOP(m1, m2, "__itruediv__", "__rtruediv__", 1, nb_inplace_true_divide);
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.true_divide);
}


static int
_array_nonzero(PyArrayObject *mp)
{
    npy_intp n;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        return PyArray_DESCR(mp)->f->nonzero(PyArray_DATA(mp), mp);
    }
    else if (n == 0) {
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array " \
                        "with more than one element is ambiguous. " \
                        "Use a.any() or a.all()");
        return -1;
    }
}



static PyObject *
array_divmod(PyArrayObject *op1, PyObject *op2)
{
    PyObject *divp, *modp, *result;
    GIVE_UP_IF_HAS_RIGHT_BINOP(op1, op2, "__divmod__", "__rdivmod__", 0, nb_divmod);

    divp = array_floor_divide(op1, op2);
    if (divp == NULL) {
        return NULL;
    }
    else if(divp == Py_NotImplemented) {
        return divp;
    }
    modp = array_remainder(op1, op2);
    if (modp == NULL) {
        Py_DECREF(divp);
        return NULL;
    }
    else if(modp == Py_NotImplemented) {
        Py_DECREF(divp);
        return modp;
    }
    result = Py_BuildValue("OO", divp, modp);
    Py_DECREF(divp);
    Py_DECREF(modp);
    return result;
}


NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be"\
                        " converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_int == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to int");
        Py_DECREF(pv);
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_int(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_float(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to a "\
                        "float; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_float == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to float");
        Py_DECREF(pv);
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_float(pv);
    Py_DECREF(pv);
    return pv2;
}

#if !defined(NPY_PY3K)

static PyObject *
array_long(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_long == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to long");
        Py_DECREF(pv);
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_long(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_oct(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_oct == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to oct");
        Py_DECREF(pv);
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_oct(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_hex(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_hex == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to hex");
        Py_DECREF(pv);
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_hex(pv);
    Py_DECREF(pv);
    return pv2;
}

#endif

static PyObject *
_array_copy_nice(PyArrayObject *self)
{
    return PyArray_Return((PyArrayObject *) PyArray_Copy(self));
}

static PyObject *
array_index(PyArrayObject *v)
{
    if (!PyArray_ISINTEGER(v) || PyArray_NDIM(v) != 0) {
        PyErr_SetString(PyExc_TypeError,
            "only integer scalar arrays can be converted to a scalar index");
        return NULL;
    }
    return PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
}


NPY_NO_EXPORT PyNumberMethods array_as_number = {
    (binaryfunc)array_add,                      /*nb_add*/
    (binaryfunc)array_subtract,                 /*nb_subtract*/
    (binaryfunc)array_multiply,                 /*nb_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)array_divide,                   /*nb_divide*/
#endif
    (binaryfunc)array_remainder,                /*nb_remainder*/
    (binaryfunc)array_divmod,                   /*nb_divmod*/
    (ternaryfunc)array_power,                   /*nb_power*/
    (unaryfunc)array_negative,                  /*nb_neg*/
    (unaryfunc)_array_copy_nice,                /*nb_pos*/
    (unaryfunc)array_absolute,                  /*(unaryfunc)array_abs,*/
    (inquiry)_array_nonzero,                    /*nb_nonzero*/
    (unaryfunc)array_invert,                    /*nb_invert*/
    (binaryfunc)array_left_shift,               /*nb_lshift*/
    (binaryfunc)array_right_shift,              /*nb_rshift*/
    (binaryfunc)array_bitwise_and,              /*nb_and*/
    (binaryfunc)array_bitwise_xor,              /*nb_xor*/
    (binaryfunc)array_bitwise_or,               /*nb_or*/
#if !defined(NPY_PY3K)
    0,                                          /*nb_coerce*/
#endif
    (unaryfunc)array_int,                       /*nb_int*/
#if defined(NPY_PY3K)
    0,                                          /*nb_reserved*/
#else
    (unaryfunc)array_long,                      /*nb_long*/
#endif
    (unaryfunc)array_float,                     /*nb_float*/
#if !defined(NPY_PY3K)
    (unaryfunc)array_oct,                       /*nb_oct*/
    (unaryfunc)array_hex,                       /*nb_hex*/
#endif

    /*
     * This code adds augmented assignment functionality
     * that was made available in Python 2.0
     */
    (binaryfunc)array_inplace_add,              /*inplace_add*/
    (binaryfunc)array_inplace_subtract,         /*inplace_subtract*/
    (binaryfunc)array_inplace_multiply,         /*inplace_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)array_inplace_divide,           /*inplace_divide*/
#endif
    (binaryfunc)array_inplace_remainder,        /*inplace_remainder*/
    (ternaryfunc)array_inplace_power,           /*inplace_power*/
    (binaryfunc)array_inplace_left_shift,       /*inplace_lshift*/
    (binaryfunc)array_inplace_right_shift,      /*inplace_rshift*/
    (binaryfunc)array_inplace_bitwise_and,      /*inplace_and*/
    (binaryfunc)array_inplace_bitwise_xor,      /*inplace_xor*/
    (binaryfunc)array_inplace_bitwise_or,       /*inplace_or*/

    (binaryfunc)array_floor_divide,             /*nb_floor_divide*/
    (binaryfunc)array_true_divide,              /*nb_true_divide*/
    (binaryfunc)array_inplace_floor_divide,     /*nb_inplace_floor_divide*/
    (binaryfunc)array_inplace_true_divide,      /*nb_inplace_true_divide*/
    (unaryfunc)array_index,                     /*nb_index */
#if PY_VERSION_HEX >= 0x03050000
    (binaryfunc)array_matrix_multiply,          /*nb_matrix_multiply*/
    (binaryfunc)array_inplace_matrix_multiply,  /*nb_inplace_matrix_multiply*/
#endif
};
