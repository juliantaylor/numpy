#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "npy_pycompat.h"


NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_unicode_typestr(char *typestr, Py_ssize_t len)
{
    PyArray_UnicodeMetaData meta;

    if (len < 2 || typestr[0] != 'U') {
        PyErr_Format(PyExc_TypeError,
                "Invalid unicode typestr \"%s\"",
                typestr);
        return NULL;
    }
    char codec[len];
    int elsize = 0;
    /* TODO U[UCS4] */
    int n = sscanf(typestr, "U%d[%s", &elsize, codec);
    if (n == 1 || n == 0) {
        meta.codec = NPY_UCS4;
        elsize = PyArray_MAX(elsize * 4, 4);
    }
    else if (n == 2) {
        if (strcmp(codec, "ucs4]") == 0) {
            meta.codec = NPY_UCS4;
            elsize = PyArray_MAX(elsize * 4, 4);
        }
        else if (strcmp(codec, "latin1]") == 0) {
            meta.codec = NPY_LATIN1;
            elsize = PyArray_MAX(elsize, 1);
        }
        else {
            PyErr_Format(PyExc_TypeError,
                         "Invalid unicode codec \"%s\"",
                         codec);
            return NULL;
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid unicode typestr \"%s\"",
                typestr);
        return NULL;
    }

    PyArray_Descr *dtype = NULL;
    PyArray_UnicodeMetaData *dt_data;
    dtype = PyArray_DescrNewFromType(NPY_UNICODE);
    if (dtype == NULL) {
        return NULL;
    }

    dtype->elsize = elsize;
    dt_data = &(((PyArray_UnicodeDTypeMetaData *)dtype->c_metadata)->meta);
    dt_data->codec = meta.codec;

    return dtype;
}

NPY_NO_EXPORT PyArray_UnicodeMetaData *
get_unicode_metadata_from_dtype(const PyArray_Descr * dtype)
{
    if (!PyDataType_ISUNICODE(dtype)) {
        PyErr_SetString(PyExc_TypeError,
                "cannot get unicode metadata from non-unicode type");
        return NULL;
    }

    return &(((PyArray_UnicodeDTypeMetaData *)dtype->c_metadata)->meta);
}

NPY_NO_EXPORT int
get_unicode_codec_itemsize(const PyArray_Descr * dtype)
{
    const PyArray_UnicodeMetaData * meta =
        get_unicode_metadata_from_dtype(dtype);
    if (meta == NULL) {
        return -1;
    }
    switch (meta->codec) {
        case NPY_UCS4:
            return 4;
        case NPY_LATIN1:
            return 1;
        default:
            assert(0);
            return 4;
    }
}

NPY_NO_EXPORT int
get_unicode_codec(const PyArray_Descr * dtype)
{
    const PyArray_UnicodeMetaData * meta =
        get_unicode_metadata_from_dtype(dtype);
    if (meta == NULL) {
        return NPY_UCS4;
    }
    return meta->codec;
}

int
same_unicode_codec(const PyArray_Descr * dtype1, const PyArray_Descr * dtype2)
{
    const PyArray_UnicodeMetaData *meta1 =
        get_unicode_metadata_from_dtype(dtype);
    const PyArray_UnicodeMetaData *meta2 =
        get_unicode_metadata_from_dtype(dtype);
    if (meta1 == NULL || meta2 == NULL) {
        return 0;
    }
    return meta1->codec == meta2->codec;
}

/* This function steals the reference 'ret' */
NPY_NO_EXPORT PyObject *
append_unicode_metastr_to_string(PyArray_Descr * descr, int skip_prefix,
                                 PyObject *ret)
{
    PyObject *res;
    PyArray_UnicodeMetaData *meta = get_unicode_metadata_from_dtype(descr);
    PyObject *prefix;
    char endian = descr->byteorder;

    if (ret == NULL || meta == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }

    if (skip_prefix) {
        prefix = PyUString_FromString("");
    }
    else {
        prefix = PyUString_FromFormat("%cU", endian);
    }

    switch (meta->codec) {
        case NPY_UCS4:
            res = PyUString_FromFormat("%d", descr->elsize / 4);
            break;
        case NPY_LATIN1:
            res = PyUString_FromFormat("%d[latin1]", descr->elsize);
            break;
        default:
            Py_DECREF(prefix);
            return NULL;
    }

    PyUString_ConcatAndDel(&prefix, res);
    PyUString_ConcatAndDel(&ret, prefix);
    return ret;
}
