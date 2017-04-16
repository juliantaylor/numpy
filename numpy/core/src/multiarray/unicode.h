#ifndef _NPY_PRIVATE_UNICODE_H_
#define _NPY_PRIVATE_UNICODE_H_

NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_unicode_typestr(char *typestr, Py_ssize_t len);

NPY_NO_EXPORT PyArray_UnicodeMetaData *
get_unicode_metadata_from_dtype(PyArray_Descr * dtype);

NPY_NO_EXPORT int
get_unicode_codec_itemsize(const PyArray_Descr * dtype);

NPY_NO_EXPORT NPY_UNICODE_CODEC
get_unicode_codec(const PyArray_Descr * dtype);

int
same_unicode_codec(const PyArray_Descr * dtype1, const PyArray_Descr * dtype2);

NPY_NO_EXPORT PyObject *
append_unicode_metastr_to_string(PyArray_Descr * descr, int skip_prefix,
                                 PyObject *ret);

#endif
