#ifndef __NPY_PARTITION_H__
#define __NPY_PARTITION_H__

#include "npy_sort.h"

/* Python include is for future object sorts */
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#define NPY_ENOMEM 1
#define NPY_ECOMP 2


int quickselect_bool(npy_bool *vec, npy_intp cnt, void *null);
int aquickselect_bool(npy_bool *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_byte(npy_byte *vec, npy_intp cnt, void *null);
int aquickselect_byte(npy_byte *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_ubyte(npy_ubyte *vec, npy_intp cnt, void *null);
int aquickselect_ubyte(npy_ubyte *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_short(npy_short *vec, npy_intp cnt, void *null);
int aquickselect_short(npy_short *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_ushort(npy_ushort *vec, npy_intp cnt, void *null);
int aquickselect_ushort(npy_ushort *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_int(npy_int *vec, npy_intp cnt, void *null);
int aquickselect_int(npy_int *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_uint(npy_uint *vec, npy_intp cnt, void *null);
int aquickselect_uint(npy_uint *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_long(npy_long *vec, npy_intp cnt, void *null);
int aquickselect_long(npy_long *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_ulong(npy_ulong *vec, npy_intp cnt, void *null);
int aquickselect_ulong(npy_ulong *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_longlong(npy_longlong *vec, npy_intp cnt, void *null);
int aquickselect_longlong(npy_longlong *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_ulonglong(npy_ulonglong *vec, npy_intp cnt, void *null);
int aquickselect_ulonglong(npy_ulonglong *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_half(npy_ushort *vec, npy_intp cnt, void *null);
int aquickselect_half(npy_ushort *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_float(npy_float *vec, npy_intp cnt, void *null);
int aquickselect_float(npy_float *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_double(npy_double *vec, npy_intp cnt, void *null);
int aquickselect_double(npy_double *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_longdouble(npy_longdouble *vec, npy_intp cnt, void *null);
int aquickselect_longdouble(npy_longdouble *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_cfloat(npy_cfloat *vec, npy_intp cnt, void *null);
int aquickselect_cfloat(npy_cfloat *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_cdouble(npy_cdouble *vec, npy_intp cnt, void *null);
int aquickselect_cdouble(npy_cdouble *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_clongdouble(npy_clongdouble *vec, npy_intp cnt, void *null);
int aquickselect_clongdouble(npy_clongdouble *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_string(npy_char *vec, npy_intp cnt, PyArrayObject *arr);
int aquickselect_string(npy_char *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int quickselect_unicode(npy_ucs4 *vec, npy_intp cnt, PyArrayObject *arr);
int aquickselect_unicode(npy_ucs4 *vec, npy_intp *ind, npy_intp cnt, npy_intp kth, void *null);


int npy_quickselect(void *base, size_t num, size_t size, npy_comparator cmp);

#endif
