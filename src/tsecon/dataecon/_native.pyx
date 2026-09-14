# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=False
# SPDX-License-Identifier: MIT
"""Thin wrapper of the DataEcon 0.4.0 public C ABI.

Declarations follow bankofcanada/DataEcon at
1a108688a044380f808bebf64079e32dbb9cd1a4, include/daec.h.
See DATAECON_LICENSE.txt. Borrowed results are copied before another native call.
"""

from libc.stdint cimport int32_t, uint32_t, int64_t
from libc.stddef cimport size_t
from libc.string cimport memcpy, memset, strlen
from cpython.bytes cimport PyBytes_FromStringAndSize

from threading import RLock

from ._codec import (series_frequency, tensor_metadata, validate_array_metadata,
                     validate_array_payload, validate_date_code, validate_matrix_metadata,
                     validate_matrix_payload, validate_metadata, validate_scalar_metadata,
                     validate_series_payload, validate_tensor_metadata, validate_tensor_payload)
from ._errors import DataEconError

cdef extern from "daec.h":
    const char *DE_VERSION
    ctypedef void *de_file
    ctypedef int64_t obj_id_t
    ctypedef int64_t axis_id_t
    ctypedef int64_t date_t
    ctypedef enum class_t:
        class_catalog
        class_tseries
        class_matrix
        class_tensor
    ctypedef enum type_t:
        type_integer
        type_unsigned
        type_date
        type_float
        type_complex
        type_string
        type_vector
        type_range
        type_tseries
        type_matrix
        type_mvtseries
        type_tensor
    ctypedef enum frequency_t:
        freq_none
        freq_unit
        freq_daily
        freq_bdaily
        freq_weekly_mon
        freq_weekly_sun7
        freq_monthly
        freq_quarterly_jan
        freq_quarterly_feb
        freq_quarterly_mar
        freq_halfyearly_jan
        freq_halfyearly_jun
        freq_yearly_jan
        freq_yearly_dec
    ctypedef enum axis_type_t:
        axis_plain
        axis_range
        axis_names
    enum:
        DE_SUCCESS
        DE_OBJ_DNE
        DE_EXISTS
        DE_MIS_ATTR
        DE_MAX_AXES
    ctypedef struct object_t:
        obj_id_t id
        obj_id_t pid
        class_t obj_class
        type_t obj_type
        const char *name
    ctypedef struct scalar_t:
        object_t object
        frequency_t frequency
        int64_t nbytes
        const void *value
    ctypedef struct axis_t:
        axis_id_t id
        axis_type_t ax_type
        int64_t length
        frequency_t frequency
        int64_t first
        const char *names
    ctypedef struct tseries_t:
        object_t object
        type_t eltype
        frequency_t elfreq
        axis_t axis
        int64_t nbytes
        const void *value
    ctypedef struct mvtseries_t:
        object_t object
        type_t eltype
        frequency_t elfreq
        axis_t axis1
        axis_t axis2
        int64_t nbytes
        const void *value
    # axis[DE_MAX_AXES]; the header's value is asserted to be five at handle
    # creation so the literal below cannot silently disagree with it.
    ctypedef struct ndtseries_t:
        object_t object
        type_t eltype
        frequency_t elfreq
        int64_t naxes
        axis_t axis[5]
        int64_t nbytes
        const void *value
    const char *de_version()
    int de_open(const char *, de_file *)
    int de_open_readonly(const char *, de_file *)
    int de_open_memory(de_file *)
    int de_close(de_file)
    int de_truncate(de_file)
    int de_load_object(de_file, obj_id_t, object_t *)
    int de_delete_object(de_file, obj_id_t)
    int de_catalog_size(de_file, obj_id_t, int64_t *)
    int de_error(char *, size_t)
    int de_clear_error()
    int de_find_object(de_file, obj_id_t, const char *, obj_id_t *)
    int de_get_attribute(de_file, obj_id_t, const char *, const char **)
    int de_set_attribute(de_file, obj_id_t, const char *, const char *)
    int de_pack_year_period_date(frequency_t, int32_t, uint32_t, date_t *)
    int de_unpack_year_period_date(frequency_t, date_t, int32_t *, uint32_t *)
    int de_pack_calendar_date(frequency_t, int32_t, uint32_t, uint32_t, date_t *)
    int de_unpack_calendar_date(frequency_t, date_t, int32_t *, uint32_t *, uint32_t *)
    int de_axis_range(de_file, int64_t, frequency_t, int64_t, axis_id_t *)
    int de_axis_plain(de_file, int64_t, axis_id_t *)
    int de_axis_names(de_file, int64_t, const char *, axis_id_t *)
    int de_store_tseries(de_file, obj_id_t, const char *, type_t, type_t,
                        frequency_t, axis_id_t, int64_t, const void *, obj_id_t *)
    int de_load_tseries(de_file, obj_id_t, tseries_t *)
    int de_store_mvtseries(de_file, obj_id_t, const char *, type_t, type_t,
                          frequency_t, axis_id_t, axis_id_t, int64_t, const void *, obj_id_t *)
    int de_load_mvtseries(de_file, obj_id_t, mvtseries_t *)
    int de_store_ndtseries(de_file, obj_id_t, const char *, type_t, type_t, frequency_t,
                           int64_t, const axis_id_t *, int64_t, const void *, obj_id_t *)
    int de_load_ndtseries(de_file, obj_id_t, ndtseries_t *)
    int de_store_scalar(de_file, obj_id_t, const char *, type_t, frequency_t,
                        int64_t, const void *, obj_id_t *)
    int de_load_scalar(de_file, obj_id_t, scalar_t *)

# DataEcon error state is process-global. The lock covers copying and error
# extraction too, and is shared by every file owned by this module.
_lock = RLock()


cdef void check(int rc, str operation, str path, object name=None) except *:
    cdef char message[4096]
    if rc == DE_SUCCESS:
        return
    memset(message, 0, sizeof(message))
    de_error(message, sizeof(message))
    message[sizeof(message) - 1] = 0
    text = (<bytes>message).decode("utf-8", "replace")
    if strlen(message) == sizeof(message) - 1:
        text += " [message may be truncated]"
    raise DataEconError(rc, operation, path, text, name)


cdef tuple unpack_date(frequency_t freq, date_t code, str path, str name):
    # Caller owns the native lock and has already checked date/payload bounds.
    cdef int32_t year = 0
    cdef uint32_t period = 0
    cdef uint32_t ppy
    if freq == freq_monthly:
        ppy = 12
    elif freq in (freq_quarterly_jan, freq_quarterly_feb, freq_quarterly_mar):
        ppy = 4
    elif freq_halfyearly_jan <= freq <= freq_halfyearly_jun:
        ppy = 2
    elif freq_yearly_jan <= freq <= freq_yearly_dec:
        ppy = 1
    else:
        raise TypeError("Unsupported DataEcon series frequency.")
    check(de_unpack_year_period_date(freq, code, &year, &period), "unpack_date", path, name)
    if not 1 <= period <= ppy or int(year) * ppy + int(period) - 1 != code:
        raise ValueError("Date does not round-trip through the native date codec.")
    return int(year), int(period)


cdef bint is_calendar(frequency_t freq) noexcept:
    return freq == freq_daily or freq == freq_bdaily or freq_weekly_mon <= freq <= freq_weekly_sun7


cdef void verify_date(frequency_t freq, int64_t code, str path, str name) except *:
    # Caller owns the native lock; validate_date_code (scalars) or validate_metadata
    # (series axes) already bounded the code for this frequency. Unit codes are
    # Julia's pass-through and see no native codec. Calendar codes must decode to
    # a year/month/day that re-encodes to the same code: the native decoder has no
    # range check of its own and the encoder only checks the year, so the explicit
    # window plus this round trip together exclude every code that wraps in either
    # direction. Year/period codes must pack from their year and period to the
    # same code and unpack back to them.
    cdef int32_t year = 0
    cdef uint32_t month = 0
    cdef uint32_t day = 0
    cdef date_t packed = 0
    cdef int32_t native_year = 0
    cdef uint32_t native_period = 0
    if freq == freq_unit:
        return
    if is_calendar(freq):
        check(de_unpack_calendar_date(freq, code, &year, &month, &day),
              "unpack_date", path, name)
        check(de_pack_calendar_date(freq, year, month, day, &packed), "pack_date", path, name)
        if packed != code:
            raise ValueError("Date does not round-trip through the native date codec.")
        return
    ppy = series_frequency(freq).periods_per_year
    year_py, period_index = divmod(int(code), ppy)
    native_year = year_py
    native_period = period_index + 1
    check(de_pack_year_period_date(freq, native_year, native_period, &packed),
          "pack_date", path, name)
    if packed != code:
        raise ValueError("Date does not round-trip through the native date codec.")
    unpack_date(freq, code, path, name)


def version():
    """Return the compiled header and loaded library versions."""
    with _lock:
        return (<bytes>DE_VERSION).decode("ascii"), (<bytes>de_version()).decode("ascii")


def abi_layout():
    """Return compiler sizes/field offsets for comparison with the Julia ABI."""
    cdef object_t obj
    cdef axis_t ax
    cdef tseries_t ts
    cdef scalar_t scal
    cdef mvtseries_t mv
    cdef ndtseries_t nd
    return {
        "max_axes": int(DE_MAX_AXES),
        "mvtseries_t": (sizeof(mvtseries_t), tuple([
            <size_t>(<char *>&mv.object - <char *>&mv),
            <size_t>(<char *>&mv.eltype - <char *>&mv),
            <size_t>(<char *>&mv.elfreq - <char *>&mv),
            <size_t>(<char *>&mv.axis1 - <char *>&mv),
            <size_t>(<char *>&mv.axis2 - <char *>&mv),
            <size_t>(<char *>&mv.nbytes - <char *>&mv),
            <size_t>(<char *>&mv.value - <char *>&mv)])),
        "ndtseries_t": (sizeof(ndtseries_t), tuple([
            <size_t>(<char *>&nd.object - <char *>&nd),
            <size_t>(<char *>&nd.eltype - <char *>&nd),
            <size_t>(<char *>&nd.elfreq - <char *>&nd),
            <size_t>(<char *>&nd.naxes - <char *>&nd),
            <size_t>(<char *>&nd.axis - <char *>&nd),
            <size_t>(<char *>&nd.nbytes - <char *>&nd),
            <size_t>(<char *>&nd.value - <char *>&nd)])),
        "enums": (sizeof(class_t), sizeof(type_t), sizeof(frequency_t), sizeof(axis_type_t)),
        "scalar_t": (sizeof(scalar_t), tuple([
            <size_t>(<char *>&scal.object - <char *>&scal),
            <size_t>(<char *>&scal.frequency - <char *>&scal),
            <size_t>(<char *>&scal.nbytes - <char *>&scal),
            <size_t>(<char *>&scal.value - <char *>&scal)])),
        "object_t": (sizeof(object_t), tuple([
            <size_t>(<char *>&obj.id - <char *>&obj),
            <size_t>(<char *>&obj.pid - <char *>&obj),
            <size_t>(<char *>&obj.obj_class - <char *>&obj),
            <size_t>(<char *>&obj.obj_type - <char *>&obj),
            <size_t>(<char *>&obj.name - <char *>&obj)])),
        "axis_t": (sizeof(axis_t), tuple([
            <size_t>(<char *>&ax.id - <char *>&ax),
            <size_t>(<char *>&ax.ax_type - <char *>&ax),
            <size_t>(<char *>&ax.length - <char *>&ax),
            <size_t>(<char *>&ax.frequency - <char *>&ax),
            <size_t>(<char *>&ax.first - <char *>&ax),
            <size_t>(<char *>&ax.names - <char *>&ax)])),
        "tseries_t": (sizeof(tseries_t), tuple([
            <size_t>(<char *>&ts.object - <char *>&ts),
            <size_t>(<char *>&ts.eltype - <char *>&ts),
            <size_t>(<char *>&ts.elfreq - <char *>&ts),
            <size_t>(<char *>&ts.axis - <char *>&ts),
            <size_t>(<char *>&ts.nbytes - <char *>&ts),
            <size_t>(<char *>&ts.value - <char *>&ts)])),
    }


cdef class FileHandle:
    cdef de_file handle
    cdef int state  # 0: never opened/closed, 1: open, 2: close failed
    cdef bint initialized
    cdef str path

    def __cinit__(self):
        self.handle = NULL
        self.state = 0
        self.initialized = False

    def __init__(self, str path, bint readonly, bint memory=False):
        # memory=True opens a private in-memory database through de_open_memory;
        # the path is then the literal ":memory:" label and no file is touched.
        cdef bytes filename = path.encode("utf-8")
        if self.initialized:
            raise ValueError("Cannot reinitialize a native DataEcon handle.")
        self.initialized = True
        self.path = path
        if b"\0" in filename:
            raise ValueError("NUL is not allowed in a DataEcon path.")
        if memory and (readonly or path != ":memory:"):
            raise ValueError("In-memory DataEcon databases are writable and labelled ':memory:'.")
        with _lock:
            if version() != ("0.4.0", "0.4.0"):
                raise ImportError("DataEcon requires matching 0.4.0 header and library.")
            if DE_MAX_AXES != 5:
                raise ImportError("DataEcon's DE_MAX_AXES differs from the five-slot declaration.")
            if memory:
                check(de_open_memory(&self.handle), "open", path)
            elif readonly:
                check(de_open_readonly(filename, &self.handle), "open", path)
            else:
                check(de_open(filename, &self.handle), "open", path)
            self.state = 1

    cdef void require_open(self) except *:
        if self.state != 1:
            raise ValueError("DataEcon handle is closed or its close failed.")

    def close(self):
        cdef int rc
        with _lock:
            if self.state == 0:
                return
            self.require_open()
            rc = de_close(self.handle)
            # Never retry after failure: upstream finalization retry is unsafe.
            self.state = 2 if rc else 0
            if rc == 0:
                self.handle = NULL
            check(rc, "close", self.path)

    def __dealloc__(self):
        if self.state == 1 and self.handle != NULL:
            # Best effort only. Explicit close/context exit is the durability API.
            with _lock:
                de_close(self.handle)
                de_clear_error()
            self.handle = NULL

    cdef tuple load_matrix(self, obj_id_t oid, str name):
        # Caller owns the native lock and has already found the object. The
        # borrowed payload and names pointer are copied into Python objects
        # before any further native call, as for one-dimensional objects.
        cdef mvtseries_t mv
        cdef const char *attribute = NULL
        cdef bytes key
        cdef int rc
        memset(&mv, 0, sizeof(mv))
        check(de_load_mvtseries(self.handle, oid, &mv), "read matrix", self.path, name)
        metadata = (int(mv.object.obj_class), int(mv.object.obj_type), int(mv.eltype),
                    int(mv.elfreq),
                    int(mv.axis1.ax_type), int(mv.axis1.length), int(mv.axis1.frequency),
                    int(mv.axis1.first),
                    int(mv.axis2.ax_type), int(mv.axis2.length), int(mv.axis2.frequency),
                    int(mv.axis2.first), int(mv.nbytes))
        validate_matrix_metadata(metadata)
        if (mv.nbytes > 0 and mv.value == NULL) or mv.object.name == NULL:
            raise ValueError("DataEcon returned a NULL matrix payload or name.")
        if mv.axis2.ax_type == axis_names and mv.axis2.names == NULL:
            raise ValueError("DataEcon returned a NULL column names axis.")
        loaded_name = (<bytes>mv.object.name).decode("utf-8")
        payload = b"" if mv.nbytes == 0 else PyBytes_FromStringAndSize(
            <const char *>mv.value, mv.nbytes)
        names = None
        if mv.axis2.ax_type == axis_names:
            names = (<bytes>mv.axis2.names).decode("utf-8")
        marker = None
        object_marker = None
        for key in (b"jeltype", b"jtype"):
            rc = de_get_attribute(self.handle, oid, key, &attribute)
            if rc == DE_MIS_ATTR:
                de_clear_error()
            else:
                check(rc, "matrix attribute", self.path, name)
                if attribute == NULL:
                    raise TypeError("DataEcon returned a NULL reconstruction attribute.")
                text = (<bytes>attribute).decode("utf-8")
                if key == b"jeltype":
                    marker = text
                else:
                    object_marker = text
        validate_matrix_payload(metadata, payload, marker, object_marker, names)
        if mv.axis1.ax_type == axis_range and mv.axis1.frequency != freq_unit:
            if is_calendar(mv.axis1.frequency):
                verify_date(mv.axis1.frequency, mv.axis1.first, self.path, name)
            else:
                unpack_date(mv.axis1.frequency, mv.axis1.first, self.path, name)
                if mv.axis1.frequency != freq_monthly and mv.axis1.length > 0:
                    unpack_date(mv.axis1.frequency, mv.axis1.first + mv.axis1.length - 1,
                                self.path, name)
        return payload, metadata, loaded_name, marker, object_marker, names

    cdef tuple load_tensor(self, obj_id_t oid, str name):
        # Caller owns the native lock and has already found the object. The
        # header and all five axis slots are validated before the borrowed value
        # pointer is read; the payload is then copied before any further native
        # call, as for the one- and two-dimensional loaders.
        cdef ndtseries_t nd
        cdef const char *attribute = NULL
        cdef bytes key
        cdef int rc
        cdef int i
        memset(&nd, 0, sizeof(nd))
        check(de_load_ndtseries(self.handle, oid, &nd), "read tensor", self.path, name)
        slots = []
        for i in range(5):
            slots.extend((int(nd.axis[i].id), int(nd.axis[i].ax_type), int(nd.axis[i].length),
                          int(nd.axis[i].frequency), int(nd.axis[i].first)))
        metadata = (int(nd.object.obj_class), int(nd.object.obj_type), int(nd.eltype),
                    int(nd.elfreq), int(nd.naxes), int(nd.nbytes), *slots)
        validate_tensor_metadata(metadata)
        if (nd.nbytes > 0 and nd.value == NULL) or nd.object.name == NULL:
            raise ValueError("DataEcon returned a NULL tensor payload or name.")
        loaded_name = (<bytes>nd.object.name).decode("utf-8")
        payload = b"" if nd.nbytes == 0 else PyBytes_FromStringAndSize(
            <const char *>nd.value, nd.nbytes)
        marker = None
        object_marker = None
        for key in (b"jeltype", b"jtype"):
            rc = de_get_attribute(self.handle, oid, key, &attribute)
            if rc == DE_MIS_ATTR:
                de_clear_error()
            else:
                check(rc, "tensor attribute", self.path, name)
                if attribute == NULL:
                    raise TypeError("DataEcon returned a NULL reconstruction attribute.")
                text = (<bytes>attribute).decode("utf-8")
                if key == b"jeltype":
                    marker = text
                else:
                    object_marker = text
        validate_tensor_payload(metadata, payload, marker, object_marker)
        return payload, metadata, loaded_name, marker, object_marker, None

    cdef obj_id_t locate(self, bytes encoded, str name, int *obj_class) except? -1:
        # One find plus one object load decides which loader applies, so a
        # two-dimensional object never reaches de_load_tseries.
        cdef obj_id_t oid = 0
        cdef object_t obj
        check(de_find_object(self.handle, 0, encoded, &oid), "find", self.path, name)
        memset(&obj, 0, sizeof(obj))
        check(de_load_object(self.handle, oid, &obj), "find", self.path, name)
        obj_class[0] = int(obj.obj_class)
        return oid

    def read(self, str name):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef tseries_t ts
        cdef const char *attribute = NULL
        cdef bytes key
        cdef int rc
        cdef int obj_class = 0
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        with _lock:
            self.require_open()
            oid = self.locate(encoded, name, &obj_class)
            if obj_class == <int>class_matrix:
                return self.load_matrix(oid, name)
            if obj_class == <int>class_tensor:
                raise TypeError(
                    "This object is an N-dimensional DataEcon array; read it with read_array."
                )
            memset(&ts, 0, sizeof(ts))
            check(de_load_tseries(self.handle, oid, &ts), "read", self.path, name)
            metadata = (int(ts.object.obj_class), int(ts.object.obj_type), int(ts.eltype),
                        int(ts.elfreq), int(ts.axis.ax_type), int(ts.axis.length),
                        int(ts.axis.frequency), int(ts.axis.first), int(ts.nbytes))
            validate_metadata(metadata)
            if (ts.nbytes > 0 and ts.value == NULL) or ts.object.name == NULL:
                raise ValueError("DataEcon returned a NULL series payload or name.")
            loaded_name = (<bytes>ts.object.name).decode("utf-8")
            payload = b"" if ts.nbytes == 0 else PyBytes_FromStringAndSize(<const char *>ts.value, ts.nbytes)
            marker = None
            object_marker = None
            # All borrowed data is now owned by Python, before further C calls.
            # Both reserved attributes are copied out as NUL-terminated C
            # strings (the ABI carries no length); each is owned before the
            # next native call. Precedence and validation happen in Python.
            for key in (b"jeltype", b"jtype"):
                rc = de_get_attribute(self.handle, oid, key, &attribute)
                if rc == DE_MIS_ATTR:
                    de_clear_error()
                else:
                    check(rc, "attribute", self.path, name)
                    if attribute == NULL:
                        raise TypeError("DataEcon returned a NULL reconstruction attribute.")
                    text = (<bytes>attribute).decode("utf-8")
                    if key == b"jeltype":
                        marker = text
                    else:
                        object_marker = text
            validate_series_payload(int(ts.axis.frequency), int(ts.eltype), int(ts.elfreq),
                                    int(ts.axis.length), payload, marker, object_marker)
            # validate_metadata bounded first and length, so the last code cannot
            # overflow. Calendar axes take the calendar round trip on the stored
            # first date only, like Julia (trailing codes are implicit and never
            # packed); year/period axes keep their unpack checks (monthly: first
            # only).
            if ts.axis.frequency == freq_unit:
                pass
            elif is_calendar(ts.axis.frequency):
                verify_date(ts.axis.frequency, ts.axis.first, self.path, name)
            else:
                unpack_date(ts.axis.frequency, ts.axis.first, self.path, name)
                if ts.axis.frequency != freq_monthly and ts.axis.length > 0:
                    unpack_date(ts.axis.frequency, ts.axis.first + ts.axis.length - 1,
                                self.path, name)
            return payload, metadata, loaded_name, marker, object_marker, None

    def read_array(self, str name):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef tseries_t ts
        cdef const char *attribute = NULL
        cdef bytes key
        cdef int rc
        cdef int obj_class = 0
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        with _lock:
            self.require_open()
            oid = self.locate(encoded, name, &obj_class)
            if obj_class == <int>class_matrix:
                return self.load_matrix(oid, name)
            if obj_class == <int>class_tensor:
                return self.load_tensor(oid, name)
            memset(&ts, 0, sizeof(ts))
            check(de_load_tseries(self.handle, oid, &ts), "read array", self.path, name)
            metadata = (int(ts.object.obj_class), int(ts.object.obj_type), int(ts.eltype),
                        int(ts.elfreq), int(ts.axis.ax_type), int(ts.axis.length),
                        int(ts.axis.frequency), int(ts.axis.first), int(ts.nbytes))
            validate_array_metadata(metadata)
            if (ts.nbytes > 0 and ts.value == NULL) or ts.object.name == NULL:
                raise ValueError("DataEcon returned a NULL array payload or name.")
            loaded_name = (<bytes>ts.object.name).decode("utf-8")
            payload = b"" if ts.nbytes == 0 else PyBytes_FromStringAndSize(<const char *>ts.value, ts.nbytes)
            marker = None
            object_marker = None
            for key in (b"jeltype", b"jtype"):
                rc = de_get_attribute(self.handle, oid, key, &attribute)
                if rc == DE_MIS_ATTR:
                    de_clear_error()
                else:
                    check(rc, "array attribute", self.path, name)
                    if attribute == NULL:
                        raise TypeError("DataEcon returned a NULL reconstruction attribute.")
                    text = (<bytes>attribute).decode("utf-8")
                    if key == b"jeltype":
                        marker = text
                    else:
                        object_marker = text
            validate_array_payload(metadata, payload, marker, object_marker)
            if ts.object.obj_type == type_range and ts.axis.ax_type == axis_range:
                verify_date(ts.axis.frequency, ts.axis.first, self.path, name)
            return payload, metadata, loaded_name, marker, object_marker, None

    cdef void replace_existing(self, obj_id_t oid, str operation, str name) except *:
        # Caller owns the native lock, has finished every Python/native validation
        # of the new value, and asked for overwrite. Julia deletes whatever exists,
        # catalogs included; Python refuses a catalog here so that recursive
        # deletion stays an explicit request through delete(recursive=True).
        # Delete-then-store is not transactional and nothing rolls back: after
        # this point the original is gone, and a failed store leaves the name
        # absent or holding a partial replacement (de_store_scalar/de_store_tseries
        # create the object row before storing the payload row).
        cdef object_t obj
        memset(&obj, 0, sizeof(obj))
        check(de_load_object(self.handle, oid, &obj), "find", self.path, name)
        if obj.obj_class == class_catalog:
            raise ValueError(
                "Cannot overwrite a catalog; delete it explicitly with recursive=True first."
            )
        check(de_delete_object(self.handle, oid), operation, self.path, name)

    def write(self, str name, frequency, first, bytes payload, bint overwrite,
              element, element_frequency, length, marker, object_marker=None):
        # first is the native date code of the first observation (the axis
        # anchor for an empty series). validate_metadata bounds it to the
        # reliable range of its frequency before it is narrowed to date_t.
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef axis_id_t axis = 0
        cdef date_t native_first = 0
        cdef frequency_t freq
        cdef int rc
        cdef int64_t native_length
        cdef type_t native_element
        cdef frequency_t native_element_frequency
        cdef bytes encoded_marker
        cdef bytes encoded_object_marker
        cdef const void *value = NULL
        cdef bint existing = False
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        if type(frequency) is not int:
            raise TypeError("Series frequency must be an integer native code.")
        if type(first) is not int:
            raise ValueError("Expected an integer first date code.")
        if type(element) is not int:
            raise TypeError("Series element must be an integer native code.")
        if type(element_frequency) is not int:
            raise TypeError("Series element frequency must be an integer native code.")
        if type(length) is not int:
            raise ValueError("Expected an integer series length.")
        validate_metadata((2, 12, element, element_frequency, 1, length, frequency, first, len(payload)))
        validate_series_payload(frequency, element, element_frequency, length, payload, marker,
                                object_marker)
        # Marker text was checked in Python (str, NUL-free); tokens and any
        # inactive text under a whole-object marker are stored verbatim.
        encoded_marker = b"" if marker is None else marker.encode("utf-8")
        encoded_object_marker = b"" if object_marker is None else object_marker.encode("utf-8")
        if frequency == 32 and not -178956970 <= first // 12 <= 178956969:
            # Historical monthly series-write guard: the last eight signed
            # 32-bit monthly codes (year 178956970) stay rejected on write.
            # Scalar dates and reads are unaffected.
            raise ValueError("Date is outside the native monthly encoding range.")
        # All Python arithmetic and bounds checks precede narrowing into C types.
        freq = <frequency_t><uint32_t>frequency
        native_first = first
        native_length = length
        native_element = <type_t><uint32_t>element
        native_element_frequency = <frequency_t><uint32_t>element_frequency
        with _lock:
            self.require_open()
            rc = de_find_object(self.handle, 0, encoded, &oid)
            if rc == DE_SUCCESS:
                if not overwrite:
                    raise DataEconError(DE_EXISTS, "write", self.path,
                                       "Object already exists.", name)
                existing = True
            elif rc != DE_OBJ_DNE:
                check(rc, "find", self.path, name)
            de_clear_error()
            verify_date(freq, native_first, self.path, name)
            if length > 0 and freq != freq_unit and not is_calendar(freq):
                # Year/period axes also check their last date; calendar axes
                # follow Julia and pack the first date only.
                unpack_date(freq, native_first + length - 1, self.path, name)
            if existing:
                # Every validation of the new series is complete; delete only now.
                self.replace_existing(oid, "write (overwrite)", name)
            check(de_axis_range(self.handle, native_length, freq, native_first, &axis),
                  "axis", self.path, name)
            if length > 0:
                value = <const char *>payload
            check(de_store_tseries(self.handle, 0, encoded, type_tseries, native_element,
                                  native_element_frequency, axis, len(payload), value, &oid),
                  "write (overwrite; original deleted, partial replacement may remain)"
                  if existing else "write (partial object may remain)", self.path, name)
            # Julia's order: the element marker first, then the whole-object
            # marker. Each is a separate native statement with no rollback.
            if marker is not None:
                check(de_set_attribute(self.handle, oid, b"jeltype", encoded_marker),
                      "write marker (unmarked object may read as a different dtype; no rollback)",
                      self.path, name)
            if object_marker is not None:
                check(de_set_attribute(self.handle, oid, b"jtype", encoded_object_marker),
                      "write object marker (the element marker, if any, is now active; "
                      "no rollback)",
                      self.path, name)

    def write_array(self, str name, object_type, axis_type, frequency, first,
                    bytes payload, bint overwrite, element, element_frequency,
                    length, marker, object_marker=None):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef axis_id_t axis = 0
        cdef frequency_t freq
        cdef int rc
        cdef const void *value = NULL
        cdef bint existing = False
        cdef bytes encoded_marker
        cdef bytes encoded_object_marker
        metadata = (2, object_type, element, element_frequency, axis_type, length,
                    frequency, first, len(payload))
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        for label, item in (("object type", object_type), ("axis type", axis_type),
                            ("frequency", frequency), ("first", first),
                            ("element", element), ("element frequency", element_frequency),
                            ("length", length)):
            if type(item) is not int:
                raise TypeError(f"Array {label} must be an integer native value.")
        validate_array_payload(metadata, payload, marker, object_marker)
        encoded_marker = b"" if marker is None else marker.encode("utf-8")
        encoded_object_marker = b"" if object_marker is None else object_marker.encode("utf-8")
        freq = <frequency_t><uint32_t>frequency
        with _lock:
            self.require_open()
            rc = de_find_object(self.handle, 0, encoded, &oid)
            if rc == DE_SUCCESS:
                if not overwrite:
                    raise DataEconError(DE_EXISTS, "write array", self.path,
                                       "Object already exists.", name)
                existing = True
            elif rc != DE_OBJ_DNE:
                check(rc, "find", self.path, name)
            de_clear_error()
            if object_type == type_range and axis_type == axis_range:
                verify_date(freq, first, self.path, name)
            if existing:
                self.replace_existing(oid, "write array (overwrite)", name)
            if object_type == type_range and axis_type == axis_range:
                check(de_axis_range(self.handle, length, freq, first, &axis),
                      "array axis", self.path, name)
            else:
                check(de_axis_plain(self.handle, length, &axis), "array axis", self.path, name)
            if len(payload) > 0:
                value = <const char *>payload
            check(de_store_tseries(self.handle, 0, encoded, <type_t><uint32_t>object_type,
                                   <type_t><uint32_t>element,
                                   <frequency_t><uint32_t>element_frequency,
                                   axis, len(payload), value, &oid),
                  "write array (overwrite; original deleted, partial replacement may remain)"
                  if existing else "write array (partial object may remain)", self.path, name)
            if marker is not None:
                check(de_set_attribute(self.handle, oid, b"jeltype", encoded_marker),
                      "write array marker (unmarked object may read differently; no rollback)",
                      self.path, name)
            if object_marker is not None:
                check(de_set_attribute(self.handle, oid, b"jtype", encoded_object_marker),
                      "write array object marker (no rollback)", self.path, name)



    def write_matrix(self, str name, object_type, element, element_frequency,
                     axis1_type, rows, frequency, first, columns, names,
                     bytes payload, bint overwrite, marker, object_marker=None):
        cdef bytes encoded = name.encode("utf-8")
        cdef bytes encoded_names
        cdef obj_id_t oid = 0
        cdef axis_id_t axis1 = 0
        cdef axis_id_t axis2 = 0
        cdef frequency_t freq
        cdef int rc
        cdef const void *value = NULL
        cdef bint existing = False
        cdef bytes encoded_marker
        cdef bytes encoded_object_marker
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        for label, item in (("object type", object_type), ("element", element),
                            ("element frequency", element_frequency),
                            ("row axis type", axis1_type), ("rows", rows),
                            ("frequency", frequency), ("first", first),
                            ("columns", columns)):
            if type(item) is not int:
                raise TypeError(f"Matrix {label} must be an integer native value.")
        if names is not None and type(names) is not str:
            raise TypeError("Matrix column names must be a string or None.")
        metadata = (3, object_type, element, element_frequency, axis1_type, rows, frequency,
                    first, 2 if names is not None else 0, columns, 0, 0, len(payload))
        validate_matrix_payload(metadata, payload, marker, object_marker, names)
        encoded_marker = b"" if marker is None else marker.encode("utf-8")
        encoded_object_marker = b"" if object_marker is None else object_marker.encode("utf-8")
        # The names axis is one NUL-terminated C string; the codec has already
        # refused newline and NUL inside a name, so nothing can be truncated.
        encoded_names = b"" if names is None else names.encode("utf-8")
        if b"\0" in encoded_names:
            raise ValueError("Matrix column names cannot contain NUL.")
        freq = <frequency_t><uint32_t>frequency
        with _lock:
            self.require_open()
            rc = de_find_object(self.handle, 0, encoded, &oid)
            if rc == DE_SUCCESS:
                if not overwrite:
                    raise DataEconError(DE_EXISTS, "write matrix", self.path,
                                       "Object already exists.", name)
                existing = True
            elif rc != DE_OBJ_DNE:
                check(rc, "find", self.path, name)
            de_clear_error()
            if axis1_type == <int>axis_range:
                verify_date(freq, first, self.path, name)
                if rows > 0 and freq != freq_unit and not is_calendar(freq):
                    unpack_date(freq, first + rows - 1, self.path, name)
            if existing:
                # Every validation of the new value is complete; delete only now.
                self.replace_existing(oid, "write matrix (overwrite)", name)
            if axis1_type == <int>axis_range:
                check(de_axis_range(self.handle, rows, freq, first, &axis1),
                      "matrix row axis", self.path, name)
            else:
                check(de_axis_plain(self.handle, rows, &axis1),
                      "matrix row axis", self.path, name)
            if names is None:
                check(de_axis_plain(self.handle, columns, &axis2),
                      "matrix column axis", self.path, name)
            else:
                check(de_axis_names(self.handle, columns, encoded_names, &axis2),
                      "matrix column axis", self.path, name)
            if len(payload) > 0:
                value = <const char *>payload
            check(de_store_mvtseries(self.handle, 0, encoded, <type_t><uint32_t>object_type,
                                     <type_t><uint32_t>element,
                                     <frequency_t><uint32_t>element_frequency,
                                     axis1, axis2, len(payload), value, &oid),
                  "write matrix (overwrite; original deleted, partial replacement may remain)"
                  if existing else "write matrix (partial object may remain)", self.path, name)
            if marker is not None:
                check(de_set_attribute(self.handle, oid, b"jeltype", encoded_marker),
                      "write matrix marker (unmarked object may read differently; no rollback)",
                      self.path, name)
            if object_marker is not None:
                check(de_set_attribute(self.handle, oid, b"jtype", encoded_object_marker),
                      "write matrix object marker (the element marker, if any, is now "
                      "active; no rollback)", self.path, name)

    def write_tensor(self, str name, object_type, element, element_frequency, shape,
                     bytes payload, bint overwrite, marker, object_marker=None):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef axis_id_t axes[5]
        cdef int64_t naxes
        cdef int i
        cdef int rc
        cdef const void *value = NULL
        cdef bint existing = False
        cdef bytes encoded_marker
        cdef bytes encoded_object_marker
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        for label, item in (("object type", object_type), ("element", element),
                            ("element frequency", element_frequency)):
            if type(item) is not int:
                raise TypeError(f"Tensor {label} must be an integer native value.")
        if type(shape) is not tuple or any(type(n) is not int for n in shape):
            raise TypeError("Tensor shape must be a tuple of integers.")
        if object_type != <int>type_tensor:
            raise TypeError("Only plain type_tensor objects are written.")
        metadata = tensor_metadata(element, element_frequency, shape, len(payload))
        # Rank, every dimension, the Python-integer product and the payload are
        # all validated here before anything is narrowed into C types.
        validate_tensor_payload(metadata, payload, marker, object_marker)
        naxes = len(shape)
        encoded_marker = b"" if marker is None else marker.encode("utf-8")
        encoded_object_marker = b"" if object_marker is None else object_marker.encode("utf-8")
        memset(axes, 0, sizeof(axes))
        with _lock:
            self.require_open()
            rc = de_find_object(self.handle, 0, encoded, &oid)
            if rc == DE_SUCCESS:
                if not overwrite:
                    raise DataEconError(DE_EXISTS, "write tensor", self.path,
                                       "Object already exists.", name)
                existing = True
            elif rc != DE_OBJ_DNE:
                check(rc, "find", self.path, name)
            de_clear_error()
            if existing:
                # Every validation of the new value is complete; delete only now.
                self.replace_existing(oid, "write tensor (overwrite)", name)
            for i in range(naxes):
                check(de_axis_plain(self.handle, <int64_t>shape[i], &axes[i]),
                      "tensor axis", self.path, name)
            if len(payload) > 0:
                value = <const char *>payload
            check(de_store_ndtseries(self.handle, 0, encoded, type_tensor,
                                     <type_t><uint32_t>element,
                                     <frequency_t><uint32_t>element_frequency,
                                     naxes, axes, len(payload), value, &oid),
                  "write tensor (overwrite; original deleted, partial replacement may remain)"
                  if existing else "write tensor (partial object may remain)", self.path, name)
            if marker is not None:
                check(de_set_attribute(self.handle, oid, b"jeltype", encoded_marker),
                      "write tensor marker (unmarked object may read differently; no rollback)",
                      self.path, name)
            if object_marker is not None:
                check(de_set_attribute(self.handle, oid, b"jtype", encoded_object_marker),
                      "write tensor object marker (the element marker, if any, is now "
                      "active; no rollback)", self.path, name)

    def read_scalar(self, str name):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef scalar_t scal
        cdef const char *attribute = NULL
        cdef bytes key
        cdef int rc
        cdef int64_t code
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        with _lock:
            self.require_open()
            check(de_find_object(self.handle, 0, encoded, &oid), "find", self.path, name)
            memset(&scal, 0, sizeof(scal))
            check(de_load_scalar(self.handle, oid, &scal), "read_scalar", self.path, name)
            metadata = (int(scal.object.obj_class), int(scal.object.obj_type),
                        int(scal.frequency), int(scal.nbytes))
            # Validation bounds nbytes (eight, or 1..MAX_BYTES for strings) before the copy.
            validate_scalar_metadata(metadata)
            if scal.value == NULL or scal.object.name == NULL:
                raise ValueError("DataEcon returned a NULL scalar payload or name.")
            payload = PyBytes_FromStringAndSize(<const char *>scal.value, scal.nbytes)
            loaded_name = (<bytes>scal.object.name).decode("utf-8")
            # Both borrowed values are owned before attribute calls.
            if scal.object.obj_type == type_date:
                # Decode the owned eight-byte snapshot with memcpy: the SQLite
                # blob carries no int64 alignment guarantee, and the borrowed
                # scal.value is not touched again. Python bounds first, then
                # the native codec must reproduce the code (Unit excepted).
                memcpy(&code, <const char *>payload, sizeof(code))
                validate_date_code(metadata[2], int(code))
                verify_date(scal.frequency, code, self.path, name)
            for key in (b"jtype", b"jeltype"):
                rc = de_get_attribute(self.handle, oid, key, &attribute)
                if rc == DE_MIS_ATTR:
                    de_clear_error()
                else:
                    check(rc, "attribute", self.path, name)
                    raise TypeError("Scalar reconstruction attributes are not supported.")
            return payload, metadata, loaded_name

    def write_scalar(self, str name, int kind, frequency, bytes payload, bint overwrite=False):
        # kind is the validated native scalar type code: 1 (signed integer at a
        # validated width, or a Duration when frequency is nonzero), 2 (unsigned
        # integer), 3 (MIT date), 4 (float at a validated width), 5 (complex) or
        # 6 (string). Date frequencies cover Unit (11), the calendar codes and
        # year/period codes. validate_scalar_metadata bounds every payload width.
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef int rc
        cdef type_t native_type
        cdef frequency_t freq
        cdef int64_t code = 0
        cdef bint existing = False
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        if type(frequency) is not int:
            raise TypeError("Scalar frequency must be an integer native code.")
        validate_scalar_metadata((1, kind, frequency, len(payload)))
        if kind == 4:
            native_type = type_float
        elif kind == 3:
            native_type = type_date
        elif kind == 6:
            native_type = type_string
        elif kind == 2:
            native_type = type_unsigned
        elif kind == 5:
            native_type = type_complex
        else:
            native_type = type_integer
        if kind == 3:
            # validate_scalar_metadata established len(payload) == 8; memcpy
            # avoids assuming the bytes object's buffer is int64-aligned.
            memcpy(&code, <const char *>payload, sizeof(code))
            validate_date_code(frequency, int(code))
        # All Python arithmetic and bounds checks precede narrowing into C types.
        freq = <frequency_t><uint32_t>frequency
        with _lock:
            self.require_open()
            rc = de_find_object(self.handle, 0, encoded, &oid)
            if rc == DE_SUCCESS:
                if not overwrite:
                    raise DataEconError(DE_EXISTS, "write_scalar", self.path,
                                       "Object already exists.", name)
                existing = True
            elif rc != DE_OBJ_DNE:
                check(rc, "find", self.path, name)
            de_clear_error()
            if kind == 3:
                verify_date(freq, code, self.path, name)
            if existing:
                # Every validation of the new value is complete; delete only now.
                self.replace_existing(oid, "write_scalar (overwrite)", name)
            check(de_store_scalar(self.handle, 0, encoded, native_type, freq,
                                  len(payload), <const char *>payload, &oid),
                  "write_scalar (overwrite; original deleted, partial replacement may remain)"
                  if existing else "write_scalar (partial object may remain)", self.path, name)

    def delete(self, str name, bint recursive=False):
        # Delete one root object. A catalog is refused unless recursive is set:
        # the native DELETE cascades through every nested catalog and object.
        # A missing name is DE_OBJ_DNE from the lookup (the native delete itself
        # would silently succeed for a stale id). Axes are not objects and stay.
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef object_t obj
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        with _lock:
            self.require_open()
            check(de_find_object(self.handle, 0, encoded, &oid), "find", self.path, name)
            memset(&obj, 0, sizeof(obj))
            check(de_load_object(self.handle, oid, &obj), "find", self.path, name)
            if obj.obj_class == class_catalog and not recursive:
                raise ValueError(
                    "Object is a catalog; pass recursive=True to delete it and everything "
                    "it contains."
                )
            check(de_delete_object(self.handle, oid), "delete", self.path, name)

    def truncate(self):
        # Reset the file to a freshly created state. The native call commits,
        # finalizes its cached statements, VACUUMs under RESET_DATABASE and
        # re-initializes the schema. A failure quarantines this handle exactly
        # like a failed close: the native statement-finalization path leaves a
        # dangling statement behind when any statement was in an error state,
        # and a later close would be a use-after-free. Nothing native is called
        # on a quarantined handle again; its resources remain until process exit.
        cdef int rc
        with _lock:
            self.require_open()
            rc = de_truncate(self.handle)
            if rc != DE_SUCCESS:
                self.state = 2
            check(rc, "truncate", self.path)

    def catalog_size(self):
        # Number of objects directly under the root catalog (native subtracts
        # the root's self-reference). Zero means the file is empty.
        cdef int64_t count = 0
        with _lock:
            self.require_open()
            check(de_catalog_size(self.handle, 0, &count), "catalog_size", self.path)
            return int(count)
