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
from libc.string cimport memset, strlen
from cpython.bytes cimport PyBytes_FromStringAndSize

from threading import RLock

from ._codec import validate_metadata
from ._errors import DataEconError

cdef extern from "daec.h":
    const char *DE_VERSION
    ctypedef void *de_file
    ctypedef int64_t obj_id_t
    ctypedef int64_t axis_id_t
    ctypedef int64_t date_t
    ctypedef enum class_t:
        class_tseries
    ctypedef enum type_t:
        type_tseries
        type_float
    ctypedef enum frequency_t:
        freq_none
        freq_monthly
    ctypedef enum axis_type_t:
        axis_range
    enum:
        DE_SUCCESS
        DE_OBJ_DNE
        DE_EXISTS
        DE_MIS_ATTR
    ctypedef struct object_t:
        obj_id_t id
        obj_id_t pid
        class_t obj_class
        type_t obj_type
        const char *name
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
    const char *de_version()
    int de_open(const char *, de_file *)
    int de_open_readonly(const char *, de_file *)
    int de_close(de_file)
    int de_error(char *, size_t)
    int de_clear_error()
    int de_find_object(de_file, obj_id_t, const char *, obj_id_t *)
    int de_get_attribute(de_file, obj_id_t, const char *, const char **)
    int de_pack_year_period_date(frequency_t, int32_t, uint32_t, date_t *)
    int de_unpack_year_period_date(frequency_t, date_t, int32_t *, uint32_t *)
    int de_axis_range(de_file, int64_t, frequency_t, int64_t, axis_id_t *)
    int de_store_tseries(de_file, obj_id_t, const char *, type_t, type_t,
                        frequency_t, axis_id_t, int64_t, const void *, obj_id_t *)
    int de_load_tseries(de_file, obj_id_t, tseries_t *)

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


def version():
    """Return the compiled header and loaded library versions."""
    with _lock:
        return (<bytes>DE_VERSION).decode("ascii"), (<bytes>de_version()).decode("ascii")


def abi_layout():
    """Return compiler sizes/field offsets for comparison with the Julia ABI."""
    cdef object_t obj
    cdef axis_t ax
    cdef tseries_t ts
    return {
        "enums": (sizeof(class_t), sizeof(type_t), sizeof(frequency_t), sizeof(axis_type_t)),
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

    def __init__(self, str path, bint readonly):
        cdef bytes filename = path.encode("utf-8")
        if self.initialized:
            raise ValueError("Cannot reinitialize a native DataEcon handle.")
        self.initialized = True
        self.path = path
        if b"\0" in filename:
            raise ValueError("NUL is not allowed in a DataEcon path.")
        with _lock:
            if version() != ("0.4.0", "0.4.0"):
                raise ImportError("DataEcon requires matching 0.4.0 header and library.")
            if readonly:
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

    def read(self, str name):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef tseries_t ts
        cdef int32_t year = 0
        cdef uint32_t month = 0
        cdef const char *attribute = NULL
        cdef bytes key
        cdef int rc
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        with _lock:
            self.require_open()
            check(de_find_object(self.handle, 0, encoded, &oid), "find", self.path, name)
            memset(&ts, 0, sizeof(ts))
            check(de_load_tseries(self.handle, oid, &ts), "read", self.path, name)
            metadata = (int(ts.object.obj_class), int(ts.object.obj_type), int(ts.eltype),
                        int(ts.elfreq), int(ts.axis.ax_type), int(ts.axis.length),
                        int(ts.axis.frequency), int(ts.axis.first), int(ts.nbytes))
            validate_metadata(metadata)
            if ts.value == NULL or ts.object.name == NULL:
                raise ValueError("DataEcon returned a NULL series payload or name.")
            loaded_name = (<bytes>ts.object.name).decode("utf-8")
            payload = PyBytes_FromStringAndSize(<const char *>ts.value, ts.nbytes)
            # All borrowed data is now owned by Python, before further C calls.
            for key in (b"jtype", b"jeltype"):
                rc = de_get_attribute(self.handle, oid, key, &attribute)
                if rc == DE_MIS_ATTR:
                    de_clear_error()
                else:
                    check(rc, "attribute", self.path, name)
                    raise TypeError("Julia reconstruction attributes are not supported yet.")
            check(de_unpack_year_period_date(freq_monthly, ts.axis.first, &year, &month),
                  "unpack_date", self.path, name)
            if not 1 <= month <= 12 or int(year) * 12 + int(month) - 1 != ts.axis.first:
                raise ValueError("Date does not round-trip through the native monthly codec.")
            return int(year), int(month), payload, metadata, loaded_name

    def write(self, str name, int32_t year, uint32_t month, bytes payload):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef axis_id_t axis = 0
        cdef date_t first = 0
        cdef int32_t decoded_year = 0
        cdef uint32_t decoded_month = 0
        cdef int rc
        cdef int64_t length = len(payload) // 8
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        if not 1 <= month <= 12 or not -178956970 <= year <= 178956969:
            raise ValueError("Date is outside the native monthly encoding range.")
        validate_metadata((2, 12, 4, 0, 1, length, 32, int(year) * 12 + month - 1, len(payload)))
        with _lock:
            self.require_open()
            rc = de_find_object(self.handle, 0, encoded, &oid)
            if rc == DE_SUCCESS:
                raise DataEconError(DE_EXISTS, "write", self.path, "Object already exists.", name)
            if rc != DE_OBJ_DNE:
                check(rc, "find", self.path, name)
            de_clear_error()
            check(de_pack_year_period_date(freq_monthly, year, month, &first),
                  "pack_date", self.path, name)
            # The native decoder has a narrower reliable range than its int64
            # signature suggests. Reject non-round-trippable dates before storage.
            for code in (first, first + length - 1):
                check(de_unpack_year_period_date(freq_monthly, code, &decoded_year, &decoded_month),
                      "unpack_date", self.path, name)
                if (not 1 <= decoded_month <= 12 or
                        int(decoded_year) * 12 + int(decoded_month) - 1 != code):
                    raise ValueError("Date does not round-trip through the native monthly codec.")
            check(de_axis_range(self.handle, length, freq_monthly, first, &axis),
                  "axis", self.path, name)
            check(de_store_tseries(self.handle, 0, encoded, type_tseries, type_float,
                                  freq_none, axis, len(payload), <const char *>payload, &oid),
                  "write (partial object may remain)", self.path, name)
