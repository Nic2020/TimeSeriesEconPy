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

from ._codec import (series_frequency, validate_date_code, validate_metadata,
                     validate_scalar_metadata)
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
    ctypedef enum type_t:
        type_integer
        type_unsigned
        type_date
        type_float
        type_complex
        type_string
        type_tseries
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
    int de_pack_year_period_date(frequency_t, int32_t, uint32_t, date_t *)
    int de_unpack_year_period_date(frequency_t, date_t, int32_t *, uint32_t *)
    int de_pack_calendar_date(frequency_t, int32_t, uint32_t, uint32_t, date_t *)
    int de_unpack_calendar_date(frequency_t, date_t, int32_t *, uint32_t *, uint32_t *)
    int de_axis_range(de_file, int64_t, frequency_t, int64_t, axis_id_t *)
    int de_store_tseries(de_file, obj_id_t, const char *, type_t, type_t,
                        frequency_t, axis_id_t, int64_t, const void *, obj_id_t *)
    int de_load_tseries(de_file, obj_id_t, tseries_t *)
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


cdef void verify_scalar_date(frequency_t freq, int64_t code, str path, str name) except *:
    # Caller owns the native lock; validate_date_code already bounded the code for
    # this frequency. Unit codes are Julia's pass-through and see no native codec.
    # Calendar codes must decode to a year/month/day that re-encodes to the same
    # code: the native decoder has no range check of its own and the encoder only
    # checks the year, so the explicit window plus this round trip together
    # exclude every code that wraps in either direction.
    cdef int32_t year = 0
    cdef uint32_t month = 0
    cdef uint32_t day = 0
    cdef date_t packed = 0
    cdef int32_t native_year = 0
    cdef uint32_t native_period = 0
    if freq == freq_unit:
        return
    if freq == freq_daily or freq == freq_bdaily or freq_weekly_mon <= freq <= freq_weekly_sun7:
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
    return {
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
            if (ts.nbytes > 0 and ts.value == NULL) or ts.object.name == NULL:
                raise ValueError("DataEcon returned a NULL series payload or name.")
            loaded_name = (<bytes>ts.object.name).decode("utf-8")
            payload = b"" if ts.nbytes == 0 else PyBytes_FromStringAndSize(<const char *>ts.value, ts.nbytes)
            # All borrowed data is now owned by Python, before further C calls.
            for key in (b"jtype", b"jeltype"):
                rc = de_get_attribute(self.handle, oid, key, &attribute)
                if rc == DE_MIS_ATTR:
                    de_clear_error()
                else:
                    check(rc, "attribute", self.path, name)
                    if attribute == NULL:
                        raise TypeError("DataEcon returned a NULL reconstruction attribute.")
                    # Copy the borrowed C string before any subsequent native call.
                    marker = <bytes>attribute
                    if key != b"jeltype" or ts.axis.length != 0 or marker != b"Float64":
                        raise TypeError("Unsupported Julia reconstruction attribute.")
            year, month = unpack_date(ts.axis.frequency, ts.axis.first, self.path, name)
            if ts.axis.frequency != freq_monthly and ts.axis.length > 0:
                unpack_date(ts.axis.frequency, ts.axis.first + ts.axis.length - 1, self.path, name)
            return int(year), int(month), payload, metadata, loaded_name

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

    def write(self, str name, frequency, year, period, bytes payload, bint overwrite=False):
        cdef bytes encoded = name.encode("utf-8")
        cdef obj_id_t oid = 0
        cdef axis_id_t axis = 0
        cdef date_t first = 0
        cdef frequency_t freq
        cdef int32_t native_year
        cdef uint32_t native_period
        cdef int rc
        cdef int64_t length = len(payload) // 8
        cdef const void *value = NULL
        cdef bint existing = False
        if not encoded or b"/" in encoded or b"\0" in encoded:
            raise ValueError("Expected a nonempty root object name without '/' or NUL.")
        ppy = series_frequency(frequency).periods_per_year
        if type(year) is not int or type(period) is not int or not 1 <= period <= ppy:
            raise ValueError("Expected an integer year and valid period.")
        if frequency == 32 and not -178956970 <= year <= 178956969:
            raise ValueError("Date is outside the native monthly encoding range.")
        expected_first = year * ppy + period - 1
        validate_metadata((2, 12, 4, 0, 1, length, frequency, expected_first, len(payload)))
        # All Python arithmetic and bounds checks precede narrowing into C types.
        freq = <frequency_t><uint32_t>frequency
        native_year = year
        native_period = period
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
            check(de_pack_year_period_date(freq, native_year, native_period, &first),
                  "pack_date", self.path, name)
            if first != expected_first:
                raise ValueError("Date does not round-trip through the native date codec.")
            unpack_date(freq, first, self.path, name)
            if length > 0:
                unpack_date(freq, first + length - 1, self.path, name)
            if existing:
                # Every validation of the new series is complete; delete only now.
                self.replace_existing(oid, "write (overwrite)", name)
            check(de_axis_range(self.handle, length, freq, first, &axis),
                  "axis", self.path, name)
            if length > 0:
                value = <const char *>payload
            check(de_store_tseries(self.handle, 0, encoded, type_tseries, type_float,
                                  freq_none, axis, len(payload), value, &oid),
                  "write (overwrite; original deleted, partial replacement may remain)"
                  if existing else "write (partial object may remain)", self.path, name)



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
                verify_scalar_date(scal.frequency, code, self.path, name)
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
                verify_scalar_date(freq, code, self.path, name)
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
