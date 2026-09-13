# DataEcon interchange

`tsecon.dataecon` reads and writes **supported numeric scalars
(Float16/32/64, Int8/16/32/64, UInt8/16/32/64, Complex64/128), string, MIT date
and Duration scalars, and numeric/Boolean TSeries over the monthly, quarterly,
half-yearly, annual, daily, business-daily and weekly frequencies, including
empty series**, through the DataEcon 0.4.0 C library. Date and duration
scalars cover every core frequency: `Unit`, `Daily`, `BDaily`, `Weekly` with
any end day, `Monthly`, `Quarterly`, `HalfYearly` and `Yearly`.
Series may also hold date, duration, Int128/UInt128 and ComplexF16 elements
through `StoredSeries` (see [Represented series elements](#represented-series-elements)).
Int128/UInt128/ComplexF16 scalars, other marker-reconstructed Julia types,
Unit-frequency series, catalogs, workspaces and general attributes are not supported yet.
Existing JSON I/O is unchanged.

Native DataEcon support is configured in the wheel workflow for CPython 3.11–3.13:
Windows x86-64, Linux x86-64 and macOS arm64. Native wheel builds and Julia
monthly, empty and scalar interchange checks have passed on all three platforms.
Quarterly and annual interchange have also passed on all three platforms.
Half-yearly interchange is included in the configured wheel checks, as are
string, date and duration scalars over every frequency and daily,
business-daily and weekly series. Successful CI builds are separate from a
published release.
The integration uses a thin
Cython extension; CFFI and Julia are not runtime dependencies. Only the native
parts are compiled: the Python file API and conversion code remain Python.

```python
import numpy as np
from tsecon import TSeries, mm
from tsecon.dataecon import open_dataecon

series = TSeries(mm(2024, 1), np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64))
# Use a new file/name: "a" appends without overwriting existing objects.
with open_dataecon("example.daec", "a") as db:
    db.write_series("sample", series)

with open_dataecon("example.daec") as db:  # read-only by default
    restored = db.read_series("sample")

assert restored.firstdate == mm(2024, 1)
assert restored.lastdate == mm(2024, 4)
np.testing.assert_array_equal(restored.values, series.values)
```

Reads return owning TSeries arrays that remain usable after closing the file.
Writes copy the input, including strided arrays and column views. NaNs are
preserved as floating-point missing values. Names must be nonempty root names
without `/` or NUL; Unicode object names are supported. **Windows file paths must
be ASCII**: the pinned native library's narrow `fopen` existence check does not
reliably recognize UTF-8 paths when reopening files. The adapter rejects such
paths before opening or creating anything.

The payload limit is 128 MiB per series. Dates must fit the native encoding;
extreme boundary years can also be rejected before native date arithmetic.
Only little-endian hosts are currently supported. Julia reconstruction attributes
are never evaluated. `jtype` is rejected; `jeltype` is accepted only when its
value is an exact supported empty-element token or the canonical Bool marker. Custom attributes
are outside this API's interchange contract.

## Quarterly series and fiscal anchors

Quarterly series support all three canonical calendars. `end_month` is the
ending month of the first quarter: 1 (January), 2 (February) or 3 (March,
the default). The anchor remains part of the frequency when data is saved.

```python
from tsecon import MIT, Quarterly

start = MIT.from_yp(Quarterly(end_month=1), 2024, 4)
quarterly = TSeries(start, np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64))
with open_dataecon("quarterly-example.daec", "a") as db:
    db.write_series("quarterly", quarterly)
with open_dataecon("quarterly-example.daec") as db:
    restored_quarterly = db.read_series("quarterly")
assert restored_quarterly.frequency == Quarterly(end_month=1)
assert restored_quarterly.firstdate == start
assert restored_quarterly.lastdate == MIT.from_yp(Quarterly(end_month=1), 2025, 3)
np.testing.assert_array_equal(restored_quarterly.values, quarterly.values)
```

Quarterly native date codes are `4 * year + period - 1`. The reliable range is
`-131200` through `2147483647`, inclusive; both observation endpoints must fit.
An empty series checks its stored first-date anchor, while its synthetic last
date may lie below that limit. Invalid dates raise `ValueError` before storage.
Integer date codes alone cannot distinguish fiscal anchors, so the adapter
preserves the native frequency codes 65, 66 and 67 explicitly. Other encodings
are rejected. Quarterly negative and zero years do not require conversion to
Python `datetime.date`.

## Half-yearly series and fiscal endings

Half-yearly series support all six `HalfYearly(end_month=...)` values from 1
through 6. The default is June. `end_month` is the ending month of the first
half-year; the second half-year ends six months later. Each year has two
periods, and the year labels the calendar year containing the first half-year's
end: `HalfYearly(end_month=3)` at 2024 period 1 covers October 2023 through
March 2024, and period 2 covers April through September 2024.

```python
from tsecon import HalfYearly

half_start = MIT.from_yp(HalfYearly(end_month=3), 2024, 1)
half = TSeries(half_start, np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64))
with open_dataecon("halfyearly-example.daec", "a") as db:
    db.write_series("halfyearly", half)
with open_dataecon("halfyearly-example.daec") as db:
    restored_half = db.read_series("halfyearly")
assert restored_half.frequency == HalfYearly(end_month=3)
assert restored_half.firstdate == half_start
assert restored_half.lastdate == MIT.from_yp(HalfYearly(end_month=3), 2025, 2)
np.testing.assert_array_equal(restored_half.values, half.values)
```

Half-yearly native date codes are `2 * year + period - 1`. The reliable range is
`-65600` through `2147483647`, inclusive; both observation endpoints must fit.
This lower limit differs from the quarterly and annual limits because the native
decoder divides by the number of periods per year. An empty series checks its
stored first-date anchor, while its synthetic last date may lie below that limit.
The adapter preserves the native frequency codes 129 through 134 explicitly; the
bare half-yearly code 128 and other encodings are rejected. A stored code below
the limit is rejected on read rather than decoded as a different date.

## Annual series and fiscal year endings

Annual series support every `Yearly(end_month=...)` value from 1 through 12.
The default is December. The year labels the calendar year in which the fiscal
period ends: `Yearly(end_month=6)` at 2024 covers July 2023 through June 2024.

```python
from tsecon import Yearly

annual_start = MIT.from_yp(Yearly(end_month=6), 2024, 1)
annual = TSeries(annual_start, np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64))
with open_dataecon("annual-example.daec", "a") as db:
    db.write_series("annual", annual)
with open_dataecon("annual-example.daec") as db:
    restored_annual = db.read_series("annual")
assert restored_annual.frequency == Yearly(end_month=6)
assert restored_annual.firstdate == annual_start
assert restored_annual.lastdate == MIT.from_yp(Yearly(end_month=6), 2027, 1)
np.testing.assert_array_equal(restored_annual.values, annual.values)
```

Annual native date codes equal the year. Both observation endpoints must fit
`-2147483648` through `2147483647`, inclusive. For an empty series only the stored
first-date anchor is checked. Invalid dates fail before storage; negative and
zero years do not require conversion to Python `datetime.date`. Fiscal endings
remain distinct even when their integer dates and values are identical.

## Daily, business-daily and weekly series

Series over `Daily`, `BDaily` and `Weekly` (any end day) use the native
calendar codec. The stored axis holds the code of the first observation and
the length; observations are consecutive codes, so a business-daily series
continues from Friday to Monday (Monday to Friday only, no holiday calendar
on either side) and a weekly series advances by seven days on its end day.
Year ends and leap days need no special handling.

```python
from tsecon import MIT, BDaily, Daily, Weekly, bdaily, weekly

business = TSeries(bdaily("2024-01-12"), np.array([1.25, -2.5], dtype=np.float64))
weeks = TSeries(weekly("2024-12-30", 3), np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64))
far = TSeries(MIT(Daily(), 3652059), np.array([1.0, 2.0], dtype=np.float64))  # from 31 Dec 9999
with open_dataecon("calendar-series-example.daec", "a") as db:
    db.write_series("business", business)
    db.write_series("weeks", weeks)
    db.write_series("far", far)
with open_dataecon("calendar-series-example.daec") as db:
    restored_business = db.read_series("business")
    restored_weeks = db.read_series("weeks")
    restored_far = db.read_series("far")
assert restored_business.frequency == BDaily()
assert restored_business.lastdate == bdaily("2024-01-15")  # Friday, then Monday
assert restored_weeks.frequency == Weekly(3)
assert restored_weeks.firstdate == weekly("2025-01-01", 3)  # the week ending Wednesday 1 January
assert restored_far.lastdate == MIT(Daily(), 3652060)  # 1 January 10000, beyond datetime
np.testing.assert_array_equal(restored_weeks.values, weeks.values)
```

The stored first date must lie inside the reliable code range of its
frequency (the same windows as the calendar date scalars: daily `-11980259`
through `11979954`, business daily `-8557114` through `8557110`, weekly
`-1711422` through `1711422`) and the native codec must reproduce it;
otherwise `ValueError` is raised before anything is stored. Only the first
date is stored and packed, on both sides: later observations are consecutive
codes, so a series may run past 31 December 32800 exactly as Julia writes and
reloads it (for example two daily values from code `11979954`, or a thousand
values straddling the maximum). The 128 MiB payload limit bounds a float64 series at
16,777,216 values, so a trailing code never exceeds the window maximum plus
16,777,215 and stays inside the signed 32-bit range the adapter checks. A
stored first date below the window, which Julia writes with a warning and
reloads as another date, is rejected on read, as are the weekly axis codes 16
and 24 through 31 that Julia never writes. Codes are never converted through
`datetime`, so years outside 1..9999 round-trip; converting such an `MIT` to a
`date` with the core's `mit_to_date` still raises. The weekly end day is part
of the frequency code, so `Weekly(1)` and `Weekly(7)` series with equal codes
stay distinct. Unit-frequency series are not supported yet.

## Empty series

An empty series retains its first-date anchor and owns an empty float64 array:

```python
empty = TSeries(mm(2024, 1), np.empty(0, dtype=np.float64))
with open_dataecon("empty-example.daec", "a") as db:
    db.write_series("empty", empty)
with open_dataecon("empty-example.daec") as db:
    restored_empty = db.read_series("empty")

assert restored_empty.firstdate == mm(2024, 1)
assert restored_empty.lastdate == mm(2023, 12)
assert restored_empty.values.shape == (0,)
```

The last date follows the core empty-range convention: one period before the
first date. It does not identify a stored observation. Empty inputs retain the
same read-only, closed-file, duplicate-name and native date-range checks.

The pinned Julia writer marks every empty series with its element type. Julia
reloads these marked empties as plain typed vectors, losing the dated wrapper;
Python retains their stored anchor. Empty int64, uint64, float64 and complex128
writes omit the redundant marker, so Julia preserves their dated wrapper too.
Other empty dtypes require a marker to preserve their width; Julia reads these
as plain vectors, while Python returns the anchored TSeries. Unmarked empties
default to int64, uint64, float64 or complex128 according to their native kind.
Unknown or contradictory reconstruction tokens are rejected without evaluation.

## Numeric and Boolean series

`write_series` preserves the native-endian NumPy dtype: int8/16/32/64,
uint8/16/32/64, float16/32/64, complex64/128 or bool. These values use the same
supported date axes described above. Reads own their writable NumPy storage;
strided writes snapshot logical values without changing width. Floating-point
NaN payloads, signed zeros and subnormals are copied without numeric conversion.
NumPy `longdouble` and `clongdouble` are rejected consistently, including when
their platform representation matches float64 or complex128.

```python
counts = TSeries(mm(2024, 11), np.array([-32768, -1, 0, 32767], dtype=np.int16))
flags = TSeries(mm(2024, 11), np.array([False, True, False], dtype=bool))
with open_dataecon("typed-series-example.daec", "a") as db:
    db.write_series("counts", counts)
    db.write_series("flags", flags)
with open_dataecon("typed-series-example.daec") as db:
    restored_counts = db.read_series("counts")
    restored_flags = db.read_series("flags")
assert restored_counts.values.dtype == np.int16
assert restored_counts.lastdate == mm(2025, 2)
assert restored_flags.values.dtype == np.bool_
np.testing.assert_array_equal(restored_counts.values, counts.values)
np.testing.assert_array_equal(restored_flags.values, flags.values)
```

Boolean series use Julia's `jeltype="Bool"` marker and canonical zero/one bytes.
Boolean scalar writes instead store unmarked Int8. Series reads accept the Bool
encoding Julia writes: signed one-byte elements with that marker and only 00/01
bytes. A Bool marker on another ordinary numeric kind or width, which Julia's
loader converts, is converted the same way: every value must be exactly zero or
one (imaginary part zero, signed zero allowed) and the result is a Boolean
`TSeries`; any other value raises `ValueError`, as Julia raises `InexactError`.
Rewriting such a series stores the canonical one-byte encoding, as Julia's
rewrite does. A Bool marker on an Int128, UInt128 or ComplexF16 payload is
preserved instead (see [Represented series elements](#represented-series-elements)).
Unmarked Int8 never becomes Bool.

Required markers are written after the payload. If a marker write fails, the
operation raises but can leave a readable object of the wrong dtype: nonempty
Bool becomes Int8, and a narrow empty array takes the native wide default.
An overwrite has already deleted the original. There is no rollback or implicit
cleanup delete; the error identifies the failed marker operation. Handle the
failure before trusting subsequent reads of that name.
After a native store or marker failure, the later close may also fail and
quarantine the owner. Do not retry that close; reopen the file to inspect what
was stored.

## Float64 and Int64 scalars

A scalar is a single value without a date axis; it is distinct from a
one-observation TSeries. Use the same file owner:

```python
with open_dataecon("scalar-example.daec", "a") as db:
    db.write_scalar("rate", 1.25)
    db.write_scalar("count", 2**53 + 1)
with open_dataecon("scalar-example.daec") as db:
    rate = db.read_scalar("rate")
    count = db.read_scalar("count")
assert type(rate) is float
assert rate == 1.25
assert type(count) is int
assert count == 9007199254740993
```

`write_scalar` accepts exact Python `float` and NumPy `float64` values, stored
as Float64, and exact Python `int` and NumPy `int64` values, stored as Int64.
Integers are packed as signed 64-bit two's complement and never pass through
floating point, so values beyond 2^53 and both signed endpoints round-trip
exactly; a Python `int` outside that range raises `ValueError`. Other supported
numeric widths and Booleans are described below. Decimal, Fraction, scalar
subclasses and arrays are rejected without implicit conversion. Float64 and
Int64 reads return an independent Python `float` or `int` according to
the stored type. NaN, infinities and the sign of zero are preserved; arbitrary
signaling-NaN states or payload bits are not an interchange guarantee.

Julia writes `Int64` values with the same metadata and reads them back as
`Int64`. Strings, dates and durations are separate scalar kinds below; the
narrower, unsigned and complex numeric widths follow next.

Scalars share the root namespace with series: existing names are never
overwritten, and wrong-class reads raise `DataEconError`. Scalar `jtype` and
`jeltype` attributes are always rejected, including the literals `Float64`
and `Int64`. The empty-series attribute exception does not apply to scalars.
Numeric scalar payloads must be exactly eight bytes with no frequency metadata.

## Narrow, unsigned and complex numeric scalars

Julia stores `Float16`/`Float32`, `Int8`/`Int16`/`Int32`, `UInt8`..`UInt64`
and `ComplexF32`/`ComplexF64` at their own byte width with no marker and reloads
them by type and width. Python mirrors that with exact NumPy scalar classes:

```python
import numpy as np

with open_dataecon("width-example.daec", "a") as db:
    db.write_scalar("half", np.float16(0.1))
    db.write_scalar("mask", np.uint64(2**64 - 1))
    db.write_scalar("small", np.int8(-128))
    db.write_scalar("z32", np.complex64(1.5 - 2.25j))
    db.write_scalar("z", 8 + 3j)
with open_dataecon("width-example.daec") as db:
    half, mask, small, z32, z = (db.read_scalar(n) for n in ("half", "mask", "small", "z32", "z"))
assert type(half) is np.float16 and half.tobytes() == b"\x66\x2e"
assert type(mask) is np.uint64 and mask == 2**64 - 1
assert type(small) is np.int8 and small == -128
assert type(z32) is np.complex64
assert type(z) is complex and z == 8 + 3j
```

`write_scalar` accepts `np.float16`, `np.float32`, `np.int8`, `np.int16`,
`np.int32`, `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`, `np.complex64`,
`np.complex128` and Python `complex` (stored as `ComplexF64`, like Julia's
`8.0 + 3.0im`). Each NumPy value is stored from its own bytes, so NaN payloads,
signed zeros and subnormals survive and nothing is widened: precision is
whatever the caller chose when it built the value (`np.float16(2049)` is
already `2048`). The C-named NumPy classes (`np.intc`, `np.uintc`,
`np.longlong`, `np.ulonglong`, and NumPy 2's `np.long`, `np.ulong`) are
accepted by their width on every platform and every supported NumPy version:
the accepted classes are discovered through the dtype type codes, not through
attribute names, and are matched by class identity. Python `int`/`float` and
`np.int64`/`np.float64` keep their Int64/Float64 behavior.

Reads return the sized NumPy class of the stored width, except the eight-byte
Int64/Float64 and sixteen-byte ComplexF64 encodings, which return Python
`int`, `float` and `complex`. A Julia `Bool` is byte-identical to `Int8` in the
file (Julia itself reloads it as `Int8`). Exact Python `bool` and NumPy
`bool_` scalars follow this convention: false stores one zero byte, true stores
one byte equal to one, with no Boolean marker. Reads return `np.int8(0)` or
`np.int8(1)`, not a Boolean. Rewriting that result preserves the stored type.
Arrays and arbitrary truth-convertible objects are not accepted as Boolean
scalars.

```python
with open_dataecon("bool-example.daec", "a") as db:
    db.write_scalar("enabled", True)
    enabled = db.read_scalar("enabled")
assert type(enabled) is np.int8
assert enabled == 1
```

`Int128`, `UInt128` and `ComplexF16` objects, which Julia can write, have no
NumPy scalar type and raise `ValueError` on read; there is no write path for
them yet. Any other width, a frequency on a numeric type, and scalar `jtype`
or `jeltype` attributes are rejected without coercion or evaluation.

## String scalars

Python `str` values are stored as the native string type: UTF-8 bytes followed
by one NUL terminator, with no frequency and no attributes. Reads return an
independent `str`.

```python
with open_dataecon("string-example.daec", "a") as db:
    db.write_scalar("label", "héllo wörld")
    db.write_scalar("note", "")
with open_dataecon("string-example.daec") as db:
    label = db.read_scalar("label")
    note = db.read_scalar("note")
assert type(label) is str
assert label == "héllo wörld"
assert note == ""
```

Only exact `str` is accepted: `bytes`, `bytearray`, `str` subclasses and other
types raise `TypeError`. A string containing NUL or a lone surrogate raises
`ValueError` before anything is written, because the native format is a C
string and Julia's loader stops at the first NUL. The payload, including the
terminator, must stay within the 128 MiB limit.

Reads are strict. The stored payload must end with NUL, contain no other NUL
and decode as valid UTF-8; otherwise `read_scalar` raises `ValueError`. The
pinned Julia loader instead truncates at an embedded NUL and returns invalid
bytes unchecked; Python never returns a silently shortened or undecodable
value. A string object whose metadata carries a frequency raises `TypeError`.

The supported subset is therefore valid-UTF-8 text without NUL. Julia's
`String` deliberately admits arbitrary bytes, and its writer stores an
embedded NUL, so a Julia file can hold string objects that Python currently
refuses rather than misreads. A lossless Python representation for those
values is planned parity work, not a permanent exclusion.
Julia stores a `Symbol` through the same string path with a `jtype="Symbol"`
marker; that marker is rejected like every other reconstruction attribute, so
Julia symbols are not read as strings. Julia `SubString` values carry a
`jtype` marker as well and are rejected the same way.

## Date and duration scalars

An `MIT` scalar is stored as the native date type with its frequency code; a
`Duration` scalar is stored as a signed 64-bit integer with the same frequency
code. Both cover `Monthly`, all three `Quarterly` anchors, all six `HalfYearly`
endings and all twelve `Yearly` endings, plus `Unit`, `Daily`, `BDaily` and
`Weekly` with every end day (see the next section).

```python
from tsecon import Duration

start = MIT.from_yp(Quarterly(end_month=1), 2024, 4)
with open_dataecon("date-example.daec", "a") as db:
    db.write_scalar("start", start)
    db.write_scalar("horizon", Duration(Quarterly(end_month=1), 8))
    db.write_scalar("lag", Duration(Yearly(), -1))
with open_dataecon("date-example.daec") as db:
    restored_start = db.read_scalar("start")
    horizon = db.read_scalar("horizon")
    lag = db.read_scalar("lag")
assert type(restored_start) is MIT and restored_start == start
assert type(horizon) is Duration and horizon.value == 8
assert horizon.frequency == Quarterly(end_month=1)
assert lag == Duration(Yearly(), -1)
```

A date is validated as a date. Its integer code must lie within the reliable
native range of its frequency, checked before any C call and confirmed by the
native pack/unpack round trip: monthly `-393600` through `2147483647`,
quarterly `-131200` through `2147483647`, half-yearly `-65600` through
`2147483647`, and annual the full signed 32-bit range. Codes outside these
limits raise `ValueError` in both directions. The pinned Julia writer stores a
below-minimum code intact and then misdates it on load with a warning; Python
rejects such a stored code instead of returning a different date. Fiscal
anchors are part of the frequency code, so `Quarterly(end_month=1)` and
`Quarterly(end_month=3)` dates with the same integer stay distinct.

A duration is a count of periods, not a date. It only needs to fit the signed
64-bit range; no date bound applies, and values beyond 2^53 round-trip exactly.
The stored type code is shared with `Int64`: a frequency of zero reads back as
a Python `int`, and a supported frequency reads back as a `Duration`.

Julia writes `MIT` and `Duration` values with exactly this metadata and loads
them back as `MIT{F}` and `Duration{F}`. Julia anchors beyond the canonical
range collapse to canonical codes on write (its `Quarterly{4}` is stored as
`Quarterly{1}`), so Python reads them as the canonical `Quarterly(end_month=1)`.
Bare family codes such as 64, 128 or 256, mixed bits, the monthly alias 33,
and date or duration payloads that are not eight bytes are rejected.

## Unit and calendar date scalars

`Unit` dates and durations are plain signed 64-bit codes. Julia stores them
without any date conversion, so Python does the same: any value from -2^63 to
2^63-1 round-trips exactly and no date validation applies (frequency code 11).

`Daily` (12), `BDaily` (13) and `Weekly` (16 plus the ISO end day, Monday 17
through Sunday 23) dates use the native calendar codec. Python never converts
these codes through `datetime`, so years outside 1..9999 are stored and read as
integer codes. The reliable code range of each frequency was verified natively
on every code: daily `-11980259` (1 March -32800) through `11979954` (31
December 32800); business daily `-8557114` (25 December -32800) through
`8557110` (29 December 32800); weekly `-1711422` through `1711422` for every
end day (the weeks ending in the last week of December -32800 and of 32800).
Codes outside these ranges raise `ValueError` in both directions, and every
accepted code must also survive the native decode/encode round trip. The
native encoder accepts years down to -32800, but its arithmetic wraps for
dates before those minimums: the pinned Julia writer stores such a value with
a warning and then reloads a different date (daily, business daily) or the
same one by coincidence (weekly). Python rejects those stored codes instead.
Julia's loader also reads a natively written code just above the maximum
unchanged, although its own writer cannot produce one; Python rejects it.

```python
from tsecon import BDaily, Daily, Unit, Weekly, bdaily, daily, weekly

with open_dataecon("calendar-example.daec", "a") as db:
    db.write_scalar("day", daily("2024-01-15"))
    db.write_scalar("business_day", bdaily("2024-01-15"))
    db.write_scalar("week", weekly("2024-01-17", 3))
    db.write_scalar("far_future", MIT(Daily(), 3652062))  # 3 January 10000
    db.write_scalar("step", MIT(Unit(), 2**40))
    db.write_scalar("horizon", Duration(BDaily(), 20))
with open_dataecon("calendar-example.daec") as db:
    assert db.read_scalar("day") == daily("2024-01-15")
    assert db.read_scalar("business_day") == bdaily("2024-01-15")
    assert db.read_scalar("week") == MIT(Weekly(3), 105558)
    assert db.read_scalar("far_future") == MIT(Daily(), 3652062)
    assert db.read_scalar("step") == MIT(Unit(), 2**40)
    assert db.read_scalar("horizon") == Duration(BDaily(), 20)
```

Business daily means Monday to Friday on both sides; neither Julia nor Python
applies a holiday calendar. Every business-daily code identifies a weekday, so
the native weekend error cannot arise from a stored code. The weekly end day is
part of the frequency code, so `Weekly(1)` and `Weekly(7)` dates with the same
integer stay distinct. Julia anchors beyond 1..7 collapse on write (`Weekly{8}`
is stored as `Weekly{1}`). The Sunday alias 16, the unused codes 14 and 15 and
weekly codes 24 through 31, which Julia never writes, are rejected.

## Deleting, overwriting, truncating and in-memory files

The defaults are unchanged: `open_dataecon(path)` is read-only and `"a"`
appends without ever replacing an object. The Julia file operations are
available as explicit calls:

```python
import numpy as np
from tsecon import TSeries, mm
from tsecon.dataecon import open_dataecon, open_dataecon_memory

with open_dataecon("fileops-example.daec", "a") as db:
    db.write_scalar("rate", 1.25)
    db.write_scalar("rate", "revised", overwrite=True)  # delete-then-store
    db.write_scalar("temporary", 7)
    db.delete("temporary")
    assert db.read_scalar("rate") == "revised"
    assert not db.is_empty()

with open_dataecon("fileops-example.daec", "w") as db:  # "w" truncates on open
    assert db.is_empty()
    db.write_series("fresh", TSeries(mm(2024, 1), np.array([1.0, 2.0])))
    db.truncate()  # same as Julia's empty!(de); the owner stays usable
    assert db.is_empty()
    db.write_scalar("after", 42)

with open_dataecon_memory() as scratch:  # Julia's opendaecmem()
    scratch.write_scalar("x", np.float32(0.1))
    assert scratch.read_scalar("x") == np.float32(0.1)
    assert scratch.path == ":memory:"
```

`delete(name)` removes one root object with its payload and attributes. A
missing name raises `DataEconError` (code -989) and the owner stays usable.
A root **catalog** (written by Julia) is refused unless you pass
`recursive=True`, in which case every nested catalog and object under it is
deleted, exactly as Julia's `delete_object` does without asking. Deleting a
series leaves its shared axis row in the file (axes are not objects), and
values returned by earlier reads remain valid because reads copy.

`overwrite=True` on `write_scalar`/`write_series` is Julia's
`opendaec(...; overwrite=true)` behavior for that one call: the existing root
scalar or series of that name is deleted and the new object stored, whatever
its class. Python validates the new value completely *before* deleting, so a
rejected value leaves the old object intact (Julia deletes first and then
fails). The operation is **not atomic** and nothing rolls back: once the
delete has run the original value is gone, and a native store failure after
it leaves the name either absent or holding a partial, unreadable replacement
(the native library creates the object row before it stores the payload, so
a failure in between leaves an object without a value; reading it raises
`DataEconError`). Such a failure is reported with the operation label
`write_scalar (overwrite; original deleted, partial replacement may remain)`
(or `write (...)` for series). An existing catalog is never overwritten
implicitly; delete it explicitly with `recursive=True`.

`truncate()` (Julia's `truncatedaec`/`empty!`) commits pending writes, resets
the file to a freshly created state, restarts object ids and leaves the owner
open; `open_dataecon(path, "w")` does the same immediately after opening
(Julia's `truncate=true`) and creates the file when it does not exist.
`is_empty()` is Julia's `isempty(de)`: true when the root catalog holds no
objects. Read-only owners reject `delete`, `truncate` and `overwrite=True`
with `ValueError` before any native call. Note that writes made through an
open owner are committed only when it closes (or truncates); another
connection opened in the meantime does not see them.

`open_dataecon_memory()` opens a private, writable, empty in-memory database
that is discarded on close; it cannot be reopened or shared with Julia, and
no file named `:memory:` is created. Passing the literal `":memory:"` to
`open_dataecon` raises `ValueError` rather than turning it into a disk path.

A native truncate failure quarantines the owner the same way a failed close
does: it cannot be used or closed again and native resources may remain until
process exit. This is deliberate: the native library's statement-finalization
path is unsafe once a statement has failed, so the adapter never makes a second
native call on such a handle.

## Closing and errors

Use a context manager or call `close()` explicitly. Successful close is
idempotent. Missing objects and duplicate names raise `DataEconError`, which
retains `code`, `operation`, `path`, `name` and `native_message`. Positive native
codes come from SQLite. Invalid or unsupported Python inputs raise
`TypeError`/`ValueError`; writes, deletions, truncation and overwrites through
an explicitly read-only owner are rejected before calling C.

A native write failure may leave a partial object or unused axis; the adapter
does not promise rollback or delete the file. A failed native close makes the
owner unusable and is not retried: upstream statement-finalization retry safety
is not guaranteed. Native resources may remain until process exit in that case.
If a context body also raised, that original exception is preserved and the
cleanup failure is attached as an exception note. Garbage-collection cleanup is
best effort, not a substitute for explicit close.

All native calls through this extension share a lock, including error capture
and copying borrowed buffers. This does not synchronize a separate binding or
Julia runtime embedded in the same process. Interchange verification runs Julia
as a separate process.

## Local Windows build

Use CPython x86-64 and Visual Studio 2022 MSVC Build Tools with a Windows SDK.
Install the project's build requirements (Cython, NumPy, setuptools, hatchling;
`build` for the wheel command). The preferred route builds DataEcon and its bundled
SQLite directly from pinned source using MSVC, matching the Cython toolchain.

Download the [pinned source ZIP](https://codeload.github.com/bankofcanada/DataEcon/zip/1a108688a044380f808bebf64079e32dbb9cd1a4)
to `downloads/dataecon-source.zip`. Its SHA-256 is
`d2b48df3a47d173c43354fe033438253bb46d878ae6c3824c6224fbd4e6a2d71`.
The helper verifies this hash before extracting or compiling. Use a new output
directory for each build; an existing one is never overwritten.

```powershell
python scripts/build_dataecon_windows.py downloads/dataecon-source.zip --output build/dataecon-source
$env:TSECON_DATAECON_ROOT = (Resolve-Path build/dataecon-source).Path
python -m build --wheel --no-isolation
```

The helper compiles the 13 DataEcon C files and bundled SQLite 3.50.2 amalgamation,
links `bin/libdaec.dll`, and generates `lib/daec.lib`. This **import library** is
the linker's description of the DLL's exported functions. It exports the 41 public
functions declared in `daec.h`, copies that header and notices, and writes
`build-info.json` with source/output hashes, compiler/SDK details and build flags.
That manifest is included with the DLL in the configured wheel.

One guarded portability adjustment is applied only to the extracted source:
`static const uint32_t EPOCH_s = 82;` becomes `#define EPOCH_s UINT32_C(82)`.
MSVC requires a constant expression in the dependent file-scope initializers;
the unsigned value, formulas and public ABI are unchanged. The compiler uses
C11 mode and its standard `/MD` shared C runtime. The native DLL therefore uses
the Microsoft C runtime, as does the Cython extension; dependency auditing and
installed-wheel checks remain required for each supported platform target.

The build steps are repeatable from pinned inputs and record the selected toolchain.
This is not yet a bit-for-bit reproducible build: linker timestamps are not
normalized, and repeated builds can have different binary hashes even with the
same compiler/SDK. Each manifest identifies its actual output binaries.
Upstream narrowing-conversion warnings are left visible; the adapter's existing
date/payload guards and limited supported scope still apply. No native source
algorithm, file-format, type support or Windows path policy is expanded here.

### Alternative: verified upstream binary

For comparison or a local fallback, obtain the
**DataEcon_jll 0.4.0+0 Windows x86-64 artifact** from its
[release](https://github.com/JuliaBinaryWrappers/DataEcon_jll.jl/releases/tag/DataEcon-v0.4.0%2B0).
The archive `DataEcon.v0.4.0.x86_64-w64-mingw32.tar.gz` has SHA-256
`c4d965008d46ab204fa35bbaee3238f3a15393850f0993e1d3cda52dba9af465`.
Verify it and extract it to a local directory. An already installed matching
Julia artifact can also supply these build inputs.

```powershell
python scripts/prepare_dataecon_windows.py <extracted-artifact-directory>
$env:TSECON_DATAECON_ROOT = (Resolve-Path build/dataecon).Path
python -m build --wheel --no-isolation
```

The preparation script checks the header and DLL hashes and generates an MSVC
**import library**, the linker's description of the DLL's exported functions.
The build hook compiles the extension against the actual header and bundles the
DLL in the wheel. It does not download anything during build or runtime. The
default build without `TSECON_DATAECON_ROOT` omits DataEcon native support.
Source distributions include both native build helpers and the adapter sources;
the external native inputs/build tools are still required to enable the feature.

Install the resulting `.whl` in a fresh environment and test outside the checkout.
For a wheel containing this native support, users need neither a compiler nor a
separate DataEcon/Julia installation. A **wheel** is the installable built package;
GitHub Actions runs build/test jobs, and publishing is a separate operation.
Windows source builds are verified locally and in wheel CI. The Linux/macOS
source-build configuration below is also verified in platform CI.

The bundled native library uses the BSD-3-Clause DataEcon license and public-domain
SQLite; notices ship beside the adapter. Cython code uses the package's MIT license.

## Linux and macOS source builds

`scripts/build_dataecon_unix.py` uses the same verified source ZIP/header and
guarded constant-expression patch. It compiles all 13 DataEcon sources and bundled
SQLite into a position-independent **static archive**, `lib/libdaec.a`. The linker
incorporates that code into the Cython extension; no separate DataEcon shared
library or runtime search path is needed. Native symbols have hidden visibility
to isolate this SQLite copy from other extensions in the process. Windows retains
its verified DLL/import-library route; both routes use the same Cython binding.

```sh
# macOS arm64 only: export MACOSX_DEPLOYMENT_TARGET=11.0
python scripts/build_dataecon_unix.py downloads/dataecon-source.zip --output build/dataecon-native
export TSECON_DATAECON_ROOT="$PWD/build/dataecon-native"
python -m build --wheel
```

The native helper needs a C compiler and `ar`. Its manifest records the compiler,
flags, source/header hashes, portability patch and static archive hash. The build
hook checks the header/archive against that manifest before linking. The archive
hash identifies a build input, not the final extension: wheel repair can modify
the extension. The archive itself does not ship in the wheel; the manifest and
license notices do. Updating native code requires rebuilding the extension.

Linux CI builds inside the explicitly selected manylinux_2_28 x86-64 container
(glibc 2.28 floor), then uses cibuildwheel's auditwheel repair. A local Linux wheel
build alone does not establish manylinux compatibility. macOS uses the explicit
`macos-15` Apple Silicon runner, targets macOS 11.0, and retains delocate repair
and deployment-target validation. Intel macOS and universal2 are outside this
matrix. See [cibuildwheel options](https://cibuildwheel.pypa.io/en/v3.4.1/options/)
and [manylinux platforms](https://github.com/pypa/manylinux).

## Wheel CI checks

Each Windows wheel job builds the pinned native source with MSVC and includes
the DLL, build manifest and notices. `scripts/check_dataecon_wheel.py` runs after
installation: it rejects imports from the source checkout, verifies the pinned
manifest and DLL hash, checks the DLL's actual loaded location, reads the Julia
fixture and writes a result for separate Julia verification. The build log and
manifest record the native runtime dependency names.

The workflow sets `TSECON_REQUIRE_DATAECON=1`, so an absent extension is a pytest
configuration error rather than an optional skip. The ordinary core-only test
jobs leave this setting unset. The checker can also verify a core-only wheel when
the setting is unset. Linux/macOS installed checks inspect the repaired extension
with `readelf`/`otool` and `nm`: external DataEcon/SQLite dependencies or exposed
native symbols fail the job. Existing ABI-layout, owning-buffer, lifecycle and
error tests run against each installed native wheel.

After the installed tests pass, each platform job checks its output with Julia
1.12.5 and the pinned TimeSeriesEcon.jl checkout. Julia and DataEcon_jll are CI
verification dependencies, not dependencies of the Python wheel. The Julia setup
helper pins DataEcon_jll to 0.4.0+0; the interchange script verifies the reference
checkout identity and nonempty/empty value, type and metadata assertions before
wheel upload. Each output contains the nonempty sample and empty series anchored
at 2024M1 and 2025M7; the Julia loader must preserve all three as dated TSeries.
The same file contains seven Float64 scalars covering finite values, signed zero,
NaN and infinities, and fourteen Int64 scalars including both signed endpoints
and values around 2^53; Julia checks their types, metadata and values as well.
It also holds 91 narrow, unsigned and complex numeric scalars (every Float16/
Float32, Int8/16/32, UInt8..UInt64 and ComplexF32/F64 group with endpoints,
signed zeros, NaN, infinities, subnormals and precision-loss values, plus a
Python `complex`) that Julia must load with the same type and bits. It holds
147 represented series objects written from `StoredSeries` containers: both
Int64 endpoints as dates and as durations for all 32 element frequencies, each
with its marked empty; annual dates on a daily axis and daily durations on an
annual axis; Int128/UInt128 endpoints and word-boundary values, the ComplexF16
bit patterns and the three marked wide empties; three rewrites of read results;
and three Bool-marked wide carriers with their explicit Boolean conversions.
Julia checks their metadata, bytes, markers and loaded values, expects its own
loader's failure on the empty date/duration series, and loads the marked wide
carriers and their conversions as Boolean series. The
combined output also records three file-operation outcomes (an Int64 scalar
overwritten by a string, a deleted scalar, a scalar replaced by a series),
and an auxiliary `fileops/cpXY-fileops.daec` file is written, reopened with
`"w"` (truncated) and refilled; Julia checks the overwrite/delete results, the
truncated auxiliary file's content and that its object ids restarted. The
workflow's Julia steps locate the primary output by globbing `*.daec` in the
interchange directory and require exactly one match, so auxiliary outputs
live in the `fileops` subdirectory; the checker refuses to leave any other
`.daec` file at the top level, and the Julia verifier checks the same layout.
It also contains fifteen strings (empty, ASCII, accented, CJK, emoji,
whitespace, punctuation and long values) that Julia must load as `String`, and
six `MIT` dates plus six `Duration` values for each of the 22 year/period
frequency families, covering typical, year-boundary, negative, zero and both
reliable-limit codes and the signed 64-bit duration endpoints; thirteen dates
plus six durations for each of the nine daily, business-daily and weekly
families, including leap day, year zero, a negative year, both window
endpoints and years beyond `datetime`; and eleven `Unit` dates and durations
including both signed 64-bit endpoints.
Quarterly objects cover all three fiscal anchors, year transitions, negative and
zero years, and nonempty/empty anchors at the supported date limits. Their
frequency and first/last dates must also survive the Julia read. Annual objects
exercise all twelve fiscal endings with the same kinds of empty and boundary
checks, including both signed 32-bit year limits. Half-yearly objects cover
all six endings with the same nonempty/empty and limit cases. Calendar series
objects (117: thirteen per daily, business-daily and weekly end-day family)
cross a year end, leap day 2024 and a weekend, include negative and zero
codes, both window endpoints, a series from 31 December 9999 into year 10000,
two values from the window maximum and a thousand values straddling it, and
empty anchors at typical and endpoint codes; Julia checks their metadata,
frequency, first/last dates and values.

For a local installed-wheel check, use a fresh output directory and the installed
environment's Python from outside the source package:

```powershell
$env:TSECON_REQUIRE_DATAECON = '1'
python scripts/check_dataecon_wheel.py --fixture tests/dataecon/fixtures/julia_monthly.daec --output-dir build/interchange-output
```

Use an absolute path to the script/fixture when running outside the checkout.
Without `--output-dir`, the checker uses `TSECON_DATAECON_OUTPUT_DIR` when set,
otherwise `build/dataecon-interchange` beside the project scripts directory.
Linux CI sets that variable to the host workspace through cibuildwheel's `/host`
mount. Its `/project` directory is a container copy, so files written there are
not automatically available to the later Julia step on the host.

The output is named for the CPython version tag (for example `cp311.daec`), and an existing
output is not overwritten. CI builds and artifact checks do not publish to PyPI;
publication remains a separate release workflow.

## Represented series elements

Julia's dated `TSeries` can hold element types NumPy has no scalar type for:
`MIT{F}` dates and `Duration{F}` spans (stored as Int64 codes with their own
element frequency), `Int128`/`UInt128` and `ComplexF16`. `read_series`
returns a `TSeries` for every ordinary numeric or Boolean element type and a
`StoredSeries` for these families, so its result type is
`TSeries | StoredSeries` (exported as `SeriesValue`). A `StoredSeries` keeps
three things: the dated anchor, an owning NumPy carrier array holding the exact
stored bytes, and a `StoredElement` descriptor naming the family, the element
frequency of a date or duration element, and any preserved marker. Dates and
durations use an `<i8` carrier; Int128/UInt128 use the structured dtype
`[("lo", "<u8"), ("hi", "<u8")]` (low word first, as stored); ComplexF16 uses
`[("real", "<f2"), ("imag", "<f2")]`. `write_series` accepts either container.

### Axis frequency versus element frequency

A series has one axis frequency, taken from its first date, and its elements
carry an independent element frequency. Julia stores both; Python keeps them
apart: `frequency` and `firstdate`/`lastdate` describe the axis, while
`element.frequency` describes the stored dates or durations. The two may
differ, as in daily durations on an annual axis or annual dates on a daily
axis.

```python
import numpy as np
from tsecon import MIT, Daily, Duration, Monthly, Yearly, mm
from tsecon.dataecon import StoredElement, StoredSeries, open_dataecon

dates = StoredSeries.from_list(
    mm(2024, 11), StoredElement.date(Monthly()), [mm(2024, 11), mm(2024, 12)]
)
spans = StoredSeries(
    MIT(Yearly(12), 2024), np.array([-5, 0, 7], dtype="<i8"), StoredElement.duration(Daily())
)
with open_dataecon("dates-example.daec", "a") as db:
    db.write_series("dates", dates)
    db.write_series("spans", spans)
with open_dataecon("dates-example.daec") as db:
    restored_dates = db.read_series("dates")
    restored_spans = db.read_series("spans")
assert isinstance(restored_dates, StoredSeries)
assert restored_dates.element == StoredElement.date(Monthly())
assert restored_dates.values.tobytes().hex() == "ea5e000000000000eb5e000000000000"
assert restored_dates.tolist() == [mm(2024, 11), mm(2024, 12)]
assert restored_spans.frequency == Yearly(12)  # the axis
assert restored_spans.element.frequency == Daily()  # the elements
assert restored_spans.lastdate == MIT(Yearly(12), 2026)
assert restored_spans.tolist() == [Duration(Daily(), -5), Duration(Daily(), 0), Duration(Daily(), 7)]
```

Element codes are never validated against date windows and never packed:
every signed 64-bit code round-trips for all 32 element frequencies (`Unit`,
`Daily`, `BDaily`, `Weekly` with any end day, `Monthly`, `Quarterly`,
`HalfYearly` and `Yearly` with any anchor), exactly as Julia stores them. Only
the axis follows the date rules of the earlier sections. `tolist()` returns
core `MIT` or `Duration` objects with the element frequency, and `from_list`
accepts only such objects of exactly that frequency.

### 128-bit integers and half-precision complex values

```python
from tsecon.dataecon import COMPLEXF16, INT128, UINT128

wide = StoredSeries.from_list(mm(2024, 11), INT128, [-(2**127), -1, 0, 2**127 - 1])
assert wide.values.dtype == np.dtype([("lo", "<u8"), ("hi", "<u8")])
assert wide.values.tobytes()[:16].hex() == "00000000000000000000000000000080"
words = StoredSeries.from_list(mm(2024, 11), UINT128, [2**64])
assert words.values.tobytes().hex() == "00000000000000000100000000000000"
halves = StoredSeries.from_list(
    mm(2024, 11), COMPLEXF16, [complex(-0.0, 1.25), complex(float("inf"), 0.0), complex(65504.0, -65504.0)]
)
assert halves.values["real"].view("<u2").tolist() == [0x8000, 0x7C00, 0x7BFF]
with open_dataecon("wide-example.daec", "a") as db:
    db.write_series("wide", wide)
    db.write_series("halves", halves)
with open_dataecon("wide-example.daec") as db:
    restored_wide = db.read_series("wide")
    restored_halves = db.read_series("halves")
assert restored_wide == wide
assert restored_wide.tolist() == [-(2**127), -1, 0, 2**127 - 1]
assert restored_halves.tolist()[0] == complex(-0.0, 1.25)
widened = restored_halves.to_complex64()
assert widened.values.dtype == np.complex64
assert np.signbit(widened.values[0].real)
```

Julia stores these values with the same bytes and reloads them as `Int128`,
`UInt128` and `ComplexF16` series; `2**64` is eight zero bytes followed by a
one, and `-2**127` is fifteen zero bytes followed by `0x80`. An empty
Int128/UInt128/ComplexF16 series writes Julia's type marker (`"Int128"` and so
on), because an unmarked empty payload of those native kinds reads as
int64/uint64/complex128 on both sides.

**ComplexF16 input is exact.** `from_list` accepts a Python `complex` only
when both finite components are exactly representable as float16 (largest
finite value 65504, signed zero preserved); infinities map to float16
infinities and NaN components to the canonical quiet NaN. Inexact narrowing or
overflow raises `ValueError`; nothing is rounded. `(np.float16, np.float16)`
pairs and carrier buffers built with the exact dtype are the bit-preserving
route, including NaN payloads, which conversion through Python `complex` does
not preserve.

```python
for inexact in (complex(0.1, 0.0), complex(70000.0, 0.0), complex(2049.0, 0.0)):
    try:
        StoredSeries.from_list(mm(2024, 11), COMPLEXF16, [inexact])
    except ValueError:
        pass
    else:
        raise AssertionError("inexact ComplexF16 input must be refused")
payload = np.array([0x7E55], dtype="<u2").view("<f2")[0]  # a NaN with a payload
bits = StoredSeries.from_list(mm(2024, 11), COMPLEXF16, [(np.float16(1.0), payload)])
assert bits.values["imag"].view("<u2").tolist() == [0x7E55]
again = StoredSeries.from_list(mm(2024, 11), COMPLEXF16, bits.tolist())
assert again.values["imag"].view("<u2").tolist() == [0x7E00]  # canonical quiet NaN
```

### Storage-preserving operations versus explicit conversions

Reading, editing the carrier in place and writing preserve the stored kind,
width, element frequency, bytes and marker; rewriting a read result reproduces
the stored metadata and payload exactly, as Julia's own rewrite does. The
conversions are explicit and never change what is stored: `tolist()`
(`MIT`/`Duration`, Python `int` or `complex`), `from_list()` (its strict
inverse), `to_complex64()` (ComplexF16 widened into a `complex64` `TSeries`)
and `to_bool()` (below). Nothing converts on read or write.

Construction copies by default, so a container never aliases the caller's
array. Explicit `copy=False` shares only a one-dimensional, C-contiguous,
writable ndarray of exactly the carrier dtype; strided or read-only input is
copied even then, and an array of another dtype is refused rather than
converted (an int64 array is never silently reinterpreted as dates of the
wrong kind). `values` is the live carrier: its contents may be edited, but a
change of its shape, dtype or strides is detected and refused by `validate()`,
by every query and before any write.

### Wide Bool markers and empty-width ambiguity

A file may carry Julia's `"Bool"` marker on an Int128, UInt128 or ComplexF16
payload. Julia's writer never produces this, but its loader converts such a
series to `Bool` when every value is exactly zero or one, and its rewrite
stores the canonical one-byte encoding. Python preserves the stored width
instead: the read result is a `StoredSeries` whose descriptor carries the
marker, its bytes are kept, and every value is validated as exactly zero or
one (a zero imaginary part and either signed zero count as zero); other
values raise `ValueError`, as Julia raises `InexactError`. `to_bool()` is the
explicit conversion; writing its result stores Julia's canonical encoding.

```python
flags = StoredSeries.from_list(mm(2024, 11), INT128.with_bool_marker(), [1, 0])
with open_dataecon("wide-bool-example.daec", "a") as db:
    db.write_series("flags", flags)
    preserved = db.read_series("flags")
    db.write_series("flags_bool", preserved.to_bool())  # canonical one-byte Bool
    restored_bool = db.read_series("flags_bool")
assert preserved == flags
assert preserved.element.marker == "Bool"
assert preserved.tolist() == [1, 0]  # the stored values, not Booleans
assert preserved.to_bool().values.tolist() == [True, False]
assert restored_bool.values.dtype == np.bool_
preserved.values["hi"][1] = 1  # 2**64 is not a Boolean value
try:
    preserved.validate()
except ValueError:
    pass
else:
    raise AssertionError("an invalid marked carrier must be refused")
assert preserved.element.marker == "Bool"  # nothing is normalized or stripped
plain = StoredSeries(preserved.firstdate, preserved.values, INT128)  # explicit: drop the marker
assert plain.tolist() == [1, 2**64]
```

An **empty** payload with the Bool marker has no width in the file: kind 1, 2
or 5 with zero bytes is the same record whether it came from Int8, Int128 or
ComplexF16. Python therefore reads it as an empty Boolean `TSeries` (Julia
gives `Bool[]`) and never fabricates a wide container; a marked wide
container cannot be empty, and emptying one in place is refused before any
write. Rewriting such an empty series stores the canonical empty Bool
encoding, as Julia does.

### Ownership, overwrite and marker-write failure residue

Reads always return an owning, writable carrier copied out of the native
payload, whatever `copy` setting the writer used; the result outlives file
closure. `StoredSeries` holds no native resource.

With `overwrite=True` the live carrier and its marker are validated, the
bytes are snapshotted and the snapshot is checked against the container's
descriptor before the existing object is deleted, so a rejected container
leaves the old object intact. The file lock does not serialize a caller's
own array mutations: a carrier resized or edited by another thread between
validation and the snapshot is detected and refused, but no atomic snapshot
of a concurrently mutated array is promised. Overwrite itself remains
delete-then-store without rollback, as for every other series.

Marker writes are separate native operations after the store. If the marker
write fails for an empty Int128/UInt128/ComplexF16 series, the residue is an
unmarked empty object that reads back as int64, uint64 or complex128; for a
Bool-marked wide carrier the residue reads as a plain unmarked carrier with
the same bytes; for an empty date or duration series the residue reads
identically in Python (the element frequency is stored separately) and stays
unloadable by the pinned Julia loader either way. Nonempty represented series
carry no marker and have no residue. The error names the failed marker
operation; there is no rollback or implicit cleanup delete, and a later close
may fail and quarantine the owner. Do not retry that close; reopen the file
to inspect what was stored.

### Compatibility restrictions and remaining unsupported capabilities

- Only canonical element-frequency codes are accepted. Julia's own writer
  emits only these; the noncanonical encodings 14, 15, 16, 24 through 31, 33,
  64, 128 and 256 are refused with `TypeError`, although Julia loads some of
  them as frequency types such as `Weekly{0}`, `Weekly{8}` or `Quarterly{0}`.
- Empty date and duration series write Julia's exact marker and read with or
  without it; the pinned Julia loader fails on either form.
- A Bool marker on date or duration elements, and every other foreign marker
  (for example `Int64` on Int16 payloads or `MIT{Monthly}` on Int64 payloads),
  is refused with `TypeError`. Julia converts some of these case by case;
  supporting them is planned parity work, not an approved exclusion.
- Int128, UInt128 and ComplexF16 scalars, Unit-frequency series axes and
  wider-than-64-bit element widths other than these three families remain
  unsupported.
- Marker text is never evaluated: markers are compared with a finite table of
  tokens.
