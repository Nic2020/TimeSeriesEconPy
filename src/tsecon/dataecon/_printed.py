# SPDX-License-Identifier: MIT
"""Julia's printed forms behind the ``Symbol`` marker routes.

Julia's DataEcon loader applies ``Symbol(value)`` to a whole object under a
``jtype`` of ``Symbol`` and ``Symbol(element)`` to each element under a
``jeltype`` of ``Symbol``. ``Symbol(x)`` is ``Symbol(string(x))``, so the
loaded name is Julia's ``print`` of the value: ``show(io, ::AbstractArray)``
for arrays (``[1, 2]``, ``Int8[1 3; 2 4]``, ``["a", "b"]``,
``Matrix{Float64}(undef, 0, 2)``), ``print(io, ::Date)`` for calendar MITs
and the core's ``str(MIT)`` for the year/period frequencies. The functions
here transcribe those rules from ``base/arrayshow.jl``, ``base/show.jl``,
``base/strings/io.jl`` and ``stdlib/Dates`` at the pinned version; every
Int64 step wraps like Julia's, every ``Char`` step follows Julia's own UTF-8
decoder, and nothing here evaluates marker text.

One thing Julia's rules cannot settle from Python alone is refused rather
than guessed: a character that this Python's Unicode tables list as
unassigned (Julia's tables may assign it, which decides whether it prints raw
or escaped). A ``BDaily`` code whose Float64 day arithmetic does not produce
an integer is Julia's own ``InexactError`` and raises ``ValueError``.
"""

from __future__ import annotations

import itertools
import math
import unicodedata
from typing import Any

import numpy as np

from tsecon.frequencies import BDaily, Daily, Frequency, Unit, Weekly
from tsecon.mit import MIT

from . import _exact
from ._metadata import MAX_INT64, MIN_INT64

# ---- Julia's Int64 arithmetic ------------------------------------------------


wrap64 = _exact.wrap_int64


def _tdiv(a: int, b: int) -> int:
    """Julia's ``div``: the quotient truncated toward zero."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def julia_yearmonthday(days: int) -> tuple[int, int, int]:
    """``Dates.yearmonthday`` exactly as Julia computes it, wrapping Int64 at every step.

    ``fld`` steps floor and ``div`` steps truncate; a day count beyond about
    ``9.2e16`` overflows ``100z`` in Julia and the result is the wrapped
    arithmetic's own (a month or day outside 1..31 is then printed as is).
    """
    z = wrap64(days + 306)
    h = wrap64(100 * z - 25)
    a = h // 3652425
    b = wrap64(a - a // 4)
    y = wrap64(100 * b + h) // 36525
    c = wrap64(b + z - wrap64(365 * y) - y // 4)
    m = _tdiv(wrap64(5 * c + 456), 153)
    d = wrap64(c - _tdiv(wrap64(153 * m - 457), 5))
    return (wrap64(y + 1), wrap64(m - 12), d) if m > 12 else (y, m, d)


def julia_date_string(days: int) -> str:
    """``print(io, ::Date)`` of a Rata Die day count: ``yyyy-mm-dd`` with Julia's padding.

    A negative year prints through ``@sprintf("%05i")`` (``-0001``), other
    years are left-padded to four digits (``0000``, ``10000``); month and day
    are padded to two.
    """
    year, month, day = julia_yearmonthday(days)
    yy = f"{year:05d}" if year < 0 else f"{year:04d}"
    return f"{yy}-{month:02d}-{day:02d}"


def calendar_days(code: int, frequency: Frequency) -> int:
    """``Dates.value(Dates.Date(m::MIT{F}))`` for a calendar frequency (``_d0`` is day 0).

    Daily is the code itself; weekly is ``code * 7 - (7 - end_day)`` in
    wrapping Int64; business daily is ``code + 2 * floor((code - 1) / 5)`` in
    Julia's Float64 arithmetic, which ``Day(::Float64)`` accepts only when the
    result is an integer within Int64 (``InexactError`` otherwise).
    """
    if isinstance(frequency, Daily):
        return code
    if isinstance(frequency, Weekly):
        return wrap64(wrap64(code * 7) - (7 - frequency.end_day))
    assert isinstance(frequency, BDaily)
    quotient = float(wrap64(code - 1)) / 5.0
    total = float(code) + 2.0 * math.floor(quotient)
    if not (total.is_integer() and MIN_INT64 <= total <= MAX_INT64):
        raise _exact._failure("InexactError", "Int64", total)
    return int(total)


def mit_string(code: int, frequency: Frequency) -> str:
    """``string(m::MIT{F})`` for any Int64 code: Julia's printed form of the element.

    Calendar frequencies print the Date Julia builds (any proleptic year,
    with Julia's wrapping arithmetic at extreme codes); Unit prints
    ``<code>U``; the year/period frequencies print through the core, whose
    ``divmod`` decomposition equals Julia's ``divrem`` with its sign fix-up
    for every Int64 code.
    """
    if isinstance(frequency, (Daily, BDaily, Weekly)):
        return julia_date_string(calendar_days(code, frequency))
    if isinstance(frequency, Unit):
        return f"{code}U"
    return str(MIT(frequency, code))


# ---- Julia's quoted strings ------------------------------------------------

_SIMPLE_ESCAPES = {7: "\\a", 8: "\\b", 9: "\\t", 10: "\\n", 11: "\\v", 12: "\\f", 13: "\\r"}
# Julia's ``isprint``: the utf8proc categories Lu..Zs (letters, marks, numbers,
# punctuation, symbols and space separators; not the controls, formats,
# surrogates, private-use, unassigned or line/paragraph separators).
_PRINTABLE_CATEGORIES = frozenset(
    (
        *("Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me"),
        *("Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"),
        *("Sm", "Sc", "Sk", "So", "Zs"),
    )
)


def julia_chars(data: bytes) -> list[tuple[int, bytes]]:
    """Split UTF-8 bytes into Julia's ``Char`` units: ``(code point or -1, bytes)``.

    Julia's ``iterate(::String)`` takes a lead byte and up to its declared
    number of continuation bytes (``10xxxxxx``), stopping early at anything
    else; a stray continuation byte, a lead byte above ``0xf7`` or a
    truncated sequence is one malformed ``Char`` of the bytes consumed, and an
    overlong encoding decodes but is printed byte by byte. The code point is
    ``-1`` for a malformed or overlong unit.
    """
    out = []
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        start = i
        i += 1
        if b < 0x80:
            out.append((b, data[start:i]))
            continue
        if b < 0xC0 or b > 0xF7:
            out.append((-1, data[start:i]))
            continue
        need = 1 if b < 0xE0 else 2 if b < 0xF0 else 3
        while need and i < n and data[i] & 0xC0 == 0x80:
            i += 1
            need -= 1
        unit = data[start:i]
        length = len(unit)
        declared = 2 if b < 0xE0 else 3 if b < 0xF0 else 4
        overlong = (
            b in (0xC0, 0xC1)
            or (b == 0xE0 and length > 1 and unit[1] < 0xA0)
            or (b == 0xF0 and length > 1 and unit[1] < 0x90)
        )
        if length != declared or overlong:
            out.append((-1, unit))
            continue
        point = b & (0x7F >> declared)
        for extra in unit[1:]:
            point = (point << 6) | (extra & 0x3F)
        out.append((point, unit))
    return out


def _is_hex_digit(point: int) -> bool:
    return 0x30 <= point <= 0x39 or 0x41 <= point <= 0x46 or 0x61 <= point <= 0x66


def _printable(point: int) -> bool:
    """Julia's ``isprint`` (categories Lu..Zs) through this Python's Unicode tables."""
    if point > 0x10FFFF:
        return False
    category = unicodedata.category(chr(point))
    if category == "Cn":
        raise ValueError(
            f"U+{point:04X} is unassigned in this Python's Unicode {unicodedata.unidata_version} "
            "tables, so whether Julia prints it raw or escaped is not known here; keep the "
            "stored text."
        )
    return category in _PRINTABLE_CATEGORIES


def julia_string_repr(data: bytes) -> str:  # noqa: PLR0912 - one branch per escape class
    r"""``repr(s::String)`` of the stored bytes: ``print_quoted`` with ``escape_string``.

    Quotes and ``$`` are backslash-escaped, the C escapes and ``\\e`` are
    used for their controls, other non-printable ASCII prints as ``\\xNN``,
    printable Unicode prints raw, non-printable Unicode as ``\\uNNNN`` /
    ``\\UNNNNNNNN`` (the minimum digits, or the full width when a hex digit
    follows) and malformed or overlong bytes as ``\\xNN`` each.
    """
    units = julia_chars(data)
    out = ['"']
    for index, (point, unit) in enumerate(units):
        following = units[index + 1][0] if index + 1 < len(units) else None
        if point < 0:
            out.append("".join(f"\\x{byte:02x}" for byte in unit))
        elif point in (0x22, 0x24):
            out.append("\\" + chr(point))
        elif point < 0x80:
            if point == 0:
                out.append(
                    "\\x00" if following is not None and 0x30 <= following <= 0x37 else "\\0"
                )
            elif point == 0x1B:
                out.append("\\e")
            elif point == 0x5C:
                out.append("\\\\")
            elif point in _SIMPLE_ESCAPES:
                out.append(_SIMPLE_ESCAPES[point])
            elif 0x20 <= point < 0x7F:
                out.append(chr(point))
            else:
                out.append(f"\\x{point:02x}")
        elif _printable(point):
            out.append(chr(point))
        else:
            full = following is not None and _is_hex_digit(following)
            if point <= 0xFFFF:
                out.append(f"\\u{point:04x}" if full else f"\\u{point:x}")
            else:
                out.append(f"\\U{point:08x}" if full else f"\\U{point:04x}")
    out.append('"')
    return "".join(out)


# ---- Julia's printed arrays ---------------------------------------------------


def _matrix_body(cells: np.ndarray[Any, Any]) -> str:
    """Rows joined by ``"; "`` and columns by ``" "``: ``_show_nonempty`` without brackets."""
    return "; ".join(" ".join(cells[row, :].tolist()) for row in range(cells.shape[0]))


def _cartesian(dims: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Julia's ``CartesianIndices`` order: the first index varies fastest."""
    return [
        tuple(reversed(index)) for index in itertools.product(*(range(n) for n in reversed(dims)))
    ]


def array_string(cells: np.ndarray[Any, Any], prefix: str, eltype: str) -> str:
    """``show(io, X::AbstractArray)`` of pre-rendered elements shaped like the Julia array.

    ``cells`` holds each element's own printed text at its Julia position (a
    column-major reshape of the stored order); ``prefix`` is the element type
    Julia writes before ``[`` when the type is not implied (``Int8``,
    ``Float32``, ``MIT{Monthly}``; nothing for Int64, Float64 and String) and
    ``eltype`` names the element type of an empty array (``Float64[]``,
    ``Matrix{String}(undef, 0, 2)``, ``Array{Int64, 3}(undef, 0, 2, 2)``).
    Vectors separate elements with ``", "``; matrices use ``" "`` and ``"; "``
    (a one-column matrix ends in ``;;``); higher ranks join their leading
    two-dimensional slices with ``;;;``, ``;;;;`` ... at each index change and
    close with ``nd + 2`` semicolons when the last dimension never changed.
    """
    shape = cells.shape
    ndim = cells.ndim
    if cells.size == 0:
        if ndim == 1:
            return f"{eltype}[]"
        name = f"Matrix{{{eltype}}}" if ndim == 2 else f"Array{{{eltype}, {ndim}}}"
        return f"{name}(undef, {', '.join(str(n) for n in shape)})"
    if ndim == 1:
        return f"{prefix}[{', '.join(cells.tolist())}]"
    if ndim == 2:
        return f"{prefix}[{_matrix_body(cells)}{'' if shape[1] > 1 else ';;'}]"
    out = [prefix, "["]
    tail = shape[2:]
    last: tuple[int, ...] | None = None
    reached_last = False
    for index in _cartesian(tail):
        if last is not None:
            changed = [previous < current for previous, current in zip(last, index, strict=True)]
            if any(changed):
                count = 2 + max(position + 1 for position, flag in enumerate(changed) if flag)
                out.append(";" * count)
                reached_last = reached_last or count == ndim
                out.append(" ")
        slice_index: tuple[Any, ...] = (slice(None), slice(None), *index)
        out.append(_matrix_body(cells[slice_index]))
        last = index
    if not reached_last:
        out.append(";" * (len(tail) + 2))
    out.append("]")
    return "".join(out)


_IMPLICIT_ELEMENTS = ("Int64", "Float64", "String")


def text_symbol(items: tuple[bytes, ...], shape: tuple[int, ...]) -> str:
    """``Symbol(::Array{String,N})``: Julia's printed String array (see :func:`array_string`)."""
    cells = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        cells[index] = julia_string_repr(item)
    return array_string(cells.reshape(shape, order="F"), "", "String")


def element_strings(  # noqa: PLR0911 - one form per element family
    values: np.ndarray[Any, Any],
    kind: str,
    dtype: np.dtype[Any],
    frequency: Frequency | None,
    width: int,
) -> list[str]:
    """``show(io, x)`` of every stored element with the array's ``typeinfo`` in effect.

    Signed integers print in decimal, unsigned ones as ``0x`` plus their full
    hex width, floats in Julia's shortest form (``show`` keeps ``f`` as the
    Float32 exponent marker: ``1.0f10``), complex values with typed components
    (``1.0f0 + 2.0f0im``, ``Float16(1.0) + Float16(2.0)im``), MIT and
    Duration elements through :func:`mit_string` and the code. ``values`` are
    the flat elements in Julia's (column-major) order.
    """
    if kind == "date":
        assert frequency is not None
        return [mit_string(int(code), frequency) for code in values.tolist()]
    if kind == "duration":
        return [str(int(code)) for code in values.tolist()]
    if kind == "int128":
        return [str(n) for n in _exact_words(values, signed=True)]
    if kind == "uint128":
        return [f"0x{n:032x}" for n in _exact_words(values, signed=False)]
    if kind == "complexf16":
        return [
            _exact.julia_complex_string(r, i, 2)
            for r, i in zip(values["real"], values["imag"], strict=True)
        ]
    if dtype.kind == "i":
        return [str(int(v)) for v in values.tolist()]
    if dtype.kind == "u":
        return [f"0x{int(v):0{2 * dtype.itemsize}x}" for v in values.tolist()]
    if dtype.kind == "f":
        return [_exact.julia_float_string(v, width, show=True) for v in values]
    return [_exact.julia_complex_string(v.real, v.imag, width) for v in values]


def _exact_words(values: np.ndarray[Any, Any], *, signed: bool) -> list[int]:
    from . import _interpret  # noqa: PLC0415 - the word codec lives with the carriers

    return _interpret.unpack_words(values, signed)


def numeric_symbol(
    values: np.ndarray[Any, Any],
    kind: str,
    dtype: np.dtype[Any],
    julia_name: str,
    frequency: Frequency | None,
) -> str:
    """``Symbol(::Array{T,N})`` of a numeric, wide, date or duration container.

    ``values`` is the stored array in its own shape (C-order NumPy view of the
    column-major Julia array); the element type prefix is written unless
    Julia implies it (``Int64``, ``Float64``).
    """
    flat = values.reshape(-1, order="F")
    width = dtype.itemsize // 2 if dtype.kind == "c" else dtype.itemsize
    cells = np.empty(flat.shape[0], dtype=object)
    for index, text in enumerate(element_strings(flat, kind, dtype, frequency, width)):
        cells[index] = text
    prefix = "" if julia_name in _IMPLICIT_ELEMENTS else julia_name
    return array_string(cells.reshape(values.shape, order="F"), prefix, julia_name)


def byte_symbol(data: bytes) -> str:
    """``Symbol(::Vector{UInt8})``: the bytes themselves name the symbol (no NUL allowed)."""
    if b"\0" in data:
        raise ValueError("Julia's Symbol name may not contain NUL (ArgumentError).")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            "Julia builds a Symbol whose name is not UTF-8; no Python str holds those bytes. "
            "Keep the stored array."
        ) from None
