# SPDX-License-Identifier: MIT
"""Finite interpretation of Julia reconstruction markers on stored series values.

Julia's DataEcon loader rebuilds a dated series from its native storage and
two optional string attributes: ``jeltype`` names an element type every value
is converted to, and ``jtype`` names a whole-object type that takes precedence
over ``jeltype`` (which is then ignored, even if unknown). Julia evaluates the
text; this module never does. Tokens are compared with finite tables, and the
conversions reproduce the pinned reference's constructor and IEEE routes:
nearest-even rounding, Float16 through Float32 for integers up to 64 bits,
Float16 through Float64 for 128-bit integers, plotting values (fractional
years) for year/period dates and durations, exact Int64 bounds for ``MIT``
targets and the Int64-only ``Duration`` constructor. Nothing here allocates a
converted array merely to check that a conversion is possible, and nothing
here touches native code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tsecon.frequencies import Frequency, HalfYearly, Monthly, Quarterly, Yearly

from ._metadata import (
    _SCALAR_FREQUENCIES,
    CODE_DTYPE,
    COMPLEXF16_DTYPE,
    INT128_DTYPE,
    JULIA_KIND_DEFAULTS,
    JULIA_NUMERIC_TYPES,
    KIND_COMPLEX,
    KIND_DATE,
    KIND_FLOAT,
    KIND_INTEGER,
    KIND_UNSIGNED,
    MAX_BYTES,
    MAX_INT64,
    MIN_INT64,
    julia_frequency_name,
)

_MASK64 = (1 << 64) - 1
_WIDE_KINDS = ("int128", "uint128", "complexf16")
_DATE_KINDS = ("date", "duration")
_WIDE_DTYPES: dict[str, np.dtype[Any]] = {
    "int128": INT128_DTYPE,
    "uint128": INT128_DTYPE,
    "complexf16": COMPLEXF16_DTYPE,
}
_WIDE_NAMES: dict[str, str] = {
    "int128": "Int128",
    "uint128": "UInt128",
    "complexf16": "ComplexF16",
}
_WIDE_KIND_BY_NAME = {name: kind for kind, name in _WIDE_NAMES.items()}
_NATIVE_KINDS: dict[str, int] = {"date": KIND_DATE, "duration": KIND_INTEGER}
_NATIVE_KINDS.update({"int128": KIND_INTEGER, "uint128": KIND_UNSIGNED, "complexf16": KIND_COMPLEX})
_DTYPE_KINDS = {"i": KIND_INTEGER, "u": KIND_UNSIGNED, "f": KIND_FLOAT, "c": KIND_COMPLEX}
_NUMERIC_NAMES = {dtype: name for name, (_, dtype) in JULIA_NUMERIC_TYPES.items()}
# Literal aliases Julia resolves to the same types on every supported platform.
ALIASES: dict[str, str] = {"Int": "Int64", "UInt": "UInt64", "Complex{Float16}": "ComplexF16"}
# Numeric targets Julia can build from MIT/Duration elements (besides Bool).
_DATE_TARGET_DTYPES = (
    np.dtype("<i8"),
    np.dtype("<f4"),
    np.dtype("<f8"),
    np.dtype("<c8"),
    np.dtype("<c16"),
)


@dataclass(frozen=True, slots=True)
class Target:
    """A resolved element family: an ordinary dtype, a wide carrier or a date family."""

    kind: str
    dtype: np.dtype[Any]
    frequency: Frequency | None = None

    @property
    def julia_name(self) -> str:
        """Julia's canonical spelling of this family, used as a comparison token."""
        if self.kind == "numeric":
            return _NUMERIC_NAMES[self.dtype]
        if self.kind in _WIDE_KINDS:
            return _WIDE_NAMES[self.kind]
        assert self.frequency is not None
        outer = "MIT" if self.kind == "date" else "Duration"
        return f"{outer}{{{julia_frequency_name(self.frequency)}}}"

    @property
    def native_kind(self) -> int:
        """The native element type code this family is stored under."""
        if self.kind == "numeric":
            return _DTYPE_KINDS[self.dtype.kind] if self.dtype.kind != "b" else KIND_INTEGER
        return _NATIVE_KINDS[self.kind]

    @property
    def is_bool(self) -> bool:
        return self.kind == "numeric" and self.dtype.kind == "b"


def numeric_target(dtype: np.dtype[Any]) -> Target:
    """Return the target for an ordinary series dtype (Bool included)."""
    return Target("numeric", dtype)


def wide_target(kind: str) -> Target:
    return Target(kind, _WIDE_DTYPES[kind])


def date_target(kind: str, frequency: Frequency) -> Target:
    return Target(kind, CODE_DTYPE, frequency)


def default_target(native_kind: int) -> Target:
    """Return the element Julia assumes for an unmarked empty payload of a numeric kind."""
    return numeric_target(JULIA_NUMERIC_TYPES[JULIA_KIND_DEFAULTS[native_kind]][1])


def _date_tokens() -> dict[str, Target]:
    tokens: dict[str, Target] = {}
    for frequency in _SCALAR_FREQUENCIES.values():
        for kind in _DATE_KINDS:
            target = date_target(kind, frequency)
            tokens[target.julia_name] = target
    return tokens


# The finite active ``jeltype`` vocabulary: 14 ordinary names, the three wide
# carriers, 64 exact date/duration tokens and three tested aliases.
ACTIVE_TOKENS: dict[str, Target] = {
    **{name: numeric_target(dtype) for name, (_, dtype) in JULIA_NUMERIC_TYPES.items()},
    **{name: wide_target(kind) for kind, name in _WIDE_NAMES.items()},
    **_date_tokens(),
}
ACTIVE_TOKENS.update({alias: ACTIVE_TOKENS[name] for alias, name in ALIASES.items()})
VECTOR_TOKENS = ("Vector", "Vector{Float64}")


def check_marker_text(marker: object, name: str) -> None:
    """Require a marker to be a plain string the native attribute API can hold."""
    if marker is None:
        return
    if type(marker) is not str:
        raise TypeError(f"The {name} marker must be a string or None.")
    if "\0" in marker:
        raise ValueError(f"The {name} marker cannot contain NUL.")


def resolve_token(token: str) -> Target | None:
    """Look a ``jeltype`` token up in the finite table; unknown text resolves to None."""
    return ACTIVE_TOKENS.get(token)


def object_interpretation(token: str, axis: Frequency, base: Target, length: int) -> str:
    """Classify a supported ``jtype`` token as ``"identity"`` or ``"vector"``.

    ``TSeries`` and the exactly spelled ``TSeries{F, T}`` /
    ``TSeries{F, T, Vector{T}}`` tokens naming this object's axis and base
    element are identity conversions in Julia; any other spelling or a
    mismatched parameter fails there and is refused here (no element
    conversion is attempted in its place). ``Vector`` and ``Vector{Float64}``
    are supported for empty ordinary numeric payloads only.
    """
    if token == "TSeries":
        return "identity"
    axis_name, element_name = julia_frequency_name(axis), base.julia_name
    if token in (
        f"TSeries{{{axis_name}, {element_name}}}",
        f"TSeries{{{axis_name}, {element_name}, Vector{{{element_name}}}}}",
    ):
        return "identity"
    if token in VECTOR_TOKENS:
        if base.kind != "numeric":
            raise TypeError(
                f"The whole-object marker {token!r} is supported for empty numeric payloads only."
            )
        if length:
            raise ValueError(
                f"The whole-object marker {token!r} applies to an empty payload only; "
                f"this series holds {length} observations."
            )
        return "vector"
    raise TypeError(
        f"Unsupported whole-object reconstruction marker {token!r}; the marker text is not "
        "evaluated and no element conversion is applied in its place."
    )


def vector_dtype(token: str, base: Target) -> np.dtype[Any]:
    """Return the NumPy dtype of an empty ``Vector`` interpretation."""
    return np.dtype("<f8") if token == "Vector{Float64}" else base.dtype


# ---- plain array and matrix whole-object markers ---------------------------

# Julia's `_apply_jtype` calls `convert(T, value)` on the whole array, so a
# parameterised container token converts every element, unlike the dated-series
# tokens above where the same spelling raises `DimensionMismatch`. These
# structural tokens name LinearAlgebra wrappers; the pinned loader can only
# rebuild them when the loading session has LinearAlgebra in `Main`.
STRUCTURE_TOKENS: tuple[str, ...] = ("Diagonal", "Symmetric", "Hermitian")
_ARRAY_IDENTITY_TOKENS: dict[int, tuple[str, ...]] = {
    1: ("Vector", "AbstractVector", "Array", "Any"),
    2: ("Matrix", "AbstractMatrix", "Array", "Any"),
}


def _array_token_element(token: str, ndim: int) -> str | None:
    """Return the element spelling inside a container token, or None."""
    outer = "Vector" if ndim == 1 else "Matrix"
    prefix = f"{outer}{{"
    if token.startswith(prefix) and token.endswith("}"):
        return token[len(prefix) : -1]
    for suffix in (f",{ndim}}}", f", {ndim}}}"):
        if token.startswith("Array{") and token.endswith(suffix):
            return token[len("Array{") : -len(suffix)]
    return None


def array_object_interpretation(token: str, base: Target, ndim: int) -> tuple[str, Target]:
    """Classify a supported array ``jtype`` token.

    Returns ``("identity", base)`` for the bare container and ``Any`` tokens
    and for a parameterised spelling naming the stored element, ``("element",
    target)`` for a parameterised spelling naming another supported element,
    and ``("structure", base)`` for the LinearAlgebra wrappers. Every other
    spelling raises ``TypeError``; the text is never evaluated.
    """
    if ndim not in _ARRAY_IDENTITY_TOKENS:
        raise TypeError("Whole-object array markers are supported for 1-D and 2-D objects only.")
    if token in _ARRAY_IDENTITY_TOKENS[ndim]:
        return "identity", base
    if ndim == 2 and token in STRUCTURE_TOKENS:
        return "structure", base
    inner = _array_token_element(token, ndim)
    target = None if inner is None else ACTIVE_TOKENS.get(inner)
    if target is None:
        raise TypeError(
            f"Unsupported whole-object reconstruction marker {token!r} for a "
            f"{ndim}-dimensional DataEcon array; the marker text is not evaluated."
        )
    return ("identity" if target == base else "element"), target


# ---- routes ---------------------------------------------------------------


def _is_integer_source(source: Target) -> bool:
    if source.kind == "numeric":
        return source.dtype.kind in "iu"
    return source.kind in ("int128", "uint128")


def check_route(source: Target, target: Target) -> None:
    """Refuse source/target pairs the pinned Julia loader has no conversion for."""
    if source == target:
        return
    if source.kind in _DATE_KINDS:
        if target.kind in _DATE_KINDS:
            raise TypeError(
                f"Julia has no conversion from {source.julia_name} elements to "
                f"{target.julia_name}; changing the element family or frequency is not "
                "a frequency conversion."
            )
        if target.kind == "numeric" and (target.is_bool or target.dtype in _DATE_TARGET_DTYPES):
            return
        raise TypeError(
            f"Julia has no conversion from {source.julia_name} elements to {target.julia_name} "
            "(only Bool, Int64, Float32, Float64, ComplexF32 and ComplexF64 are supported)."
        )
    if target.kind == "date":
        if _is_integer_source(source):
            return
        raise TypeError(
            f"Julia builds {target.julia_name} elements from integer sources only, not from "
            f"{source.julia_name}."
        )
    if target.kind == "duration":
        if source.kind == "numeric" and source.dtype == np.dtype("<i8"):
            return
        raise TypeError(
            f"Julia builds {target.julia_name} elements from Int64 sources only, not from "
            f"{source.julia_name}."
        )
    # Every numeric/wide source converts to every numeric/wide target, subject
    # to the value checks below.


# ---- value checks (masks only; no converted array is allocated) -----------


def _fail(source: Target, target: Target, index: int, reason: str) -> ValueError:
    return ValueError(
        f"The stored {source.julia_name} values cannot be interpreted as {target.julia_name}: "
        f"observation {index} {reason}."
    )


def _first_bad(valid: np.ndarray[Any, Any]) -> int | None:
    if bool(np.all(valid)):
        return None
    # Return the first False directly instead of allocating an intp index for
    # every invalid observation.
    return int(np.argmin(valid))


def _components(values: np.ndarray[Any, Any], source: Target) -> tuple[Any, Any]:
    """Real and (possibly None) imaginary views of a source array."""
    if source.kind == "complexf16":
        return values["real"], values["imag"]
    if source.kind == "numeric" and values.dtype.kind == "c":
        return values.real, values.imag
    return values, None


def _check_imag_zero(values: np.ndarray[Any, Any], source: Target, target: Target) -> None:
    _, imag = _components(values, source)
    if imag is not None:
        index = _first_bad(imag == 0)
        if index is not None:
            raise _fail(source, target, index, "has a nonzero imaginary part")


def check_bool(values: np.ndarray[Any, Any], source: Target) -> None:
    """Every value must be exactly zero or one (signed zeros count as zero)."""
    target = numeric_target(np.dtype("?"))
    if source.kind in ("int128", "uint128"):
        valid = (values["hi"] == 0) & (values["lo"] <= 1)
    else:
        real, imag = _components(values, source)
        valid = (real == 0) | (real == 1)
        if imag is not None:
            valid &= imag == 0
    index = _first_bad(valid)
    if index is not None:
        raise _fail(source, target, index, "is not exactly zero or one")


def bool_flags(values: np.ndarray[Any, Any], source: Target) -> np.ndarray[Any, Any]:
    """Return the Boolean reading of a carrier that passed :func:`check_bool`."""
    if source.kind in ("int128", "uint128"):
        return np.asarray(values["lo"] == 1, dtype=bool)
    real, _ = _components(values, source)
    return np.asarray(real == 1, dtype=bool)


def _int_bounds(target: Target) -> tuple[int, int]:
    if target.kind == "int128":
        return -(1 << 127), (1 << 127) - 1
    if target.kind == "uint128":
        return 0, (1 << 128) - 1
    if target.kind == "date":
        return MIN_INT64, MAX_INT64
    info = np.iinfo(target.dtype)
    return int(info.min), int(info.max)


def _check_words_in_range(
    values: np.ndarray[Any, Any], source: Target, target: Target, low: int, high: int
) -> None:
    """Range-check 128-bit words against integer bounds using word arithmetic only."""
    lo, hi = values["lo"], values["hi"]
    negative = (hi >= (1 << 63)) if source.kind == "int128" else np.zeros(len(lo), dtype=bool)
    if source.kind == "int128":
        # Nonnegative values: hi < 2**63; negative values: two's complement words.
        upper_words, upper_lo = divmod(high, 1 << 64) if high >= 0 else (0, 0)
        valid = ~negative & ((hi < upper_words) | ((hi == upper_words) & (lo <= upper_lo)))
        if low < 0:
            low_words, low_lo = divmod(low % (1 << 128), 1 << 64)
            valid |= negative & ((hi > low_words) | ((hi == low_words) & (lo >= low_lo)))
    else:
        upper_words, upper_lo = divmod(high, 1 << 64)
        valid = (hi < upper_words) | ((hi == upper_words) & (lo <= upper_lo))
    index = _first_bad(valid)
    if index is not None:
        raise _fail(source, target, index, "is outside the target range")


def check_integer(values: np.ndarray[Any, Any], source: Target, target: Target) -> None:
    """Integral, finite, imaginary-free values inside the target's exact bounds."""
    low, high = _int_bounds(target)
    if source.kind in ("int128", "uint128"):
        _check_words_in_range(values, source, target, low, high)
        return
    if source.kind in _DATE_KINDS:
        return  # full Int64 codes reach Int64 (and Int aliases) exactly
    _check_imag_zero(values, source, target)
    real, _ = _components(values, source)
    if real.dtype.kind in "iu":
        info = np.iinfo(real.dtype)
        valid = np.ones(len(real), dtype=bool)
        if low > info.min:
            valid &= real >= real.dtype.type(low)
        if high < info.max:
            valid &= real <= real.dtype.type(high)
        index = _first_bad(valid)
        if index is not None:
            raise _fail(source, target, index, "is outside the target range")
        return
    finite = np.isfinite(real)
    index = _first_bad(finite)
    if index is not None:
        raise _fail(source, target, index, "is not finite")
    index = _first_bad(real == np.floor(real))
    if index is not None:
        raise _fail(source, target, index, "is not an integer")
    limit = float(np.finfo(real.dtype).max)
    valid = np.ones(len(real), dtype=bool)
    if float(low) >= -limit:
        valid &= real >= real.dtype.type(float(low))
    if float(high + 1) <= limit:
        # ``high + 1`` is a power of two, exactly representable; ``high`` may not be.
        valid &= real < real.dtype.type(float(high + 1))
    index = _first_bad(valid)
    if index is not None:
        raise _fail(source, target, index, "is outside the target range")


def check_values(values: np.ndarray[Any, Any], source: Target, target: Target) -> None:
    """Raise ``ValueError`` if any stored value cannot be interpreted as the target."""
    if not len(values) or source == target:
        return
    if target.is_bool:
        check_bool(values, source)
    elif target.kind in ("int128", "uint128", "date") or (
        target.kind == "numeric" and target.dtype.kind in "iu"
    ):
        check_integer(values, source, target)
    elif target.kind == "numeric" and target.dtype.kind == "f":
        _check_imag_zero(values, source, target)
    # Complex, ComplexF16 and Duration targets accept every value of an accepted route.


def check_output_capacity(length: int, target: Target) -> None:
    """Refuse an explicit conversion whose output would exceed the payload limit."""
    if length * target.dtype.itemsize > MAX_BYTES:
        raise ValueError(
            f"Interpreting {length} observations as {target.julia_name} would exceed the "
            f"{MAX_BYTES} byte payload limit; nothing was allocated."
        )


# ---- conversion kernels ---------------------------------------------------


def unpack_words(values: np.ndarray[Any, Any], signed: bool) -> list[int]:
    """Python integers from Int128/UInt128 words (low word first)."""
    lo = values["lo"].astype(object)
    hi = values["hi"].astype(object)
    combined = lo | (hi << 64)
    if signed:
        combined = np.where(values["hi"] >= (1 << 63), combined - (1 << 128), combined)
    return [int(v) for v in combined]


def pack_words(items: list[int]) -> np.ndarray[Any, Any]:
    """Int128/UInt128 words from Python integers already checked for range."""
    out = np.empty(len(items), dtype=INT128_DTYPE)
    for index, item in enumerate(items):
        unsigned = item % (1 << 128)
        out[index] = (unsigned & _MASK64, (unsigned >> 64) & _MASK64)
    return out


def round_to_precision(number: int, precision: int) -> float:
    """Nearest-even rounding of an integer to ``precision`` significant bits.

    The result is returned as a Python float, which represents it exactly for
    the 24-bit (Float32) precision used here; ``float(number)`` already
    performs the 53-bit rounding for Float64.
    """
    if number == 0:
        return 0.0
    sign, magnitude = (-1, -number) if number < 0 else (1, number)
    shift = magnitude.bit_length() - precision
    if shift <= 0:
        return float(sign * magnitude)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return sign * math.ldexp(float(quotient), shift)


def _cast(values: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """Cast with IEEE overflow to infinity, as Julia's ``fptrunc`` produces."""
    with np.errstate(all="ignore"):
        return np.asarray(values.astype(dtype, copy=True))


def _uint64_to_float(values: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """Correctly rounded ``uitofp`` for 64-bit unsigned values, without relying on the C cast.

    Values below 2**63 convert as signed integers. Larger values halve with a
    sticky low bit, convert, and double: the discarded bit lies below the
    rounding position of either floating format, so the sticky bit preserves
    the nearest-even decision exactly.
    """
    big = values >= np.uint64(1 << 63)
    halved = (values >> np.uint64(1)) | (values & np.uint64(1))
    signed = np.where(big, halved, values).view("<i8")
    converted = signed.astype(dtype)
    return np.asarray(np.where(big, converted * dtype.type(2), converted))


def _real_to_float(
    real: np.ndarray[Any, Any], source: Target, dtype: np.dtype[Any]
) -> np.ndarray[Any, Any]:
    """Convert a real-valued view along Julia's route for the target float width."""
    if source.kind in ("int128", "uint128"):
        numbers = unpack_words(real, source.kind == "int128")
        if dtype == np.dtype("<f4"):
            rounded = np.array([round_to_precision(n, 24) for n in numbers], dtype="<f8")
        else:
            # Float64(Int128) rounds once; Float16(Int128) goes through Float64.
            rounded = np.array([float(n) for n in numbers], dtype="<f8")
        return _cast(rounded, dtype)
    if real.dtype.kind == "u" and real.dtype.itemsize == 8:
        wide = _uint64_to_float(real, np.dtype("<f4") if dtype == np.dtype("<f2") else dtype)
        return _cast(wide, dtype) if dtype == np.dtype("<f2") else wide
    if real.dtype.kind in "iu":
        if dtype == np.dtype("<f2"):
            # Float16(x::Integer) = Float16(Float32(x)) for integers up to 64 bits.
            return _cast(real.astype("<f4"), dtype)
        return _cast(real, dtype)
    return _cast(real, dtype)


def _plotting_values(
    codes: np.ndarray[Any, Any], source: Target, dtype: np.dtype[Any]
) -> np.ndarray[Any, Any]:
    """Julia's floating conversion of MIT/Duration elements (used for plotting).

    Year/period dates give ``year + (period - 1) / ppy`` and durations give
    ``code / ppy``, evaluated with Julia's separate integer division and
    floating operations; Unit and calendar codes convert directly. Float32
    output converts the Float64 intermediate, as Julia's ``convert`` does.
    """
    frequency = source.frequency
    if not isinstance(frequency, (Monthly, Quarterly, HalfYearly, Yearly)):
        return _cast(codes, dtype)
    ppy = frequency.periods_per_year
    if source.kind == "date":
        years, periods = np.divmod(codes, ppy)
        value = years.astype("<f8") + periods.astype("<f8") / np.float64(ppy)
    else:
        value = codes.astype("<f8") / np.float64(ppy)
    return _cast(value, dtype)


def _to_float(
    values: np.ndarray[Any, Any], source: Target, dtype: np.dtype[Any]
) -> np.ndarray[Any, Any]:
    if source.kind in _DATE_KINDS:
        return _plotting_values(values, source, dtype)
    real, _ = _components(values, source)
    return _real_to_float(real, source, dtype)


def _imag_to_float(
    values: np.ndarray[Any, Any], source: Target, dtype: np.dtype[Any]
) -> np.ndarray[Any, Any]:
    _, imag = _components(values, source)
    if imag is None:
        return np.zeros(len(values), dtype=dtype)
    return _cast(imag, dtype)


def _to_integer(
    values: np.ndarray[Any, Any], source: Target, target: Target
) -> np.ndarray[Any, Any]:
    """Integer output for a checked route: exact after :func:`check_integer`."""
    dtype = target.dtype
    if source.kind in ("int128", "uint128"):
        if target.kind in ("int128", "uint128"):
            return np.asarray(values.copy())
        lo = values["lo"]
        return np.asarray(lo.view("<i8").astype(dtype) if dtype.kind == "i" else lo.astype(dtype))
    if source.kind in _DATE_KINDS:
        return np.asarray(values.astype(dtype, copy=True))
    real, _ = _components(values, source)
    if target.kind in ("int128", "uint128"):
        if real.dtype.kind in "iu":
            out = np.empty(len(real), dtype=INT128_DTYPE)
            if real.dtype.kind == "i":
                out["lo"] = real.astype("<i8").view("<u8")
                out["hi"] = np.where(real < 0, np.uint64(_MASK64), np.uint64(0))
            else:
                out["lo"] = real.astype("<u8")
                out["hi"] = 0
            return out
        return pack_words([int(v) for v in real.astype("<f8").tolist()])
    return np.asarray(real.astype(dtype, copy=True))


def _to_codes(values: np.ndarray[Any, Any], source: Target) -> np.ndarray[Any, Any]:
    """Int64 element codes for a checked MIT/Duration route."""
    if source.kind in ("int128", "uint128"):
        return np.asarray(values["lo"].view("<i8").copy())
    if source.kind in _DATE_KINDS:
        return np.asarray(values.copy())
    return np.asarray(values.astype(CODE_DTYPE, copy=True))


def _to_complex(
    values: np.ndarray[Any, Any], source: Target, target: Target
) -> np.ndarray[Any, Any]:
    """Complex output: each component follows the float route of its width."""
    if target.kind == "complexf16":
        out = np.empty(len(values), dtype=COMPLEXF16_DTYPE)
        out["real"] = _to_float(values, source, np.dtype("<f2"))
        out["imag"] = _imag_to_float(values, source, np.dtype("<f2"))
        return out
    component = np.dtype("<f4") if target.dtype.itemsize == 8 else np.dtype("<f8")
    out = np.empty(len(values), dtype=target.dtype)
    out.real = _to_float(values, source, component)
    out.imag = _imag_to_float(values, source, component)
    return out


def convert_values(
    values: np.ndarray[Any, Any], source: Target, target: Target
) -> np.ndarray[Any, Any]:
    """Return a fresh array of the target carrier dtype; the caller checked the values."""
    if not len(values) or source == target:
        # Julia builds an empty typed vector for any target (no cast is involved);
        # an identity target copies the stored values.
        return np.asarray(values.copy()) if len(values) else np.empty(0, dtype=target.dtype)
    if target.is_bool:
        return bool_flags(values, source)
    if target.kind in _DATE_KINDS:
        return _to_codes(values, source)
    if target.kind in ("int128", "uint128") or target.dtype.kind in "iu":
        return _to_integer(values, source, target)
    if target.kind == "complexf16" or target.dtype.kind == "c":
        return _to_complex(values, source, target)
    return _to_float(values, source, target.dtype)
