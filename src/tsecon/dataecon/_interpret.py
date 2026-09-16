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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from tsecon.frequencies import Frequency, HalfYearly, Monthly, Quarterly, Yearly

from . import _exact
from ._exact import round_to_precision
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
# Exact carriers Julia's loader builds but its writer cannot store (it
# recurses): rationals, integer complexes and rational complexes of a bit
# integer parameter (``Complex{Bool}`` included). Their structured dtypes hold
# the exact components; the 128-bit parameters nest the two-word carrier.
# ``datedcomplex`` is ``Complex{MIT{F}}`` or ``Complex{Duration{F}}``: Julia's
# ``convert(Complex{T}, x)`` on an MIT/Duration element (or an integer under
# the explicit token) builds ``Complex(T(x), zero(T))``, a pair of two Int64
# codes; the ``parameter`` is ``MIT`` or ``Duration`` and the target carries
# the frequency.
_EXTENDED_KINDS = ("rational", "intcomplex", "rationalcomplex", "datedcomplex")
_EXTENDED_NAMES = {
    "rational": "Rational{{{}}}",
    "intcomplex": "Complex{{{}}}",
    "rationalcomplex": "Complex{{Rational{{{}}}}}",
}
_EXTENDED_FIELDS = {
    "rational": ("num", "den"),
    "intcomplex": ("re", "im"),
    "rationalcomplex": ("re_num", "re_den", "im_num", "im_den"),
    "datedcomplex": ("re", "im"),
}
_DATED_PARAMETERS = {"MIT": "date", "Duration": "duration"}
_PARAMETER_DTYPES: dict[str, np.dtype[Any]] = {
    **{f"Int{bits}": np.dtype(f"<i{bits // 8}") for bits in (8, 16, 32, 64)},
    **{f"UInt{bits}": np.dtype(f"<u{bits // 8}") for bits in (8, 16, 32, 64)},
    "Int128": INT128_DTYPE,
    "UInt128": INT128_DTYPE,
    "Bool": np.dtype("?"),
}
_NATIVE_KINDS: dict[str, int] = {"date": KIND_DATE, "duration": KIND_INTEGER}
_NATIVE_KINDS.update({"int128": KIND_INTEGER, "uint128": KIND_UNSIGNED, "complexf16": KIND_COMPLEX})
_DTYPE_KINDS = {"i": KIND_INTEGER, "u": KIND_UNSIGNED, "f": KIND_FLOAT, "c": KIND_COMPLEX}
_NUMERIC_NAMES = {dtype: name for name, (_, dtype) in JULIA_NUMERIC_TYPES.items()}
# Literal aliases Julia resolves to the same types on every supported platform
# (each verified individually against the pinned loader).
ALIASES: dict[str, str] = {
    "Int": "Int64",
    "UInt": "UInt64",
    "Complex{Float16}": "ComplexF16",
    "Complex{Float32}": "ComplexF32",
    "Complex{Float64}": "ComplexF64",
}
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
    """A resolved element family: an ordinary dtype, a wide carrier or a date family.

    The extended families carry a ``parameter``: the integer parameter of an
    exact ``rational``/``intcomplex``/``rationalcomplex`` carrier, the token
    of an ``empty``-only or ``opaque`` marker, or the ``Date``/``DateTime``
    token of a ``datetime`` result.
    """

    kind: str
    dtype: np.dtype[Any]
    frequency: Frequency | None = None
    parameter: str | None = None

    @property
    def julia_name(self) -> str:  # noqa: PLR0911 - one spelling per family
        """Julia's canonical spelling of this family, used as a comparison token."""
        if self.kind == "numeric":
            return _NUMERIC_NAMES[self.dtype]
        if self.kind in _WIDE_KINDS:
            return _WIDE_NAMES[self.kind]
        if self.kind == "datedcomplex":
            assert self.frequency is not None
            return f"Complex{{{self.parameter}{{{julia_frequency_name(self.frequency)}}}}}"
        if self.kind in _EXTENDED_KINDS:
            return _EXTENDED_NAMES[self.kind].format(self.parameter)
        if self.kind in ("bigint", "bigfloat", "symbol", "char"):
            return {"bigint": "BigInt", "bigfloat": "BigFloat", "symbol": "Symbol", "char": "Char"}[
                self.kind
            ]
        if self.kind in ("datetime", "empty", "opaque"):
            assert self.parameter is not None
            return self.parameter
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
# carriers, 64 exact date/duration tokens and five tested aliases.
ACTIVE_TOKENS: dict[str, Target] = {
    **{name: numeric_target(dtype) for name, (_, dtype) in JULIA_NUMERIC_TYPES.items()},
    **{name: wide_target(kind) for kind, name in _WIDE_NAMES.items()},
    **_date_tokens(),
}
ACTIVE_TOKENS.update({alias: ACTIVE_TOKENS[name] for alias, name in ALIASES.items()})
# Whole-object tokens on a dated series. Julia's loader applies `convert(T,
# value)` to the loaded TSeries: the bare/qualified container name, the exactly
# spelled parameterised forms naming this axis and element (both spacings, with
# `Vector{T}` as the optional third parameter) and the abstract
# `AbstractVector`/`Any` are identities; a mismatched axis or element parameter
# is a `MethodError` (no element conversion is attempted in its place); the
# plain vector spellings below are `DimensionMismatch` on a nonempty series and
# build a typed empty vector on an empty one. Every spelling was verified
# individually against the pinned loader over eighteen element families.
TSERIES_IDENTITY_TOKENS = (
    "TSeries",
    " TSeries",
    "TimeSeriesEcon.TSeries",
    "AbstractVector",
    "Any",
)
_EMPTY_VECTOR_ELEMENTS: tuple[str, ...] = (
    *(name for name in JULIA_NUMERIC_TYPES),
    *_WIDE_NAMES.values(),
)
# The bare vector spellings and `Vector{T}`/`Array{T}`/`Array{T,1}`/
# `Array{T, 1}` for the eighteen exact element names, plus `Vector{Any}`.
VECTOR_TOKENS: dict[str, str | None] = {"Vector": None, "Array": None, "Vector{Any}": "Any"}
for _element in _EMPTY_VECTOR_ELEMENTS:
    for _spelling in (
        f"Vector{{{_element}}}",
        f"Array{{{_element}}}",
        f"Array{{{_element},1}}",
        f"Array{{{_element}, 1}}",
    ):
        VECTOR_TOKENS[_spelling] = _element


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
    """Classify a supported ``jtype`` token as ``"identity"``, ``"vector"`` or ``"display"``.

    ``Symbol`` is ``"display"``: Julia's ``Symbol(::TSeries)`` is the series'
    display text, whose row selection and column widths depend on the
    loading session's ``LINES``/``COLUMNS``, so no fixed value reconstructs
    it (see :func:`display_text_error`). The identity spellings
    (:data:`TSERIES_IDENTITY_TOKENS`, ``TSeries{F}`` and
    the exact ``TSeries{F, T}`` / ``TSeries{F,T}`` / ``TSeries{F, T,
    Vector{T}}`` forms naming this object's axis and base element) are
    identity conversions in Julia; any other spelling or a mismatched
    parameter fails there and is refused here (no element conversion is
    attempted in its place). The plain vector spellings
    (:data:`VECTOR_TOKENS`) build a typed empty vector from an empty numeric
    or wide payload only.
    """
    if token in TSERIES_IDENTITY_TOKENS:
        return "identity"
    if token == "Symbol":
        return "display"
    axis_name, element_name = julia_frequency_name(axis), base.julia_name
    if token in (
        f"TSeries{{{axis_name}}}",
        f"TSeries{{{axis_name}, {element_name}}}",
        f"TSeries{{{axis_name},{element_name}}}",
        f"TSeries{{{axis_name}, {element_name}, Vector{{{element_name}}}}}",
    ):
        return "identity"
    if token in VECTOR_TOKENS:
        if base.kind in _DATE_KINDS:
            raise TypeError(
                f"The whole-object marker {token!r} is supported for empty numeric payloads only."
            )
        if length:
            raise ValueError(
                f"The whole-object marker {token!r} applies to an empty payload only; "
                f"this series holds {length} observations."
            )
        return "vector"
    # Any other spelling (a mismatched parameter, a type Julia fails on or one
    # this table does not know) is preserved opaquely: the marker text is not
    # evaluated, no element conversion is applied in its place and explicit
    # interpretation raises.
    return "opaque"


# Whole-object tokens on a dated matrix. Julia's loader applies `convert(T,
# value)` to the loaded MVTSeries: the bare and qualified container names, the
# exact parameterised spellings naming this axis and element (with `Matrix{T}`
# as the optional third parameter) and the abstract `AbstractMatrix`/`Any`
# are identities; a mismatched axis or element parameter is a `MethodError`
# (no element conversion is attempted), the plain `Matrix`/`Array`/bit-array
# spellings are `DimensionMismatch`, and the LinearAlgebra wrappers,
# `TSeries` and `Vector` fail too. Only the identities are accepted here.
MVTSERIES_IDENTITY_TOKENS = (
    "MVTSeries",
    "TimeSeriesEcon.MVTSeries",
    "AbstractMatrix",
    "AbstractArray",
    "Any",
)


def _element_spellings(element_name: str) -> tuple[str, ...]:
    """Return the exact name plus the verified aliases Julia resolves to it in a parameter."""
    return (element_name, *(alias for alias, name in ALIASES.items() if name == element_name))


def mvtseries_object_interpretation(token: str, axis: Frequency, base: Target) -> str:
    """Classify a supported ``jtype`` token on an MVTSeries as ``"identity"`` or ``"display"``.

    ``Symbol`` is ``"display"`` (the display text; see
    :func:`display_text_error`). Identities are the container spellings in
    :data:`MVTSERIES_IDENTITY_TOKENS`,
    ``MVTSeries{F}``, the exact ``MVTSeries{F, T}`` (either spacing) and
    ``MVTSeries{F, T, Matrix{T}}`` forms naming this axis and element, and
    ``MVTSeries{F, A}`` for a verified alias ``A`` of the element
    (``Complex{Float16}`` for ComplexF16).
    """
    if token in MVTSERIES_IDENTITY_TOKENS:
        return "identity"
    if token == "Symbol":
        return "display"
    axis_name, element_name = julia_frequency_name(axis), base.julia_name
    if token in (
        f"MVTSeries{{{axis_name}}}",
        f"MVTSeries{{{axis_name}, {element_name}}}",
        f"MVTSeries{{{axis_name},{element_name}}}",
        f"MVTSeries{{{axis_name}, {element_name}, Matrix{{{element_name}}}}}",
    ):
        return "identity"
    if token in tuple(
        f"MVTSeries{{{axis_name}, {alias}}}" for alias in _element_spellings(element_name)[1:]
    ):
        return "identity"
    return "opaque"


def vector_dtype(token: str, base: Target) -> np.dtype[Any]:
    """Return the NumPy dtype of an empty vector interpretation.

    The bare ``Vector``/``Array`` keep the base carrier dtype, ``Vector{Any}``
    gives an object dtype, and a named element gives its carrier dtype
    (Boolean for ``Bool``, the structured word/pair dtypes for the wide
    families).
    """
    element = VECTOR_TOKENS[token]
    if element is None:
        return base.dtype
    if element == "Any":
        return np.dtype(object)
    return ACTIVE_TOKENS[element].dtype


# ---- plain array and matrix whole-object markers ---------------------------

# Julia's `_apply_jtype` calls `convert(T, value)` on the whole array, so a
# parameterised container token converts every element, unlike the dated-series
# tokens above where the same spelling raises `DimensionMismatch`. These
# structural tokens name LinearAlgebra wrappers; the pinned loader can only
# rebuild them when the loading session has LinearAlgebra in `Main`, and it
# has no conversion from a rank-3 or higher array to any of them.
STRUCTURE_TOKENS: tuple[str, ...] = ("Diagonal", "Symmetric", "Hermitian")
# DataEcon stores at most five axes (DE_MAX_AXES); ranks three to five are
# `type_tensor` objects whose bare container spellings are `Array` and
# `AbstractArray`.
MAX_AXES = 5
_ARRAY_IDENTITY_TOKENS: dict[int, tuple[str, ...]] = {
    1: ("Vector", "AbstractVector", "Array", "Any"),
    2: ("Matrix", "AbstractMatrix", "Array", "Any"),
    **dict.fromkeys(range(3, MAX_AXES + 1), ("AbstractArray", "Array", "Any")),
}
# Julia's own writer stores a `BitArray` with `jtype` naming the rank and
# `jeltype = "Bool"`; the loader converts exact 0/1 values back to a BitArray.
_BIT_ARRAY_TOKENS: dict[int, tuple[str, ...]] = {
    n: (f"BitArray{{{n}}}",) + (("BitVector",) if n == 1 else ("BitMatrix",) if n == 2 else ())
    for n in range(1, MAX_AXES + 1)
}


def _array_token_element(token: str, ndim: int) -> str | None:
    """Return the element spelling inside a container token, or None.

    ``Vector{T}``/``Matrix{T}`` name ranks one and two, ``Array{T,N}`` and
    ``Array{T, N}`` any rank, and the rank-free ``Array{T}`` a plain vector
    (the only rank on which that spelling was verified).
    """
    if ndim in (1, 2):
        prefix = "Vector{" if ndim == 1 else "Matrix{"
        if token.startswith(prefix) and token.endswith("}"):
            return token[len(prefix) : -1]
    for suffix in (f",{ndim}}}", f", {ndim}}}"):
        if token.startswith("Array{") and token.endswith(suffix):
            return token[len("Array{") : -len(suffix)]
    if ndim == 1 and token.startswith("Array{") and token.endswith("}") and "," not in token:
        return token[len("Array{") : -1]
    return None


def array_object_interpretation(  # noqa: PLR0911 - one class per token family
    token: str, base: Target, ndim: int
) -> tuple[str, Target]:
    """Classify a supported array ``jtype`` token.

    Returns ``("identity", base)`` for the bare container and ``Any`` tokens
    and for a parameterised spelling naming the stored element, ``("element",
    target)`` for a parameterised spelling naming another supported element
    (the Julia bit-array tokens name ``Bool``), ``("structure", base)`` for
    the LinearAlgebra wrappers on a matrix, ``("printed", base)`` for
    ``Symbol`` (Julia's ``Symbol(string(array))``, the printed array;
    ``("bytes", base)`` on a UInt8 vector, whose bytes name the symbol
    directly) and ``("opaque", target)`` for every other spelling, which is
    preserved; the text is never evaluated.
    """
    if ndim not in _ARRAY_IDENTITY_TOKENS:
        raise TypeError(
            f"Whole-object array markers are supported for one- to {MAX_AXES}-dimensional "
            "objects only."
        )
    if token in _ARRAY_IDENTITY_TOKENS[ndim]:
        return "identity", base
    if token == "Symbol":
        if ndim == 1 and base.kind == "numeric" and base.dtype == np.dtype("<u1"):
            return "bytes", base
        return "printed", base
    if ndim == 2 and token in STRUCTURE_TOKENS:
        return "structure", base
    if token in _BIT_ARRAY_TOKENS[ndim]:
        return "element", ACTIVE_TOKENS["Bool"]
    inner = _array_token_element(token, ndim)
    target = None if inner is None else ACTIVE_TOKENS.get(inner)
    if target is None:
        # Preserved opaquely: the text is not evaluated and interpretation raises.
        return "opaque", opaque_target(token, base)
    return ("identity" if target == base else "element"), target


# ---- routes ---------------------------------------------------------------


def _is_integer_source(source: Target) -> bool:
    if source.kind == "numeric":
        return source.dtype.kind in "iu"
    return source.kind in ("int128", "uint128")


def check_route(source: Target, target: Target) -> None:  # noqa: PLR0911 - one rule per family
    """Refuse source/target pairs the pinned Julia loader has no conversion for."""
    if source == target:
        return
    if source.kind in _DATE_KINDS:
        if target.kind == "datedcomplex":
            check_dated_complex_route(source, target)
            return
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
    if target.kind == "datedcomplex":
        check_dated_complex_route(source, target)
        return
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
    if target.kind == "char":
        if source.kind in _DATE_KINDS:
            raise TypeError(f"Julia has no Char conversion from {source.julia_name} elements.")
        return
    # Every numeric/wide source converts to every numeric/wide target, subject
    # to the value checks below.


def check_dated_complex_route(source: Target, target: Target) -> None:
    """``Complex{MIT{F}}``/``Complex{Duration{F}}`` follow the ``MIT{F}``/``Duration{F}`` routes.

    The same-family, same-frequency element is the identity component; an
    MIT complex also builds from any integer source (``MIT{F}(Int64(n))``)
    and a Duration complex from an Int64 source only; floats, complexes and
    every other date family have no method.
    """
    assert target.frequency is not None
    parameter_kind = _DATED_PARAMETERS[target.parameter or ""]
    if source.kind == parameter_kind and source.frequency == target.frequency:
        return
    if source.kind in _DATE_KINDS:
        raise TypeError(
            f"Julia has no {target.julia_name} conversion from {source.julia_name} elements."
        )
    if parameter_kind == "date" and _is_integer_source(source):
        return
    if (
        parameter_kind == "duration"
        and source.kind == "numeric"
        and source.dtype == np.dtype("<i8")
    ):
        return
    raise TypeError(
        f"Julia has no {target.julia_name} conversion from {source.julia_name} elements."
    )


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
    if target.kind in _EXTENDED_KINDS:
        _drain(_iter_extended(values, source, target))  # raises where Julia's constructor does
        return
    if target.kind == "bigint":
        _drain(_iter_bigint(values, source))
        return
    if target.kind == "bigfloat":
        _drain(_iter_bigfloat(values, source))
        return
    if target.kind == "datetime":
        _drain(_iter_rata_die(values, source, target))
        return
    if target.kind == "char":
        _drain(_iter_chars(values, source))
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


def check_payload_bytes(nbytes: int, description: str) -> None:
    """Refuse a prospective payload beyond the limit before anything is allocated.

    ``nbytes`` is computed by the caller in Python integers from the input's
    shape, so an oversized request never reaches NumPy.
    """
    if nbytes > MAX_BYTES:
        raise ValueError(
            f"{description} would need {nbytes} bytes, more than the {MAX_BYTES} byte "
            "payload limit; nothing was allocated."
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


def convert_values(  # noqa: PLR0911 - one return per target family
    values: np.ndarray[Any, Any], source: Target, target: Target
) -> np.ndarray[Any, Any]:
    """Return a fresh array of the target carrier dtype; the caller checked the values."""
    if not len(values) or source == target:
        # Julia builds an empty typed vector for any target (no cast is involved);
        # an identity target copies the stored values.
        return np.asarray(values.copy()) if len(values) else np.empty(0, dtype=target.dtype)
    if target.kind in _EXTENDED_KINDS:
        return _pack_extended(list(_iter_extended(values, source, target)), target)
    if target.kind == "bigint":
        return _object_array(_iter_bigint(values, source), len(values))
    if target.kind == "bigfloat":
        return _object_array(_iter_bigfloat(values, source), len(values))
    if target.kind == "datetime":
        return _datetime_values(values, source, target)
    if target.kind == "char":
        return _char_values(values, source)
    if target.is_bool:
        return bool_flags(values, source)
    if target.kind in _DATE_KINDS:
        return _to_codes(values, source)
    if target.kind in ("int128", "uint128") or target.dtype.kind in "iu":
        return _to_integer(values, source, target)
    if target.kind == "complexf16" or target.dtype.kind == "c":
        return _to_complex(values, source, target)
    return _to_float(values, source, target.dtype)


# ---- extended element families ----------------------------------------------
#
# Julia's loader applies the same ``convert`` to every element of an array
# that it applies to a scalar, so these routes are the scalar routes of
# ``_scalars.StoredScalar`` element by element: ``Rational{T}`` and
# ``Complex{Rational{T}}`` through Julia's width-aware ``rationalize``,
# integer ``Complex{T}`` through ``T(x)``, ``BigInt``/``BigFloat`` as exact
# Python ``int``/``Decimal`` objects, and ``Date``/``DateTime`` through
# ``unix2datetime`` (plain arrays only: a dated series needs ``Number``
# elements). The results are contiguous structured carriers or object arrays;
# nothing here evaluates marker text.

# Julia spellings the pinned loader resolves to the exact token on the right,
# each individually verified against the reference (qualified names Julia
# exports through Base/Core/TimeSeriesEcon/Dates and the recorded spacing
# variants). The literal text is preserved on rewrite; this table only decides
# interpretation. No general grammar: an unlisted spelling is opaque.
SPELLINGS: dict[str, str] = {
    "Dates.Date": "Date",
    "Dates.DateTime": "DateTime",
    "Rational{Int}": "Rational{Int64}",
    "Complex{Int}": "Complex{Int64}",
    "Base.Int64": "Int64",
    "Core.Int64": "Int64",
    "Main.Int64": "Int64",
    "Main.Base.Int64": "Int64",
    "TimeSeriesEcon.Int64": "Int64",
    "Base.Float64": "Float64",
    " Int64": "Int64",
    "Int64 ": "Int64",
    " Int64 ": "Int64",
    "Float64 ": "Float64",
}


def resolve_spelling(marker: str) -> str:
    """Return the exact token a verified alternative spelling stands for (else the text)."""
    return SPELLINGS.get(marker, marker)


ABSTRACT_TOKENS = ("Any", "Number", "Real", "Integer", "Signed", "Unsigned", "AbstractFloat")
UNION_TOKENS = ("Union{Int64,Float64}", "Union{Int64, Float64}")
DATE_TOKENS = {"Date": "D", "DateTime": "ms", "Dates.Date": "D", "Dates.DateTime": "ms"}
# Tokens Julia loads from an empty payload as a typed empty vector and from a
# nonempty one not at all (``MethodError``), mapped to the empty NumPy dtype
# of the interpretation (object where no faithful native dtype exists).
EMPTY_ONLY_TOKENS: dict[str, np.dtype[Any]] = {
    "Symbol": np.dtype("<U1"),
    "String": np.dtype("<U1"),
    "Char": np.dtype(object),
    "Date": np.dtype("<M8[D]"),
    "DateTime": np.dtype("<M8[ms]"),
    "Dates.Date": np.dtype("<M8[D]"),
    "Dates.DateTime": np.dtype("<M8[ms]"),
    **dict.fromkeys(
        (
            "MIT",
            "Duration",
            "MIT{Quarterly}",
            "MIT{Frequency}",
            "Union{}",
            "Nothing",
            "Missing",
            "Vector{Int64}",
            "BigInt",
            "BigFloat",
            "Rational",
            "Complex",
            *ABSTRACT_TOKENS,
            *UNION_TOKENS,
        ),
        np.dtype(object),
    ),
}
_SIGNED_DTYPES = {np.dtype(f"<u{n}"): np.dtype(f"<i{n}") for n in (1, 2, 4, 8)}
_UNSIGNED_DTYPES = {signed: unsigned for unsigned, signed in _SIGNED_DTYPES.items()}


def parameter_dtype(parameter: str) -> np.dtype[Any]:
    """Return the NumPy dtype of one component of an exact carrier of this parameter."""
    return _PARAMETER_DTYPES[parameter]


def carrier_dtype(kind: str, parameter: str) -> np.dtype[Any]:
    """Return the structured carrier dtype of an exact family (``rational``, ...)."""
    component = CODE_DTYPE if kind == "datedcomplex" else parameter_dtype(parameter)
    return np.dtype([(name, component) for name in _EXTENDED_FIELDS[kind]])


def dated_complex_target(parameter: str, frequency: Frequency) -> Target:
    """Return the ``Complex{MIT{F}}`` (``parameter="MIT"``) or ``Complex{Duration{F}}`` target."""
    if parameter not in _DATED_PARAMETERS:
        raise TypeError(f"Unknown dated complex parameter {parameter!r}.")
    return Target("datedcomplex", carrier_dtype("datedcomplex", parameter), frequency, parameter)


def _dated_complex_token(token: str) -> Target | None:
    """Resolve ``Complex{MIT{F}}``/``Complex{Duration{F}}`` through the finite date table."""
    if not (token.startswith("Complex{") and token.endswith("}")):
        return None
    inner = ACTIVE_TOKENS.get(token[len("Complex{") : -1])
    if inner is None or inner.kind not in _DATE_KINDS:
        return None
    assert inner.frequency is not None
    return dated_complex_target("MIT" if inner.kind == "date" else "Duration", inner.frequency)


def extended_target(kind: str, parameter: str) -> Target:
    """Return the target of an exact carrier family; ``TypeError`` for an unknown parameter."""
    if kind not in _EXTENDED_KINDS:
        raise TypeError(f"Unknown exact carrier kind {kind!r}.")
    table = _exact.COMPLEX_PARAMETERS if kind == "intcomplex" else _exact.RATIONAL_PARAMETERS
    if parameter not in table:
        raise TypeError(f"Unknown {_EXTENDED_NAMES[kind].format('T')} parameter {parameter!r}.")
    return Target(kind, carrier_dtype(kind, parameter), None, parameter)


def empty_target(token: str, dtype: np.dtype[Any]) -> Target:
    return Target("empty", dtype, None, token)


def opaque_target(token: str, base: Target) -> Target:
    return Target("opaque", base.dtype, base.frequency, token)


def _element_width(source: Target) -> int:
    """Return the float width Julia's arithmetic runs in for a float or complex source."""
    if source.kind == "complexf16":
        return 2
    return source.dtype.itemsize // 2 if source.dtype.kind == "c" else source.dtype.itemsize


# Validation walks the values element by element without retaining them
# (nothing is allocated beyond a bounded chunk); conversion collects the same
# iterators into the result carrier.
_CHUNK = 1 << 16


def _drain(items: Iterator[Any]) -> None:
    """Consume an element iterator for its checks only, retaining nothing."""
    for _ in items:
        pass


def _object_array(items: Iterator[Any], length: int) -> np.ndarray[Any, Any]:
    out = np.empty(length, dtype=object)
    for index, item in enumerate(items):
        out[index] = item
    return out


def _iter_integers(values: np.ndarray[Any, Any], source: Target) -> Iterator[int]:
    """Python integers of an integer, wide or date/duration source, one bounded chunk at a time."""
    for start in range(0, len(values), _CHUNK):
        chunk = values[start : start + _CHUNK]
        if source.kind in ("int128", "uint128"):
            yield from unpack_words(chunk, source.kind == "int128")
        else:
            yield from (int(v) for v in chunk.tolist())


def _iter_reals(values: np.ndarray[Any, Any], source: Target) -> Iterator[tuple[Any, Any]]:
    """``(real, imag)`` NumPy scalars of a float or complex source in their own width."""
    real, imag = _components(values, source)
    zero = real.dtype.type(0)
    for index in range(len(real)):
        yield real[index], (zero if imag is None else imag[index])


def _is_float_source(source: Target) -> bool:
    return source.kind == "complexf16" or (source.kind == "numeric" and source.dtype.kind in "fc")


def _iter_extended(  # noqa: PLR0912 - one branch per source family
    values: np.ndarray[Any, Any], source: Target, target: Target
) -> Iterator[tuple[int, ...]]:
    """Julia's exact components for every element, or ``ValueError`` where it fails.

    Integer sources take ``T(n)`` (a ``Rational`` is ``n//1``); float sources
    rationalize in their own width or convert through ``T(x)``; complex
    sources need a zero imaginary part except under ``Complex{Rational{T}}``,
    which rationalizes both components; MIT/Duration sources have a method
    only for the ``Int64`` parameter (and ``Complex{Bool}`` on a 0/1 code).
    """
    kind, parameter = target.kind, target.parameter
    assert parameter is not None
    if kind == "datedcomplex":
        yield from _iter_dated_complex(values, source, target)
        return
    if source.kind in _DATE_KINDS:
        if parameter != "Int64" and not (kind == "intcomplex" and parameter == "Bool"):
            raise TypeError(
                f"Julia has no {target.julia_name} conversion from {source.julia_name} elements."
            )
        for code in _iter_integers(values, source):
            if kind == "rational":
                yield (code, 1)
            elif kind == "intcomplex":
                yield (_exact.integer_from_integer(code, parameter), 0)
            else:
                yield (code, 1, 0, 1)
        return
    if _is_integer_source(source):
        for number in _iter_integers(values, source):
            if kind == "rational":
                yield _exact.rational_from_integer(number, parameter)
            elif kind == "intcomplex":
                yield _exact.integer_complex_from_integer(number, parameter)
            else:
                yield (*_exact.rational_from_integer(number, parameter), 0, 1)
        return
    width = _element_width(source)
    for re, im in _iter_reals(values, source):
        if kind == "rationalcomplex":
            real_part = _exact.rationalize(re, parameter, width)
            yield (*real_part, *_exact.rationalize(im, parameter, width))
            continue
        if kind == "intcomplex":
            yield _exact.integer_complex_from_floats(float(re), float(im), parameter)
            continue
        if im != 0:
            raise _exact._failure("InexactError", target.julia_name, complex(float(re), float(im)))
        yield _exact.rationalize(re, parameter, width)


def _iter_dated_complex(
    values: np.ndarray[Any, Any], source: Target, target: Target
) -> Iterator[tuple[int, int]]:
    """``Complex(T(x), zero(T))`` per element: the Int64 code and a zero code."""
    check_dated_complex_route(source, target)
    for number in _iter_integers(values, source):
        yield _exact.integer_from_integer(number, "Int64"), 0


def _two_words(target: Target) -> bool:
    return target.kind != "datedcomplex" and parameter_dtype(target.parameter or "") == INT128_DTYPE


def _pack_extended(components: list[tuple[int, ...]], target: Target) -> np.ndarray[Any, Any]:
    """Pack exact components into the structured carrier of ``target``."""
    assert target.parameter is not None
    out = np.empty(len(components), dtype=target.dtype)
    fields = _EXTENDED_FIELDS[target.kind]
    if _two_words(target):
        for position, name in enumerate(fields):
            out[name] = pack_words([item[position] for item in components])
    else:
        for position, name in enumerate(fields):
            out[name] = [item[position] for item in components]
    return out


def unpack_extended(values: np.ndarray[Any, Any], target: Target) -> list[tuple[int, ...]]:
    """Python integers of every component of an exact carrier (128-bit words unpacked)."""
    assert target.parameter is not None
    fields = _EXTENDED_FIELDS[target.kind]
    if _two_words(target):
        columns = [unpack_words(values[name], target.parameter == "Int128") for name in fields]
    else:
        columns = [[int(v) for v in values[name].tolist()] for name in fields]
    return list(zip(*columns, strict=True))


def _iter_bigint(values: np.ndarray[Any, Any], source: Target) -> Iterator[int]:
    """``BigInt(x)`` per element: exact integers; floats must be integral with no imaginary part."""
    if source.kind in _DATE_KINDS:
        raise TypeError(f"Julia has no BigInt conversion from {source.julia_name} elements.")
    if _is_integer_source(source):
        yield from _iter_integers(values, source)
        return
    for real, imag in _iter_reals(values, source):
        re, im = float(real), float(imag)
        if im != 0 or not np.isfinite(re) or re != np.floor(re):
            raise _exact._failure("InexactError", "BigInt", complex(re, im) if im else re)
        yield int(re)


def _iter_bigfloat(values: np.ndarray[Any, Any], source: Target) -> Iterator[Any]:
    """``BigFloat(x)`` per element: the exact value of an integer or float (zero imaginary part)."""
    if source.kind in _DATE_KINDS:
        raise TypeError(f"Julia has no BigFloat conversion from {source.julia_name} elements.")
    if _is_integer_source(source):
        for number in _iter_integers(values, source):
            yield _exact.exact_decimal(number)
        return
    for real, imag in _iter_reals(values, source):
        if imag != 0:
            raise _exact._failure("InexactError", "BigFloat", complex(float(real), float(imag)))
        yield _exact.exact_decimal(float(real))


def _iter_chars(values: np.ndarray[Any, Any], source: Target) -> Iterator[str]:
    """``Char(x)`` per element: ``Char(UInt32(x))`` for a valid code point, else Julia's error.

    Integers, integral finite floats and complexes with a zero imaginary part
    convert through ``UInt32`` (``InexactError`` otherwise); ``Char(::UInt32)``
    accepts code points below ``0x200000`` (``CodePointError`` above) and
    builds an invalid ``Char`` for ``0x110000..0x1fffff`` and the surrogates,
    which Python represents only up to ``0x10ffff`` (a surrogate is a valid
    ``str`` character; a larger point is refused explicitly). The result is
    an object array of one-character ``str`` (a NumPy ``str_`` array cannot
    hold ``U+0000``).
    """
    if source.kind in _DATE_KINDS:
        raise TypeError(f"Julia has no Char conversion from {source.julia_name} elements.")
    if _is_integer_source(source):
        points: Iterator[Any] = _iter_integers(values, source)
    else:
        points = _iter_reals(values, source)
    for item in points:
        if _is_integer_source(source):
            number = int(item)
        else:
            real, imag = item
            if imag != 0:
                raise _exact._failure("InexactError", "UInt32", complex(float(real), float(imag)))
            if not (np.isfinite(real) and float(real) == np.floor(real)):
                raise _exact._failure("InexactError", "UInt32", float(real))
            number = int(float(real))
        yield _char_of(number)


def _char_of(number: int) -> str:
    if not 0 <= number < 1 << 32:
        raise _exact._failure("InexactError", "UInt32", number)
    if number >= 0x200000:
        raise ValueError(
            f"Julia raises CodePointError: {number:#x} is not a valid Char code point."
        )
    if number > 0x10FFFF:
        raise ValueError(
            f"Julia builds the invalid Char U+{number:X} (above U+10FFFF), which no Python str "
            "character represents; keep the stored container."
        )
    return chr(number)


def _char_values(values: np.ndarray[Any, Any], source: Target) -> np.ndarray[Any, Any]:
    # An object array of one-character str: a NumPy str_ array cannot hold U+0000.
    return _object_array(_iter_chars(values, source), len(values))


def display_text_error(container: str) -> TypeError:
    """Return the refusal of a whole-object ``Symbol`` on a dated container.

    Julia's ``Symbol(::TSeries)``/``Symbol(::MVTSeries)`` is the object's
    display text, which selects rows and pads columns from the loading
    session's ``displaysize`` (``LINES``/``COLUMNS``); the pinned loader
    returned different names for the same stored object under different
    settings, so the marker is preserved and no value is reconstructed.
    """
    return TypeError(
        f"Julia's Symbol of a {container} is its display text, which depends on the loading "
        "session's LINES/COLUMNS; the marker is preserved and not interpreted."
    )


def check_union_values(
    values: np.ndarray[Any, Any], source: Target, failure: str = "MethodError"
) -> None:
    """Refuse a complex source with a nonzero imaginary part before any real-valued route.

    ``Union{Int64,Float64}`` has no method for it (``TypeError``); a date
    constructor's ``isreal`` check raises ``InexactError`` (``ValueError``).
    """
    _, imag = _components(values, source)
    if imag is not None and bool(np.any(imag != 0)):
        if failure == "InexactError":
            raise ValueError(
                f"Julia raises InexactError: {source.julia_name} elements with a nonzero "
                "imaginary part have no real value to convert."
            )
        raise TypeError(
            f"Julia has no Union{{Int64,Float64}} conversion from {source.julia_name} elements "
            "with a nonzero imaginary part."
        )


def _iter_rata_die(values: np.ndarray[Any, Any], source: Target, target: Target) -> Iterator[int]:
    """``unix2datetime`` per element as Rata Die milliseconds (plain arrays only).

    A Duration is a Signed count and multiplies like an Int64; an MIT has no
    method.
    """
    if source.kind == "date":
        raise TypeError(
            f"Julia has no {target.julia_name} conversion from {source.julia_name} elements."
        )
    if _is_integer_source(source) or source.kind == "duration":
        for number in _iter_integers(values, source):
            if source.kind in ("int128", "uint128"):
                yield _exact.rata_die_ms_from_unix_wide_integer(
                    number, signed=source.kind == "int128"
                )
            elif source.dtype == np.dtype("<u8"):
                yield _exact.rata_die_ms_from_unix_uint64(number)
            else:
                yield _exact.rata_die_ms_from_unix_integer(number)
        return
    width = _element_width(source)
    for real, imag in _iter_reals(values, source):
        if imag != 0:
            raise _exact._failure(
                "InexactError", target.julia_name, complex(float(real), float(imag))
            )
        if width == 8:
            yield _exact.rata_die_ms_from_unix_seconds(float(real))
        else:
            yield _exact.rata_die_ms_from_unix_narrow(real, width)


def _datetime_values(
    values: np.ndarray[Any, Any], source: Target, target: Target
) -> np.ndarray[Any, Any]:
    """``datetime64[D]``/``[ms]`` of every element (an instant NumPy cannot hold raises)."""
    unit = target.dtype.str[-3:].strip("[]")
    counts = []
    for value in _iter_rata_die(values, source, target):
        count = (
            value // _exact.MS_PER_DAY - _exact.UNIX_EPOCH_MS // _exact.MS_PER_DAY
            if unit == "D"
            else value - _exact.UNIX_EPOCH_MS
        )
        if not MIN_INT64 < count <= MAX_INT64:
            raise ValueError(
                "An element is NumPy's NaT sentinel or outside its Int64 count; keep the "
                "stored container instead."
            )
        counts.append(count)
    return np.array(counts, dtype="<i8").astype(target.dtype)


def _sign_changed(source: Target, signed: bool) -> Target | None:
    """Return the same-width target of the other signedness, or None where there is none."""
    if source.kind == "int128":
        return None if signed else wide_target("uint128")
    if source.kind == "uint128":
        return wide_target("int128") if signed else None
    table = _SIGNED_DTYPES if signed else _UNSIGNED_DTYPES
    dtype = table.get(source.dtype)
    return None if dtype is None else numeric_target(dtype)


def abstract_route(token: str, base: Target) -> Target | None:  # noqa: PLR0911, PLR0912
    """Resolve an abstract token on a nonempty source: the identity (None) or a conversion target.

    ``TypeError`` names a route Julia lacks. Value checks (an integral float
    under ``Integer``, a zero imaginary part under ``Real``, a nonnegative
    value under ``Unsigned``) are the ordinary ones of the returned target.
    """
    is_date = base.kind in _DATE_KINDS
    is_float = _is_float_source(base)
    is_complex = base.kind == "complexf16" or (base.kind == "numeric" and base.dtype.kind == "c")
    if token in ("Any", "Number"):
        return None
    if token in UNION_TOKENS:
        if base.kind == "numeric" and base.dtype in (np.dtype("<i8"), np.dtype("<f8")):
            return None
        if base.kind == "numeric" and base.dtype == np.dtype("<c16"):
            return numeric_target(np.dtype("<f8"))
        raise TypeError(f"Julia has no {token} conversion from {base.julia_name} elements.")
    component = np.dtype(f"<f{_element_width(base)}") if is_complex else None
    if token == "Real":
        return numeric_target(component) if component is not None else None
    if token == "AbstractFloat":
        if component is not None:
            return numeric_target(component)
        return None if is_float else numeric_target(np.dtype("<f8"))
    if token == "Integer":
        if is_date or _is_integer_source(base):
            return None
        return numeric_target(np.dtype("<i8"))
    if token == "Signed":
        if is_date or base.kind == "int128" or (base.kind == "numeric" and base.dtype.kind == "i"):
            return None
        if is_float:
            return numeric_target(np.dtype("<i8"))
        return _sign_changed(base, signed=True)
    assert token == "Unsigned"
    if is_date:
        raise TypeError(f"Julia has no Unsigned conversion from {base.julia_name} elements.")
    if base.kind == "uint128" or (base.kind == "numeric" and base.dtype.kind == "u"):
        return None
    if is_float:
        return numeric_target(np.dtype("<u8"))
    return _sign_changed(base, signed=False)


def bare_rational_target(base: Target) -> Target:
    """``Rational`` without a parameter: the source's own integer type, else ``Int64``."""
    if base.kind in _DATE_KINDS:
        raise TypeError(f"Julia has no Rational conversion from {base.julia_name} elements.")
    if _is_integer_source(base):
        return extended_target("rational", base.julia_name)
    if _element_width(base) == 2:
        raise TypeError(f"Julia has no Rational conversion from {base.julia_name} elements.")
    return extended_target("rational", "Int64")


def bare_complex_target(base: Target) -> Target | None:
    """``Complex`` without a parameter: ``Complex{IntT}`` for integers, else ``ComplexF{W}``."""
    if base.kind in _DATE_KINDS:
        assert base.frequency is not None
        return dated_complex_target("MIT" if base.kind == "date" else "Duration", base.frequency)
    if _is_integer_source(base):
        return extended_target("intcomplex", base.julia_name)
    if base.kind == "complexf16" or base.dtype.kind == "c":
        return None
    return ACTIVE_TOKENS[{2: "ComplexF16", 4: "ComplexF32", 8: "ComplexF64"}[base.dtype.itemsize]]


def extended_token_target(  # noqa: PLR0911, PLR0912 - one branch per finite token family
    token: str, base: Target, *, plain: bool
) -> tuple[str, Target | None]:
    """Classify a token outside the finite element table on a nonempty source.

    Returns ``("identity", None)``, ``("element", target)`` for a conversion
    (the exact carriers and the dated complexes included), ``("object",
    target)`` for BigInt/BigFloat, ``("datetime", target)``, ``("symbol",
    target)`` and ``("char", target)`` for the plain-array routes, or
    ``("opaque", target)`` for text the tables do not know. ``TypeError``
    names a route Julia lacks.
    """
    token = resolve_spelling(token)
    dated = _dated_complex_token(token)
    if dated is not None:
        return "element", dated
    parameter = _exact.rational_parameter(token)
    if parameter is not None:
        return "element", extended_target("rational", parameter)
    parameter = _exact.integer_complex_parameter(token)
    if parameter is not None:
        return "element", extended_target("intcomplex", parameter)
    parameter = _exact.rational_complex_parameter(token)
    if parameter is not None:
        return "element", extended_target("rationalcomplex", parameter)
    if token == "Rational":
        return "element", bare_rational_target(base)
    if token == "Complex":
        target = bare_complex_target(base)
        return ("identity", None) if target is None else ("element", target)
    if token in ABSTRACT_TOKENS or token in UNION_TOKENS:
        target = abstract_route(token, base)
        return ("identity", None) if target is None else ("element", target)
    if token == "BigInt":
        return "object", Target("bigint", np.dtype(object))
    if token == "BigFloat":
        return "object", Target("bigfloat", np.dtype(object))
    if token in DATE_TOKENS:
        if not plain:
            raise TypeError(
                f"Julia has no {token} conversion for a dated series or MVTSeries (its elements "
                "must be numbers); plain arrays load it."
            )
        return "datetime", Target("datetime", np.dtype(f"<M8[{DATE_TOKENS[token]}]"), None, token)
    if token == "Symbol":
        if not plain:
            raise TypeError(
                "Julia has no Symbol conversion for a dated series or MVTSeries (its elements "
                "must be numbers); plain arrays load it."
            )
        return "symbol", Target("symbol", np.dtype(object))
    if token == "Char":
        if not plain:
            raise TypeError(
                "Julia has no Char conversion for a dated series or MVTSeries (its elements "
                "must be numbers); plain arrays load it."
            )
        return "char", Target("char", np.dtype(object))
    if token in EMPTY_ONLY_TOKENS:
        raise TypeError(
            f"Julia loads the element marker {token!r} from an empty payload only; this "
            "payload holds values."
        )
    return "opaque", opaque_target(token, base)
