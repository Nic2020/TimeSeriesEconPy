"""The exact storage helpers: what they build, what they refuse and when they allocate.

``rational_storage``, ``integer_complex_storage`` and
``rational_complex_storage`` turn Python exact values into the Float64 or
ComplexF64 storage Julia reloads under the matching marker. They accept an
iterable, an object array of any shape or an exact carrier, and they check
the prospective payload against the limit from the captured shape (or while
consuming an iterable) before anything is flattened, copied or allocated.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import repeat

import numpy as np
import pytest

from tsecon.dataecon import (
    IntegerComplex,
    RationalComplex,
    StoredArray,
    StoredElement,
    StoredSeries,
    _represented,
    integer_complex_storage,
    rational_complex_storage,
    rational_storage,
)
from tsecon.frequencies import Monthly
from tsecon.mit import MIT

ANCHOR = MIT(Monthly(), 24288)
HELPERS = {
    "rational": (rational_storage, lambda: Fraction(1, 2), 8, "<f8"),
    "intcomplex": (integer_complex_storage, lambda: IntegerComplex(1, -2), 16, "<c16"),
    "rationalcomplex": (
        rational_complex_storage,
        lambda: RationalComplex(Fraction(1, 2), Fraction(-3, 4)),
        16,
        "<c16",
    ),
}


class _NoCopy(np.ndarray):
    """An object-array view whose whole-array copies fail: sizing must come first."""

    def copy(self, *args, **kwargs):
        raise AssertionError("copy was reached before the capacity check")

    def astype(self, *args, **kwargs):
        raise AssertionError("astype was reached before the capacity check")

    def reshape(self, *args, **kwargs):
        raise AssertionError("reshape was reached before the capacity check")


@pytest.fixture
def no_large_allocation(monkeypatch):
    """Fail any ``np.empty``/``np.zeros``/``np.array`` request beyond the patched limit."""
    real = {name: getattr(np, name) for name in ("empty", "zeros")}

    def spy(name):
        def wrapped(shape, dtype=float, *args, **kwargs):
            size = int(np.prod(shape)) if not isinstance(shape, int) else shape
            if size * np.dtype(dtype).itemsize > _represented.MAX_BYTES:
                raise AssertionError(f"np.{name}({shape!r}) was reached before validation")
            return real[name](shape, dtype, *args, **kwargs)

        return wrapped

    for name in real:
        monkeypatch.setattr(np, name, spy(name))


# ---- what the helpers build ---------------------------------------------------


def test_rational_storage_builds_the_float64_carrier_julia_reloads():
    values, element = rational_storage([Fraction(1, 2), Fraction(-3, 4), Fraction(5)])
    assert values.dtype == np.dtype("<f8")
    assert values.tolist() == [0.5, -0.75, 5.0]
    assert element == StoredElement.numeric("<f8", "Rational{Int64}")
    reloaded = StoredArray(values, element).to_interpreted()
    assert isinstance(reloaded, StoredArray)
    assert reloaded.element.kind == "rational"
    assert reloaded.tolist() == [Fraction(1, 2), Fraction(-3, 4), Fraction(5)]
    # An exact carrier goes back to storage unchanged, as does an object array of any shape.
    assert rational_storage(reloaded.values)[0].tolist() == values.tolist()
    grid = np.array([[Fraction(1, 2), Fraction(1)], [Fraction(-2), Fraction(3, 4)]], dtype=object)
    shaped, _ = rational_storage(grid)
    assert shaped.shape == (2, 2)
    assert shaped.flags.c_contiguous
    assert shaped.tolist() == [[0.5, 1.0], [-2.0, 0.75]]


def test_rational_storage_is_exact_by_default_and_lossy_only_on_request():
    with pytest.raises(ValueError, match="pass exact=False"):
        rational_storage([Fraction(1, 3)])
    lossy, element = rational_storage([Fraction(1, 3)], exact=False)
    assert lossy.tolist() == [1 / 3]
    assert element.marker == "Rational{Int64}"
    assert rational_storage([Fraction(1, 3)], parameter="Int8")[1].marker == "Rational{Int8}"
    with pytest.raises(TypeError, match="Unknown Rational parameter"):
        rational_storage([Fraction(1, 2)], parameter="Int7")
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        rational_storage([0.5])
    with pytest.raises(ValueError, match="Float64 storage range"):
        rational_storage([Fraction(10**400)])


def test_integer_and_rational_complex_storage():
    pairs, element = integer_complex_storage([IntegerComplex(2**54, 1), IntegerComplex(-3, 0)])
    assert pairs.dtype == np.dtype("<c16")
    assert pairs.tolist() == [complex(2**54, 1), complex(-3, 0)]
    assert element == StoredElement.numeric("<c16", "Complex{Int64}")
    reloaded = StoredArray(pairs, element).to_interpreted()
    assert reloaded.tolist() == [IntegerComplex(2**54, 1), IntegerComplex(-3, 0)]
    assert integer_complex_storage(reloaded.values)[0].tolist() == pairs.tolist()
    with pytest.raises(ValueError, match="InexactError rebuilding Int8"):
        integer_complex_storage([IntegerComplex(200, 0)], parameter="Int8")
    with pytest.raises(TypeError, match="IntegerComplex objects"):
        integer_complex_storage([1 + 2j])

    mixed, mixed_element = rational_complex_storage(
        [RationalComplex(Fraction(1, 2), Fraction(-3, 4))]
    )
    assert mixed.tolist() == [complex(0.5, -0.75)]
    assert mixed_element.marker == "Complex{Rational{Int64}}"
    reloaded = StoredArray(mixed, mixed_element).to_interpreted()
    assert reloaded.tolist() == [RationalComplex(Fraction(1, 2), Fraction(-3, 4))]
    assert rational_complex_storage(reloaded.values)[0].tolist() == mixed.tolist()
    with pytest.raises(ValueError, match="pass exact=False"):
        rational_complex_storage([RationalComplex(Fraction(1, 3), Fraction(0))])
    with pytest.raises(TypeError, match="RationalComplex objects"):
        rational_complex_storage([Fraction(1, 2)])


def test_dated_containers_take_the_storage_pairs():
    values, element = rational_storage([Fraction(1, 2), Fraction(3)])
    series = StoredSeries(ANCHOR, values, element)
    carrier = series.to_interpreted()
    assert isinstance(carrier, StoredSeries)
    assert carrier.tolist() == [Fraction(1, 2), Fraction(3)]
    with pytest.raises(TypeError, match="rational_storage"):
        carrier.element.written_marker(2)
    with pytest.raises(TypeError, match="rational_storage"):
        carrier.element.native_kind  # noqa: B018 - the property raises


# ---- capacity: checked from the captured shape before any copy or allocation ---


@pytest.mark.parametrize("family", list(HELPERS))
def test_small_limit_refuses_the_oversized_output_before_allocating(monkeypatch, family):
    helper, make, itemsize, dtype = HELPERS[family]
    monkeypatch.setattr(_represented, "MAX_BYTES", 2 * itemsize)
    fits, _ = helper([make(), make()])
    assert fits.dtype == np.dtype(dtype)
    assert len(fits) == 2
    with pytest.raises(ValueError, match="nothing was allocated"):
        helper([make(), make(), make()])
    # An object array is sized from its shape: the same three elements, any layout.
    with pytest.raises(ValueError, match=r"3 .*nothing was allocated"):
        helper(np.array([make(), make(), make()], dtype=object))
    with pytest.raises(ValueError, match=r"4 .*nothing was allocated"):
        helper(np.array([[make(), make()], [make(), make()]], dtype=object))
    monkeypatch.setattr(_represented, "MAX_BYTES", itemsize - 1)
    with pytest.raises(ValueError, match="nothing was allocated"):
        helper([make()])
    assert len(helper([])[0]) == 0


@pytest.mark.usefixtures("no_large_allocation")
@pytest.mark.parametrize("family", list(HELPERS))
def test_broadcast_inputs_are_sized_before_any_copy(monkeypatch, family):
    helper, make, itemsize, _ = HELPERS[family]
    monkeypatch.setattr(_represented, "MAX_BYTES", 4 * itemsize)
    # A fitting broadcast view is read through its flat iterator: no whole copy.
    fitting = np.broadcast_to(np.array([make()], dtype=object), (2, 2)).view(_NoCopy)
    values, _ = helper(fitting)
    assert values.shape == (2, 2)
    # An oversized broadcast (a single object behind millions of elements) is
    # refused from its shape; neither the object array nor the output is touched.
    oversized = np.broadcast_to(np.array([make()], dtype=object), (20_000_000,)).view(_NoCopy)
    with pytest.raises(ValueError, match=r"20000000 .*nothing was allocated"):
        helper(oversized)
    # An exact carrier is sized the same way before it is unpacked.
    carrier = StoredArray(*helper([make(), make()])).to_interpreted().values
    wide = np.broadcast_to(carrier, (3, 2)).view(_NoCopy)
    with pytest.raises(ValueError, match=r"6 .*nothing was allocated"):
        helper(wide)
    assert helper(np.broadcast_to(carrier, (2, 2)))[0].shape == (2, 2)


@pytest.mark.parametrize("family", list(HELPERS))
def test_an_unbounded_iterable_is_cut_off_at_the_limit(monkeypatch, family):
    helper, make, itemsize, _ = HELPERS[family]
    monkeypatch.setattr(_represented, "MAX_BYTES", 3 * itemsize)
    consumed = 0

    def counting():
        nonlocal consumed
        for item in repeat(make()):
            consumed += 1
            yield item

    with pytest.raises(ValueError, match=r"more than 3 .*not consumed further"):
        helper(counting())
    assert consumed == 4


def test_the_real_limit_is_the_payload_limit():
    # 16,777,216 Float64 values fill the 128 MiB limit exactly; one more is refused
    # from the shape alone (the broadcast never materialises).
    limit = _represented.MAX_BYTES // 8
    with pytest.raises(ValueError, match=f"{limit + 1} Fraction values"):
        rational_storage(np.broadcast_to(np.array([Fraction(1)], dtype=object), (limit + 1,)))
    limit = _represented.MAX_BYTES // 16
    with pytest.raises(ValueError, match=f"{limit + 1} IntegerComplex values"):
        integer_complex_storage(
            np.broadcast_to(np.array([IntegerComplex(1, 1)], dtype=object), (limit + 1,))
        )
