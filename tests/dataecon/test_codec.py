"""Validation and owning conversion tests that do not require a native build."""

import numpy as np
import pytest

from tsecon import MIT, TSeries, mm
from tsecon.dataecon._codec import MAX_BYTES, decode_series, encode_series, validate_metadata
from tsecon.frequencies import Unit


@pytest.mark.parametrize(
    ("series", "exception"),
    [
        (np.ones(4), TypeError),
        (TSeries(MIT(Unit(), 4048), np.ones(4)), TypeError),
        (TSeries(mm(2024, 1), np.ones(4, dtype=object)), TypeError),
        (TSeries(mm(2024, 1), np.ones(4, dtype="S4")), TypeError),
        (TSeries(mm(2024, 1), np.ones(4, dtype=">f8")), TypeError),
        (TSeries(mm(2024, 1), np.array([], dtype=object)), TypeError),
        (TSeries(MIT(Unit(), 4048), np.array([], dtype=np.float64)), TypeError),
    ],
)
def test_reject_unsupported_input(series, exception):
    with pytest.raises(exception):
        encode_series(series)


@pytest.mark.parametrize(
    ("position", "value", "exception"),
    [
        (0, 1, TypeError),
        (1, 10, TypeError),
        (2, 6, TypeError),
        (3, 32, TypeError),
        (4, 0, TypeError),
        (6, 64, TypeError),
        (5, -1, ValueError),
        (5, 0, ValueError),
        (5, 2**62, ValueError),
        (8, -1, ValueError),
        (8, 31, ValueError),
        (8, MAX_BYTES + 8, ValueError),
        (7, -(2**31) - 1, ValueError),
        (7, 2**31 - 2, ValueError),
    ],
)
def test_validate_before_pointer_read(position, value, exception):
    metadata = [2, 12, 4, 0, 1, 4, 32, 24288, 32]
    metadata[position] = value
    with pytest.raises(exception):
        validate_metadata(tuple(metadata))


def test_strided_snapshot_and_owning_decode():
    source = np.arange(8, dtype=np.float64)
    series = TSeries(mm(2024, 1), source[::2])
    frequency, first, payload, element, element_frequency, length, marker = encode_series(series)
    assert (frequency, first) == (32, 24288)
    source[:] = -1
    result = decode_series(frequency, first, payload, element, element_frequency, length, marker)
    assert result.firstdate == mm(2024, 1)
    assert result.values.flags.owndata
    np.testing.assert_array_equal(result.values, [0.0, 2.0, 4.0, 6.0])
    result.values[0] = 10
    assert np.frombuffer(payload, dtype=np.float64)[0] == 0


@pytest.mark.parametrize("anchor", [mm(2024, 1), mm(2025, 7)])
def test_empty_codec_preserves_anchor_and_owns_values(anchor):
    source = TSeries(anchor, np.empty(0, dtype=np.float64))
    frequency, first, payload, element, element_frequency, length, marker = encode_series(source)
    assert payload == b""
    assert first == anchor.value
    result = decode_series(frequency, first, payload, element, element_frequency, length, marker)
    assert result.firstdate == anchor
    assert result.lastdate == anchor - 1
    assert result.values.shape == (0,)
    assert result.values.dtype == np.float64
    assert result.values.flags.owndata


@pytest.mark.parametrize("first", [-(2**31) - 1, 2**31])
def test_empty_anchor_bounds_are_checked_independently(first):
    with pytest.raises(ValueError, match="date range"):
        validate_metadata((2, 12, 4, 0, 1, 0, 32, first, 0))
