"""Validation and owning conversion tests that do not require a native build."""

import numpy as np
import pytest

from tsecon import TSeries, mm, qq
from tsecon.dataecon._codec import MAX_BYTES, decode_series, encode_series, validate_metadata


@pytest.mark.parametrize(
    ("series", "exception"),
    [
        (np.ones(4), TypeError),
        (TSeries(qq(2024, 1), np.ones(4)), TypeError),
        (TSeries(mm(2024, 1), np.ones(4, dtype=np.float32)), TypeError),
        (TSeries(mm(2024, 1), np.ones(4, dtype=np.int64)), TypeError),
        (TSeries(mm(2024, 1), np.ones(4, dtype=">f8")), TypeError),
        (TSeries(mm(2024, 1), np.array([], dtype=np.float64)), ValueError),
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
        (2, 1, TypeError),
        (3, 32, TypeError),
        (4, 0, TypeError),
        (6, 67, TypeError),
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
    year, month, payload = encode_series(series)
    source[:] = -1
    result = decode_series(year, month, payload)
    assert result.firstdate == mm(2024, 1)
    assert result.values.flags.owndata
    np.testing.assert_array_equal(result.values, [0.0, 2.0, 4.0, 6.0])
    result.values[0] = 10
    assert np.frombuffer(payload, dtype=np.float64)[0] == 0
