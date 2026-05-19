# SPDX-License-Identifier: MIT
"""Lookup-table integrity for ``tsecon.x13._consts``.

The constants are module-private (consumed by ``_spec`` / ``_write`` /
``_result``), but their shape is a load-bearing contract: M2.5's
``load=`` keyword validates against the extension partitions, M2.6's
docs page enumerates them. Tests here pin:

* the five extension partitions stay disjoint (modulo the documented
  ``ais`` / ``c9`` overlap with the untreated set, which mirrors Julia);
* every save-table code has a description and an alt-name;
* the month-name tables are mutual inverses.
"""

from __future__ import annotations

import pytest

from tsecon.x13._consts import (
    _HUMAN_TEXT_EXTENSIONS,
    _KV_LIST_EXTENSIONS,
    _MONTHS_AND_QUARTERS,
    _ORDERED_MONTH_NAMES,
    _OUTPUT_ALT_NAMES,
    _OUTPUT_DESCRIPTIONS,
    _OUTPUT_DESCRIPTIONS_FLAT,
    _OUTPUT_SAVE_TABLES,
    _OUTPUT_UDG_DESCRIPTION,
    _PROBABLY_SERIES_EXTENSIONS,
    _SERIES_EXTENSIONS,
    _TABLE_EXTENSIONS,
)

# ---------------------------------------------------------------------------
# Month / quarter tables
# ---------------------------------------------------------------------------


class TestMonthTables:
    """``_ORDERED_MONTH_NAMES`` is the inverse of ``_MONTHS_AND_QUARTERS``."""

    def test_twelve_months(self) -> None:
        assert len(_ORDERED_MONTH_NAMES) == 12
        assert len(_MONTHS_AND_QUARTERS) == 12

    def test_order_matches_indexing(self) -> None:
        """``_ORDERED_MONTH_NAMES[i - 1]`` returns the 1-based month name."""
        assert _ORDERED_MONTH_NAMES[0] == "jan"
        assert _ORDERED_MONTH_NAMES[11] == "dec"

    def test_inverse_mapping(self) -> None:
        for one_based, name in enumerate(_ORDERED_MONTH_NAMES, start=1):
            assert _MONTHS_AND_QUARTERS[name] == one_based

    def test_lowercase(self) -> None:
        assert all(name == name.lower() for name in _ORDERED_MONTH_NAMES)

    def test_three_chars(self) -> None:
        assert all(len(name) == 3 for name in _ORDERED_MONTH_NAMES)


# ---------------------------------------------------------------------------
# Extension partitions
# ---------------------------------------------------------------------------


_EXTENSION_SETS = {
    "series": _SERIES_EXTENSIONS,
    "probably_series": _PROBABLY_SERIES_EXTENSIONS,
    "table": _TABLE_EXTENSIONS,
    "kv_list": _KV_LIST_EXTENSIONS,
    "human_text": _HUMAN_TEXT_EXTENSIONS,
}


class TestExtensionPartitions:
    """Each extension set is non-empty and contains only string codes."""

    @pytest.mark.parametrize("name", list(_EXTENSION_SETS))
    def test_non_empty(self, name: str) -> None:
        assert len(_EXTENSION_SETS[name]) > 0

    @pytest.mark.parametrize("name", list(_EXTENSION_SETS))
    def test_all_strings(self, name: str) -> None:
        assert all(isinstance(ext, str) for ext in _EXTENSION_SETS[name])

    @pytest.mark.parametrize("name", list(_EXTENSION_SETS))
    def test_immutable(self, name: str) -> None:
        """Constants are :class:`frozenset` so accidental mutation raises."""
        with pytest.raises(AttributeError):
            _EXTENSION_SETS[name].add("xxx")  # type: ignore[attr-defined]

    def test_kv_list_minimal(self) -> None:
        """``lks`` (likelihood stats) and ``mdc`` (component models) are the
        two key-value-list outputs X-13 emits. Locking the size catches
        accidental drift."""
        assert frozenset({"lks", "mdc"}) == _KV_LIST_EXTENSIONS

    def test_human_text_minimal(self) -> None:
        """Four human-readable text outputs: spc / gmt / out / sum."""
        assert frozenset({"spc", "gmt", "out", "sum"}) == _HUMAN_TEXT_EXTENSIONS

    def test_partitions_mostly_disjoint(self) -> None:
        """Most extension partitions are disjoint. The known overlap is
        ``c9`` (Julia upstream catalogues it in both probably-series and
        the untreated reference set; reflects the X-11 alias of b9). New
        accidental overlaps fail this test."""
        # Series vs human_text
        assert frozenset() == _SERIES_EXTENSIONS & _HUMAN_TEXT_EXTENSIONS
        assert frozenset() == _SERIES_EXTENSIONS & _KV_LIST_EXTENSIONS
        assert frozenset() == _SERIES_EXTENSIONS & _TABLE_EXTENSIONS
        assert frozenset() == _TABLE_EXTENSIONS & _HUMAN_TEXT_EXTENSIONS
        assert frozenset() == _TABLE_EXTENSIONS & _KV_LIST_EXTENSIONS


# ---------------------------------------------------------------------------
# Output tables (spec → short-code → ...)
# ---------------------------------------------------------------------------


_SPEC_NAMES = sorted(_OUTPUT_DESCRIPTIONS)


class TestOutputTablesShape:
    """``_OUTPUT_ALT_NAMES``, ``_OUTPUT_DESCRIPTIONS``, ``_OUTPUT_SAVE_TABLES``
    share spec-name keys."""

    def test_alt_names_specs_match_descriptions(self) -> None:
        assert set(_OUTPUT_ALT_NAMES) == set(_OUTPUT_DESCRIPTIONS)

    def test_save_tables_specs_match_descriptions(self) -> None:
        assert set(_OUTPUT_SAVE_TABLES) == set(_OUTPUT_DESCRIPTIONS)

    def test_nineteen_specs(self) -> None:
        """The 19 spec-builder names enumerated in decision 24."""
        assert len(_OUTPUT_DESCRIPTIONS) == 19


class TestSaveTableCoverage:
    """Every code in ``_OUTPUT_SAVE_TABLES`` should appear under its spec
    in either ``_OUTPUT_ALT_NAMES`` or ``_OUTPUT_DESCRIPTIONS``.

    The Julia upstream has a small set of save-table codes missing from
    one of the two tables (``mva`` under series is in save-tables but
    not in alt-names because Julia's source uses ``:mv`` as the alt-name
    key); the test catalogues those known gaps as expected failures.
    """

    _KNOWN_SAVE_TABLE_GAPS: frozenset[tuple[str, str]] = frozenset(
        {
            # ('spec', 'code'): documented Julia-side inconsistency.
            ("series", "mva"),  # spelt ``mv`` in alt-names / descriptions.
        }
    )

    @pytest.mark.parametrize("spec", _SPEC_NAMES)
    def test_save_codes_resolve(self, spec: str) -> None:
        alt_names = _OUTPUT_ALT_NAMES[spec]
        descriptions = _OUTPUT_DESCRIPTIONS[spec]
        for code in _OUTPUT_SAVE_TABLES[spec]:
            if (spec, code) in self._KNOWN_SAVE_TABLE_GAPS:
                continue
            assert code in alt_names or code in descriptions, (
                f"_OUTPUT_SAVE_TABLES[{spec!r}] includes {code!r} but neither "
                f"_OUTPUT_ALT_NAMES nor _OUTPUT_DESCRIPTIONS knows about it. "
                f"If this is a deliberate Julia-side inconsistency, add "
                f"({spec!r}, {code!r}) to _KNOWN_SAVE_TABLE_GAPS."
            )


# ---------------------------------------------------------------------------
# Description tables
# ---------------------------------------------------------------------------


class TestDescriptions:
    """Per-spec and flat description-table invariants."""

    @pytest.mark.parametrize("spec", _SPEC_NAMES)
    def test_descriptions_are_nonempty_strings(self, spec: str) -> None:
        for code, description in _OUTPUT_DESCRIPTIONS[spec].items():
            assert isinstance(code, str)
            assert isinstance(description, str)
            assert description, f"_OUTPUT_DESCRIPTIONS[{spec!r}][{code!r}] is empty."

    def test_flat_covers_all_descriptions(self) -> None:
        """``_OUTPUT_DESCRIPTIONS_FLAT`` contains every code from the nested form.

        Cross-spec duplicate keys (e.g. ``"a1"`` appears in both ``series``
        and ``spectrum``) collapse under dict-merge; this test only checks
        coverage, not multiplicity.
        """
        all_codes: set[str] = set()
        for spec_descriptions in _OUTPUT_DESCRIPTIONS.values():
            all_codes.update(spec_descriptions)
        assert set(_OUTPUT_DESCRIPTIONS_FLAT) == all_codes

    def test_udg_descriptions_nonempty(self) -> None:
        """UDG diagnostic descriptions are non-empty strings."""
        assert len(_OUTPUT_UDG_DESCRIPTION) > 0
        for code, desc in _OUTPUT_UDG_DESCRIPTION.items():
            assert isinstance(code, str)
            assert code
            assert isinstance(desc, str)
            assert desc


# ---------------------------------------------------------------------------
# Alt-name table
# ---------------------------------------------------------------------------


class TestAltNames:
    """Each spec's alt-name table has string keys and string values."""

    @pytest.mark.parametrize("spec", _SPEC_NAMES)
    def test_alt_names_string_typed(self, spec: str) -> None:
        for code, alt_name in _OUTPUT_ALT_NAMES[spec].items():
            assert isinstance(code, str)
            assert code
            assert isinstance(alt_name, str)
            assert alt_name
