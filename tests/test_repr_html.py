# SPDX-License-Identifier: MIT
"""Snapshot tests for the ``_repr_html_`` methods on :class:`~tsecon.TSeries`,
:class:`~tsecon.MVTSeries`, and :class:`~tsecon.Workspace`.

The Python ``_repr_html_`` protocol is how Jupyter / VS Code interactive /
MS Fabric / Databricks notebooks render objects: returning HTML routes
the object through the host's rich display instead of the plain-text
``__repr__``. The Julia upstream's parallel is
``Base.show(io, ::MIME"text/html", ::TSeries)`` (and the MVTSeries /
Workspace analogues) — used by IJulia for the same purpose.

Each test locks the HTML output as a **literal string** (not against an
external golden file) so any future drift surfaces as an in-test diff
the reviewer can see in one place. Snapshot rendering rules:

* TSeries — ``head_8 / ⋮ / tail_8`` truncation once ``n > 20``.
* MVTSeries — row truncation per the ``_DEFAULT_DISPLAY_HEIGHT - 6`` rule
  shared with the text repr; no column truncation (notebooks scroll).
* Workspace — no truncation; one ``<tr>`` per member.

See [decision 26 — _repr_html_ strategy] for the rationale behind the
chosen semantics + plain-``<table>`` (no CSS class / inline style) shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsecon import MVTSeries, TSeries, Workspace, qq

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canonical_tseries() -> TSeries:
    """4-element Quarterly TSeries starting 2020Q1."""
    return TSeries(qq(2020, 1), np.array([100.0, 101.2, 102.3, 103.5]))


@pytest.fixture
def canonical_mvtseries() -> MVTSeries:
    """3×2 Quarterly MVTSeries with columns gdp / cpi starting 2020Q1."""
    return MVTSeries(qq(2020, 1), ["gdp", "cpi"], np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


@pytest.fixture
def canonical_workspace() -> Workspace:
    """3-entry Workspace mixing scalar / str / TSeries values."""
    return Workspace(
        alpha=1,
        beta="hello",
        t=TSeries(qq(2020, 1), np.array([1.0, 2.0])),
    )


# ---------------------------------------------------------------------------
# TSeries
# ---------------------------------------------------------------------------


class TestTSeriesReprHtml:
    def test_canonical_4q(self, canonical_tseries: TSeries) -> None:
        expected = (
            "<table>\n"
            '<thead><tr><th colspan="2">4-element TSeries{Quarterly} with range 2020Q1:2020Q4</th></tr></thead>\n'  # noqa: E501
            "<tbody>\n"
            "<tr><th>2020Q1</th><td>100.0</td></tr>\n"
            "<tr><th>2020Q2</th><td>101.2</td></tr>\n"
            "<tr><th>2020Q3</th><td>102.3</td></tr>\n"
            "<tr><th>2020Q4</th><td>103.5</td></tr>\n"
            "</tbody>\n"
            "</table>"
        )
        assert canonical_tseries._repr_html_() == expected

    def test_empty(self) -> None:
        t = TSeries(qq(2020, 1), np.array([], dtype=np.float64))
        expected = (
            "<table>\n"
            "<thead><tr><th>Empty TSeries{Quarterly} starting 2020Q1</th></tr></thead>\n"
            "</table>"
        )
        assert t._repr_html_() == expected

    def test_long_series_truncates_to_head_8_dots_tail_8(self) -> None:
        """``n=24`` should render 8 head rows, a ⋮ separator, 8 tail rows."""
        t = TSeries(qq(2020, 1), np.arange(24, dtype=np.float64) + 100.0)
        expected = (
            "<table>\n"
            '<thead><tr><th colspan="2">24-element TSeries{Quarterly} with range 2020Q1:2025Q4</th></tr></thead>\n'  # noqa: E501
            "<tbody>\n"
            "<tr><th>2020Q1</th><td>100.0</td></tr>\n"
            "<tr><th>2020Q2</th><td>101.0</td></tr>\n"
            "<tr><th>2020Q3</th><td>102.0</td></tr>\n"
            "<tr><th>2020Q4</th><td>103.0</td></tr>\n"
            "<tr><th>2021Q1</th><td>104.0</td></tr>\n"
            "<tr><th>2021Q2</th><td>105.0</td></tr>\n"
            "<tr><th>2021Q3</th><td>106.0</td></tr>\n"
            "<tr><th>2021Q4</th><td>107.0</td></tr>\n"
            '<tr><td colspan="2">⋮</td></tr>\n'
            "<tr><th>2024Q1</th><td>116.0</td></tr>\n"
            "<tr><th>2024Q2</th><td>117.0</td></tr>\n"
            "<tr><th>2024Q3</th><td>118.0</td></tr>\n"
            "<tr><th>2024Q4</th><td>119.0</td></tr>\n"
            "<tr><th>2025Q1</th><td>120.0</td></tr>\n"
            "<tr><th>2025Q2</th><td>121.0</td></tr>\n"
            "<tr><th>2025Q3</th><td>122.0</td></tr>\n"
            "<tr><th>2025Q4</th><td>123.0</td></tr>\n"
            "</tbody>\n"
            "</table>"
        )
        assert t._repr_html_() == expected

    def test_truncation_threshold_inclusive_at_20(self) -> None:
        """``n=20`` is the largest size that renders unabbreviated."""
        t = TSeries(qq(2020, 1), np.arange(20, dtype=np.float64))
        html_out = t._repr_html_()
        assert "⋮" not in html_out
        # 20 data rows + thead row + table tags + tbody tags.
        assert html_out.count("<tr>") == 21

    def test_truncation_fires_at_21(self) -> None:
        """``n=21`` is the smallest size that truncates."""
        t = TSeries(qq(2020, 1), np.arange(21, dtype=np.float64))
        html_out = t._repr_html_()
        assert '<tr><td colspan="2">⋮</td></tr>' in html_out
        # head 8 + ellipsis + tail 8 + thead = 18 rows.
        assert html_out.count("<tr>") == 18

    def test_returned_object_is_str(self, canonical_tseries: TSeries) -> None:
        assert isinstance(canonical_tseries._repr_html_(), str)


# ---------------------------------------------------------------------------
# MVTSeries
# ---------------------------------------------------------------------------


class TestMVTSeriesReprHtml:
    def test_canonical_3x2(self, canonical_mvtseries: MVTSeries) -> None:
        expected = (
            "<table>\n"
            '<thead><tr><th colspan="3">3×2 MVTSeries{Quarterly} with range 2020Q1:2020Q3 and variables (gdp,cpi)</th></tr>\n'  # noqa: E501
            "<tr><th></th><th>gdp</th><th>cpi</th></tr></thead>\n"
            "<tbody>\n"
            "<tr><th>2020Q1</th><td>1</td><td>2</td></tr>\n"
            "<tr><th>2020Q2</th><td>3</td><td>4</td></tr>\n"
            "<tr><th>2020Q3</th><td>5</td><td>6</td></tr>\n"
            "</tbody>\n"
            "</table>"
        )
        assert canonical_mvtseries._repr_html_() == expected

    def test_empty(self) -> None:
        m = MVTSeries(qq(2020, 1), [], np.zeros((0, 0)))
        expected = (
            "<table>\n"
            '<thead><tr><th colspan="1">0×0 MVTSeries{Quarterly} with range 2020Q1:2019Q4 and no variables</th></tr></thead>\n'  # noqa: E501
            "</table>"
        )
        assert m._repr_html_() == expected

    def test_long_series_truncates_with_vdots(self) -> None:
        """``nrows=25 > _DEFAULT_DISPLAY_HEIGHT-6=18`` triggers vertical truncation."""
        data = np.arange(25 * 2, dtype=np.float64).reshape(25, 2)
        m = MVTSeries(qq(2020, 1), ["a", "b"], data)
        html_out = m._repr_html_()
        assert '<tr><td colspan="3">⋮</td></tr>' in html_out
        # 9 top rows + ⋮ + 9 bot rows + 2 thead rows = 21 tr starts.
        assert html_out.count("<tr>") == 21

    def test_column_name_truncation(self) -> None:
        """Column names ≥ 10 chars truncate to 10 + ellipsis (mirrors text repr)."""
        m = MVTSeries(
            qq(2020, 1),
            ["short", "very_long_column_name"],
            np.array([[1.0, 2.0]]),
        )
        html_out = m._repr_html_()
        assert "<th>short</th>" in html_out
        # First 10 chars + … (the `_NAME_TRUNCATE_AT` rule).
        assert "<th>very_long_…</th>" in html_out
        assert "very_long_column_name" not in html_out.replace("<th>very_long_column_name</th>", "")

    def test_html_escapes_special_chars_in_column_names(self) -> None:
        m = MVTSeries(qq(2020, 1), ["x<y", "a&b"], np.array([[1.0, 2.0]]))
        html_out = m._repr_html_()
        assert "<th>x&lt;y</th>" in html_out
        assert "<th>a&amp;b</th>" in html_out


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class TestWorkspaceReprHtml:
    def test_canonical_3_member(self, canonical_workspace: Workspace) -> None:
        expected = (
            "<table>\n"
            '<thead><tr><th colspan="2">Workspace with 3 variables</th></tr></thead>\n'
            "<tbody>\n"
            "<tr><th>alpha</th><td>1</td></tr>\n"
            "<tr><th>beta</th><td>&#x27;hello&#x27;</td></tr>\n"
            "<tr><th>t</th><td>2-element TSeries{Quarterly}</td></tr>\n"
            "</tbody>\n"
            "</table>"
        )
        assert canonical_workspace._repr_html_() == expected

    def test_empty(self) -> None:
        w = Workspace()
        expected = "<table>\n<thead><tr><th>Empty Workspace</th></tr></thead>\n</table>"
        assert w._repr_html_() == expected

    def test_single_member_uses_singular(self) -> None:
        """Header reads ``Workspace with 1 variable`` (no plural ``s``)."""
        w = Workspace(only=42)
        expected = (
            "<table>\n"
            '<thead><tr><th colspan="2">Workspace with 1 variable</th></tr></thead>\n'
            "<tbody>\n"
            "<tr><th>only</th><td>42</td></tr>\n"
            "</tbody>\n"
            "</table>"
        )
        assert w._repr_html_() == expected

    def test_html_escapes_special_chars_in_keys(self) -> None:
        # Workspace stores ASCII identifier keys by default; this test
        # checks the escape pipeline still fires for the value column,
        # which is the realistic injection surface.
        w = Workspace(plain_key="value with <tag> & ampersand")
        html_out = w._repr_html_()
        assert "&lt;tag&gt;" in html_out
        assert "&amp;" in html_out
