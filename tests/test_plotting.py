# SPDX-License-Identifier: MIT
"""Tests for ``tsecon.plotting``.

Both backends are installed in the dev environment (the package's
``[matplotlib]`` and ``[plotly]`` extras are pulled in via the dev group),
so we exercise each in turn. The matplotlib backend is forced to the
``Agg`` non-interactive backend at module load.

The Julia ``plotrecipes.jl`` recipes are exercised in the Plots.jl test
suite; here we mirror their *behavioural* contract — frequency uniformity,
range intersection, mit_loc within-period offsets, panel grid, multi-
dataset overlay, missing-variable graceful fall-through, and dispatcher
arity — rather than the underlying recipe machinery (which is Julia
specific).
"""

from __future__ import annotations

import datetime as _dt

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.ticker import FuncFormatter

import tsecon
import tsecon.plotting as plotting_pkg
from tsecon import MITRange, MVTSeries, TSeries, mitrange, mm, qq
from tsecon.frequencies import (
    BDaily,
    Daily,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
)
from tsecon.mit import MIT, daily
from tsecon.plotting import (
    BackendNotAvailableError,
    available_backends,
    plot,
    resolve_backend,
)
from tsecon.plotting._common import (
    build_xy,
    check_uniform_frequency,
    classify_inputs,
    intersect_ranges,
    mit_formatter,
    mit_offset,
    normalize_label,
    normalize_vars,
    panel_grid,
    xaxis_kind,
    yp_tick_positions,
)


@pytest.fixture(autouse=True)
def _close_figs() -> None:
    """Close every matplotlib figure between tests to keep memory bounded."""
    yield
    plt.close("all")


# ===========================================================================
# Shared helpers (_common.py)
# ===========================================================================


class TestMitOffset:
    def test_left_is_zero_for_every_frequency(self) -> None:
        for f in [Quarterly(), Monthly(), Yearly(), HalfYearly(), Weekly(), Daily(), BDaily()]:
            assert mit_offset("left", f) == 0.0

    def test_middle_for_yp_is_half_over_n(self) -> None:
        assert mit_offset("middle", Yearly()) == 0.5
        assert mit_offset("middle", HalfYearly()) == 0.25
        assert mit_offset("middle", Quarterly()) == 0.125
        assert mit_offset("middle", Monthly()) == pytest.approx(0.5 / 12)

    def test_right_for_yp_is_one_over_n(self) -> None:
        assert mit_offset("right", Yearly()) == 1.0
        assert mit_offset("right", Quarterly()) == 0.25
        assert mit_offset("right", Monthly()) == pytest.approx(1.0 / 12)

    def test_middle_for_non_yp_is_half(self) -> None:
        for f in [Daily(), BDaily(), Weekly()]:
            assert mit_offset("middle", f) == 0.5

    def test_right_for_non_yp_is_one(self) -> None:
        for f in [Daily(), BDaily(), Weekly()]:
            assert mit_offset("right", f) == 1.0

    def test_invalid_mit_loc_raises(self) -> None:
        with pytest.raises(ValueError, match="mit_loc"):
            mit_offset("center", Quarterly())  # type: ignore[arg-type]


class TestMitFormatter:
    def test_yp_at_grid_returns_mit_string(self) -> None:
        f = mit_formatter("left", Quarterly())
        assert f(2020.0) == "2020Q1"
        assert f(2020.25) == "2020Q2"
        assert f(2020.5) == "2020Q3"
        assert f(2020.75) == "2020Q4"
        assert f(2021.0) == "2021Q1"

    def test_yp_offgrid_returns_suffix_and_warns(self) -> None:
        f = mit_formatter("left", Quarterly())
        with pytest.warns(UserWarning, match="not aligned"):
            label = f(2020.1)
        assert label.endswith("+")
        # Second off-grid value reuses the closure's warned flag (one-shot).
        # We can't assert "no warning" here without filterwarnings, but the
        # closure should still return the suffix.
        assert f(2020.4).endswith("+")

    def test_yp_yearly_formatter(self) -> None:
        f = mit_formatter("left", Yearly())
        assert f(2020.0) == "2020Y"
        assert f(2021.0) == "2021Y"

    def test_non_yp_raises(self) -> None:
        with pytest.raises(TypeError, match="YP-only"):
            mit_formatter("left", Daily())

    def test_middle_offset_centres_format(self) -> None:
        # With mit_loc=middle on Quarterly, 2020.125 is the middle of 2020Q1,
        # so the formatter should round to 2020Q1.
        f = mit_formatter("middle", Quarterly())
        assert f(2020.125) == "2020Q1"
        assert f(2020.375) == "2020Q2"


class TestYpTickPositions:
    def test_quarterly_three_years(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2022, 4))
        vals, txt = yp_tick_positions(rng, "left")
        # With 12 quarters and target_n=8, stride=4 (one year) gives 3 ticks.
        assert vals == [2020.0, 2021.0, 2022.0]
        assert txt == ["2020Q1", "2021Q1", "2022Q1"]

    def test_quarterly_two_years_uses_per_period_stride(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2021, 4))
        vals, _txt = yp_tick_positions(rng, "left")
        # 8 quarters fits under target_n=8 with stride 1; gets 8 ticks.
        assert len(vals) == 8

    def test_empty_range(self) -> None:
        empty = MITRange(qq(2020, 4), qq(2020, 1))  # start > stop
        assert yp_tick_positions(empty, "left") == ([], [])

    def test_middle_offset_shifts_tickvals(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2022, 4))
        vals_left, _ = yp_tick_positions(rng, "left")
        vals_mid, _ = yp_tick_positions(rng, "middle")
        diffs = [m - l_ for m, l_ in zip(vals_mid, vals_left, strict=True)]
        assert all(d == pytest.approx(0.125) for d in diffs)

    def test_non_yp_raises(self) -> None:
        rng = mitrange(daily(_dt.date(2024, 1, 1)), daily(_dt.date(2024, 1, 10)))
        with pytest.raises(TypeError):
            yp_tick_positions(rng, "left")


class TestIntersectRanges:
    def test_overlap(self) -> None:
        a = mitrange(qq(2020, 1), qq(2022, 4))
        b = mitrange(qq(2021, 1), qq(2023, 4))
        c = intersect_ranges(a, b)
        assert c == mitrange(qq(2021, 1), qq(2022, 4))

    def test_disjoint_yields_empty(self) -> None:
        a = mitrange(qq(2020, 1), qq(2020, 4))
        b = mitrange(qq(2021, 1), qq(2021, 4))
        c = intersect_ranges(a, b)
        assert len(c) == 0

    def test_three_args(self) -> None:
        a = mitrange(qq(2020, 1), qq(2025, 4))
        b = mitrange(qq(2021, 1), qq(2024, 4))
        c = mitrange(qq(2022, 1), qq(2023, 4))
        assert intersect_ranges(a, b, c) == mitrange(qq(2022, 1), qq(2023, 4))

    def test_mixed_frequencies_raise(self) -> None:
        a = mitrange(qq(2020, 1), qq(2020, 4))
        b = mitrange(mm(2020, 1), mm(2020, 12))
        with pytest.raises(TypeError, match="frequencies"):
            intersect_ranges(a, b)


class TestXAxisKind:
    def test_yp(self) -> None:
        for f in [Yearly(), HalfYearly(), Quarterly(), Monthly()]:
            assert xaxis_kind(f) == "yp"

    def test_date(self) -> None:
        for f in [Daily(), BDaily(), Weekly()]:
            assert xaxis_kind(f) == "date"

    def test_numeric(self) -> None:
        assert xaxis_kind(Unit()) == "numeric"


class TestClassifyInputs:
    def test_all_tseries(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
        assert classify_inputs([t, t]) == "tseries"

    def test_all_mvtseries(self) -> None:
        m = MVTSeries(qq(2020, 1), ["a"], np.ones((3, 1)))
        assert classify_inputs([m, m]) == "mvtseries"

    def test_empty_raises(self) -> None:
        with pytest.raises(TypeError):
            classify_inputs([])

    def test_mixed_raises(self) -> None:
        t = TSeries(qq(2020, 1), [1.0])
        m = MVTSeries(qq(2020, 1), ["a"], np.ones((1, 1)))
        with pytest.raises(TypeError, match="mix"):
            classify_inputs([t, m])


class TestNormalizeLabel:
    def test_none_broadcasts_none(self) -> None:
        assert normalize_label(None, 3) == [None, None, None]

    def test_str_broadcasts(self) -> None:
        assert normalize_label("a", 3) == ["a", "a", "a"]

    def test_sequence_passthrough(self) -> None:
        assert normalize_label(["a", "b", "c"], 3) == ["a", "b", "c"]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Number of labels"):
            normalize_label(["a", "b"], 3)

    def test_non_string_in_seq_raises(self) -> None:
        with pytest.raises(TypeError):
            normalize_label([1, 2, 3], 3)  # type: ignore[list-item]


class TestNormalizeVars:
    def _datasets(self) -> list[MVTSeries]:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        a = MVTSeries(rng, ["x", "y"], np.ones((4, 2)))
        b = MVTSeries(rng, ["y", "z"], np.ones((4, 2)))
        return [a, b]

    def test_union_default_preserves_first_seen_order(self) -> None:
        pairs = normalize_vars(None, self._datasets())
        assert [name for name, _ in pairs] == ["x", "y", "z"]
        assert [title for _, title in pairs] == ["x", "y", "z"]

    def test_explicit_strings(self) -> None:
        pairs = normalize_vars(["y", "x"], self._datasets())
        assert pairs == [("y", "y"), ("x", "x")]

    def test_pair_form_supplies_title(self) -> None:
        pairs = normalize_vars([("y", "Yield"), ("x", "Excess")], self._datasets())
        assert pairs == [("y", "Yield"), ("x", "Excess")]

    def test_cap_at_ten(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        names = [f"v{i}" for i in range(11)]
        m = MVTSeries(rng, names, np.ones((4, 11)))
        with pytest.raises(ValueError, match="Too many"):
            normalize_vars(None, [m])


class TestPanelGrid:
    @pytest.mark.parametrize(
        ("nvars", "expected"),
        [
            (1, (1, 1)),
            (2, (1, 2)),
            (3, (1, 3)),
            (4, (2, 2)),
            (5, (2, 3)),
            (6, (2, 3)),
            (7, (3, 3)),
            (8, (3, 3)),
            (9, (3, 3)),
            (10, (5, 2)),
        ],
    )
    def test_shapes(self, nvars: int, expected: tuple[int, int]) -> None:
        assert panel_grid(nvars) == expected

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            panel_grid(0)


class TestBuildXy:
    def test_yp_x_uses_floats_plus_offset(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        t = TSeries(rng, [1.0, 2.0, 3.0, 4.0])
        x, y, kind = build_xy(t, trange=None, mit_loc="left")
        assert kind == "yp"
        np.testing.assert_array_equal(x, np.array([2020.0, 2020.25, 2020.5, 2020.75]))
        np.testing.assert_array_equal(y, [1.0, 2.0, 3.0, 4.0])

    def test_yp_middle_offset(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        t = TSeries(rng, [1.0, 2.0, 3.0, 4.0])
        x, _, _ = build_xy(t, trange=None, mit_loc="middle")
        np.testing.assert_allclose(x, np.array([2020.0, 2020.25, 2020.5, 2020.75]) + 0.125)

    def test_yp_right_offset(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        t = TSeries(rng, [1.0, 2.0, 3.0, 4.0])
        x, _, _ = build_xy(t, trange=None, mit_loc="right")
        np.testing.assert_allclose(x, np.array([2020.0, 2020.25, 2020.5, 2020.75]) + 0.25)

    def test_daily_x_is_date_objects(self) -> None:
        rng = mitrange(daily(_dt.date(2024, 1, 1)), daily(_dt.date(2024, 1, 5)))
        t = TSeries(rng, [1.0, 2.0, 3.0, 4.0, 5.0])
        x, y, kind = build_xy(t, trange=None, mit_loc="left")
        assert kind == "date"
        assert x[0] == _dt.date(2024, 1, 1)
        assert x[-1] == _dt.date(2024, 1, 5)
        assert y.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_trange_intersection(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2022, 4))
        t = TSeries(rng, np.arange(len(rng), dtype=float))
        x, y, _ = build_xy(t, trange=mitrange(qq(2021, 1), qq(2021, 4)), mit_loc="left")
        np.testing.assert_array_equal(x, [2021.0, 2021.25, 2021.5, 2021.75])
        np.testing.assert_array_equal(y, [4.0, 5.0, 6.0, 7.0])

    def test_trange_disjoint_returns_empty(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        t = TSeries(rng, [1.0, 2.0, 3.0, 4.0])
        x, y, _ = build_xy(t, trange=mitrange(qq(2030, 1), qq(2030, 4)), mit_loc="left")
        assert len(x) == 0
        assert len(y) == 0

    def test_unit_frequency_uses_numeric(self) -> None:
        rng = MITRange(MIT(Unit(), 0), MIT(Unit(), 3))
        t = TSeries(rng, [1.0, 2.0, 3.0, 4.0])
        x, _, kind = build_xy(t, trange=None, mit_loc="middle")
        assert kind == "numeric"
        np.testing.assert_array_equal(x, [0.5, 1.5, 2.5, 3.5])


class TestCheckUniformFrequency:
    def test_returns_shared_frequency(self) -> None:
        t1 = TSeries(qq(2020, 1), [1.0, 2.0])
        t2 = TSeries(qq(2021, 1), [3.0, 4.0])
        assert check_uniform_frequency([t1, t2]) is Quarterly()

    def test_mismatched_raises(self) -> None:
        t1 = TSeries(qq(2020, 1), [1.0, 2.0])
        t2 = TSeries(mm(2020, 1), [3.0, 4.0])
        with pytest.raises(TypeError, match="frequency"):
            check_uniform_frequency([t1, t2])


# ===========================================================================
# Backend resolution
# ===========================================================================


class TestResolveBackend:
    def test_auto_returns_first_available(self) -> None:
        assert resolve_backend("auto") in ("matplotlib", "plotly")

    def test_matplotlib_resolves_to_matplotlib(self) -> None:
        assert resolve_backend("matplotlib") == "matplotlib"

    def test_plotly_resolves_to_plotly(self) -> None:
        assert resolve_backend("plotly") == "plotly"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("seaborn")  # type: ignore[arg-type]

    def test_missing_backend_raises_with_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate matplotlib being uninstalled by stubbing _is_installed.
        def fake_installed(name: str) -> bool:
            return name != "matplotlib"

        monkeypatch.setattr(plotting_pkg, "_is_installed", fake_installed)
        with pytest.raises(BackendNotAvailableError, match="not installed"):
            resolve_backend("matplotlib")

    def test_auto_with_no_backends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(plotting_pkg, "_is_installed", lambda _name: False)
        with pytest.raises(BackendNotAvailableError, match="No plotting backend"):
            resolve_backend("auto")


class TestAvailableBackends:
    def test_returns_in_preference_order(self) -> None:
        backends = available_backends()
        # matplotlib is preferred over plotly when both are installed.
        if "matplotlib" in backends and "plotly" in backends:
            assert backends.index("matplotlib") < backends.index("plotly")


# ===========================================================================
# Matplotlib backend
# ===========================================================================


class TestMatplotlibTSeries:
    def test_single_yp_returns_figure(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        fig = plot(t, backend="matplotlib")
        assert fig.__class__.__name__ == "Figure"
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        line = ax.lines[0]
        np.testing.assert_array_equal(line.get_ydata(), [1.0, 2.0, 3.0, 4.0])

    def test_many_tseries_overlay(self) -> None:
        t1 = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        t2 = TSeries(qq(2020, 1), [4.0, 3.0, 2.0, 1.0])
        fig = plot(t1, t2, backend="matplotlib", label=["a", "b"])
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        assert [line.get_label() for line in ax.lines] == ["a", "b"]
        # Legend was created because labels are non-None.
        assert ax.get_legend() is not None

    def test_yp_axis_has_funcformatter(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        fig = plot(t, backend="matplotlib")
        assert isinstance(fig.axes[0].xaxis.get_major_formatter(), FuncFormatter)

    def test_daily_axis_is_date(self) -> None:
        rng = mitrange(daily(_dt.date(2024, 1, 1)), daily(_dt.date(2024, 1, 10)))
        t = TSeries(rng, np.arange(len(rng), dtype=float))
        fig = plot(t, backend="matplotlib")
        ax = fig.axes[0]
        # Date axis: tick locator is the default (no FuncFormatter applied).
        assert not isinstance(ax.xaxis.get_major_formatter(), FuncFormatter)

    def test_trange_limits_data(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2022, 4))
        t = TSeries(rng, np.arange(len(rng), dtype=float))
        fig = plot(t, backend="matplotlib", trange=mitrange(qq(2021, 1), qq(2021, 4)))
        ydata = fig.axes[0].lines[0].get_ydata()
        np.testing.assert_array_equal(ydata, [4.0, 5.0, 6.0, 7.0])

    def test_mit_loc_middle_shifts_x(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        fig_left = plot(t, backend="matplotlib", mit_loc="left")
        fig_mid = plot(t, backend="matplotlib", mit_loc="middle")
        diffs = fig_mid.axes[0].lines[0].get_xdata() - fig_left.axes[0].lines[0].get_xdata()
        np.testing.assert_allclose(diffs, [0.125, 0.125, 0.125, 0.125])

    def test_title_and_labels_set(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0])
        fig = plot(
            t,
            backend="matplotlib",
            title="My title",
            xlabel="time",
            ylabel="value",
        )
        ax = fig.axes[0]
        assert ax.get_title() == "My title"
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "value"

    def test_legend_disabled(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0])
        fig = plot(t, backend="matplotlib", label="a", legend=False)
        assert fig.axes[0].get_legend() is None

    def test_mixed_frequencies_raise(self) -> None:
        t1 = TSeries(qq(2020, 1), [1.0])
        t2 = TSeries(mm(2020, 1), [1.0])
        with pytest.raises(TypeError, match="frequency"):
            plot(t1, t2, backend="matplotlib")

    def test_empty_after_trange_is_empty_axes(self) -> None:
        t = TSeries(qq(2020, 1), [1.0])
        fig = plot(t, backend="matplotlib", trange=mitrange(qq(2030, 1), qq(2030, 4)))
        assert len(fig.axes[0].lines) == 0

    def test_passthrough_kwargs_to_plot(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
        fig = plot(t, backend="matplotlib", linewidth=4, linestyle="--")
        line = fig.axes[0].lines[0]
        assert line.get_linewidth() == 4
        assert line.get_linestyle() == "--"


class TestMatplotlibMVTSeriesPanel:
    def test_single_dataset_default_layout(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        mvts = MVTSeries(rng, ["x", "y", "z"], np.arange(12, dtype=float).reshape(4, 3))
        fig = plot(mvts, backend="matplotlib")
        assert len(fig.axes) == 3
        assert [ax.get_title() for ax in fig.axes] == ["x", "y", "z"]

    def test_two_datasets_share_panels(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        a = MVTSeries(rng, ["x", "y"], np.arange(8, dtype=float).reshape(4, 2))
        b = MVTSeries(rng, ["x", "y"], np.arange(8, 16, dtype=float).reshape(4, 2))
        fig = plot(a, b, backend="matplotlib", label=["a", "b"])
        assert len(fig.axes) == 2
        for ax in fig.axes:
            assert len(ax.lines) == 2  # one per dataset
            assert [line.get_label() for line in ax.lines] == ["a", "b"]

    def test_missing_variable_skipped_per_dataset(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        a = MVTSeries(rng, ["x", "y"], np.ones((4, 2)))
        b = MVTSeries(rng, ["y", "z"], np.ones((4, 2)))
        fig = plot(a, b, backend="matplotlib")
        # Three subplots (x, y, z), one trace where the dataset has the var.
        assert len(fig.axes) == 3
        titles = [ax.get_title() for ax in fig.axes]
        ax_by_var = dict(zip(titles, fig.axes, strict=True))
        assert len(ax_by_var["x"].lines) == 1  # only a
        assert len(ax_by_var["y"].lines) == 2  # both
        assert len(ax_by_var["z"].lines) == 1  # only b

    def test_vars_kwarg_selects_subset_and_order(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        mvts = MVTSeries(rng, ["x", "y", "z"], np.arange(12, dtype=float).reshape(4, 3))
        fig = plot(mvts, backend="matplotlib", vars=["z", "x"])
        assert [ax.get_title() for ax in fig.axes] == ["z", "x"]

    def test_vars_pair_form_supplies_subplot_title(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        mvts = MVTSeries(rng, ["x"], np.ones((4, 1)))
        fig = plot(mvts, backend="matplotlib", vars=[("x", "Eks")])
        assert fig.axes[0].get_title() == "Eks"

    def test_default_labels_data_n(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        a = MVTSeries(rng, ["x"], np.ones((4, 1)))
        b = MVTSeries(rng, ["x"], np.zeros((4, 1)))
        fig = plot(a, b, backend="matplotlib")
        legend_labels = [line.get_label() for line in fig.axes[0].lines]
        assert legend_labels == ["data1", "data2"]

    def test_supertitle(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        mvts = MVTSeries(rng, ["x"], np.ones((4, 1)))
        fig = plot(mvts, backend="matplotlib", title="Overview")
        assert fig._suptitle.get_text() == "Overview"

    def test_vars_kwarg_rejected_for_tseries(self) -> None:
        t = TSeries(qq(2020, 1), [1.0])
        with pytest.raises(TypeError, match="vars="):
            plot(t, backend="matplotlib", vars=["x"])

    def test_mixed_types_raise(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        t = TSeries(rng, np.ones(4))
        mvts = MVTSeries(rng, ["x"], np.ones((4, 1)))
        with pytest.raises(TypeError, match="mix"):
            plot(t, mvts, backend="matplotlib")


# ===========================================================================
# Plotly backend
# ===========================================================================


class TestPlotlyTSeries:
    def test_single_yp_returns_figure(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        fig = plot(t, backend="plotly")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        np.testing.assert_array_equal(fig.data[0].y, [1.0, 2.0, 3.0, 4.0])

    def test_yp_tickvals_text_set(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2022, 4))
        t = TSeries(rng, np.arange(len(rng), dtype=float))
        fig = plot(t, backend="plotly")
        # xaxis tickmode='array' with our YP ticktext labels.
        assert fig.layout.xaxis.tickmode == "array"
        # At least one tick text should be a YP MIT string.
        assert any("Q" in str(t) or "Y" in str(t) for t in fig.layout.xaxis.ticktext)

    def test_daily_returns_date_axis(self) -> None:
        rng = mitrange(daily(_dt.date(2024, 1, 1)), daily(_dt.date(2024, 1, 5)))
        t = TSeries(rng, np.arange(len(rng), dtype=float))
        fig = plot(t, backend="plotly")
        # Plotly's xaxis tickmode is not set for date traces; first x is a date.
        assert isinstance(fig.data[0].x[0], _dt.date)

    def test_many_tseries_multiple_traces(self) -> None:
        t1 = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        t2 = TSeries(qq(2020, 1), [4.0, 3.0, 2.0, 1.0])
        fig = plot(t1, t2, backend="plotly", label=["a", "b"])
        assert len(fig.data) == 2
        assert [tr.name for tr in fig.data] == ["a", "b"]

    def test_layout_title_and_labels(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0])
        fig = plot(
            t,
            backend="plotly",
            title="My title",
            xlabel="time",
            ylabel="value",
        )
        assert fig.layout.title.text == "My title"
        assert fig.layout.xaxis.title.text == "time"
        assert fig.layout.yaxis.title.text == "value"

    def test_figsize_pixels(self) -> None:
        t = TSeries(qq(2020, 1), [1.0])
        fig = plot(t, backend="plotly", figsize=(8, 6))
        assert fig.layout.width == 800
        assert fig.layout.height == 600


class TestPlotlyMVTSeriesPanel:
    def test_single_dataset_subplot_titles(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        mvts = MVTSeries(rng, ["x", "y", "z"], np.arange(12, dtype=float).reshape(4, 3))
        fig = plot(mvts, backend="plotly")
        # plotly stores subplot titles in layout.annotations
        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["x", "y", "z"]

    def test_two_datasets_legend_once_per_dataset(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        a = MVTSeries(rng, ["x", "y"], np.ones((4, 2)))
        b = MVTSeries(rng, ["x", "y"], np.zeros((4, 2)))
        fig = plot(a, b, backend="plotly", label=["a", "b"])
        # 4 traces (2 vars x 2 datasets) but only 2 with showlegend=True.
        showlegend_count = sum(bool(tr.showlegend) for tr in fig.data)
        assert showlegend_count == 2

    def test_missing_var_skipped_per_dataset(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        a = MVTSeries(rng, ["x", "y"], np.ones((4, 2)))
        b = MVTSeries(rng, ["y", "z"], np.ones((4, 2)))
        fig = plot(a, b, backend="plotly")
        # Subplots: x (1 trace), y (2 traces), z (1 trace) → 4 traces total.
        assert len(fig.data) == 4

    def test_vars_kwarg_subset(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        mvts = MVTSeries(rng, ["x", "y", "z"], np.arange(12, dtype=float).reshape(4, 3))
        fig = plot(mvts, backend="plotly", vars=["z"])
        assert len(fig.data) == 1
        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["z"]


# ===========================================================================
# Method bindings on TSeries / MVTSeries
# ===========================================================================


class TestMethodBindings:
    def test_tseries_plot_method(self) -> None:
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        fig = t.plot(backend="matplotlib")
        assert fig.__class__.__name__ == "Figure"
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [1.0, 2.0, 3.0, 4.0])

    def test_mvtseries_plot_method(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2020, 4))
        mvts = MVTSeries(rng, ["x", "y"], np.ones((4, 2)))
        fig = mvts.plot(backend="matplotlib")
        assert len(fig.axes) == 2

    def test_top_level_re_export(self) -> None:
        # tsecon.plot is the top-level entry point.
        t = TSeries(qq(2020, 1), [1.0])
        fig = tsecon.plot(t, backend="matplotlib")
        assert fig.__class__.__name__ == "Figure"


# ===========================================================================
# Dispatcher arity
# ===========================================================================


class TestDispatcherArity:
    def test_empty_call_raises(self) -> None:
        with pytest.raises(TypeError, match="at least one"):
            plot()

    def test_unknown_backend_raises(self) -> None:
        t = TSeries(qq(2020, 1), [1.0])
        with pytest.raises(ValueError, match="Unknown backend"):
            plot(t, backend="seaborn")  # type: ignore[arg-type]
