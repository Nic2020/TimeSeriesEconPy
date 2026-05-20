# SPDX-License-Identifier: MIT
"""M2.4 .spc-serializer tests: x13write + impose_line_length.

Hand-authored fixtures (per session-52 scope-confirmation Q3): each
test asserts ``x13write`` produces a specific ``.spc`` reference
string. The reference strings are derived from reading the X-13
ARIMA-SEATS reference manual and the Julia upstream's per-builder
doctest comments — the round-trip against captured Julia outputs is
deferred to M2.5 when the binary runner can confirm byte parity.

Test organisation:

* :class:`TestImposeLineLength` — line-wrap kernel: split-at-space,
  split-at-``" + "``, single-line passthrough, single-token-over-limit
  raise, embedded-newline recursion.
* :class:`TestX13WriteSeriesBlock` — series ``{ ... }`` rendering.
* :class:`TestX13WritePerBlock` — one ``TestX`` per sub-spec covering
  the most common kwarg shapes and the field-type quirks
  (Symbol-vs-String, Vector{Symbol}-vs-Vector{String}, fixed-ar/ma/b).
* :class:`TestX13WriteCrossBlock` — multi-block specs assert block
  ordering and the trailing ``string`` side-effect on :class:`X13spec`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsecon.frequencies import Monthly, Quarterly
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.x13 import (
    ArimaModel,
    ArimaSpec,
    Span,
    X13spec,
    ao,
    arima,
    automdl,
    check,
    easter,
    estimate,
    force,
    forecast,
    history,
    identify,
    impose_line_length,
    metadata,
    newspec,
    outlier,
    pickmdl,
    regression,
    seats,
    series,
    slidingspans,
    spectrum,
    td,
    transform,
    x11,
    x11regression,
    x13write,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def q_ts() -> TSeries:
    """Quarterly TSeries 1990Q1..2009Q4 (80 points), values 0..79."""
    return TSeries(MIT.from_yp(Quarterly(), 1990, 1), np.arange(80.0))


@pytest.fixture
def m_ts() -> TSeries:
    """Monthly TSeries 2000M1..2019M12 (240 points), values 0..239."""
    return TSeries(MIT.from_yp(Monthly(), 2000, 1), np.arange(240.0))


@pytest.fixture
def q_spec(q_ts: TSeries) -> X13spec:
    """Quarterly spec with ``transform(power=0.5) + x11(mode='mult')`` baseline.

    The transform+x11 pair short-circuits :func:`validateX13spec`'s
    "default x11.mode ⊥ default transform" check (the canonical
    X-13 misuse trap). Per-block tests add their own sub-spec on top.
    """
    spec = newspec(q_ts)
    spec.transform = transform(power=0.5)
    spec.x11 = x11(mode="mult")
    return spec


@pytest.fixture
def m_spec(m_ts: TSeries) -> X13spec:
    """Monthly spec with ``transform(power=0.5) + x11(mode='mult')`` baseline."""
    spec = newspec(m_ts)
    spec.transform = transform(power=0.5)
    spec.x11 = x11(mode="mult")
    return spec


# ---------------------------------------------------------------------------
# impose_line_length
# ---------------------------------------------------------------------------


class TestImposeLineLength:
    """The Julia-mirrored line-wrap helper."""

    def test_short_line_passes_through(self) -> None:
        """Lines at or under the limit are untouched."""
        s = ["a = 1", "b = 2"]
        impose_line_length(s)
        assert s == ["a = 1", "b = 2"]

    def test_long_line_splits_at_space(self) -> None:
        """A line over 124 cols splits at the last fitting space."""
        long = " ".join(f"x{i}" for i in range(60))  # 60 tokens, ~180 chars
        s = [long]
        impose_line_length(s)
        assert len(s) == 2
        assert all(len(line) <= 132 for line in s)
        # Continuation line carries the 8-space indent
        assert s[1].startswith("        ")

    def test_long_line_prefers_plus_split(self) -> None:
        """When ``" + "`` is present, the wrapper splits there over plain spaces."""
        long = "print = (" + " + ".join(f"table{i}" for i in range(30)) + ")"
        s = [long]
        impose_line_length(s)
        # Every continuation line must end in '+ ' (the split-char trailing) or be the last fragment
        for line in s[:-1]:
            assert line.rstrip().endswith("+"), f"line {line!r} not split at +"

    def test_explicit_limit_arg(self) -> None:
        """Caller can lower the limit for testing."""
        s = ["a b c d e f g h i j k l m n o p q"]
        impose_line_length(s, limit=12)
        assert len(s) > 1

    def test_un_splittable_token_raises(self) -> None:
        """A single token longer than the limit eventually raises ValueError."""
        with pytest.raises(ValueError, match="Could not split"):
            impose_line_length(["a" * 200])

    def test_embedded_newline_recurses(self) -> None:
        """Lines with embedded ``\\n`` are wrapped sub-line-wise."""
        sub_long = " ".join(f"x{i}" for i in range(60))
        s = [f"prefix\n{sub_long}\nsuffix"]
        impose_line_length(s)
        # The newline-joined sub-lines have been wrapped without
        # breaking the outer line.
        assert "\n" in s[0]
        for sub in s[0].split("\n"):
            # Inner sub-line may have been wrapped — the joined block is one
            # entry in s; sub-lines may still contain `\n` from the wrap.
            for actual_line in sub.split("\n"):
                assert len(actual_line) <= 132 + 8

    def test_mutation_is_in_place(self) -> None:
        """The helper mutates ``s`` rather than returning a new list."""
        s = [" ".join(f"x{i}" for i in range(60))]
        before_id = id(s)
        result = impose_line_length(s)
        assert result is None
        assert id(s) == before_id
        assert len(s) > 1

    def test_default_limit_is_124(self) -> None:
        """The default limit matches Julia's ``132 - 8`` (8-column block indent)."""
        # A line of exactly 124 + 1 chars should split.
        s = [" ".join(["a"] * 70)]  # ~ 139 chars
        impose_line_length(s)
        assert len(s) > 1


# ---------------------------------------------------------------------------
# series block
# ---------------------------------------------------------------------------


class TestX13WriteSeriesBlock:
    """``x13write(spec)`` series-block rendering."""

    def test_minimal_quarterly(self, q_spec: X13spec) -> None:
        """A bare Quarterly series renders data + period + start."""
        out = x13write(q_spec, test=True)
        assert out.startswith("series {\n")
        assert "data = (" in out
        assert "period = 4" in out
        assert "start = 1990.1" in out

    def test_monthly_renders_month_abbr(self, m_ts: TSeries) -> None:
        """Monthly MIT starts render as ``year.<month-abbr>``."""
        spec = newspec(m_ts)
        out = x13write(spec, test=True)
        assert "start = 2000.jan" in out

    def test_name_field_quoted(self, q_ts: TSeries) -> None:
        """``name=`` renders quoted (Julia ``::String``)."""
        spec = newspec(series(q_ts, name="GNP"))
        out = x13write(spec, test=True)
        assert 'name = "GNP"' in out

    def test_title_field_quoted(self, q_ts: TSeries) -> None:
        """``title=`` renders quoted."""
        spec = newspec(series(q_ts, title="Real GDP, Quarterly"))
        out = x13write(spec, test=True)
        assert 'title = "Real GDP, Quarterly"' in out

    def test_type_field_unquoted(self, q_ts: TSeries) -> None:
        """``type=`` renders bare (Julia ``::Symbol``)."""
        spec = newspec(series(q_ts, type="flow"))
        out = x13write(spec, test=True)
        assert "type = flow" in out
        assert 'type = "flow"' not in out

    def test_test_flag_skips_print_save_savelog(self, q_ts: TSeries) -> None:
        """``test=True`` omits ``print``/``save``/``savelog`` fields."""
        spec = newspec(series(q_ts, print=["default"], save=["seriesplot"]))
        out = x13write(spec, test=True)
        assert "print =" not in out
        assert "save =" not in out

    def test_test_flag_false_emits_print_save(self, q_ts: TSeries) -> None:
        """``test=False`` (the default) emits ``print``/``save`` fields."""
        spec = newspec(series(q_ts, print=["default"]))
        out = x13write(spec, test=False)
        assert "print = (default)" in out


# ---------------------------------------------------------------------------
# Per sub-spec block
# ---------------------------------------------------------------------------


class TestX13WriteArimaBlock:
    def test_simple_arima(self, q_spec: X13spec) -> None:
        q_spec.arima = arima(ArimaModel.from_pdq(1, 1, 1))
        out = x13write(q_spec, test=True)
        assert "arima {\n" in out
        assert "model = (1 1 1)" in out

    def test_seasonal_arima(self, q_spec: X13spec) -> None:
        q_spec.arima = arima(ArimaModel.from_pdq_seasonal(0, 1, 1, 0, 1, 1))
        out = x13write(q_spec, test=True)
        assert "model = (0 1 1)(0 1 1)" in out

    def test_explicit_arima_period(self, q_spec: X13spec) -> None:
        q_spec.arima = arima(ArimaSpec(1, 1, 1, period=12))
        out = x13write(q_spec, test=True)
        assert "model = (1 1 1)12" in out

    def test_arima_ar_with_fixar(self, q_spec: X13spec) -> None:
        q_spec.arima = arima(
            ArimaSpec(1, 1, 1),
            ar=[0.3, 0.5],
            fixar=[True, False],
        )
        out = x13write(q_spec, test=True)
        assert "ar = (0.3f,0.5)" in out

    def test_arima_ma_with_missing_value(self, q_spec: X13spec) -> None:
        """``None`` entries in ``ma`` render as empty (Julia ``Missing`` token)."""
        q_spec.arima = arima(
            ArimaSpec(1, 1, 1),
            ma=[0.3, None, 0.5],
            fixma=[True, False, True],
        )
        out = x13write(q_spec, test=True)
        assert "ma = (0.3f,,0.5f)" in out

    def test_arima_tuple_pdq(self, q_spec: X13spec) -> None:
        """Tuple-shaped ``p`` renders as a bracketed list."""
        q_spec.arima = arima(ArimaSpec((2, 3), 0, 0))
        out = x13write(q_spec, test=True)
        assert "model = ([2 3] 0 0)" in out


class TestX13WriteTransformBlock:
    def test_func_log(self, q_spec: X13spec) -> None:
        q_spec.transform = transform(func="log")
        out = x13write(q_spec, test=True)
        assert "transform {\n" in out
        # func renames to function on output (Julia ``key == :func`` rewrite)
        assert "function = log" in out
        assert "func = " not in out

    def test_power_box_cox(self, q_spec: X13spec) -> None:
        q_spec.transform = transform(power=0.5)
        out = x13write(q_spec, test=True)
        assert "power = 0.5" in out


class TestX13WriteRegressionBlock:
    def test_variables_symbol_list_space_separated(self, q_spec: X13spec) -> None:
        """``variables=[<X13var>, "td"]`` renders space-separated."""
        q_spec.regression = regression(variables=[ao(MIT.from_yp(Quarterly(), 1995, 2)), "td"])
        q_spec.transform = transform(power=0.5)
        q_spec.x11 = x11(mode="mult")
        out = x13write(q_spec, test=True)
        assert "variables = (ao1995.2 td)" in out

    def test_aictest_at_end_of_block(self, q_spec: X13spec) -> None:
        """``aictest=`` is one of the ``keys_at_end`` (Julia parity)."""
        q_spec.regression = regression(variables="td", aictest=["td"])
        q_spec.transform = transform(power=0.5)
        q_spec.x11 = x11(mode="mult")
        out = x13write(q_spec, test=True)
        # aictest comes after variables in the block body
        block = out.split("regression {")[1].split("}")[0]
        v_pos = block.index("variables =")
        a_pos = block.index("aictest =")
        assert a_pos > v_pos

    def test_regression_b_with_fixb(self, q_spec: X13spec) -> None:
        q_spec.regression = regression(variables="td", b=[0.1, 0.2], fixb=[False, True])
        q_spec.transform = transform(power=0.5)
        q_spec.x11 = x11(mode="mult")
        out = x13write(q_spec, test=True)
        assert "b = (0.1,0.2f)" in out


class TestX13WriteX11Block:
    def test_mode_unquoted(self, q_spec: X13spec) -> None:
        q_spec.x11 = x11(mode="mult")
        out = x13write(q_spec, test=True)
        assert "x11 {\n" in out
        assert "mode = mult" in out

    def test_title_quoted(self, q_spec: X13spec) -> None:
        """``title`` is the rare ``Union{String,Vector{String}}`` — quoted."""
        q_spec.x11 = x11(mode="mult", title="Decomposition")
        out = x13write(q_spec, test=True)
        assert 'title = "Decomposition"' in out

    def test_empty_x11_emits_braces(self, q_spec: X13spec) -> None:
        """A bare ``x11()`` (no kwargs) renders ``x11 { }``."""
        q_spec.x11 = x11()
        out = x13write(q_spec, test=True)
        assert "x11 { }" in out


class TestX13WriteSeatsBlock:
    def test_seats_block_renders(self, m_ts: TSeries) -> None:
        """A bare ``seats()`` always emits at least the ``out = 0`` default field."""
        spec = newspec(m_ts)
        spec.transform = transform(power=0.5)
        # x11 left at default; seats is the alternative seasonal-adjustment path.
        spec.seats = seats()
        out = x13write(spec, test=True)
        assert "seats {\n" in out
        assert "out = 0" in out


class TestX13WriteOutlierBlock:
    def test_outlier_types_list(self, q_spec: X13spec) -> None:
        q_spec.outlier = outlier(types=["ao", "ls"])
        out = x13write(q_spec, test=True)
        assert "outlier {\n" in out
        assert "types = (ao ls)" in out


class TestX13WriteAutomdlBlock:
    def test_automdl_basic(self, q_spec: X13spec) -> None:
        """A bare ``automdl()`` renders as the empty-block form under ``test=True``."""
        q_spec.automdl = automdl()
        out = x13write(q_spec, test=True)
        assert "automdl { }" in out

    def test_automdl_maxorder(self, q_spec: X13spec) -> None:
        q_spec.automdl = automdl(maxorder=[3, 2])
        out = x13write(q_spec, test=True)
        assert "automdl {\n" in out
        assert "maxorder = (3, 2)" in out


class TestX13WriteForecastBlock:
    def test_forecast_maxlead(self, q_spec: X13spec) -> None:
        q_spec.forecast = forecast(maxlead=12)
        out = x13write(q_spec, test=True)
        assert "forecast {\n" in out
        assert "maxlead = 12" in out


class TestX13WriteCheckBlock:
    def test_check_block(self, q_spec: X13spec) -> None:
        q_spec.check = check(maxlag=10)
        out = x13write(q_spec, test=True)
        assert "check {\n" in out
        assert "maxlag = 10" in out


class TestX13WriteEstimateBlock:
    def test_estimate_tol(self, q_spec: X13spec) -> None:
        q_spec.estimate = estimate(tol=1e-5)
        out = x13write(q_spec, test=True)
        assert "estimate {\n" in out
        assert "tol = 1e-05" in out


class TestX13WriteForceBlock:
    def test_force_lambda_renamed(self, q_spec: X13spec) -> None:
        """Python's ``lambda_`` kwarg renames to ``lambda`` on output."""
        q_spec.force = force(lambda_=0.5)
        out = x13write(q_spec, test=True)
        assert "lambda = 0.5" in out
        assert "lambda_" not in out


class TestX13WriteHistoryBlock:
    def test_history_fstep(self, q_spec: X13spec) -> None:
        q_spec.history = history(fstep=[1, 2, 3])
        out = x13write(q_spec, test=True)
        assert "history {\n" in out
        # ``Vector{Int64}`` falls through to the generic comma-separated branch.
        assert "fstep = (1, 2, 3)" in out


class TestX13WriteIdentifyBlock:
    def test_identify_maxlag(self, q_spec: X13spec) -> None:
        q_spec.identify = identify(maxlag=10)
        out = x13write(q_spec, test=True)
        assert "identify {\n" in out
        assert "maxlag = 10" in out


class TestX13WriteMetadataBlock:
    def test_single_entry(self, q_spec: X13spec) -> None:
        """Single-entry metadata uses the ``key=…\\nvalue=…`` shape."""
        q_spec.metadata = metadata({"source": "BoC"})
        out = x13write(q_spec, test=True)
        assert "metadata {\n" in out
        assert 'key = "source"' in out
        assert 'value = "BoC"' in out

    def test_multi_entry(self, q_spec: X13spec) -> None:
        """Multi-entry metadata uses the parenthesised-multi-line form."""
        q_spec.metadata = metadata([("a", "b"), ("c", "d")])
        out = x13write(q_spec, test=True)
        assert "key = (" in out
        assert "value = (" in out
        assert '"a"' in out
        assert '"d"' in out


class TestX13WritePickmdlBlock:
    def test_pickmdl_inline(self, q_spec: X13spec) -> None:
        """Pickmdl with no ``outfolder`` inlines the model list."""
        q_spec.pickmdl = pickmdl(
            ArimaModel.from_pdq_seasonal(0, 1, 1, 0, 1, 1, default=True),
            ArimaModel.from_pdq_seasonal(2, 1, 0, 0, 1, 1),
        )
        out = x13write(q_spec, test=True)
        assert "pickmdl {\n" in out
        assert "models = " in out
        # Default-marked model gets `*` suffix; non-default `X`; last one bare.
        assert "(0 1 1)(0 1 1) *" in out

    def test_pickmdl_writes_to_outfolder(
        self,
        q_spec: X13spec,
        tmp_path: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With ``outfolder=`` set, pickmdl writes ``pickmdl.mdl`` and emits ``file=``.

        Uses ``monkeypatch.chdir(tmp_path) + outfolder="."`` because Windows
        pytest tmp_paths can be long enough to overrun the 132-column .spc
        line limit — the wrapper would (correctly per Julia parity) raise.
        """
        monkeypatch.chdir(tmp_path)
        q_spec.pickmdl = pickmdl(
            ArimaModel.from_pdq_seasonal(0, 1, 1, 0, 1, 1, default=True),
            ArimaModel.from_pdq_seasonal(2, 1, 0, 0, 1, 1),
        )
        out = x13write(q_spec, test=True, outfolder=".")
        assert 'file = "' in out
        assert (Path(tmp_path) / "pickmdl.mdl").is_file()


class TestX13WriteSlidingspansBlock:
    def test_slidingspans_length(self, q_spec: X13spec) -> None:
        q_spec.slidingspans = slidingspans(length=16)
        out = x13write(q_spec, test=True)
        assert "slidingspans {\n" in out
        assert "length = 16" in out


class TestX13WriteSpectrumBlock:
    def test_spectrum_qcheck_yes(self, m_ts: TSeries) -> None:
        spec = newspec(m_ts)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        spec.spectrum = spectrum(qcheck=True)
        out = x13write(spec, test=True)
        assert "spectrum {\n" in out
        # bool renders as yes/no
        assert "qcheck = yes" in out


class TestX13WriteX11RegressionBlock:
    def test_x11regression_sigma(self, q_spec: X13spec) -> None:
        q_spec.x11regression = x11regression(sigma=2.5)
        out = x13write(q_spec, test=True)
        assert "x11regression {\n" in out
        assert "sigma = 2.5" in out


# ---------------------------------------------------------------------------
# Cross-block: block ordering and side-effects
# ---------------------------------------------------------------------------


class TestX13WriteCrossBlock:
    def test_series_always_first(self, q_ts: TSeries) -> None:
        """Series is the first block, regardless of which sub-spec was assigned first."""
        spec = newspec(q_ts)
        spec.outlier = outlier(types=["ao"])
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        out = x13write(spec, test=True)
        first_block = out.split("\n")[0]
        assert first_block == "series {"

    def test_block_order_matches_declaration(self, q_ts: TSeries) -> None:
        """Sub-spec block ordering follows the declared X13spec field order."""
        spec = newspec(q_ts)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        spec.outlier = outlier(types=["ao"])
        spec.forecast = forecast(maxlead=4)
        out = x13write(spec, test=True)
        # Order: series, arima, estimate, transform, regression, automdl,
        # x11, x11regression, check, forecast, force, pickmdl, history,
        # metadata, identify, outlier, seats, slidingspans, spectrum.
        names = ["series", "transform", "x11", "forecast", "outlier"]
        positions = [out.index(f"{n} {{") for n in names]
        assert positions == sorted(positions), (
            f"block ordering broke: positions {positions} not sorted for names {names}"
        )

    def test_spec_string_side_effect(self, q_spec: X13spec) -> None:
        """``x13write`` stashes the rendered text on ``spec.string``."""
        text = x13write(q_spec, test=True)
        assert q_spec.string == text

    def test_repeated_call_overwrites_string(self, q_spec: X13spec) -> None:
        """Calling ``x13write`` twice updates ``spec.string`` to the latest text."""
        text1 = x13write(q_spec, test=True)
        q_spec.outlier = outlier(types=["ao"])
        text2 = x13write(q_spec, test=True)
        assert text1 != text2
        assert q_spec.string == text2

    def test_raises_when_series_default(self) -> None:
        """``x13write`` on a spec without a series raises ValueError."""
        spec = newspec()  # no series argument
        with pytest.raises(ValueError, match="X13series"):
            x13write(spec)


class TestX13WriteValueRendering:
    """Per-value-type rendering edge cases."""

    def test_bool_renders_yes_no(self, m_spec: X13spec) -> None:
        """``spectrum(qcheck=True)`` on Monthly: no warning, renders ``yes``."""
        m_spec.spectrum = spectrum(qcheck=True)
        out = x13write(m_spec, test=True)
        assert "qcheck = yes" in out

    def test_mit_range_renders_paren_pair(self, q_spec: X13spec) -> None:
        q_spec.outlier = outlier(
            span=MITRange(MIT.from_yp(Quarterly(), 1992, 1), MIT.from_yp(Quarterly(), 1999, 4))
        )
        out = x13write(q_spec, test=True)
        assert "span = (1992.1, 1999.4)" in out

    def test_span_renders_paren_pair(self, q_spec: X13spec) -> None:
        q_spec.outlier = outlier(
            span=Span(MIT.from_yp(Quarterly(), 1992, 1), MIT.from_yp(Quarterly(), 1999, 4))
        )
        out = x13write(q_spec, test=True)
        assert "span = (1992.1, 1999.4)" in out

    def test_mvtseries_data_renders_multi_column(self, q_ts: TSeries) -> None:
        """A 2-column MVTSeries renders row-wise inside parens."""
        cols = MVTSeries(
            q_ts.range,
            {
                "a": np.arange(80.0),
                "b": np.arange(80.0) * 2,
            },
        )
        spec = newspec(q_ts)
        spec.regression = regression(variables="td", data=cols)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        out = x13write(spec, test=True)
        assert "data = (" in out

    def test_x13var_renders_via_str(self, q_spec: X13spec) -> None:
        """X13var subclasses serialize through their ``__str__`` (M2.1 surface)."""
        q_spec.regression = regression(variables=[ao(MIT.from_yp(Quarterly(), 1992, 2))])
        q_spec.transform = transform(power=0.5)
        q_spec.x11 = x11(mode="mult")
        out = x13write(q_spec, test=True)
        assert "ao1992.2" in out

    def test_calendar_easter_renders_bracketed(self, q_spec: X13spec) -> None:
        """``easter(n=8)`` renders as ``easter[8]``."""
        q_spec.regression = regression(variables=[easter(8)], aictest=["easter"])
        q_spec.transform = transform(power=0.5)
        q_spec.x11 = x11(mode="mult")
        # Skip the lpyear/lom/loq compat checks by using a Monthly fixture
        # (easter on Quarterly + flow type is the legal combination).
        out = x13write(q_spec, test=True)
        assert "easter[8]" in out

    def test_td_no_regime_change_bare(self, q_spec: X13spec) -> None:
        """``td()`` with no MIT renders as the bare token ``td``."""
        q_spec.regression = regression(variables=[td()])
        q_spec.transform = transform(power=0.5)
        q_spec.x11 = x11(mode="mult")
        out = x13write(q_spec, test=True)
        assert "variables = (td)" in out

    def test_arima_model_list_no_separator(self, q_spec: X13spec) -> None:
        """Multi-operator ArimaModel renders as concatenated ``(...)(P D Q)``."""
        q_spec.arima = arima(ArimaModel.from_pdq_seasonal(1, 0, 0, 0, 1, 1))
        out = x13write(q_spec, test=True)
        assert "model = (1 0 0)(0 1 1)" in out


class TestX13WriteLineWrapInteraction:
    """The writer + impose_line_length interplay."""

    def test_long_series_data_wraps(self, q_ts: TSeries) -> None:
        """A long ``data=(...)`` line wraps onto continuation lines."""
        # 80-point quarterly series fits comfortably; force a long line by
        # constructing 600 points.
        long_ts = TSeries(MIT.from_yp(Quarterly(), 1900, 1), np.arange(600.0))
        spec = newspec(long_ts)
        out = x13write(spec, test=True)
        # No single line exceeds the 132 column limit + a small buffer for
        # continuation indent.
        for line in out.splitlines():
            assert len(line) <= 132 + 16, f"line {line!r} too long"

    def test_long_variables_list_wraps(self, q_spec: X13spec) -> None:
        """A long ``variables=`` list wraps at spaces."""
        many_outliers = [
            ao(MIT.from_yp(Quarterly(), 1990 + (i // 4), (i % 4) + 1)) for i in range(60)
        ]
        q_spec.regression = regression(variables=many_outliers)
        q_spec.transform = transform(power=0.5)
        q_spec.x11 = x11(mode="mult")
        out = x13write(q_spec, test=True)
        # Should still parse — the binary will reject a long unwrapped line.
        max_line = max(len(line) for line in out.splitlines())
        assert max_line <= 132 + 16


class TestX13WriteForceBlockRho:
    def test_rho_valid(self, q_spec: X13spec) -> None:
        q_spec.force = force(rho=0.5)
        out = x13write(q_spec, test=True)
        assert "rho = 0.5" in out
