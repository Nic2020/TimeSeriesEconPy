# SPDX-License-Identifier: MIT
"""Drive the 4-column (tsecon / Julia / pandas / polars) comparison harness.

Reads the scenario registries from :mod:`scenarios` (tsecon),
:mod:`scenarios_pandas`, and :mod:`scenarios_polars`; invokes
``julia/runner.jl`` for each scenario via :mod:`subprocess`; and writes a
side-by-side comparison table. Pandas / polars scenarios are optional —
the harness lights up additional columns automatically when each module
imports successfully. A scenario that isn't implemented in a given backend
appears as ``n/a`` in its column.

Two output forms:

* ``results/<date>_<sha>.json`` — full numerical record, one record per
  scenario with one block per available backend.
* Markdown table to stdout (and optionally to a file via ``--markdown``).

Usage::

    uv run python benchmarks/compare/run.py                 # all backends, all scenarios
    uv run python benchmarks/compare/run.py --only rec_ar2_100,shift_quarterly_lag1
    uv run python benchmarks/compare/run.py --python-only   # tsecon column only (fast smoke)
    uv run python benchmarks/compare/run.py --julia-only    # skip Python
    uv run python benchmarks/compare/run.py --no-pandas     # drop the pandas column
    uv run python benchmarks/compare/run.py --no-polars     # drop the polars column
    uv run python benchmarks/compare/run.py --seconds 2     # cap per-scenario budget

If ``julia`` is not on ``PATH``, the script prints a warning and runs only
the Python columns; the Julia column is reported as ``n/a``. The intent:
the script is always runnable on a developer machine, even one without
Julia / pandas / polars installed — extra columns light up automatically
once those backends are available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scenarios import DESCRIPTION, RUN, SETUP  # noqa: E402

# Optional DataFrame-backend scenario modules. Imported lazily so missing
# libraries (pandas / polars not installed) degrade gracefully to "n/a"
# columns in the comparison table.
try:
    from scenarios_pandas import RUN as RUN_PD
    from scenarios_pandas import SETUP as SETUP_PD

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on optional install
    SETUP_PD = {}
    RUN_PD = {}
    _PANDAS_AVAILABLE = False

try:
    from scenarios_polars import RUN as RUN_PL
    from scenarios_polars import SETUP as SETUP_PL

    _POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on optional install
    SETUP_PL = {}
    RUN_PL = {}
    _POLARS_AVAILABLE = False

# Windows console default codec is cp1252 which doesn't cover the Unicode
# chars (µ, em-dashes, arrows) the descriptions and `format_seconds` emit.
# Force stdout/stderr to UTF-8 with `replace` fallback so a stray non-ASCII
# glyph in a description never crashes the harness mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO_ROOT = HERE.parent.parent
RESULTS_DIR = HERE / "results"
JULIA_DIR = HERE / "julia"


@dataclass(frozen=True)
class TimingResult:
    """One language's timing for one scenario."""

    median_seconds: float
    min_seconds: float
    samples: int


def _time_in_process(
    name: str,
    seconds: float,
    setup_registry: dict[str, Callable[[], Any]],
    run_registry: dict[str, Callable[[Any], Any]],
) -> TimingResult:
    """Time ``run_registry[name](setup_registry[name]())`` in the current process.

    Uses :func:`timeit.Timer.autorange` to find a sample size whose total
    cost is ~0.2 s, then runs ``repeat`` rounds until the total wall time
    exceeds ``seconds`` (capped). Records minimum + median of the per-sample
    cost, matching the statistics ``BenchmarkTools.@benchmark`` reports.

    Shared by the tsecon / pandas / polars columns — only Julia goes
    through ``time_julia_scenario`` and its subprocess boundary.
    """
    state = setup_registry[name]()
    run = run_registry[name]

    def _stmt() -> None:
        run(state)

    # Warmup so first-call costs (import-time stragglers, attr-cache fills)
    # aren't charged to a measured sample.
    _stmt()

    timer = timeit.Timer(_stmt)
    number, _ = timer.autorange()
    raw: list[float] = []
    total = 0.0
    while total < seconds:
        t = timer.timeit(number=number)
        raw.append(t / number)
        total += t
        if len(raw) > 5 and total >= seconds * 0.5:
            # Already have a decent number of samples; bail when budget hit.
            pass
    return TimingResult(
        median_seconds=statistics.median(raw),
        min_seconds=min(raw),
        samples=len(raw) * number,
    )


def time_python_scenario(name: str, seconds: float) -> TimingResult:
    """Time the tsecon scenario ``name``. See :func:`_time_in_process`."""
    return _time_in_process(name, seconds, SETUP, RUN)


def time_pandas_scenario(name: str, seconds: float) -> TimingResult | None:
    """Time the pandas scenario ``name``, or return ``None`` if absent."""
    if name not in SETUP_PD:
        return None
    return _time_in_process(name, seconds, SETUP_PD, RUN_PD)


def time_polars_scenario(name: str, seconds: float) -> TimingResult | None:
    """Time the polars scenario ``name``, or return ``None`` if absent."""
    if name not in SETUP_PL:
        return None
    return _time_in_process(name, seconds, SETUP_PL, RUN_PL)


def time_julia_scenario(name: str, seconds: float, julia_exe: str) -> TimingResult | None:
    """Time the scenario in Julia via ``julia/runner.jl``.

    Returns ``None`` when Julia isn't available or the invocation fails;
    the table prints ``n/a`` rather than aborting the whole run.
    """
    with tempfile.NamedTemporaryFile(
        mode="r", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        out_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [
                julia_exe,
                f"--project={JULIA_DIR}",
                str(JULIA_DIR / "runner.jl"),
                "--scenario",
                name,
                "--output",
                str(out_path),
                "--seconds",
                str(seconds),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=max(120.0, seconds * 4 + 60.0),
        )
        if proc.returncode != 0:
            print(
                f"  ! julia failed for {name} (exit {proc.returncode}): "
                f"{proc.stderr.strip()[-200:]}",
                file=sys.stderr,
            )
            return None
        payload = json.loads(out_path.read_text())
        return TimingResult(
            median_seconds=float(payload["median_seconds"]),
            min_seconds=float(payload["min_seconds"]),
            samples=int(payload["samples"]),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"  ! julia invocation error for {name}: {exc}", file=sys.stderr)
        return None
    finally:
        out_path.unlink(missing_ok=True)


def find_julia() -> str | None:
    """Return the ``julia`` executable path, or ``None`` if not installed."""
    return shutil.which("julia")


def format_seconds(s: float) -> str:
    """Format a duration as a human-friendly string (ns / µs / ms / s)."""
    if s < 1e-6:
        return f"{s * 1e9:.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} µs"
    if s < 1.0:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def _cell(t: TimingResult | None) -> str:
    """Render one timing into a Markdown table cell (``n/a`` if absent)."""
    return "n/a" if t is None else format_seconds(t.median_seconds)


def _ratio(py: TimingResult | None, jl: TimingResult | None) -> str:
    """Render a Python/Julia ratio (``n/a`` if either side is absent)."""
    if py is None or jl is None:
        return "n/a"
    return f"{py.median_seconds / jl.median_seconds:.2f}x"


def render_markdown(
    scenarios: list[str],
    py_results: dict[str, TimingResult],
    jl_results: dict[str, TimingResult | None],
    pd_results: dict[str, TimingResult | None],
    pl_results: dict[str, TimingResult | None],
    include_pandas: bool,
    include_polars: bool,
) -> str:
    """Render the comparison table as a Markdown string.

    Columns: Scenario | Description | tsecon | Julia | (Pandas | Polars |)
    Ratio (Py / Jl). Pandas / polars columns are emitted only when their
    backend imported successfully.
    """
    headers = ["Scenario", "Description", "tsecon (median)", "Julia (median)"]
    aligns = ["", "", "---:", "---:"]
    if include_pandas:
        headers.append("Pandas (median)")
        aligns.append("---:")
    if include_polars:
        headers.append("Polars (median)")
        aligns.append("---:")
    headers.append("Ratio (Py / Jl)")
    aligns.append("---:")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|" + "|".join(aligns[2:]) + "|",
    ]
    for name in scenarios:
        py = py_results.get(name)
        jl = jl_results.get(name)
        cells = [
            f"`{name}`",
            DESCRIPTION[name],
            _cell(py),
            _cell(jl),
        ]
        if include_pandas:
            cells.append(_cell(pd_results.get(name)))
        if include_polars:
            cells.append(_cell(pl_results.get(name)))
        cells.append(_ratio(py, jl))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _timing_dict(t: TimingResult | None) -> dict[str, float | int] | None:
    if t is None:
        return None
    return {
        "median_seconds": t.median_seconds,
        "min_seconds": t.min_seconds,
        "samples": t.samples,
    }


def build_payload(
    py_results: dict[str, TimingResult],
    jl_results: dict[str, TimingResult | None],
    pd_results: dict[str, TimingResult | None],
    pl_results: dict[str, TimingResult | None],
    julia_available: bool,
    pandas_available: bool,
    polars_available: bool,
) -> dict[str, object]:
    """Build the JSON payload for ``results/``.

    One record per scenario; each record holds one block per backend
    (``python`` / ``julia`` / ``pandas`` / ``polars``). Absent backends
    serialise as JSON ``null`` rather than being omitted, so future
    consumers can rely on the schema shape.
    """
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": sys.version,
        "julia_available": julia_available,
        "pandas_available": pandas_available,
        "polars_available": polars_available,
        "scenarios": [
            {
                "name": name,
                "description": DESCRIPTION[name],
                "python": _timing_dict(py_results.get(name)),
                "julia": _timing_dict(jl_results.get(name)),
                "pandas": _timing_dict(pd_results.get(name)),
                "polars": _timing_dict(pl_results.get(name)),
            }
            for name in py_results
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of scenario names to run (default: all).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="Per-scenario time budget in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Run only the tsecon column (skip Julia, pandas, polars).",
    )
    parser.add_argument("--julia-only", action="store_true", help="Skip Python timing.")
    parser.add_argument(
        "--no-pandas",
        action="store_true",
        help="Skip the pandas column even if pandas is installed.",
    )
    parser.add_argument(
        "--no-polars",
        action="store_true",
        help="Skip the polars column even if polars is installed.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="If set, also write the table to this file.",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="Skip writing the JSON results file."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison harness end-to-end."""
    args = parse_args(argv)
    scenarios = (
        list(SETUP.keys()) if args.only is None else [s.strip() for s in args.only.split(",")]
    )
    unknown = set(scenarios) - SETUP.keys()
    if unknown:
        print(f"error: unknown scenarios: {sorted(unknown)}", file=sys.stderr)
        return 2

    julia_exe = None if args.python_only else find_julia()
    if not args.python_only and julia_exe is None:
        print(
            "note: `julia` not on PATH — running Python column only. "
            "Install Julia and `Pkg.instantiate()` in benchmarks/compare/julia "
            "to enable side-by-side numbers.",
            file=sys.stderr,
        )
    include_pandas = _PANDAS_AVAILABLE and not args.no_pandas and not args.python_only
    include_polars = _POLARS_AVAILABLE and not args.no_polars and not args.python_only
    if not args.python_only and not _PANDAS_AVAILABLE:
        print(
            "note: `pandas` not importable — pandas column will read n/a. "
            "Install with `uv sync --extra pandas` to enable it.",
            file=sys.stderr,
        )
    if not args.python_only and not _POLARS_AVAILABLE:
        print(
            "note: `polars` not importable — polars column will read n/a. "
            "Install with `uv sync --extra polars` to enable it.",
            file=sys.stderr,
        )

    py_results: dict[str, TimingResult] = {}
    jl_results: dict[str, TimingResult | None] = {}
    pd_results: dict[str, TimingResult | None] = {}
    pl_results: dict[str, TimingResult | None] = {}

    for name in scenarios:
        print(f"running {name} ...", flush=True)
        if not args.julia_only:
            py_results[name] = time_python_scenario(name, args.seconds)
            print(
                f"  python: median={format_seconds(py_results[name].median_seconds)} "
                f"(min={format_seconds(py_results[name].min_seconds)}, "
                f"samples={py_results[name].samples})"
            )
        if julia_exe is not None:
            jl_results[name] = time_julia_scenario(name, args.seconds, julia_exe)
            if jl_results[name] is not None:
                print(
                    f"  julia : median={format_seconds(jl_results[name].median_seconds)} "  # type: ignore[union-attr]
                    f"(min={format_seconds(jl_results[name].min_seconds)}, "  # type: ignore[union-attr]
                    f"samples={jl_results[name].samples})"  # type: ignore[union-attr]
                )
        if include_pandas and not args.julia_only:
            pd_results[name] = time_pandas_scenario(name, args.seconds)
            if pd_results[name] is not None:
                print(
                    f"  pandas: median={format_seconds(pd_results[name].median_seconds)} "  # type: ignore[union-attr]
                    f"(min={format_seconds(pd_results[name].min_seconds)}, "  # type: ignore[union-attr]
                    f"samples={pd_results[name].samples})"  # type: ignore[union-attr]
                )
            else:
                print("  pandas: n/a (not registered for this scenario)")
        if include_polars and not args.julia_only:
            pl_results[name] = time_polars_scenario(name, args.seconds)
            if pl_results[name] is not None:
                print(
                    f"  polars: median={format_seconds(pl_results[name].median_seconds)} "  # type: ignore[union-attr]
                    f"(min={format_seconds(pl_results[name].min_seconds)}, "  # type: ignore[union-attr]
                    f"samples={pl_results[name].samples})"  # type: ignore[union-attr]
                )
            else:
                print("  polars: n/a (not registered for this scenario)")

    if not args.julia_only:
        table = render_markdown(
            scenarios,
            py_results,
            jl_results,
            pd_results,
            pl_results,
            include_pandas=include_pandas,
            include_polars=include_polars,
        )
        print()
        print(table)
        if args.markdown is not None:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(table + "\n", encoding="utf-8")
            print(f"\nWrote markdown table to {args.markdown}", file=sys.stderr)

    if not args.no_json and not args.julia_only:
        payload = build_payload(
            py_results,
            jl_results,
            pd_results,
            pl_results,
            julia_available=julia_exe is not None,
            pandas_available=include_pandas,
            polars_available=include_polars,
        )
        RESULTS_DIR.mkdir(exist_ok=True)
        try:
            sha = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sha = "nogit"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
        out = RESULTS_DIR / f"{timestamp}_{sha}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote results to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
