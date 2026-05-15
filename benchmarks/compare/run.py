# SPDX-License-Identifier: MIT
"""Drive the Julia ↔ Python comparison harness.

Reads the scenario registry from :mod:`scenarios`, times each scenario in
Python with :mod:`timeit`, invokes ``julia/runner.jl`` for each scenario via
:mod:`subprocess`, and writes a side-by-side comparison table.

Two output forms:

* ``results/<date>_<sha>.json`` — full numerical record.
* Markdown table to stdout (and optionally to a file via ``--markdown``).

Usage::

    uv run python benchmarks/compare/run.py                 # all scenarios
    uv run python benchmarks/compare/run.py --only rec_ar2_100,shift_quarterly_lag1
    uv run python benchmarks/compare/run.py --python-only   # skip Julia (fast smoke)
    uv run python benchmarks/compare/run.py --julia-only    # skip Python
    uv run python benchmarks/compare/run.py --seconds 2     # cap per-scenario budget

If ``julia`` is not on ``PATH``, the script prints a warning and runs only
the Python column; the Julia column is reported as ``n/a``. The intent: the
script is always runnable on a developer machine, even one without Julia
installed — the cross-language number lights up automatically once Julia is
available.
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scenarios import DESCRIPTION, RUN, SETUP  # noqa: E402

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


def time_python_scenario(name: str, seconds: float) -> TimingResult:
    """Time ``RUN[name](SETUP[name]())`` in Python.

    Uses :func:`timeit.Timer.autorange` to find a sample size whose total
    cost is ~0.2 s, then runs ``repeat`` rounds until the total wall time
    exceeds ``seconds`` (capped). Records minimum + median of the per-sample
    cost, matching the statistics ``BenchmarkTools.@benchmark`` reports.
    """
    state = SETUP[name]()
    run = RUN[name]

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


def render_markdown(
    scenarios: list[str],
    py_results: dict[str, TimingResult],
    jl_results: dict[str, TimingResult | None],
) -> str:
    """Render the comparison table as a Markdown string."""
    lines = [
        "| Scenario | Description | Python (median) | Julia (median) | Ratio (Py / Jl) |",
        "|---|---|---:|---:|---:|",
    ]
    for name in scenarios:
        py = py_results[name]
        jl = jl_results.get(name)
        if jl is None:
            ratio = "n/a"
            jl_cell = "n/a"
        else:
            ratio = f"{py.median_seconds / jl.median_seconds:.2f}x"
            jl_cell = format_seconds(jl.median_seconds)
        lines.append(
            f"| `{name}` | {DESCRIPTION[name]} | "
            f"{format_seconds(py.median_seconds)} | {jl_cell} | {ratio} |"
        )
    return "\n".join(lines)


def build_payload(
    py_results: dict[str, TimingResult],
    jl_results: dict[str, TimingResult | None],
    julia_available: bool,
) -> dict[str, object]:
    """Build the JSON payload for ``results/``."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": sys.version,
        "julia_available": julia_available,
        "scenarios": [
            {
                "name": name,
                "description": DESCRIPTION[name],
                "python": {
                    "median_seconds": py.median_seconds,
                    "min_seconds": py.min_seconds,
                    "samples": py.samples,
                },
                "julia": (
                    None
                    if jl_results.get(name) is None
                    else {
                        "median_seconds": jl_results[name].median_seconds,  # type: ignore[union-attr]
                        "min_seconds": jl_results[name].min_seconds,  # type: ignore[union-attr]
                        "samples": jl_results[name].samples,  # type: ignore[union-attr]
                    }
                ),
            }
            for name, py in py_results.items()
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
    parser.add_argument("--python-only", action="store_true", help="Skip Julia invocations.")
    parser.add_argument("--julia-only", action="store_true", help="Skip Python timing.")
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

    py_results: dict[str, TimingResult] = {}
    jl_results: dict[str, TimingResult | None] = {}

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

    if not args.julia_only:
        table = render_markdown(scenarios, py_results, jl_results)
        print()
        print(table)
        if args.markdown is not None:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(table + "\n", encoding="utf-8")
            print(f"\nWrote markdown table to {args.markdown}", file=sys.stderr)

    if not args.no_json and not args.julia_only:
        payload = build_payload(py_results, jl_results, julia_exe is not None)
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
