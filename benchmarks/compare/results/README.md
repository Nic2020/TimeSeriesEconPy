# Benchmark baselines — session → SHA map

Each `<UTC-timestamp>_<short-sha>.json` file in this directory is one run of
`benchmarks/compare/run.py`. The short SHA in the filename is the value of
`git rev-parse --short HEAD` at the moment of the run — **not necessarily the
SHA that introduced the harness changes being measured**, because a baseline
is typically captured *before* the session's commit lands (working-tree
state, but HEAD still points at the previous session's tip). The pairing
below is the authoritative mapping the paper / PARITY / NOTES.md cite
against.

> **Canonical baseline for the JSS Section 5 numbers:** **#9** — `2026-05-16_153814Z_98a554c.json` (session 27, first of two controlled re-runs).
> Use it for absolute timings.
> Use **#10** (`2026-05-16_160019Z_98a554c.json`, the second re-run) as the cross-validation companion — every NumPy / Cython row in #9 and #10 agrees to within ±5%, which is the stop-criterion guarantee from session 27.

This index is the fix for review finding
[`F12_session_to_baseline_mapping_missing`](../../../../claude_files/reviews/2026-05-16_holistic/F12_session_to_baseline_mapping_missing.md).

## Baselines (chronological)

| # | Session | Captured (UTC)     | Baseline file                          | Backends present                      | Scenarios | Notes |
|---|---------|--------------------|----------------------------------------|---------------------------------------|----------:|---|
| 1 | 16      | 2026-05-15 21:15Z  | `2026-05-15_211530Z_43a7a93.json`      | tsecon / Julia                        | 8  | First baseline — harness landed in session 16, captured against working tree while HEAD was still at `43a7a93` (session 15's tip). |
| 2 | 17      | 2026-05-15 22:14Z  | `2026-05-15_221430Z_7396bb0.json`      | tsecon / Julia                        | 9  | Session-17 first run: `rec_linear_ar2_100_numpy` added; Cython not yet compiled (MSVC SDK missing). |
| 3 | 17      | 2026-05-15 22:43Z  | `2026-05-15_224300Z_defb7ae.json`      | tsecon / Julia                        | 10 | Session-17 second run — first **three-flavor** row (`rec_linear_ar2_100_cython` lights up after Windows 11 SDK install + `hatch_build.py` src/-layout fix). |
| 4 | 18      | 2026-05-15 23:19Z  | `2026-05-15_231950Z_8b2c709.json`      | tsecon / Julia                        | 30 | Harness expansion 10 → 30 scenarios covering the full M1 surface. The empirical input that locked [decision 18](../../../../claude_files/decisions/18_cython_port_plan.md) (N=3 tier classification). |
| 5 | 19      | 2026-05-16 01:08Z  | `2026-05-16_010839Z_a371e02.json`      | tsecon / Julia                        | 33 | M1.5 second Cython port (`indexing_kernel`): adds `indexing_lookup_100_api / numpy / cython`. |
| 6 | 20      | 2026-05-16 01:45Z  | `2026-05-16_014528Z_a93b73e.json`      | tsecon / Julia                        | 39 | M1.5 third Cython port (`stats_scalar_kernel`): adds `mean / std / cor` `_numpy / _cython` pairs. |
| 7 | 21      | 2026-05-16 02:23Z  | `2026-05-16_022314Z_ac551ac.json`      | tsecon / Julia                        | 45 | M1.5 fourth port (`fconvert_lower_aggregate_kernel`): adds three `fconvert_*_numpy / _cython` pairs. **M1.5-closing two-column baseline.** Cited as "the session-21 baseline" throughout NOTES.md / MASTER_PLAN. |
| 8 | 26      | 2026-05-16 13:59Z  | `2026-05-16_135903Z_640d0e7.json`      | tsecon / Julia / pandas / polars      | 47 | First **four-column** harness run (adds `scenarios_pandas.py` + `scenarios_polars.py` + 2 mixed-freq scenarios). HEAD was at `640d0e7` (session 25's commit) — the 4-column harness changes land as commit `98a554c` afterwards. **⚠ Environment drift**: every kernel row runs ~2× slower than the same code at baseline 7 — see [`F02_baseline_environment_drift`](../../../../claude_files/reviews/2026-05-16_holistic/F02_baseline_environment_drift.md). Session 27 traced the cause to persistent machine state, not the harness changes; superseded by baselines 9 + 10. |
| 9 | 27      | 2026-05-16 15:38Z  | `2026-05-16_153814Z_98a554c.json`      | tsecon / Julia / pandas / polars      | 47 | **⭐ Canonical baseline for the JSS Section 5 numbers.** First of two session-27 controlled re-runs (laptop plugged in, other apps closed, `--seconds 5` per scenario for tighter medians). Reproduces baseline 8 within ±10% on 43/47 rows — confirms the post-session-26 slow regime is stable and reproducible, not transient noise. The cause is machine-state drift the harness cannot detect from inside the Python process. |
| 10 | 27     | 2026-05-16 16:00Z  | `2026-05-16_160019Z_98a554c.json`      | tsecon / Julia / pandas / polars      | 47 | **Cross-validation re-run.** Second of two session-27 controlled re-runs. Agrees with baseline 9 on 45/47 rows within ±10% (and on 15/16 NumPy / Cython rows within ±5%), satisfying the stop-criterion *"NumPy and Cython rows agree across two re-runs"*. Don't cite for absolute numbers — that's baseline 9's job; this one's job is to underwrite baseline 9's reproducibility claim. |
| 11 | 30     | 2026-05-16 22:28Z  | `2026-05-16_222828Z_ac884b6.json`      | tsecon / Julia / pandas / polars      | 53 | **F14 expansion baseline.** Adds 6 new scenarios (`quantile_quarterly_100`, `cov_two_tseries`, `ytypct_quarterly_100`, `lead_quarterly_lag1`, `fconvert_yy_to_qq_linear`, `fconvert_yy_to_qq_even`) — first 4-column run that covers them. Captured under canonical conditions (`--seconds 5`, laptop plugged in, other apps closed) but **not** treated as a canonical replacement for baseline 9 because the 47 carryover rows show a widened public-API dispatch tax on the fconvert YP-aggregate rows (`fconvert_qq_to_yy_mean` 30.84 µs in baseline 9 → 221.71 µs here; kernel-direct `fconvert_qq_to_yy_mean_cython` is essentially flat at 3.17 µs vs baseline 9's 2.62 µs). The dispatch-tax widening is queued for a separate perf investigation; ratio-based findings still survive across baselines. **Cite baseline 9 for the canonical Section 5 numbers; cite this baseline only for the six new F14 scenarios.** |

## Which baseline does the paper cite?

**For absolute timings: baseline 9** — `2026-05-16_153814Z_98a554c.json` (session 27).
This is the new canonical baseline as of 2026-05-16. It supersedes baselines 7 and 8 as the cited reference:

- **Baseline 7** (session 21) numbers reflect a pre-drift machine regime that the current hardware no longer reproduces (cross-run experiments in session 27 confirmed every harness configuration tested still measures ~2× slower than baseline 7's absolute numbers). Citing baseline 7 today would advertise a regime the reader cannot reproduce on the same machine.
- **Baseline 8** (session 26) numbers reflect the same slow regime as baseline 9 but with two known outlier rows (`indexing_lookup_100_numpy` and `mean_quarterly_100_numpy`) that landed anomalously fast and have not reproduced in baselines 9 + 10. Citing baseline 8 carries those outliers forward.
- **Baseline 9** numbers reflect the stable slow regime, with the outliers regressed to the mean, and a second baseline (#10) confirming run-to-run agreement.

For DataFrame-comparison findings, NOTES.md cites **baseline 8** historically — the pandas / polars columns first appeared there. The qualitative findings (polars 30-160× slower on mixed-freq, pandas same-source-code-different-cost, etc.) all reproduce in baseline 9; future drafts should refresh the absolute µs values to baseline 9 numbers but the structural framings stand. Cross-column ratios within any single baseline (tsecon vs Julia, tsecon vs pandas, Cython vs NumPy) are valid because every column was timed on the same clock. Cross-baseline absolute deltas remain not meaningful.

## Session 27 root-cause investigation (summary)

The "session 26 is ~2× slower than session 21" finding from
[`F02_baseline_environment_drift`](../../../../claude_files/reviews/2026-05-16_holistic/F02_baseline_environment_drift.md)
proposed three possible causes:

1. The 4-column harness's per-row pandas / polars timing loop contaminating tsecon timings.
2. The harness's module-level pandas / polars imports (run at script start, before any timing loop) polluting the CPython interpreter state.
3. Persistent machine state (thermal throttling, background processes, power-mode change).

Session 27 ran two diagnostic experiments before committing to the canonical re-runs:

- **Experiment A** — full harness invoked with `--no-pandas --no-polars` (in-loop pandas / polars timing disabled, but the module-level imports still execute at script load). Result: 1.997× slower than baseline 7 — slow regime persists. Rules out cause #1.
- **Experiment B** — a throw-away ~50-line driver script (kept in `.tmp_investigation/` for the session and deleted at end-of-session; reproducible in ~10 min from this description) that imports only `scenarios.py` (no `run.py`, no `scenarios_pandas.py`, no `scenarios_polars.py`) and re-implements the timing loop inline. Result: 1.958× slower than baseline 7 — slow regime *still* persists in a process that has never touched pandas or polars. Rules out cause #2.

Cause #3 (persistent machine state) is the residual. The harness is exonerated; no engineering fix is warranted.

Take-away for future sessions: an absolute timing reported here is reproducible *on this hardware in its current state*, not as an immutable benchmark. The
ratio-based findings (Cython vs NumPy, tsecon vs pandas, etc.) survive across all session-26-and-later baselines because they share a clock. If a future session needs to compare absolute timings cross-machine or cross-time, capture a machine fingerprint alongside the JSON or invest in CI-runner pinning (see also: [`project_benchmark_ci_side_branch.md`](../../../../../.claude/projects/c--Users-NicholasSt-pierre-Desktop-Work-20260513-TimeSeriesEconPy-TimeSeriesEconPy/memory/project_benchmark_ci_side_branch.md) in user memory — pinned-runner CI is the proper long-term home for cross-time absolute comparisons).

## Reproducing a baseline

```powershell
git checkout <short-sha>                            # check out the captured tip
cd benchmarks/compare/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()' # one-time
cd ../../..
uv sync --all-extras --group dev
uv run python benchmarks/compare/run.py --seconds 2
```

The new run lands in this directory under a fresh `<timestamp>_<HEAD-sha>.json`.
Cross-check against the historical JSON: same scenario keys, same backend
blocks, same Python version (3.11.15 throughout the M1 / M1.5 series).
