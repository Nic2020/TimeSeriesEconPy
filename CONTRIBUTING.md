# Contributing to TimeSeriesEconPy

Thanks for your interest. This file describes the dev setup and the rules
for landing a change.

## Setup

You need [`uv`](https://docs.astral.sh/uv/) and Python 3.11 or newer.

```bash
# Install uv (one-time)
# Windows:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo and install dev deps
git clone https://github.com/Nic2020/TimeSeriesEconPy
cd TimeSeriesEconPy
uv sync --all-extras --group dev

# Install pre-commit hooks
uv run pre-commit install
```

## Workflow

1. Create a branch off `main`.
2. Make your changes.
3. Run the checks locally:

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy
   uv run pytest
   ```

4. Open a pull request. CI runs the same checks. A merge requires green CI.

## Code style

- **Linter / formatter:** `ruff`. Configured in `pyproject.toml`. Pre-commit
  hooks auto-format on commit.
- **Type checker:** `mypy --strict` on `src/`. Test code is relaxed.
- **Docstrings:** NumPy style. See existing modules for examples.
- **Tests:** `pytest` + `hypothesis` for property tests. Aim for invariants,
  not just examples.

## Architectural decisions

The design has a series of recorded decisions in
`claude_files/decisions/`. Read those before proposing structural changes.
Open an issue first if you want to revise an accepted decision.

## Reporting bugs

Open a GitHub issue with a minimal reproduction. Please include your Python
version, `tsecon` version, OS, and the smallest failing snippet you can
produce.
