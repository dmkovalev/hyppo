# Contributing

## Development setup

```bash
uv sync --all-extras
```

This installs `hyppo` (distributed as `gedanken`) with all optional
extras (`gui`, `mcp`, `coa`, `gp`, `data`, `docs`, `dev`).

## Running tests

```bash
# full suite (~2 min)
.venv/Scripts/python -m pytest tests -q

# golden tests only (<1 s) — contract with the dissertation/papers
.venv/Scripts/python -m pytest tests/test_golden_claims.py -q
```

## Lint and type-check

```bash
uvx ruff check hyppo tests
uvx ruff format --check hyppo tests
uvx mypy hyppo
```

All three must report zero findings before a change is merged.

## Golden-test contract

`tests/test_golden_claims.py` pins the dissertation's and papers'
claims (e.g. the Norne hypothesis graph: 16 nodes, 18 edges, DAG depth 10;
Algorithm 2 incremental-equals-full-rebuild; Algorithm 4 plan
correctness and minimality; operation-count complexity bounds; rule 5
acyclicity) to the real implementation.

**Contract**: if you change an algorithm, its golden tests must pass.
If a golden test looks outdated, fix the paper/dissertation text first,
then update the test — never the other way around.

## Code style

- Python: `ruff check --fix` + `ruff format`; type hints on public
  functions; `mypy` clean on touched code.
- Docstrings: Google style, English.
- No wave/round/phase markers or other process artifacts in code, comments,
  or commit trees.
