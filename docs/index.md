# gedanken

> gedanken — a platform for virtual (thought) experiments over hypothesis
> lattices

Reference implementation of the virtual experiment management platform
described in Chapter 3 of the dissertation. The distribution is published
as `gedanken` on PyPI; the import path remains `hyppo`.

## Installation

```bash
pip install gedanken
```

For local development:

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## GUI

Launch the web GUI locally:

```bash
pip install "gedanken[gui]"
hyppo-gui
```

Opens a browser at http://127.0.0.1:8787 with a preloaded `norne-brugge`
demo. Walk the full virtual-experiment lifecycle: define hypotheses →
graph → plan → run → compare → iterate.

## MCP server

```bash
# stdio (Claude Code / Desktop)
hyppo-mcp

# streamable HTTP for cross-MCP callers
hyppo-mcp --transport http --port 8082
```

## Golden-test contract

`tests/test_golden_claims.py` pins every checkable claim in the
dissertation papers (Algorithm 1 graph shape, Algorithm 2 incremental ==
full rebuild, Algorithm 4 plan correctness/minimality, Theorem 1,
operation-count complexity, procedural acyclicity) to a real platform
call. Changing an algorithm requires the golden tests to still pass; if a
test is stale, the paper text is corrected first, then the test.

See [Architecture](architecture.md) for the module map and
[API Reference](api/core.md) for generated docstrings.
