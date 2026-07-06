# Hyppo — Hypothesis Platform for Virtual Experiments

Reference implementation of the virtual experiment management platform
described in Chapter 3 of the dissertation.

## Installation

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## GUI

Launch the web GUI locally:

    pip install "hyppo[gui]"
    hyppo gui

Opens a browser at http://127.0.0.1:8787 with a preloaded `norne-brugge`
demo. Walk the full virtual-experiment lifecycle: define hypotheses →
graph → plan → run → compare → iterate. The legacy `hyppo.streamlit`
module is deprecated.

## Architecture

8 components corresponding to Section 3.1:

| Component | Module | Section |
|-----------|--------|---------|
| Core | `hyppo.core` | 3.1.1 |
| Manager | `hyppo.manager` | 3.1.2 |
| HypothesisGenerator | `hyppo.generator` | 3.1.3 |
| COAConstructor | `hyppo.coa` | 3.1.4 |
| LatticesConstructor | `hyppo.lattice_constructor` | 3.1.5 |
| Planner | `hyppo.planner` | 3.1.6 |
| Runner | `hyppo.runner` | 3.1.7 |
| MetadataRepository | `hyppo.metadata_repository` | 3.1.8 |

## MCP server

Hyppo exposes 8 typed actions and a `Lattice Steward` persona via MCP.

```bash
# stdio (Claude Code / Desktop)
uv run python -m hyppo.mcp

# streamable HTTP for cross-MCP callers (e.g. the wfonto bridge)
uv run python -m hyppo.mcp --transport http --port 8082
```

After connecting, clients see tools `mcp__hyppo__BuildVirtualExperiment`,
`...DiffHypothesisStates`, `...RegisterHypothesisVersion`, etc., and the
persona resource `hyppo://personas/lattice_steward.md`.

Write actions (`RegisterHypothesisVersion`, `MarkRunWithVersion`) require
the `hypothesis_version` + `hypothesis_run_link` tables to be present in
the wfdb-backed Postgres. Until that migration ships (sub-project B of the
2026-05-25 hyppo×wfonto bridge spec), write actions return a structured
`{"error":"not_implemented", "detail":"blocked on sub-project B"}` and
non-blocking agents route around them.
