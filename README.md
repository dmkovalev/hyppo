# gedanken

[![CI](https://github.com/dmkovalev/hyppo-ref/actions/workflows/ci.yml/badge.svg)](https://github.com/dmkovalev/hyppo-ref/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)

> 🇷🇺 [Русская версия](README.ru.md)

> gedanken — a platform for virtual (thought) experiments over hypothesis
> lattices

Reference implementation of the virtual experiment management platform
described in the dissertation. The distribution is published as
`gedanken` on PyPI; the import path remains `hyppo`.

Documentation site: <https://dmkovalev.github.io/hyppo-ref/> (placeholder —
published once the `mkdocs` deployment workflow ships).

## What is this?

Scientific and engineering models are usually explored as competing
families of hypotheses — alternative equations for the same phenomenon,
each with its own assumptions and cost to (re)compute. `gedanken` builds a
dependency lattice over such hypotheses (Algorithm 1), plans the minimal
set of models that must be recomputed after a change (Algorithm 4), runs
and caches results, and tracks each hypothesis's epistemic status
(competing, superseded, corroborated) as evidence accumulates.

## Installation

```bash
pip install gedanken
```

For local development:

```bash
uv sync --all-extras
```

This installs `hyppo` (distributed as `gedanken`) with all optional
extras (`gui`, `mcp`, `coa`, `gp`, `data`, `docs`, `dev`).

## Quickstart

```python
from hyppo.coa._base import Equation, Structure
from hyppo.lattice_constructor._base import HypothesisLattice
from hyppo.planner._base import Planner
from hyppo.storage._base import Database


class Hypothesis:
    def __init__(self, name, formula):
        self.name = name
        self.structure = Structure([Equation(formula=formula)])


class Workflow:
    def __init__(self, tasks):
        self._tasks = tasks

    def get_tasks(self):
        return self._tasks


h1 = Hypothesis("H1", "q = a*p")     # q is the output of H1
h2 = Hypothesis("H2", "wct = q*2")   # wct depends on q -> H2 derived_by H1

workflow = Workflow([[h1], [h2]])
lattice = HypothesisLattice([h1, h2], workflow)
G = lattice.lattice
print(G.number_of_nodes(), G.number_of_edges())  # 2 1

Database.set_root(".hyppo_demo_db")
plan = Planner(db=Database).plan(configuration={}, lattice=lattice)
print(sorted(h.name for h in plan.needs_execution))  # ['H1', 'H2']
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

Opens a browser at http://127.0.0.1:8787 with the bundled Norne demo
(16 hypotheses, oil waterflooding). Walk the full virtual-experiment
lifecycle: define hypotheses -> graph -> plan -> run -> compare -> iterate.

## Architecture

| Component | Module | Purpose |
|-----------|--------|---------|
| Core | `hyppo.core` | OWL ontology + epistemic status of hypotheses |
| Manager | `hyppo.manager` | Lifecycle orchestration of virtual experiments |
| HypothesisGenerator | `hyppo.generator` | Baseline hypothesis generation (linear regression + optional genetic programming) |
| COAConstructor | `hyppo.coa` | Causal ordering of equation systems |
| LatticesConstructor | `hyppo.lattice_constructor` | Algorithm 1 — hypothesis lattice |
| Planner | `hyppo.planner` | Minimal recomputation plan (Algorithm 4) |
| Runner | `hyppo.runner` | Model execution with retries |
| MetadataRepository | `hyppo.metadata_repository` | Result cache shared by planner and runner |
| Versioning | `hyppo.versioning` | Hypothesis/run version tracking |
| Actions | `hyppo.actions` | Typed operations exposed to callers |
| MCP server | `hyppo.mcp` | MCP tool + resource surface (see below) |
| GUI | `hyppo.gui` | Web GUI (see above) |

## MCP server

Hyppo exposes 8 typed actions and a `Lattice Steward` persona via MCP.

```bash
# stdio (Claude Code / Desktop)
hyppo-mcp

# streamable HTTP for cross-MCP callers
hyppo-mcp --transport http --port 8082
```

After connecting, clients see tools `mcp__hyppo__BuildVirtualExperiment`,
`...DiffHypothesisStates`, `...RegisterHypothesisVersion`, etc., and the
persona resource `hyppo://personas/lattice_steward.md`.

Write actions (`RegisterHypothesisVersion`, `MarkRunWithVersion`) persist
to an embedded SQLite store (aiosqlite, file `hyppo_versions.db` in the
current working directory by default) out of the box — no setup needed.
Set `DATABASE_URL` to point at an external database (e.g. Postgres) for
production deployments.

## How to cite

See [`CITATION.cff`](CITATION.cff) — reference implementation
accompanying the dissertation of D. Kovalev.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, lint/type-check
commands, and test layout. Golden-test contract: `tests/test_golden_claims.py`
pins the dissertation's and papers' claims to the implementation — if you
change an algorithm, its golden tests must pass; if a golden test looks
outdated, fix the paper/dissertation text first, then the test.

## License

MIT — see [`LICENSE`](LICENSE).

See also [`CHANGELOG.md`](CHANGELOG.md) for release history.
