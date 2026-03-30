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
