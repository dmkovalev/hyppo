# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-07-09

Publication-prep release: packaging, tooling, and documentation brought up
to a publishable state. No behaviour changes to the platform's public
algorithms (hypothesis lattice construction, planning, running).

### Changed

- Distribution renamed to `gedanken` on PyPI (import path stays `hyppo`;
  `pip install gedanken[gui]`).
- `requires-python` lowered to `>=3.11` (no 3.12+-only syntax in the codebase).
- `hyppo.versioning` subpackage extracted from `hyppo.mcp` (`version_store.py`,
  `_db.py`) — breaks the `hyppo.actions` <-> `hyppo.mcp` import cycle.
- `hyppo.storage` fixed: added missing `cloudpickle` dependency, English
  Google-style docstrings, `logging` instead of `print`, roundtrip tests.
- webui packaged: built SPA committed under `hyppo/gui/static/`, served
  from a package-relative path; ships inside the wheel.
- README quickstart corrected (`hyppo-gui` / `hyppo-mcp` console scripts,
  not `hyppo gui`).

### Added

- `mkdocs` + `mkdocstrings` documentation site (quickstart, architecture,
  API reference).
- `ruff` and `mypy` configuration; codebase brought to zero findings on both.
- Recursive import test (`tests/test_package_imports.py`) covering every
  submodule and the `actions`/`mcp` cycle guard.
- Publication metadata: `CITATION.cff`, `CONTRIBUTING.md`, this changelog.

### Removed

- `hyppo.streamlit` (deprecated, zero production importers).
- `hyppo.core._virtual_experiment` (dead dataclass; the OWL `VirtualExperiment`
  class remains the canonical Definition 1 carrier).
- Root clutter: `HOW_TO_USE.md`, stray experiment scripts moved to
  `experiments/norne/`.

### Fixed

- `NotImplementedError` (was `raise NotImplemented()`) in
  `hyppo.core._hypothesis`.
- `datetime.now(UTC)` (was `datetime.utcnow()`) in `hyppo.actions.version`.
- Invalid `# noqa` directive in `hyppo/gui/api/runs.py`.

## [1.0.0] - 2025

Baseline dissertation reference implementation: core hypothesis lattice
platform (Algorithms 1-4, Theorem 1), COA constructor, planner, runner,
GUI, and MCP server, as described in the dissertation.
