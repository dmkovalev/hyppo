# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.1] - 2026-07-09

Audit-fix release: closes five structural defects found by an adversarial
review of the 1.1.0 codebase (versioning was broken out of the box, the
recomputation cascade was implemented four times with diverging semantics,
AIC/BIC were computed in three places under stringly-typed metrics dicts,
the ontology used a single module-global owlready2 world, and a data blob
lived inside the importable package). No public algorithm changes beyond
the ones listed below.

### Fixed

- `hyppo.versioning`: engine-per-URL caching + automatic schema creation on
  first use; default persistence now writes to a file `hyppo_versions.db`
  in the current working directory (previously an in-memory database
  recreated on every call, so every write was invisible to the next read —
  "no such table" on first real use). `DATABASE_URL` still overrides.
- `hyppo.manager`: recomputation-set semantics now match `hyppo.planner`
  exactly — a cache miss cascades to all `derived_by` descendants, and a
  cached result with `r2` below threshold prunes the hypothesis and its
  descendants from both the recompute and cached-result sets. Previously
  a cache miss did not cascade and a low-R² hit did not prune descendants.
  `gui.run_iteration` can now return fewer hypotheses when a subtree is
  pruned this way.
- `hyppo.storage.Database.save` now re-raises pickling failures instead of
  swallowing them silently.

### Changed

- Unified the pure two-way recomputation cascade to a single
  implementation, `hyppo.coa.HypothesisGraph.plan`, exposed as
  `hyppo.coa.plan_cascade`; the GUI's plan-preview delegates to it instead
  of carrying its own duplicate traversal.
- Unified AIC/BIC computation to a single source (`hyppo.comparison`);
  `hyppo.core` exposes thin delegating `get_aic`/`get_bic`. Removed the
  unused, non-standard `NonLinearModel.compute_aic`/`compute_bic` (no
  callers).
- Introduced `hyppo.core.Metrics` (a `TypedDict` for `r2`/`aic`/`bic`/`mse`)
  and typed the ~20 sites that read or write model metrics with it.
- `hyppo` ontology/schema definition extracted into factory functions
  (`define_ve_schema`, `create_ve_world`) so a virtual experiment can be
  built in a fresh, isolated owlready2 `World`;
  `norne_adapter.build_oil_virtual_experiment` is now safely re-callable
  within one process. The default-world `virtual_experiment_onto` used by
  the golden tests is unchanged.
- Moved `hyppo/gui/real_data.json` out of the importable package to
  `scripts/gui_real_data.json`; the GUI's real-data API resolves it via
  `HYPPO_REAL_DATA`, then the scripts path, then a 404 fallback.

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
