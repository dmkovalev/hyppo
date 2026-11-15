# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.3.1] - 2026-11-15

### Fixed

- Cascade planning no longer repeatedly merges the accumulated recompute set;
  vertices enter the cascade once, restoring O(V + E) work with constant-time
  set operations. Regression checks cover disconnected nodes, shared descendants,
  and preservation of cached independent branches.
- Runner requires both a better combined score and an AIC/AICc margin for
  supersession; Manager forwards `ranking_weights`. Comparable groups use AICc
  when any member has `n/k < 40`; pass `n` and total likelihood parameter count
  `k` (including variance). Legacy metrics without these fields retain AIC.
- Chained keys hash each node once while preserving existing cache digests;
  invalid topological input now raises instead of silently omitting a source.
- Incremental insertion checks reachability from the smaller boundary, rejects
  cycles before mutation, and documents the separate precondition-check cost.
- Regression tests cover ranking disagreement, AICc reversals, cache-key
  compatibility, hashing work, and insertion correctness.

- Rivalry no longer suppresses dependency (definition def:derived): an output
  collision is recorded as `competes`, but the derivability edge
  `Out(h_a) ∩ In(h_b) ≠ ∅` is still built; only a counter-supply (order
  conflict) gives no edge. Before, a rival that reads the other rival's
  output was left out of the cascade and kept a stale cached result.
- Marking completeness (prop:marking): `_cascade` no longer subtracts the
  set of newly invalidated hypotheses, so a hypothesis superseded in the same
  run as its source is still marked stale and recomputed.
- `pending_stale` / `invalid_seen` are persisted in the repository (table
  `marks`): a new Manager on the same database keeps unprocessed marks.
- `Manager._partition` is a single topological pass, O(|V| + |E|)
  (prop:plan_lower), instead of a descendants walk per cache miss.
- `Manager.orchestrate(structures=...)` accepts declared inputs
  (`{"equations": [...], "inputs": {...}}`, condition У2).

### Added

- BGM and HCP experiment adapters, equation-derived graphs, consistency checks,
  and domain regression tests for the dissertation's Chapter 4.
- Persisted revision epochs in chained cache keys prevent reuse of results
  predating a status event, including after a parameter is restored.
- `Manager(plan_mode="safe")` recomputes exactly N (thm:safe);
  `MetadataRepository.evict`. Experiment
  `examples/research/planning/scripts/eviction_safe_vs_cascade.py`.
- `hyppo.planner.mincost.plan_min_cost`: minimum-cost safe plan with read
  costs and a target set via a minimum s-t cut (thm:safe_cost).
- Tests of the plan theorems: `test_safe_plan_theorem`, `test_safe_plan_minimax`
  (worst-case optimality among adaptive plans), `test_planner_mincost`,
  `test_marking_completeness`.
- Level-parallel execution of P_ne: `Runner(max_workers=k)` /
  `Manager(max_workers=k)` runs the hypotheses of one topological level in a
  thread pool; the next level starts after the whole level has finished.
  Repository writes stay in the calling thread. Default `max_workers=1`
  keeps sequential execution (and the timings of the dissertation).

## [1.3.0] - 2026-10-12

Release used for the numbers of the dissertation (chapters 3-4).

### Added

- Algorithm 2 (`HypothesisGraph.add_hypothesis`) is incremental and faithful
  to thm:build (III): reachability check in both directions, fallback to a
  full rebuild when the new task opens new paths.
- Actuality of stored results per definition def:12: chained cache keys
  (`hyppo.planner.actuality`, `Manager.cache_keys`, `cache_keys_for`) and
  StaleHypothesis markers in the plan.
- Bridge epistemic status -> `InvalidHypothesis` -> RL cascade -> next plan
  (`apply_epistemic_status`, `Manager.pending_stale`); the cascade fires on
  a status change. Witness test of hidden staleness (prop:hidden_stale).
- Condition 5 as a supply map; supply cycles reported as merge candidates.
- Non-terminal derived cascade (thm:integrity IV); baseline experiment
  against a file cache (`examples/research/baseline_filecache`).

### Changed

- Algorithm 1 follows the dissertation's derivability (edge only along the
  task order, condition У3 has priority).
- Planning benchmarks use declared inputs; ER density matched to BA(m=2).

### Changed (breaking)

- "Lattice" terminology removed: the hypothesis structure is a graph
  (derivability arcs form a DAG), not an algebraic lattice.
  - package `hyppo.lattice_constructor` -> `hyppo.graph_constructor`;
    class `HypothesisLattice` -> `GraphConstructor`; `build_lattice()` ->
    `build_graph()`; attribute `.lattice` -> `.hyp_graph`;
  - `MetadataRepository.save_lattice` / `find_nearest_lattice` ->
    `save_graph` / `find_nearest_graph`; SQLite table `lattices` -> `graphs`
    (`lattice_id` -> `graph_id`);
  - action `GetHypothesisLattice` -> `GetHypothesisGraph`
    (`get_hypothesis_graph`, `GetHypothesisGraphInput`, `GraphPayload`,
    `GraphEdge`); MCP persona `lattice_steward` -> `graph_steward`;
  - ontology `ResearchLattice` -> `ResearchGraph`,
    `has_for_lattice_hypothesis` -> `has_for_graph_hypothesis`;
  - keyword arguments and dict keys `lattice` -> `hyp_graph`,
    `lattice_edges` -> `graph_edges`.

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
