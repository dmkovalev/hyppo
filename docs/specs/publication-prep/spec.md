# publication-prep

## Context
`hyppo` — research platform (hypothesis lattices / virtual experiments, dissertation reference code). Core healthy: 329 tests green, 85% coverage, layered architecture. Not publishable as-is: PyPI name taken, broken `storage` module (missing cloudpickle dep), webui not packaged, no linters/CI/docs pipeline, 140 ruff violations, 75 mypy errors, root clutter, missing publication metadata.

## Goals
- G1: installable from PyPI as `gedanken` (USER-APPROVED FINAL; import stays `hyppo`); `pip install gedanken[gui]` gives a working `hyppo-gui` with web UI.
- G2: `ruff check` and `ruff format --check` — 0 findings on `hyppo/` + `tests/`.
- G3: `mypy hyppo` — 0 errors (pragmatic per-module overrides for owlready2 dynamic bases allowed).
- G4: `mkdocs build --strict` green; site = quickstart + architecture + API reference (mkdocstrings, Google-style docstrings, English).
- G5: all modules importable (`hyppo.storage._base` fixed); recursive import test in suite.
- G6: repo root clean: no experiment scripts/results/junk tracked in root; publication artifacts present (CITATION.cff, CHANGELOG, CONTRIBUTING, correct README).
- G7: CI (GitHub Actions): ruff + mypy + pytest (3.11/3.12/3.13) + mkdocs build.
- G8: golden tests stay green throughout (`tests/test_golden_claims.py` — contract with the papers, CLAUDE.md).

## Non-goals
- No new features; no GUI/MCP behaviour changes.
- No consolidation of the three `VirtualExperiment` representations beyond deleting the dead dataclass (OWL class stays the canonical Definition 1).
- No domain-registry abstraction for `actions` (single domain `oil_waterflood` exists; rule of three not met) — documented as known limitation.
- No rewrite of all 190 plain docstrings — Google-style upgrade only for public API of `core`, `runner`, `storage`.
- No i18n docs (English only); Russian papers stay in thesis repo.
- No PyPI upload itself (release is a manual user action).

## Load
Research tool: lattices of 16–~1000 hypotheses, single-user local GUI, no RPS concerns. CI runtime budget: full pytest ~2 min. No perf work needed.

## Design
Distribution name `gedanken` (see G1), top-level package `hyppo` unchanged. README/docs tagline: "gedanken — a platform for virtual (thought) experiments over hypothesis lattices". Version single-sourced: `pyproject.toml` `dynamic = ["version"]` ← `hyppo.__version__` = "1.1.0".

Module changes:
- `hyppo/versioning/` (new): `version_store.py` + `_db.py` moved from `hyppo/mcp/` — breaks the `actions ⇄ mcp` import cycle (`actions/version.py:21` [VERIFIED], `mcp/tools.py:11` [VERIFIED]). `hyppo/mcp` keeps thin re-export shims? No — imports updated at all call sites, no shims (№2).
- `hyppo/storage/_base.py` [VERIFIED :1 imports missing cloudpickle]: add `cloudpickle` to core deps; `print`/`debug=True` → `logging`, `debug` removed; Russian docstrings → English Google style; fake doctests → real examples; `Database` exported from `hyppo/storage/__init__.py`; tests added.
- Deletions: `hyppo/streamlit/` (+ `tests/gui/test_streamlit_deprecated.py`) — zero production importers [VERIFIED recon §7]; `hyppo/core/_virtual_experiment.py` — dead dataclass, zero importers [VERIFIED recon]; `HOW_TO_USE.md` — boilerplate junk.
- Moves: `full_experiment.py`, `norne_battery.py`, `norne_battery_results.json` → `experiments/norne/`; untracked debug scripts `test_c3_*.py` deleted from disk; untracked data files (`_*.csv`, `*.h5`, `*.jar`, …) left as-is (needed by experiments, gitignored).
- Bug fixes: `raise NotImplemented()` → `NotImplementedError` (`core/_hypothesis.py:84,94` [VERIFIED]); `datetime.utcnow()` → `datetime.now(UTC)` (`actions/version.py:83` [VERIFIED]); unclosed sqlite connection (`coa/causal.py:85` [VERIFIED]); invalid noqa (`gui/api/runs.py:17` [VERIFIED recon §1]).
- webui packaging: built frontend copied to `hyppo/gui/static/` at build time; `app.py:43` [VERIFIED parents[2] escape] → serve from `importlib.resources`-safe package path with explicit warning when missing; `hyppo/gui/static/**` in package-data; `webui/dist` stays gitignored, `hyppo/gui/static` tracked (built artifact committed — small SPA, keeps pip install working without node).
- `ontology/__init__.py` [VERIFIED :18-25 import *] → explicit imports (kills 29 F405 + F403).
- Tooling config in `pyproject.toml`: `[tool.ruff]` (target 3.11, E/F/I), `[tool.mypy]` (per-module overrides for `hyppo.ontology.*` owlready2 dynamics), dev deps consolidated into `[project.optional-dependencies].dev` only; new extra `docs` (mkdocs-material, mkdocstrings[python]).
- `requires-python = ">=3.11"` [VERIFIED recon §3: no 3.12+ syntax in code].
- Docs: `mkdocs.yml` + `docs/index.md` (quickstart), `docs/architecture.md`, `docs/api/*.md` (mkdocstrings). Fix stale numbers in `docs/gui_demo_spec.md:24-25` [VERIFIED: 17 edges/depth 5 → must be 18/depth 10 per test_golden_claims.py].
- CI: `.github/workflows/ci.yml` — lint job (ruff, mypy), test matrix (3.11–3.13, `pytest tests -q`), docs job (`mkdocs build --strict`). Java/HermiT needed by tests → `actions/setup-java`.

### Public API
No signature changes. `hyppo.storage.Database` and `hyppo.versioning.version_store` become the new import paths (internal consumers updated).

## Acceptance (EARS)
- WHEN `uv sync --all-extras` then `python -c "import pkgutil,importlib,hyppo; [importlib.import_module(m.name) for m in pkgutil.walk_packages(hyppo.__path__, 'hyppo.')]"` runs, THE SYSTEM SHALL import every module with zero errors.
- WHEN `ruff check hyppo tests` and `ruff format --check hyppo tests` run, THE SYSTEM SHALL report 0 findings.
- WHEN `mypy hyppo` runs with repo config, THE SYSTEM SHALL report 0 errors.
- WHEN `mkdocs build --strict` runs, THE SYSTEM SHALL build the site with 0 warnings.
- WHEN `python -m pytest tests -q` runs, THE SYSTEM SHALL pass 100% (golden tests included), with deleted-module tests removed.
- WHEN the wheel is built (`uv build`), THE SYSTEM SHALL contain `hyppo/gui/static/index.html` and `py.typed`; dist name SHALL be `gedanken`.
- WHEN `hyppo-gui` starts from an installed wheel, THE SYSTEM SHALL serve the SPA (not silently API-only).
- IF `hyppo/gui/static` assets are absent at runtime, THEN THE SYSTEM SHALL log an explicit warning naming the build command.
- WHEN `import hyppo.actions` executes, THE SYSTEM SHALL not import `hyppo.mcp` (cycle broken; assert via import test).

## Testing strategy
- Layers: unit (storage, versioning move, bug fixes), integration (existing suite stays green; wheel-content check), system-smoke (recursive import test), contract (golden tests untouched = G8).
- G1→wheel-content unit test; G5→`tests/test_package_imports.py`; G2/G3/G4→CI commands as criteria (not pytest); G8→existing `test_golden_claims.py`.
- Risk-high zone: version_store move (persistence) → existing `test_version_store_*` must pass unmodified except import paths.
- Classification: breaking (import-path updates + deleted streamlit test); no big-bang R0 needed — obsolete tests deleted within their waves.

## Alternatives
- Rename import to `hyppo_ve` — rejected: breaks all dissertation code/papers for zero benefit (dist/import names may differ).
- Runtime npm build for webui — rejected: forces node on pip users; committing built assets is simpler (№2).
- Fix `hyppo/streamlit` — rejected: deprecated, zero importers.
- Keep dead `core/_virtual_experiment.py` as public API — rejected: zero consumers, OWL class is the Definition 1 carrier; deletion is reversible via git.
- Strict mypy on ontology without overrides — rejected: owlready2 metaclass magic untypeable; overrides documented.
- Sphinx — rejected: user chose mkdocs.

## Risks
- Committing built `hyppo/gui/static` → drift vs webui source; mitigated by CI note + build doc. Accepted.
- mypy fixes in `gui`/`ontology` may need `# type: ignore[<code>]` — allowed with codes, no bare ignores.
- `requires-python >=3.11` claimed but CI is the only 3.11/3.12 verification (local venv is 3.13).
- AFK defaults (name `hyppo-ve`, EN docs, >=3.11, storage fixed) — user may override at GATE; rename wave isolated for cheap redo.
- Node/npm availability for one-time webui build unknown at plan time — executor checks; absent → wave partial, HANDOFF flag.

## Waves
W1 packaging metadata · W2 deletions+moves · W3 micro bug fixes · W4 versioning move (cycle break) · W5 storage fix · W6 import test · W7 ruff sweep · W8 mypy zero · W9 webui packaging · W10 docstrings core/runner · W11 mkdocs site · W12 publication artifacts+README · W13 CI. Detail in plan.md.
