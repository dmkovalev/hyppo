# plan — publication-prep / initial-setup

## Test strategy
Classification: breaking (import-path updates for versioning move; streamlit test deleted with module). No R0: obsolete tests are deleted inside the waves that obsolete them (W2, W4).
Obsolete: tests/gui/test_streamlit_deprecated.py (W2); import paths in tests/conftest.py, tests/test_version_store_*.py, tests/test_action_*.py updated in W4 (all importers of hyppo.mcp.version_store/_db; NOT test_mcp_*.py — those never import version_store).

## Rounds
R1: W1, W2, W3
R2: W4, W5, W6
R3: W7
R4: W8
R5: W9, W10, W12
R6: W11
R7: W13

## Phases
A (foundation): R1, R2
B (quality): R3, R4, R5
C (docs+ci): R6, R7

### W1: packaging metadata [risk: low] [tier: normal] [round: R1] [phase: A] [parallel-with: W2, W3] [policy: keep-green] [layer: unit]
- Goal: pyproject publication-ready; dist renamed; version single-sourced; tooling configs in place.
- Files: pyproject.toml, hyppo/__init__.py, hyppo/py.typed (new), tests/test_packaging.py (new)
- Contracts: [VERIFIED pyproject.toml:26-28 console scripts hyppo-gui/hyppo-mcp]; [VERIFIED recon §3 no 3.12+ syntax]; [VERIFIED recon §6 dev deps duplicated in two sections]
- What: [F1] name="gedanken" (user-approved final), requires-python=">=3.11", license="MIT", authors, readme, urls, classifiers, keywords; [F2] dynamic version from hyppo.__version__ (set to 1.1.0): remove static `version=` from [project], add `dynamic=["version"]` + `[tool.setuptools.dynamic] version={attr="hyppo.__version__"}` (build-backend is setuptools>=75 [VERIFIED pyproject.toml:33-35]); [F3] add cloudpickle to core deps; consolidate dev deps into [project.optional-dependencies].dev only; add extra docs=[mkdocs-material, mkdocstrings[python]]; [F4] [tool.ruff] target-version=py311, lint select E,F,I; [tool.mypy] python_version=3.11 + per-module overrides hyppo.ontology.* (disable_error_code for owlready2 dynamic bases), strict-ish elsewhere; [F5] py.typed + package-data entry.
- Tests: tests/test_packaging.py — version consistency (importlib.metadata vs hyppo.__version__ when installed; fallback: pyproject dynamic present), py.typed exists.
- Tests obsolete: —
- Test impact: 1 new file, ≤3 tests.
- Criterion: `uv sync --all-extras` ok; `python -c "import hyppo; print(hyppo.__version__)"` == 1.1.0; `uv build` produces gedanken-1.1.0 wheel.
- Read-set: spec §Design, §Acceptance; recon HANDOFF §1-3,6.

### W2: deletions and moves [risk: low] [tier: normal] [round: R1] [phase: A] [parallel-with: W1, W3] [policy: delete] [layer: —]
- Goal: dead code and root clutter gone.
- Files: hyppo/streamlit/** (delete), tests/gui/test_streamlit_deprecated.py (delete), hyppo/core/_virtual_experiment.py (delete), HOW_TO_USE.md (delete), full_experiment.py + norne_battery.py + norne_battery_results.json (git mv → experiments/norne/), test_c3_cwa.py + test_c3_cwa_b.py + test_c3_dl.py (delete from disk, untracked), notebooks/test.ipynb (delete — 2-cell draft)
- Contracts: [VERIFIED recon §7 streamlit: zero production importers]; [VERIFIED session: core/_virtual_experiment imported by nobody]; [VERIFIED recon §4 root strays tracked]
- What: delete/move as listed; fix hardcoded `F:\git-repos\diss\...` sys.path lines in moved scripts to relative; grep-verify zero remaining references to deleted modules.
- Tests: `python -m pytest tests -q` full pass after deletion.
- Tests obsolete: tests/gui/test_streamlit_deprecated.py
- Test impact: 1 file deleted.
- Criterion: `grep -rn "streamlit\|_virtual_experiment" hyppo tests examples --include=*.py` → only unrelated hits (none for hyppo.streamlit); pytest green.
- Read-set: spec §Design deletions/moves; recon §4,7.

### W3: micro bug fixes [risk: low] [tier: normal] [round: R1] [phase: A] [parallel-with: W1, W2] [policy: keep-green] [layer: unit]
- Goal: known small defects fixed.
- Files: hyppo/core/_hypothesis.py, hyppo/actions/version.py, hyppo/gui/api/runs.py, tests/conftest.py (only if warning fix lands there)
- Contracts: [VERIFIED core/_hypothesis.py:84,94 raise NotImplemented()]; [VERIFIED actions/version.py:83 datetime.utcnow]; [VERIFIED recon §1 invalid noqa gui/api/runs.py:17]; [VERIFIED critic#2: causal.py:85 is pure algorithm — "unclosed sqlite" ResourceWarning under coverage is owlready2 global world, NOT causal.py]
- What: NotImplementedError; datetime.now(UTC); fix noqa directive; investigate ResourceWarning under `pytest --cov` — if source is owlready2 global world, close it in a session-scoped conftest fixture or document as upstream (one HANDOFF line), do NOT touch coa/.
- Tests: existing suite; add assertion for NotImplementedError type in tests/test_core.py if trivially insertable, else rely on suite.
- Tests obsolete: —
- Test impact: ≤2 files touched.
- Criterion: pytest green; NotImplementedError raised (not TypeError) on the two methods.
- Read-set: spec §Design bug fixes.

### W4: versioning move — break actions⇄mcp cycle [risk: medium] [tier: normal] [round: R2] [phase: A] [parallel-with: W5, W6] [policy: rewrite] [layer: integration]
- Goal: version_store out of mcp; no package cycle.
- Files: hyppo/versioning/ (new: __init__.py, version_store.py, _db.py), hyppo/mcp/version_store.py + hyppo/mcp/_db.py (delete), hyppo/actions/version.py (import update), tests/conftest.py (from hyppo.mcp._db import Base → hyppo.versioning._db), tests/test_version_store_integration.py, tests/test_version_store_contract.py, tests/test_action_register_version.py, tests/test_action_resolve_stale.py, tests/test_action_mark_run.py, tests/test_action_list_versions.py, tests/test_action_get_version.py (all `from hyppo.mcp import version_store` → `from hyppo.versioning import version_store`)
- Contracts: [VERIFIED actions/version.py:21 from hyppo.mcp import version_store]; [VERIFIED mcp/version_store.py:16 imports mcp._db]; [VERIFIED grep: 7 test files + conftest.py import hyppo.mcp.version_store/_db]; [VERIFIED mcp/tools.py:11 imports hyppo.actions — cycle counterpart, one-way, stays; mcp/resources.py + test_mcp_resources.py do NOT import version_store — no edit]
- What: git mv both files to hyppo/versioning/; update all importers (actions + conftest + 6 action-test files monkeypatching version_store — patch target must be the SAME module object actions imports, i.e. hyppo.versioning.version_store); no re-export shims in mcp; update stale path in actions/version.py:4 docstring + version_store.py:4 docstring.
- Tests: existing version-store/mcp tests pass with updated imports; cycle assertion added to tests/test_package_imports.py (W6 owns file — here only verify manually via `python -c "import hyppo.actions; import sys; assert 'hyppo.mcp' not in sys.modules"`).
- Tests obsolete: — (paths updated in place)
- Test impact: conftest + 7 test files, import lines only (monkeypatch-target module path).
- Criterion: pytest green; cycle-free assertion passes.
- Read-set: spec §Design versioning; DECISIONS cycle entry.

### W5: storage fix [risk: low] [tier: normal] [round: R2] [phase: A] [parallel-with: W4, W6] [policy: keep-green] [layer: unit]
- Goal: hyppo.storage importable, tested, clean.
- Files: hyppo/storage/_base.py, hyppo/storage/__init__.py, tests/test_storage.py (new)
- Contracts: [VERIFIED storage/_base.py:1 import cloudpickle missing from deps — dep added in W1]; [VERIFIED :64-65 root=Path('./'), debug=True, print-based logging]
- What: English Google docstrings; logging.getLogger instead of print; drop debug flag; real doctest-free examples; export Database; unit tests (save/load roundtrip in tmp_path, set_root, missing-file error).
- Tests: tests/test_storage.py ≥4 cases.
- Tests obsolete: —
- Test impact: 1 new file.
- Criterion: `python -c "from hyppo.storage import Database"` ok; pytest tests/test_storage.py green.
- Read-set: spec §Design storage; storage/_base.py.

### W6: recursive import test [risk: low] [tier: normal] [round: R2] [phase: A] [parallel-with: W4, W5] [policy: keep-green] [layer: system]
- Goal: every module importable, forever.
- Files: tests/test_package_imports.py (new)
- Contracts: —
- What: pkgutil.walk_packages over hyppo, importlib.import_module each; second test: `import hyppo.actions` does not pull hyppo.mcp (subprocess isolation).
- Tests: the file itself.
- Tests obsolete: —
- Test impact: 1 new file, 2 tests.
- Criterion: pytest tests/test_package_imports.py green (requires W1 cloudpickle + W4 move + W5 fix — same/prior rounds).
- Read-set: spec §Acceptance import bullets.

### W7: ruff sweep [risk: medium] [tier: normal] [round: R3] [phase: B] [parallel-with: —] [policy: keep-green] [layer: —]
- Goal: G2 — ruff check + format 0 findings.
- Files: hyppo/** and tests/** (format touches ~89 files); hyppo/ontology/__init__.py (explicit imports replace import *)
- Contracts: [VERIFIED recon §1: 140 errors — 58 F401, 29 F405, 13 F811, 25 E701; 89 files reformat]; [VERIFIED ontology/__init__.py:18-24 import * (line 25+ is explicit import, keep)]
- What: replace ontology import * with explicit names (kills F403/F405); `ruff check --fix`; manual F811/F401 review (unused imports in __init__ may be intentional re-exports → __all__ or noqa with code); `ruff format`.
- Tests: full suite after sweep.
- Tests obsolete: —
- Test impact: 0 new; whole-tree format diff.
- Criterion: `ruff check hyppo tests` 0; `ruff format --check hyppo tests` 0; pytest green (golden included).
- Read-set: spec §Goals G2; recon §1; [tool.ruff] from W1.

### W8: mypy zero [risk: medium] [tier: high] [round: R4] [phase: B] [parallel-with: —] [policy: keep-green] [layer: —]
- Goal: G3 — mypy hyppo 0 errors.
- Files: hyppo/** (annotations/type: ignore[code] where needed), pyproject.toml ([tool.mypy] override tuning only)
- Contracts: [VERIFIED recon §2: 75 errors in 13 files; dominant: owlready2 invalid base class in ontology, no-redef, var-annotated; concentrated ontology+gui]
- What: per-module overrides for hyppo.ontology.* (owlready2 dynamic bases: disable_error_code=[misc, valid-type] as needed); real fixes in gui/core/others; targeted `# type: ignore[code]` with codes only.
- Tests: full suite.
- Tests obsolete: —
- Test impact: 0.
- Criterion: `mypy hyppo` exit 0; pytest green.
- Read-set: spec §Goals G3, §Risks; recon §2; [tool.mypy] from W1.

### W9: webui packaging [risk: medium] [tier: normal] [round: R5] [phase: B] [parallel-with: W10, W12] [policy: keep-green] [layer: integration]
- Goal: G1 — installed wheel serves the SPA.
- Files: webui/ (build only, no source edits), hyppo/gui/static/** (new, committed build), hyppo/gui/app.py, pyproject.toml (package-data line), tests/gui/test_static.py, webui/README.md (build-refresh note)
- Contracts: [VERIFIED app.py:43 parents[2]/webui/dist escapes package]; [VERIFIED recon §5 dist gitignored, build = tsc -b && vite build]
- What: npm ci + build in webui/ (node availability checked first; absent → partial + HANDOFF); copy dist → hyppo/gui/static/; app.py serves package-relative static dir, logs warning naming `npm run build` when missing; update tests/gui/test_static.py for new path; wheel-content assertion in tests/test_packaging.py.
- Tests: tests/gui/test_static.py; tests/test_packaging.py wheel check.
- Tests obsolete: —
- Test impact: 2 files touched.
- Criterion: `uv build`; unzip wheel contains hyppo/gui/static/index.html; `hyppo-gui` smoke (start, GET / returns HTML, stop).
- Read-set: spec §Design webui; recon §5.

### W10: docstrings core+runner [risk: low] [tier: normal] [round: R5] [phase: B] [parallel-with: W9, W12] [policy: keep-green] [layer: —]
- Goal: public API of core and runner documented Google-style English (mkdocstrings input).
- Files: hyppo/core/_base.py, hyppo/core/_hypothesis.py, hyppo/core/_epistemic.py, hyppo/core/_workflow.py, hyppo/core/__init__.py, hyppo/runner/_base.py
- Contracts: [VERIFIED session: core 12% public docstrings, runner 33%]
- What: Google-style docstrings (Args/Returns/Raises) for public classes/methods; keep dissertation cross-references ("Algorithm N", "Definition N"); no logic changes.
- Tests: existing suite (no behaviour change).
- Tests obsolete: —
- Test impact: 0.
- Criterion: pytest green; ruff green; interrogate-style spot check: every public def/class in listed files has a docstring (grep/AST one-liner in HANDOFF).
- Read-set: spec §Non-goals (scope cap), §Design docs.

### W12: publication artifacts + README [risk: low] [tier: normal] [round: R5] [phase: B] [parallel-with: W9, W10] [policy: keep-green] [layer: —]
- Goal: G6 — publication files present and truthful.
- Files: CITATION.cff (new), CHANGELOG.md (new), CONTRIBUTING.md (new), README.md, docs/gui_demo_spec.md
- Contracts: [VERIFIED README.md:22-23 `hyppo gui` wrong — entry point is hyppo-gui]; [VERIFIED gui_demo_spec.md:24-25 stale 17 edges/depth 5 vs golden 18/depth 10]
- What: CITATION.cff (author dmkovalev, v1.1.0, dissertation reference); CHANGELOG (1.1.0 with publication-prep summary, 1.0.0 baseline); CONTRIBUTING (dev setup via uv, test/lint commands, golden-test contract from CLAUDE.md); README: fix commands (hyppo-gui, hyppo-mcp), `pip install gedanken` quickstart + tagline "gedanken — a platform for virtual (thought) experiments over hypothesis lattices", badges (CI, license, python), link to docs site; gui_demo_spec numbers → 18 edges / depth 10.
- Tests: —
- Tests obsolete: —
- Test impact: 0.
- Criterion: files exist; `grep -n "hyppo gui" README.md` → 0 hits; gui_demo_spec matches golden values.
- Read-set: spec §Goals G6; CLAUDE.md golden contract section.

### W11: mkdocs site [risk: low] [tier: normal] [round: R6] [phase: C] [parallel-with: —] [policy: keep-green] [layer: —]
- Goal: G4 — docs build strict-clean.
- Files: mkdocs.yml (new), docs/index.md (new), docs/architecture.md (new), docs/api/*.md (new), docs/gui_demo_spec.md (nav include as-is)
- Contracts: [ASSUMED mkdocstrings handles Google style via griffe default — verify at execution]
- What: mkdocs-material theme; nav: Home (quickstart from README), Architecture (module map + layering from this iteration's knowledge), API Reference (mkdocstrings pages per subpackage: core, coa, lattice_constructor, planner, runner, manager, storage, versioning, comparison, ontology, actions, adapters, gui, mcp), GUI demo spec; exclude docs/specs and docs/superpowers from nav and strict warnings.
- Tests: —
- Tests obsolete: —
- Test impact: 0.
- Criterion: `mkdocs build --strict` exit 0.
- Read-set: spec §Design docs; W10 HANDOFF.

### W13: CI [risk: low] [tier: normal] [round: R7] [phase: C] [parallel-with: —] [policy: keep-green] [layer: —]
- Goal: G7 — GitHub Actions pipeline.
- Files: .github/workflows/ci.yml (new)
- Contracts: [ASSUMED tests need Java for HermiT — verify: tests/test_owl_reasoning.py imports owlready2 reasoner; setup-java@v4 temurin 17]
- What: jobs: lint (ruff check + format --check + mypy, py3.11), test (matrix 3.11/3.12/3.13, uv sync --all-extras, pytest tests -q, setup-java), docs (mkdocs build --strict). Trigger: push + PR to main.
- Tests: —
- Tests obsolete: —
- Test impact: 0.
- Criterion: `uvx yamllint .github/workflows/ci.yml` (or python yaml.safe_load) clean; every command in ci.yml runs green locally (except matrix pythons — 3.13 only locally).
- Read-set: spec §Goals G7, §Risks python-matrix.
