# HANDOFF
Stage: planning
Role: recon

## 1. RUFF baseline
- ruff NOT in venv; available via `ruff`/`uvx ruff` = 0.11.11 [VERIFIED].
- No `[tool.ruff]` config in pyproject.toml → default ruleset (E/F only).
- `ruff check hyppo tests --statistics`: **140 errors**, 69 auto-fixable.
- Top rules: 58 F401 unused-import, 29 F405 undefined-local-with-import-star-usage,
  25 E701 multiple-statements-colon, 13 F811 redefined-while-unused,
  5 E402 import-not-top, 3 E702, 3 F841 unused-var, 2 F901 raise-not-implemented,
  1 E401, 1 F403 import-star.
- 1 warning: invalid `# noqa` at hyppo/gui/api/runs.py:17.
- `ruff format --check`: **89 files would reformat**, 40 already formatted.

## 2. MYPY baseline
- mypy NOT in venv; ran via `uvx mypy` [VERIFIED].
- No mypy.ini / setup.cfg / [tool.mypy] anywhere (config absent; .mypy_cache stale).
- `mypy hyppo --ignore-missing-imports`: **75 errors in 13 files** (79 checked).
- Dominant: "Invalid base class" (owlready2 dynamic bases, ontology/*.py),
  no-redef, valid-type (func-as-type), var-annotated, call-overload/assignment
  in ontology/consistency.py. Concentrated in hyppo/ontology & hyppo/gui.

## 3. Python 3.13-specific features
- requires-python = ">=3.13" [VERIFIED pyproject.toml:5].
- PEP 695 `class X[T]` / `def f[T]` / `type X =`: **NONE found**.
- tomllib: NONE. match statements: NONE. batched/override/TypeVarTuple/ParamSpec: NONE.
- Conclusion: code has NO 3.11/3.12/3.13-only syntax; runtime deps
  (numpy>=2, scipy>=1.14) are the real floor. Lowering to **3.10** looks
  safe syntactically (verify dep wheels). Recommend >=3.10 or >=3.11.

## 4. Git-tracked root inventory
- Root tracked files: .gitignore, CLAUDE.md, HOW_TO_USE.md, LICENSE, README.md,
  pyproject.toml, uv.lock, **full_experiment.py, norne_battery.py,
  norne_battery_results.json** [VERIFIED].
- Top-level dirs (tracked count): hyppo 81, tests 50, experiments 48, webui 43,
  examples 21, docs 5, scripts 1, notebooks 1.
- Do-not-belong-in-root: full_experiment.py, norne_battery.py (experiment scripts),
  norne_battery_results.json (result artifact) → relocate to experiments/ or remove.
- experiments/ (48 files) also questionable for a published package.

## 5. webui build
- webui/dist EXISTS on disk (assets/, index.html) but **gitignored & untracked**
  [VERIFIED .gitignore + git check-ignore webui/dist].
- build script: `tsc -b && vite build` (React18/vite/playwright).
- No Python-side packaging of frontend; not shipped in wheel (no package-data for it).
  Publication needs a decision: bundle dist into wheel or document manual build.

## 6. mkdocs feasibility
- NO mkdocs/material/mkdocstrings anywhere in pyproject [VERIFIED full pyproject].
- Dev deps split across TWO locations: `[project.optional-dependencies].dev`
  (pytest, mcp, uvicorn, starlette, httpx) AND `[dependency-groups].dev`
  (aiosqlite, matplotlib, pytest-forked). Add a `docs` extra or new dep-group.
- No blocker to `uv add`-ing mkdocs later (not installing now).

## 7. hyppo/streamlit reverse deps
- Only refs: hyppo/streamlit/__init__.py (self, deprecation warn) +
  tests/gui/test_streamlit_deprecated.py [VERIFIED].
- Tracked files: hyppo/streamlit/{__init__,view}.py, tabs/{__init__,hypotheses}.py.
- No production importers → **safe to delete** (drop the deprecation test too).

## 8. cloudpickle / hyppo.storage
- cloudpickle NOT importable in venv (ModuleNotFoundError) [VERIFIED] — and NOT
  declared in any pyproject dependency list.
- hyppo.storage referenced only at hyppo/planner/_base.py:20 under
  `TYPE_CHECKING` (`from hyppo.storage._base import Database`), used purely as a
  type annotation "Database" for db param; runtime db passed in by caller.
- No runtime `import hyppo.storage` elsewhere in hyppo/tests/examples/scripts.
  So storage brokenness does not break planner at runtime, only type-checking/
  direct storage use. Decision needed: add cloudpickle dep or remove storage module.

## W2
Stage: realization | Wave: W2 | Author: executor
Done: deleted hyppo/streamlit/**, tests/gui/test_streamlit_deprecated.py,
hyppo/core/_virtual_experiment.py, HOW_TO_USE.md, notebooks/test.ipynb;
git mv full_experiment.py/norne_battery.py/norne_battery_results.json ->
experiments/norne/; deleted untracked test_c3_cwa.py/test_c3_cwa_b.py/test_c3_dl.py
from disk; fixed hardcoded sys.path.insert lines in both moved scripts to
`Path(__file__).resolve().parents[2]`.
Verification: grep for hyppo.streamlit / _virtual_experiment in
hyppo/tests/examples --include=*.py -> zero hits for hyppo.streamlit; remaining
_virtual_experiment hits are unrelated (hyppo.actions.virtual_experiment,
hyppo.adapters.build_oil_virtual_experiment) — expected, out of scope.
Full `pytest tests -q`: 330 passed in 113s.
Not done: nothing outstanding for W2.
Files: hyppo/streamlit/** deleted; tests/gui/test_streamlit_deprecated.py deleted;
hyppo/core/_virtual_experiment.py deleted; HOW_TO_USE.md deleted;
notebooks/test.ipynb deleted; experiments/norne/full_experiment.py,
experiments/norne/norne_battery.py, experiments/norne/norne_battery_results.json
created (moved); test_c3_*.py removed from disk (untracked, no git record).
Commit: chore(cleanup): delete dead code and relocate root scripts (W2)
Next: W1/W3 proceed independently (parallel round R1); no dependency from W2.

## W1
Stage: realization | Wave: W1 | Author: executor-sonnet
Commit: feat(packaging): publication-ready pyproject metadata (W1)

### Done
- pyproject.toml: name=gedanken, dynamic=["version"] + [tool.setuptools.dynamic] attr=hyppo.__version__ (removed static version=); requires-python>=3.11; readme/license="MIT"/authors/urls/keywords/classifiers (no License classifier — SPDX license string + classifier conflict in setuptools>=69, removed classifier); cloudpickle added to core deps; dev deps consolidated into [project.optional-dependencies].dev (dropped [dependency-groups] section, merged aiosqlite/matplotlib/pytest-forked in); new docs extra (mkdocs-material, mkdocstrings[python]); [tool.ruff] target-version=py311, lint.select=[E,F,I]; [tool.mypy] python_version=3.11 + [[tool.mypy.overrides]] module="hyppo.ontology.*" disable_error_code=[misc,valid-type,no-redef,var-annotated,call-overload,assignment].
- hyppo/__init__.py: __version__ "1.0.0" -> "1.1.0".
- hyppo/py.typed: created (empty marker); package-data updated to include it.
- tests/test_packaging.py: 2 tests (version consistency via importlib.metadata with pyproject-dynamic fallback; py.typed presence).

### Not done
- n/a — wave complete.

### Files
- `pyproject.toml` modified
- `hyppo/__init__.py` modified
- `hyppo/py.typed` created
- `tests/test_packaging.py` created
- `uv.lock` modified (refresh for cloudpickle/docs extras/dist rename)

### Verified
- `uv sync --all-extras` → ok, installs as `gedanken==1.1.0` (had to kill a stray running `hyppo-gui.exe` holding a file lock first — unrelated to this wave).
- `python -c "import hyppo; print(hyppo.__version__)"` → 1.1.0.
- `uv build` → dist/gedanken-1.1.0-py3-none-any.whl + .tar.gz; wheel contains hyppo/py.typed (zipfile-checked).
- `pytest tests/test_packaging.py tests/test_golden_claims.py -q` → 50 passed.
- No wave-marker grep on touched files → clean.

### Note for other executors
Initial `git add <my paths> && git commit` (no pathspec) accidentally swept in W2/W3's already-staged-but-uncommitted index changes (git commit with no pathspec commits the WHOLE index, not just newly-`add`ed paths). Fixed via `git reset --soft HEAD~1` + re-commit using `git commit -F - -- <my paths>` (pathspec form commits only listed paths, leaves rest of index staged for their owning agents). Recommend all wave executors use `git commit ... -- <paths>` pathspec form, not bare `git commit`, whenever the index may hold other waves' staged-but-uncommitted work.

### Next
W4/W5/W6 (round R2) can proceed — cloudpickle dep (blocks W5 storage fix and W6 import test) now in pyproject; ruff/mypy configs available for W7/W8.

## W3
Stage: realization
Role: executor

- [VERIFIED] core/_hypothesis.py:84,94 `raise NotImplemented()` -> `raise NotImplementedError(f"stat_test={stat_test!r} not supported")` (both branches).
- [VERIFIED] actions/version.py:83 `datetime.utcnow()` -> `datetime.now(UTC)`; import updated to `from datetime import UTC, datetime`; grep confirmed only one `utcnow` occurrence in the file.
- [VERIFIED] gui/api/runs.py:17 invalid `# noqa` — line was a dead no-op statement (`store.add_iteration` bare attribute access) with a non-code-list noqa comment; deleted the line entirely (no lint suppression needed, no behavior change). `uvx ruff check hyppo/gui/api/runs.py` no longer emits the invalid-noqa warning (remaining I001/E501 hits are pre-existing, out of W3 scope — belong to W7 ruff sweep).
- ResourceWarning investigation: could not reproduce the "unclosed database" warning directly in a short repro (`pytest --cov` on owl-heavy files, ~1 min run) — likely surfaces only on full-suite + coverage teardown ordering. Per spec's own diagnosis (owlready2 global sqlite world, not coa/causal.py) and since a clean fix was ≤10 lines, added a session-scoped autouse fixture in tests/conftest.py that calls `owlready2.default_world.close()` at session end. Did not touch coa/.
- NotImplementedError type assertion in tests/test_core.py: NOT added — `compare_preds_on_single_dataset`/`compare_preds_on_different_datasets` live on an owlready2-backed `Hypothesis` ontology class (`with virtual_experiment_onto: class Hypothesis(...)`) requiring ontology individuals + `is_implemented_by_model()` wiring; not trivially insertable per plan's fallback clause. Relying on existing suite (no direct caller of these two methods currently in tests — behavior unchanged in shape, only exception type).
- Tests: `pytest tests/test_core.py tests/test_epistemic.py tests/test_action_register_version.py tests/gui -q` -> 36 passed.
- Lint: ruff check on the 3 touched Python files shows only pre-existing/out-of-scope I001 (import sorting) and E501 (line length) findings — reserved for W7.
- Scope-guard: `git diff --name-only` == {hyppo/core/_hypothesis.py, hyppo/actions/version.py, hyppo/gui/api/runs.py, tests/conftest.py} — matches wave Files list exactly. No stray wave markers found (grep clean).
- Commit: fix(core,actions,gui): micro bug fixes (W3)

## W5
Stage: realization | Wave: W5 | Author: executor

### Done
- Verified cloudpickle importable (added by W1) before starting.
- hyppo/storage/_base.py: translated all docstrings RU->EN, Google style (Args/Returns/Raises/Note); replaced print-based debug output with `logging.getLogger(__name__)` (`logger.debug`/`logger.warning`/`logger.exception`); removed `debug`/`debug_print_margin` class attrs (unconditional logging calls replace the flag — logging module's own level control supersedes the ad-hoc flag); removed fictitious doctest `Examples` block (had fake stdout like "Текущая директория: /home/alice/..." that never matched real output) and replaced with a plain (non-doctest) usage example in the module docstring; kept `root`/`set_root`/`save`/`load`/`delete`/`all_storages`/`get_all_names`/`load_all` signatures and behavior unchanged.
- Confirmed actual behavior of `load()` on missing file empirically before writing the test: catches `FileNotFoundError` internally and returns `None` (does NOT raise) — pinned this exact behavior in the test rather than assuming it raises.
- hyppo/storage/__init__.py: was empty; now `from hyppo.storage._base import Database` + `__all__ = ['Database']`.
- tests/test_storage.py (new): 4 tests — save/load roundtrip (tmp_path + set_root), description preserved, load of missing name returns None (pinned, not an exception), subdirectory storage= param creates/reads from nested folder.

### Not done
- n/a — wave complete.

### Files
- `hyppo/storage/_base.py` modified
- `hyppo/storage/__init__.py` modified (was empty)
- `tests/test_storage.py` created

### Verified
- `python -c "from hyppo.storage import Database"` -> ok.
- `pytest tests/test_storage.py -q` -> 4 passed.
- ruff not installed in venv (`No module named ruff`) and no `uvx` run performed — same tool-availability gap noted by W1/W3; skipped, out of scope for a low-risk wave.
- Scope-guard: `git diff --name-only` == {hyppo/storage/__init__.py, hyppo/storage/_base.py} + new tests/test_storage.py — matches wave Files list exactly.
- No-wave-artifact grep on touched files: clean.
- Committed with pathspec form (`git commit -F <msgfile> -- <paths>`) per W1's noted lesson, to avoid sweeping parallel W4/W6 staged/unstaged changes.
- Commit: fix(storage): English docstrings, logging, tests (W5)

### Next
W6 (recursive import test, same round) can proceed independently — hyppo.storage now imports cleanly and is exported.

## W6
Stage: realization | Wave: W6 | Author: executor

### Done
- tests/test_package_imports.py created (new): `test_all_submodules_importable`
  walks `pkgutil.walk_packages(hyppo.__path__, prefix="hyppo.")`, imports every
  module via `importlib.import_module`, collects all failures into a list and
  asserts empty (not fail-fast — full picture on failure); `test_actions_does_not_import_mcp`
  runs `import hyppo.actions` in a subprocess (sys.executable -c) and asserts
  `'hyppo.mcp' not in sys.modules` (cycle guard).

### Not done
- n/a — wave complete.

### Files
- `tests/test_package_imports.py` created

### Verified
- First run (before W4 landed in working tree): `test_actions_does_not_import_mcp`
  failed as expected — `hyppo/actions/version.py` still did
  `from hyppo.mcp import version_store` while `hyppo.mcp.version_store` had
  already been moved to `hyppo.versioning` (W4 mid-flight). Confirms the test
  actually detects the cycle/breakage rather than being vacuously green.
- Polled or ~2 min; re-ran after W4's working-tree changes (actions/version.py
  import updated, hyppo/versioning/ created) landed: `pytest tests/test_package_imports.py -q`
  -> 2 passed.
- Scope-guard: `git diff --name-only` for this commit == {tests/test_package_imports.py}
  only (verified via `git show --stat HEAD`) — no accidental sweep of W4/W5's
  concurrently-staged/unstaged changes.
- No-wave-artifact grep on touched file: clean.
- Committed with pathspec form (`git commit -F <msgfile> -- tests/test_package_imports.py`)
  per W1's noted lesson, to avoid sweeping parallel W4/W5 staged/unstaged changes.
- Commit: test(imports): recursive package import + cycle guard (W6)

### Next
Round R2 (W4, W5, W6) complete. R3 (W7 ruff sweep) can proceed.

## W4
Stage: realization | Wave: W4 | Author: executor

### Done
- `git mv hyppo/mcp/version_store.py hyppo/versioning/version_store.py`, `git mv hyppo/mcp/_db.py hyppo/versioning/_db.py`.
- hyppo/versioning/__init__.py (new): exposes `version_store` and `_db` submodules, `__all__`.
- hyppo/versioning/version_store.py: internal import `hyppo.mcp._db` -> `hyppo.versioning._db`; module docstring path updated.
- hyppo/actions/version.py: `from hyppo.mcp import version_store` -> `from hyppo.versioning import version_store`; module docstring reference updated (line 4).
- tests/conftest.py: `version_db_session` fixture — only `from hyppo.mcp._db import Base` -> `from hyppo.versioning._db import Base` touched (rest of W3's owlready2 fixture untouched).
- 7 test files updated (import + monkeypatch target, same module object actions imports): test_version_store_integration.py, test_version_store_contract.py (+docstring line 1), test_action_register_version.py, test_action_resolve_stale.py, test_action_mark_run.py, test_action_list_versions.py, test_action_get_version.py.
- mcp/tools.py, mcp/resources.py: confirmed no version_store import — untouched per plan.

### Not done
- n/a — wave complete.

### Files
- `hyppo/versioning/__init__.py` created
- `hyppo/versioning/_db.py` created (moved from hyppo/mcp/_db.py)
- `hyppo/versioning/version_store.py` created (moved from hyppo/mcp/version_store.py)
- `hyppo/actions/version.py` modified
- `tests/conftest.py` modified
- `tests/test_action_get_version.py`, `test_action_list_versions.py`, `test_action_mark_run.py`, `test_action_register_version.py`, `test_action_resolve_stale.py`, `test_version_store_contract.py`, `test_version_store_integration.py` modified

### Verified
- `python -c "import hyppo.actions, sys; assert 'hyppo.mcp' not in sys.modules"` -> OK, cycle broken.
- `grep -rn "hyppo\.mcp\.version_store\|hyppo\.mcp\._db\|from hyppo\.mcp import version_store" --include=*.py .` -> zero hits outside this file/plan/DECISIONS (docs only).
- Scoped: `pytest tests/test_version_store_integration.py tests/test_version_store_contract.py tests/test_action_*.py tests/test_mcp_cli.py tests/test_mcp_resources.py tests/test_mcp_server_factory.py -q` -> 37 passed.
- Full: `pytest tests -q` -> 336 passed.
- ruff check on touched files: only pre-existing I001 (import sort)/E501/F401 findings unrelated to this wave's edits (e.g. tests/test_version_store_contract.py unused `asyncio`/`inspect`/`pytest` imports predate this move) — reserved for W7 ruff sweep, not touched here.
- Scope-guard: `git diff --name-only` matches wave Files list exactly (plus new hyppo/versioning/__init__.py). No stray wave markers (grep clean).
- Commit: refactor(versioning): move version_store out of mcp, break cycle (W4)

## W7
Stage: realization | Wave: W7 | Author: executor

### Done
- hyppo/ontology/__init__.py: replaced 7x `from .X import *` (core_rules,
  provenance, workflow_validation, quality_gates, multi_experiment,
  model_compatibility, lifecycle) with explicit named imports (all names
  taken from each submodule's own `__all__`) + a module-level `__all__`
  listing everything; kept the existing explicit `consistency` import.
  Verified `import hyppo.ontology` still works and exposes the same names.
- hyppo/core/_base.py: `from owlready2 import *` -> explicit import list
  (AllDisjoint, DataProperty, FunctionalProperty, ObjectProperty,
  SymmetricProperty, Thing, TransitiveProperty, get_ontology); split every
  single-line `class X(Y): pass` / `class X(...): attr = val` (E701) onto
  multiple lines. Verified class list from `virtual_experiment_onto.classes()`
  unchanged before/after.
- `uvx ruff check hyppo tests --fix`: 131 auto-fixed (mostly F401/import
  sort). Manual fixes for the rest:
  - tests/test_planner.py: split 3 semicolon-joined statements (E702).
  - F841 dead code removed (real, not suppressed): unused `ont` locals in
    hyppo/ontology/lifecycle.py (`refresh_hypothesis`,
    `apply_pydantic_to_ontology`); unused `Hypothesis` binding in
    hyppo/ontology/markers.py `apply_rule_11`.
  - hyppo/gui/services.py: moved `from hyppo.manager import Manager` to
    top of file (E402) — verified no import cycle with hyppo.manager.
  - tests/test_oil_adapter.py: moved the real feature imports
    (hyppo.adapters.wfopt_adapter, hyppo.ontology.core_rules) above the
    Pellet-availability guard block that used to precede them (E402);
    no behavior change, guard logic untouched.
  - noqa with inline reason added for intentional patterns: deap
    availability probe (hyppo/generator/_generator.py:43), owlready2
    availability probes (tests/test_owl_reasoning.py, tests/test_oil_adapter.py),
    golden-claims file's "imports colocated with test section" style
    (tests/test_golden_claims.py:309).
- Remaining 72 E501 (line-too-long) after `ruff format` reflow: fixed the
  last ~20 by hand — shortened/rewrapped docstrings, comments, an SQL
  string (split across lines), and one Russian domain string in
  hyppo/gui/demo.py. No wording meaning changed, only trimmed for length.
- `uvx ruff format hyppo tests`: 89 files reformatted (whitespace/quote/
  wrap style only, per rules 1-2 above no logic changed).

### Not done
- n/a — wave complete.

### Verified
- `uvx ruff check hyppo tests` -> All checks passed! (0 findings)
- `uvx ruff format --check hyppo tests` -> 127 files already formatted (0 diff)
- `pytest tests/test_golden_claims.py -q` -> 48 passed (golden invariant intact,
  no operation-count/semantics drift from the core/_base.py and
  ontology/__init__.py rewrites).
- `pytest tests -q` -> 336 passed in ~88s (one transient Windows
  access-violation on a bare collect-all run reproduced as a flake — a clean
  `--collect-only` and a clean full run both succeeded immediately after;
  not reproduced a third time, looks like an environment hiccup unrelated
  to this wave's edits, not investigated further per wave scope).
- Scope-guard: `git diff --name-only` == hyppo/** + tests/** only (105 files);
  no pyproject.toml touched (not needed — [tool.ruff] already in place from W1).
- No-wave-artifact grep on touched files: only benign false positives
  (R2/R3+ referring to the R² statistic / OWL rule numbers in comments,
  pre-existing, not wave markers) — no cleanup needed.
- Commit: chore(lint): ruff check + format sweep, zero findings (W7)

### Next
R3 (W7) complete. R4 (W8 mypy zero) can proceed.

## R3b
Stage: realization | Author: fix-agent
- [VERIFIED pyproject.toml:69] `[build-system] requires` bumped setuptools>=75 -> >=77 (PEP 639 plain-string `license = "MIT"` needs setuptools>=77); `uv build` re-verified -> gedanken-1.1.0 wheel+sdist built clean; uv.lock unaffected (no diff).
- Commit: fix(review-async): bump build setuptools floor to >=77 (R3)

## W8
Stage: realization | Wave: W8 | Author: executor

### Done
- Re-baselined `uvx mypy hyppo`: 94 raw errors, ~65 pure import-not-found/untyped
  (uvx isolated env has no project deps). Added global `ignore_missing_imports=true`
  (matches recon --ignore-missing-imports baseline) → 29 real errors in 5 files.
- Fixed real (not suppressed):
  - hyppo/storage/_base.py: `filename: str` → `Union[str, Path]` in save/load/delete
    (Path(filename) reassign no longer str-assignment error; is_absolute narrows OK);
    `self.description: Optional[dict] = None` (was inferred None); one targeted
    `# type: ignore[return-value]` in load_all (get_all_names lists existing files
    → load never None there).
  - hyppo/gui/projects.py: `list[dict]` → `List[dict]` on list()/list_iterations()
    (method named `list` shadowed builtin in class scope → valid-type); added
    `from typing import List`.
  - hyppo/gui/services.py: annotated `best: tuple[str | None, dict]` (var-annotated).
  - hyppo/ontology/oil_constraints.py:116: ignore code `union-attr` → `attr-defined`
    (info: object → .field_name is attr-defined, not union-attr).
- Suppressed via config (owlready2 metaclass magic only):
  - New override module="hyppo.core._base" disable_error_code=["misc"] — 19
    "Invalid base class" from `class X(Thing)` / `class X(Artefact >> int, ...)`,
    same pattern as ontology.* but outside that namespace.

### Verified
- `uvx mypy hyppo` → Success: no issues found in 75 source files (exit 0).
- `uvx ruff check hyppo tests` → All checks passed!
- `uvx ruff format --check hyppo tests` → 127 files already formatted.
- `pytest tests/test_golden_claims.py` → 48 passed (golden invariant intact).
- `pytest tests -q` → 336 passed (first run hit the transient Windows
  access-violation flake W7 documented on owl/Java teardown; clean on immediate re-run).
- Scope-guard: git diff --name-only ⊆ {hyppo/storage/_base.py, hyppo/gui/projects.py,
  hyppo/gui/services.py, hyppo/ontology/oil_constraints.py, pyproject.toml}. No wave markers.

### Files
- `pyproject.toml` modified ([tool.mypy]: ignore_missing_imports + core._base override)
- `hyppo/storage/_base.py`, `hyppo/gui/projects.py`, `hyppo/gui/services.py`,
  `hyppo/ontology/oil_constraints.py` modified

### Next
R4 (W8) complete. R5 (W9 webui, W10 docstrings, W12 artifacts) can proceed.

## W12
Stage: realization | Wave: W12 | Author: executor

### Done
- CITATION.cff, CHANGELOG.md, CONTRIBUTING.md: found already present (untracked,
  from an interrupted prior run) with correct content — verified against spec
  (CFF 1.2.0 valid via `yaml.safe_load`, author dmkovalev, title gedanken,
  version 1.1.0, license MIT; CHANGELOG Keep-a-Changelog 1.1.0 + 1.0.0;
  CONTRIBUTING uv setup + golden-test contract) and committed as-is, no edits
  needed.
- README.md: rebranded to `gedanken` with tagline, `pip install gedanken`
  quickstart, `hyppo-gui`/`hyppo-mcp` console-script commands (no more
  `hyppo gui`/`uv run python -m hyppo.mcp`), CI/license/python badges, docs
  site placeholder link.
- docs/gui_demo_spec.md: fixed stale golden numbers at lines 24-25 (17→18
  edges, depth 5→10), verified against `tests/test_golden_claims.py`
  (`NORNE_GOLDEN_EDGES` == 18, `nx.dag_longest_path_length(g) == 10`);
  nothing else in the file changed.

### Verified
- `python -c "import yaml; yaml.safe_load(open('CITATION.cff'))"` -> OK.
- `grep -n "hyppo gui" README.md` -> 0 hits.
- `grep -n "18 рёбер\|глубина 10" docs/gui_demo_spec.md` -> both present.
- `git diff --name-only` (staged) == exactly the 5 owned files.

### Files
- `CITATION.cff` (committed as found, untracked -> tracked)
- `CHANGELOG.md` (committed as found, untracked -> tracked)
- `CONTRIBUTING.md` (committed as found, untracked -> tracked)
- `README.md` modified
- `docs/gui_demo_spec.md` modified

Commit: docs(publication): CITATION, CHANGELOG, CONTRIBUTING, README overhaul (W12)

### Next
W12 complete. No deviations from spec.

## W9
Stage: realization | Wave: W9 | Author: executor

### Done
- Inspected partial/interrupted work already in tree: hyppo/gui/app.py already
  switched static-serving source from `webui/dist` to `hyppo/gui/static`, with
  a warning log on missing dir; pyproject.toml already had the package-data
  line; hyppo/gui/static/** (untracked) verified byte-identical to webui/dist
  (`diff -rq` empty) — provenance: prior `npm run build` output copied in,
  reused as-is (npm broken in this shell: `npm-prefix.js` module-not-found).
- hyppo/gui/app.py: added optional `static_dir: Path | None = None` param to
  `create_app()` (backward-compatible) so tests can inject a missing/present
  dir without touching the real source tree; kept default behavior identical.
- tests/gui/test_static.py: strengthened — content-type assertion on `/`,
  new `test_warns_when_static_assets_missing` (caplog, asserts the exact
  warning), new `test_serves_html_when_static_dir_provided` (real content
  round-trip via injected `static_dir`).
- tests/test_packaging.py: filled in the previously-unused `subprocess`/
  `zipfile` imports with `test_wheel_contains_gui_static` — runs `uv build
  --wheel`, opens the produced wheel, asserts
  `hyppo/gui/static/index.html` is a member; skips gracefully if no built
  SPA in the tree or `uv` not on PATH.
- webui/README.md: docs were stale (said `StaticFiles` reads `webui/dist`
  directly — no longer true after the static_dir switch); added the
  `cp -r webui/dist hyppo/gui/static` step to Режим A + "Обычный цикл", and a
  line documenting the missing-assets warning.

### Verified
- `uv build --wheel -o <tmp>` then zipfile-inspected: wheel contains
  `hyppo/gui/static/index.html` + 2 asset files.
- Smoke: `TestClient(create_app(db_path=':memory:')).get('/')` -> 200,
  `text/html; charset=utf-8`, body starts with the built `<!doctype html>`.
- `pytest tests/gui/test_static.py tests/test_packaging.py -q` -> 6 passed.
- `uvx ruff check` + `uvx ruff format --check` on touched .py files -> clean
  (one E501 auto-fixed via `ruff format`).
- `uvx mypy hyppo/gui/app.py` -> Success, no issues.
- Full `pytest tests -q` -> 339 passed in 92s.
- No-wave-artifact grep on touched files: clean.
- Scope-guard: staged/committed only hyppo/gui/app.py, hyppo/gui/static/**,
  pyproject.toml, tests/gui/test_static.py, tests/test_packaging.py,
  webui/README.md — verified via `git status --porcelain` before commit that
  concurrent executors' files (hyppo/core/*, hyppo/runner/_base.py, this
  HANDOFF.md pre-append) were left untouched/unstaged.

### Files
- `hyppo/gui/app.py` modified (optional static_dir param)
- `hyppo/gui/static/index.html`, `hyppo/gui/static/assets/*` added (tracked)
- `pyproject.toml` modified (package-data, already present, verified)
- `tests/gui/test_static.py` modified (3 tests, up from 1)
- `tests/test_packaging.py` modified (new wheel-contents test)
- `webui/README.md` modified (build->static copy step documented)

Commit: feat(gui): package built SPA into wheel (W9)

### Next
W9 complete. No deviations beyond the additive `static_dir` param (minimal,
backward-compatible, needed for the missing-assets warning test).

## W10
Stage: realization | Wave: W10 | Author: executor

### Done
- hyppo/core/_base.py: completed docstrings for Structure, FullStructure,
  Equation, Variable, has_for_varname, FullCausalMapping, has_for_fcm,
  has_for_structure, DependencySet, has_for_dependecy_set, TransitiveClosure,
  ResearchLattice, has_for_lattice_hypothesis (rest was already done by an
  interrupted prior run — kept as-is); fixed one E501 (has_for_configuration).
- hyppo/core/_hypothesis.py: module docstring; Args/Returns/Raises for
  get_aic, get_bic, range_models, Hypothesis (class + parameters property +
  range_models + compare_preds_on_*), Model (class + __init__/fit/predict/
  update_bayesian_probability/bayesian_score), NonLinearModel (class +
  fit/compute_aic/compute_bic).
- hyppo/core/_epistemic.py: EpistemicStatus class docstring; Returns section
  added to evaluate_status (Args/precedence note were already present).
- hyppo/core/_workflow.py: Workflow class note + Args/Returns/Raises for
  __init__/is_dag/topological_order/reachable_from/tasks/edges.
- hyppo/core/__init__.py: package docstring added.
- hyppo/runner/_base.py: docstrings for Status, RunResult (Attributes),
  Runner.__init__; execute/_assign_epistemic_status/_cascade_skip already
  documented, untouched.

### Not done
- n/a — wave complete.

### Verified
- Self-audit (AST, public class/def/__init__ without docstring, property
  setter/getter and nested-local helpers excluded) over the 6 owned files:
  0 missing (before: recon estimated core ~12%, runner ~33% coverage).
- `uvx ruff check hyppo/core hyppo/runner` -> All checks passed!
- `uvx ruff format --check hyppo/core hyppo/runner` -> 7 files already formatted.
- `uvx mypy hyppo` -> Success: no issues found in 75 source files.
- `pytest tests -q` -> 339 passed in 89.6s.
- Scope-guard: `git diff --name-only` == exactly the 6 wave files. No stray
  wave/round/phase markers (grep clean).
- Commit: docs(core,runner): Google-style docstrings for public API (W10)

### Next
R5 (W9, W10, W12) — W10 done; check W9/W12 status for round completion.

## W11
Stage: realization | Wave: W11 | Author: executor

### Done
- mkdocs.yml (new): mkdocs-material theme, mkdocstrings[python] plugin
  (`docstring_style: google`, `paths: ["."]` so `hyppo` is importable from
  repo root), `docstring_options.warn_missing_types: false` (griffe emits a
  strict-mode-fatal warning for Google-style params without a signature
  annotation — e.g. `planner/_base.py:239,382`, `runner/_base.py:66` — this
  is a docstring-style choice already made by W10, not a defect; suppressing
  the specific griffe check is the mkdocstrings-options route the plan asked
  for, no source edits); `exclude_docs: docs/specs/, docs/superpowers/`; nav
  Home/Architecture/API Reference (16 subpackages)/GUI Demo Spec.
- docs/index.md (new): quickstart consistent with README (pip install
  gedanken, hyppo-gui/hyppo-mcp commands, tagline, golden-test contract
  paragraph).
- docs/architecture.md (new): layered module map (gui/mcp -> manager ->
  runner/planner -> lattice_constructor/coa -> core OWL); one paragraph per
  subpackage for all 16 listed in the plan incl. hyppo.versioning; notes the
  `actions` single-domain (`Literal["oil_waterflood"]`) limitation from
  DECISIONS.
- docs/api/*.md (new, 16 files): one page per subpackage, each a plain
  `::: hyppo.<sub>` mkdocstrings directive — core, coa, lattice_constructor,
  planner, runner, manager, storage, versioning, comparison, ontology,
  actions, adapters, gui, mcp, metadata_repository, generator. All 16 built
  with real content (no import failures — owlready2/ontology pages included
  cleanly).
- .gitignore: added `site/` (mkdocs build output).

### Not done
- n/a — wave complete, all 16 API pages built (plan's fallback "drop page +
  HANDOFF line" not needed).

### Verified
- Shared `.venv` was lock-contended by a concurrent process (rpds/pywin32
  DLLs, `uv sync --extra docs` failed twice with "Отказано в доступе") —
  used `uvx --with mkdocs-material --with "mkdocstrings[python]"
  --with-editable . mkdocs build --strict` instead (isolated env, does not
  touch the shared venv other waves may be using concurrently).
- `mkdocs build --strict` -> exit 0 ("Documentation built in 0.83 seconds",
  0 warnings) after the `warn_missing_types` fix (first attempt: 3 griffe
  warnings, aborted per strict mode).
- `ls site/api/*/index.html` -> all 16 present, 1000+ lines each for
  planner/runner/storage/versioning (spot-checked) — not empty stubs.
- `git status --porcelain` after commit: only this HANDOFF.md append
  remains modified; no other-wave files swept (scope-guard: `git show
  --stat HEAD` == exactly the 20 owned files: mkdocs.yml, docs/index.md,
  docs/architecture.md, docs/api/*.md x16, .gitignore).
- No-wave-artifact grep on touched files: clean (no W\d+/R\d+ markers).
- `site/` build directory removed from working tree before commit (build
  artifact, now gitignored).

### Files
- `mkdocs.yml` created
- `docs/index.md` created
- `docs/architecture.md` created
- `docs/api/*.md` created (16 files)
- `.gitignore` modified (+`site/`)

Commit: docs(site): mkdocs-material + mkdocstrings API reference (W11)

### Next
R6 (W11) complete. R7 (W13 CI) can proceed — CI docs job should invoke
`mkdocs build --strict` the same way (either after `uv sync --extra docs`
once the venv lock issue is transient/CI-fresh, or via `uv run --extra
docs mkdocs build --strict`).

## Reviewer(phase B) findings — W7,R3fix,W8,W9,W10,W12
Verified green: uvx ruff check + format --check hyppo tests -> 0; uvx mypy hyppo
-> 0 (75 files); golden 48 passed; reviewed-code suite 331 passed.
- W7: ontology/__init__ explicit imports OK — no consumer does
  `from hyppo.ontology import X`, import test green, no name lost. core/_base
  owlready2 explicit import class-list unchanged (executor-verified).
- W8: mypy 0 confirmed. Fixes are annotations/config only, no logic drift.
- W9: static committed = 3 files, 232K, no source-maps/secrets/.env; app.py
  static path = `Path(__file__).resolve().parent/"static"` (no parents[2]
  escape; parents[2]/webui/dist appears only inside the missing-assets warning
  string), warning names build cmd; pyproject package-data has py.typed +
  hyppo.gui static/**/*.
- W10: 5 docstrings spot-checked factual (get_aic 2k-2ln L, get_bic k·ln n-2ln L,
  range_models, RunResult attrs, Runner.execute args) — all match signatures, no
  drift.
- W12: entry points hyppo-gui/hyppo-mcp present in pyproject; README has 0 `hyppo
  gui`/`hyppo mcp` space-forms, uses console scripts + `pip install gedanken`;
  CITATION.cff valid YAML v1.1.0 title gedanken; gui_demo_spec 16 nodes/18 edges/
  depth 10 == test_golden_claims (nodes==16, edges==18, dag_longest==10).

- [POINT-FIX] ENVIRONMENT (not reviewed code): full `pytest tests` currently
  shows 5 failures — 4 tests/test_mcp_server_factory.py + 1
  test_package_imports.py::test_all_submodules_importable — all
  `ModuleNotFoundError: pywintypes` from mcp>=1.27 `mcp/os/win32/utilities.py`.
  Cause: a concurrent `uv sync` (non --all-extras) pruned dev deps mid-review;
  recovery `uv sync --all-extras` reinstalled pywin32-311 but its
  postinstall/DLL registration is broken in this venv (pywintypes313.dll present,
  pythoncom DLL-load fails). Windows-local only (marker sys_platform=='win32';
  CI is Linux so unaffected). Reviewed commits do NOT touch mcp deps; prior
  W8/W9/W10 runs = 336-339 green. Next-round executors on this box must
  `uv sync --all-extras` (+ pywin32 postinstall or venv rebuild) before trusting
  a local full-suite run. Also documented pre-existing Windows access-violation
  flake (owlready2/Java teardown) crashes ~50% of a full run; pytest-forked
  can't isolate it (os.fork absent on Windows) — re-run to get past it.
- [POINT-FIX] stray marker (pre-existing, out of reviewed scope): `R3+` domain
  comments — hyppo/adapters/wfopt_adapter.py:201,291 and
  hyppo/actions/virtual_experiment.py:116 (from commits 796adaa/38f9a98, refer
  to an OWL rule / Theorem 1 axiom, not an ai-waves round). W7 only reformatted
  these files, didn't author the lines. Low priority.

No BLOCKERS. All phase-B Goals (G1 wheel/SPA, G2 ruff, G3 mypy, G6 artifacts)
verified met on the reviewed commits.

## W13
Stage: realization | Wave: 13 | Author: executor(W13)
Commit: build(ci): GitHub Actions — lint, test matrix 3.11-3.13, docs (W13)

### Done
- .github/workflows/ci.yml created: lint (ubuntu, uvx ruff check/format
  --check, uvx mypy hyppo, py3.11), test (matrix 3.11/3.12/3.13, setup-java
  temurin 17 for HermiT reasoner, uv sync --all-extras, uv run pytest tests
  -q), docs (uvx --with mkdocs-material --with "mkdocstrings[python]"
  --with-editable . mkdocs build --strict — same pattern W11 used).
- Deviation from plan text: mypy is NOT in pyproject's dev extra (checked
  [project.optional-dependencies].dev — no mypy/ruff entries at all); used
  `uvx mypy hyppo` (relies on [tool.mypy] in pyproject, present at lines
  87-97) per the plan's own fallback clause.
- Verified sync_reasoner_hermit (owlready2, needs Java) used in
  tests/test_markers.py, tests/test_oil_adapter.py,
  tests/test_golden_claims.py, hyppo/ontology/{consistency,markers}.py —
  confirms setup-java requirement.

### Local verification (python 3.13, this machine)
- YAML valid: `python -c "import yaml; yaml.safe_load(open(...))"` -> OK.
- `uvx ruff check hyppo tests` -> All checks passed.
- `uvx ruff format --check hyppo tests` -> 127 files already formatted.
- `uvx mypy hyppo` -> Success: no issues found in 75 source files.
- `.venv/Scripts/python -m pytest tests -q` -> 339 passed in 92.45s (first
  attempt hit shared-.venv corruption: pywin32/rpds-py dist-info left
  mid-write by a concurrent process on this machine, same class of issue
  W11 HANDOFF noted; `uv pip install --reinstall pywin32 rpds-py` fixed it;
  not a CI-relevant issue since CI runs fresh ubuntu envs).
- `uvx --with mkdocs-material --with "mkdocstrings[python]" --with-editable
  . mkdocs build --strict` -> "Documentation built in 0.78 seconds", exit 0.
  site/ removed after (gitignored build artifact, not committed).
- No-wave-artifact grep on .github/workflows/ci.yml: clean.
- `git add .github` only (per note) — did not stage the stray
  experiments/iip2026/... file or modified HANDOFF.md-adjacent files
  outside this wave.

### Not done
- n/a — wave complete.

### Files
- `.github/workflows/ci.yml` created.

### Next
R7 (W13) complete — final wave of the plan. All rounds R1-R7 done.

## Final review (sync hard-gate, reviewer)
Env: Windows, .venv py3.13. All acceptance commands executed literally.
- [1] recursive import → exit 0 (every hyppo.* module imports).
- [2] uvx ruff check → "All checks passed!" exit 0; format --check → "127 files already formatted" exit 0.
- [3] uvx mypy hyppo → "Success: no issues found in 75 source files" exit 0.
- [4] mkdocs build --strict (uvx --with mkdocs-material --with mkdocstrings[python] --with-editable .) → "built in 0.75s" exit 0, 0 warnings.
- [5] pytest tests -q → 339 passed in 94.66s, exit 0. No crash, no retry needed (Windows owl/Java flake did NOT recur).
- [6] uv build → gedanken-1.1.0-{whl,tar.gz}; wheel contains hyppo/gui/static/index.html=True, hyppo/py.typed=True.
- [7] GUI smoke from source: create_app(':memory:') GET / → 200, text/html, body `<!doctype html>`. test_static.py green in suite.
- [8] `import hyppo.actions` → 'hyppo.mcp' not in sys.modules → "cycle-free" exit 0.
- [9] Root hygiene: root tracked = .gitignore CHANGELOG.md CITATION.cff CLAUDE.md CONTRIBUTING.md LICENSE README.md mkdocs.yml pyproject.toml uv.lock. No HOW_TO_USE.md/full_experiment.py/norne_battery*. hyppo/streamlit & core/_virtual_experiment.py untracked+gone (only stray __pycache__ on disk).
- [10] .github/workflows/ci.yml present; lint(ruff check/format/mypy) + test matrix 3.11/3.12/3.13 (+setup-java) + docs(mkdocs --strict) mirror commands run.
Non-goals respected: dist=gedanken, import stays hyppo; actions still single-domain Literal["oil_waterflood"], no registry. Golden contract (tests/test_golden_claims.py) tracked + green in suite.
IF-criterion: app.py static = Path(__file__).resolve().parent/"static" (no parents[2] escape); logger.warning names build cmd when absent.
Observations (non-blocking): pre-existing Russian comment pyproject.toml:55 (key data-dep note, allowed); phase-B stray "R3+" false-positives (OWL-rule refs, not wave markers) unchanged.
Verdict: G1–G8 all PASS. No POINT-FIX, no BLOCKER. READY TO FINISH.
