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
