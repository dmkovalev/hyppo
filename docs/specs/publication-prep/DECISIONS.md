# DECISIONS — publication-prep

## naming/dist-name: PyPI distribution name
- Decision: DIST_NAME = `gedanken` (user-approved, final). Import package stays `hyppo`; Hyppo remains the dissertation-project name, `gedanken` brands the post-defense evolution (Gedankenexperiment ≈ virtual experiment).
- Rejected: `compex` — collision with Compex sports brand + CompEx oil&gas certification (project's own domain); `vexlab` — "vex" connotation, weaker pedigree.
- Reason: `hyppo` taken on PyPI (neurodata); dist/import names may legally differ.
- Alternative: candidates checked free on PyPI: vexlab, velab, exvitro, insilex, gedanken, compex, conjecta, hypothetica, probanda, hyppo-ve, hyppo-lattice, hyppolab, hyppo-platform; taken: virtex, numex, simex.

## python/floor: requires-python >=3.11
- Decision: lower from >=3.13 to >=3.11 (AFK default, Recommended).
- Reason: no 3.12+ syntax in code [recon §3]; 3.13-only floor cuts audience for nothing.
- Alternative: >=3.10 — EOL 2026-10, instantly stale.

## storage/fate: fix, not delete
- Decision: add cloudpickle dep, EN docstrings, logging, tests (AFK default, Recommended).
- Reason: part of the platform design (planner references Database type); deletion loses reference subsystem.
- Alternative: delete module — minimal but amputates documented design.

## docs/language: English
- Decision: mkdocs site + all docstrings unified in English (AFK default, Recommended).
- Reason: open-source/scientific-software standard; RU papers remain in thesis repo.
- Alternative: bilingual i18n — double maintenance for a research tool.

## webui/packaging: commit built SPA into hyppo/gui/static
- Decision: build once, commit dist output inside the package; app.py serves package-relative path.
- Reason: pip users must not need node; SPA is small; simplest thing that works (№2).
- Alternative: build-time npm hook in CI wheel job — deferred until release automation exists.

## core/dead-dataclass: delete core/_virtual_experiment.py
- Decision: delete (zero importers; OWL class in core/_base.py:65 is the Definition 1 carrier).
- Reason: №2 minimum code; reversible via git.
- Alternative: promote to public API — no consumer demands it.

## arch/cycle: version_store moves mcp → hyppo/versioning
- Decision: new subpackage hyppo/versioning owns version_store.py + _db.py; no re-export shims.
- Reason: breaks actions⇄mcp package cycle by data ownership, not import-order luck.
- Alternative: move into hyppo/storage — mixes sqlite versioning with pickle object store.

## arch/domain-leak: keep oil_waterflood Literal in actions
- Decision: no domain registry; document as known limitation.
- Reason: rule of three not met (one domain); abstraction would be speculative (№5).
- Alternative: registry pattern — revisit at second domain.

## mypy/scope: zero errors with per-module overrides
- Decision: [tool.mypy] strict-ish globally; hyppo.ontology.* gets disable_error_code for owlready2 dynamic-base magic; ignores must carry codes.
- Reason: owlready2 metaclasses are untypeable; blanket strict would force bare-ignore litter.
- Alternative: exclude ontology entirely — hides real errors in its pure-python parts.

## W4/importers: correct version_store move file-set (critic)
- Decision: W4 updates conftest.py + 6 test_action_*.py (all monkeypatch/import hyppo.mcp.version_store/_db); drop mcp/tools.py, mcp/resources.py, test_mcp_resources.py from the list (they never import version_store).
- Reason: no-shims move + grep shows 8 importers, not 3; monkeypatch target must equal the module actions imports (hyppo.versioning.version_store) or tests silently fail.
- Alternative: keep re-export shim in mcp — rejected earlier (№2), and shim would not fix monkeypatch-target mismatch anyway.

## W1/dynamic-version: setuptools attr table (critic)
- Decision: name the required `[tool.setuptools.dynamic] version={attr="hyppo.__version__"}` table explicitly; remove static version=.
- Reason: build-backend is setuptools.build_meta; `dynamic=["version"]` alone is insufficient without the dynamic table.
- Alternative: hatchling — not configured; would change build-system for no gain.

## tests/root-strays: delete untracked test_c3_*.py, keep untracked data files
- Decision: debug scripts deleted; gitignored data (_*.csv, *.h5, *.jar) untouched.
- Reason: scripts superseded by golden tests; data still feeds experiments/.
- Alternative: move scripts to experiments/ — dead weight, logic already in tests.
