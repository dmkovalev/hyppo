# checklist — spec self-validation

- [x] Every EARS criterion testable (commands/exit codes given)
- [x] Goals measurable (G1-G8 each maps to a command or artifact)
- [x] Non-goals explicit (no VE consolidation, no domain registry, no full docstring rewrite, no i18n, no PyPI upload)
- [x] Load stated (§Load: research scale, no perf work)
- [x] No NEEDS-CLARIFICATION in spec — EXCEPT dist name PENDING (user re-opened; isolated to W1/W12, cheap redo)
- [x] Every wave traces to a Goal (W1→G1, W2→G6, W3→G5/G8, W4→G5, W5→G5, W6→G5, W7→G2, W8→G3, W9→G1, W10→G4, W11→G4, W12→G6, W13→G7)
- [x] Same-round waves have disjoint Files (R1: pyproject+__init__ / deletions / 4 fix files — W1 owns pyproject, W9 adds one line in R5 but different round; R2: versioning / storage / new test file; R5: gui+webui / core+runner / root md files — disjoint)
- [x] Contracts marked; [ASSUMED] only in W11 (mkdocstrings default) and W13 (Java in CI) — both verified at execution, low risk
- [x] Testing strategy section present; golden-test contract (G8) respected per CLAUDE.md invariant
- [x] Project invariants (CLAUDE.md): graph built only by Algorithm 1 — untouched; golden tests must pass — G8; numbering note — untouched
- [x] Wave density: 13 waves / 7 rounds; format sweep and mypy isolated (whole-tree diffs); rest merged dense
