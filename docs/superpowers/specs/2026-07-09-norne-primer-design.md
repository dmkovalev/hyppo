# Norne primer — end-to-end CLI walkthrough (design)

Date: 2026-07-09. Status: approved by user.

## Purpose
A single self-contained script that walks a reader through the ENTIRE virtual-experiment
lifecycle on the Norne HybridCRM case — the same content the GUI demo covers, but as a
readable, runnable command-line narrative. Target audience: a new user after
`git clone` (script lives in examples/, not in the wheel — user decision "2 + readme").

## Deliverables
- `examples/norne_primer.py` — one linear script, 7 numbered acts, `--pause` flag
  (Enter between acts), exit 0 + `PRIMER OK` on success, non-zero + message on any
  self-check failure. No external data files; synthetic equations/axes identical in
  spirit to `hyppo/gui/demo.py` (but built via public APIs, NOT importing gui private
  helpers).
- `examples/README.md` — new section: what each act shows, how to run, expected output tail.

## Acts
1. **VE tuple ⟨O, H, M, R, W, C⟩** (Definition 1): print ontology O (Reservoir/Well/
   Hypothesis/Model classes, derived_by/injects_into properties), 16 hypotheses H
   (paper numbering H1–H16: liquid H1–H8, watercut H9–H14, frac H15, oil H16) with
   equation formulas, models M, mapping R, workflow W, config space C (16 axes,
   |C| = 221184, constraint C1 note).
2. **Algorithm 1**: `HypothesisLattice(hypotheses, workflow)` → print derived_by edges,
   nodes/edges/DAG depth. Expected: 16 / 18 / 10.
3. **Algorithm 2 (Lemma 2)**: add a 17th hypothesis incrementally vs full rebuild;
   compare edge sets; print `incremental == rebuild: True`.
4. **Algorithm 4 + Theorem 1**: cascade-recompute plan for "H8 changed" via
   hyppo.planner; Theorem 1 demo: correctness (plan ⊇ transitive descendants — check
   against an independent reachability oracle) and ⊆-minimality (dropping any element
   breaks correctness — exhaustive loop); print `correct: True`, `minimal: True`.
5. **Run**: Runner executes in topological order; print per-hypothesis status and
   epistemic status transitions.
6. **Compare**: hyppo.comparison (sign test / AIC / BIC) on a competing hypothesis pair.
7. **Self-check**: assert golden values (16 nodes / 18 edges / depth 10) + act-3
   equivalence + act-4 theorem checks; `PRIMER OK`.

## Constraints / notes
- Source of truth for API calls and golden values: `docs/gui_demo_spec.md`,
  `tests/test_golden_claims.py`, `examples/norne_alg1_lattice.py`.
- CLAUDE.md-era invariant (now in gui_demo_spec/CONTRIBUTING): graph built ONLY by
  Algorithm 1 (`HypothesisLattice`), output variable = LHS of equation formula.
- ruff/mypy must stay green (repo is at zero); English comments; no wave markers.
- No pytest addition (examples are not under test); self-check is built into the script.

## Alternatives rejected
- Console entry point `hyppo-primer` in the wheel — user chose examples/ + README.
- Step-per-file package — loses single-narrative readability (YAGNI).
- Reusing `hyppo.gui.demo` internals — couples primer to GUI private helpers.

## Testing
Manual run criterion: `python examples/norne_primer.py` exits 0, tail = `PRIMER OK`;
`--pause` waits on stdin. Full suite stays green (script imports only public APIs).
