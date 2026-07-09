# GUI Demo Specification: Algorithms 1-4, Theorem 1, Complexities, Rule 5

Purpose: reproduce in the GUI every checkable claim from the papers via live
platform calls. **Source of truth — `tests/test_golden_claims.py`**: every
demo must converge with the golden test of the same name. Nothing is to be
overridden locally — only real calls into hyppo.

Hypothesis notation: in the code — historical names (`H11…H15`, `H12b`,
`GRP`, `H19`), in the СВД paper — continuous numbering H1-H16.
Correspondence: `H11→H9, H12→H10, H12b→H11, H13→H12, H14→H13, H15→H14,
GRP→H15, H19→H16`. In the GUI, show the paper's numbering; the code names
go in the tooltip.

---

## Demo 1. Algorithm 1 — automatic construction of the hypothesis graph

```python
from hyppo.lattice_constructor._base import HypothesisLattice
from hyppo.coa._base import Equation, Structure
# 16 formulas of NORNE_HYPS — copy from tests/test_golden_claims.py
G = HypothesisLattice(hypotheses, workflow).lattice   # networkx.DiGraph
```

**Golden**: exactly 18 edges `NORNE_GOLDEN_EDGES` (see test
`test_alg1_norne_graph_matches_figure`), 16 nodes, DAG, depth 10.

**Visualization**: graph (two branches + merge, as in Fig. 3 of the paper).
On each edge — an explanation of the derivation: "output `I_agg` (H1) enters
equation H2". A hypothesis's output = the left-hand side of its formula
(fix `5907005` — do not change!).

## Demo 2. Algorithm 2 — incremental addition

```python
lat = HypothesisLattice(hypotheses[:-1], workflow)   # without H19
lat.add_hypothesis(h19)                              # incrementally
```

**Golden**: the graph after addition == full rebuild on all 16
(`test_alg2_incremental_equals_full_rebuild`); new edges: `H8→H19`,
`H15→H19`.

**Visualization**: an "add hypothesis" button → only the new edges appear
and get highlighted; a causal-merge counter shows |H| (not |H|²).

## Demo 3. Algorithm 3 — two-stage well-posedness check

```python
from hyppo.ontology.consistency import check_consistency, Status
res = check_consistency(ve, onto, lattice, run_hermit=..., artefacts=..., configurations=...)
```

Stage B (structural, `run_hermit=False`, instant):

| Scenario | Input | Expected status |
|---|---|---|
| correct | DAG + consistent artefacts + finite domains | `OK` |
| cycle | `{0:{1},1:{2},2:{0}}` | `C3` + cycle witness |
| flow break | edge with `Out(i)∩In(j)=∅` | `C4` + culprit edge |
| infinite domain | object without `finite=True` | `C5` |
| double violation | cycle + break | `C3` (check order is fixed) |

Stage A (semantic, HermiT, ~seconds — show a spinner):

- **detectable**: a hypothesis with two different models under a functional
  property + AllDifferent → "ontology is inconsistent" (C2);
- **key demonstration of OWA**: a hypothesis *without* a model — the
  reasoner says "consistent"! Show a banner: "the open world does not see
  the absence — this check is performed by the marker layer (rule 9)."
  This is the executable justification for the three-layer architecture
  (test `test_alg3_stage_a_owa_cannot_see_missing_model`).

Run stage A in an isolated `owlready2.World()` — do not touch the
application's shared ontology.

## Demo 4. Algorithm 4 — recompute plan + cascade

```python
from hyppo.coa.graph import HypothesisGraph
g = HypothesisGraph.from_edges(n, edges)   # topology from Demo 1
pne = g.plan(cached)                       # set to recompute
```

**Golden**: `plan(cached)` == "uncached + everything reachable from them"
(`test_alg4_plan_matches_reachability_oracle`).

**Visualization**: the user clicks to mark cached nodes → P_ne is
highlighted. Paper scenario: invalidating H1 colors the whole liquid branch
(H2-H5, H8) and H16, the water-cut branch is untouched; a workover (H15)
colors only H5, H8, H16.

## Demo 5. Theorem 1 — minimality of the plan

Interactive "propose your own plan": the user picks an arbitrary set P.
The GUI checks the two correctness conditions (`_is_valid_plan` from the
golden tests): P contains all uncached nodes; P is closed under
descendants (cascade property A2). It then shows:

- P is incorrect → point out the violated condition;
- P is correct → highlight that `plan(cached) ⊆ P`, conclusion: "your plan
  contains the minimal one; extra vertices: P \ plan". Minimality holds
  both by vertex count and by any non-negative cost.

**Golden**: `test_theorem1_plan_is_minimal_correct_plan` (on graphs n≤6 —
an honest brute-force over all subsets; in the GUI, only show the
brute-force for small graphs).

## Demo 6. Complexities — via operation counters, not seconds

Instrumentation: wrap `hyppo.coa.causal.is_complete` with a counter
(monkeypatch for the duration of the demo); for Alg. 4 — a counter of
adjacency traversals.

| Algorithm | Input (chain of length n) | Counter | Law |
|---|---|---|---|
| Alg. 1 `build()` | n = 10/20/40 | completeness checks | exactly n(n−1)/2 → ×4 at ×2 |
| Alg. 2 `add_hypothesis` | n = 10/20/40 | unions | exactly n → ×2 at ×2 |
| Alg. 4 `plan()` | n = 200/400/800 | adjacency traversals | ~V+E → ×2 at ×2 |

**Visualization**: bar charts "counter vs theoretical curve" labeled
O(|H|²·s·v) / O(|H|·s·v) / O(V+E). Do not use wall-clock time — only
counters (deterministic).

## Demo 7. Rule 5 — procedural acyclicity

```python
from hyppo.ontology.consistency import _find_cycle_via_kahn
```

An attempt to add an edge that closes a cycle → rejection with a witness
shown (the sequence of vertices in the cycle). Banner: "a transitive
property cannot be declared asymmetric in OWL (the simplicity
restriction) — acyclicity is checked procedurally on every link addition"
[СВД §3.2].

## Demo 8. Series of 80 trials (systematic detection check)

Data source: **`norne_battery_results.json`** (in the repo root) — results
of 80 real HermiT runs, generated by `norne_battery.py`. Do NOT run the
series live in the interface (247 s) — load the ready-made JSON; optionally
a "recompute" button with a progress bar that calls the script in the
background.

JSON structure: `rule4[]` — 16 records `{source, stale[], match}`;
`rule7[]` — 64 records `{run_uses, invalid, derived_stale, expected,
verdict}`; `summary` — totals.

**Visualization**:
- rule 4 — a table "source → number of stale items" (monotonically
  decreasing from 12 for H1 to 0 for the forecast) with green checkmarks
  matching the oracle; on hover, highlight the corresponding cascade on the
  Demo 1 graph;
- rule 7 — a 4×16 matrix (run × candidate), cells TP/TN, no errors;
- summary banner: "102 classifications and 64 pairs — 0 discrepancies
  with graph reachability; this verifies implementation conformance to the
  spec, not a statistical metric" [СВД, §Systematic verification].

**Golden**: `summary` must yield rule4_classifications=102,
rule4_mismatches=0, rule7_tp=43, rule7_tn=21, rule7_errors=0.

---

## GUI self-check

After implementation, run and confirm the demos converge with golden:

```bash
.venv/Scripts/python -m pytest tests/test_golden_claims.py -q          # all
.venv/Scripts/python -m pytest tests/test_golden_claims.py -q -m "not reasoner"  # fast only
```

A discrepancy between a demo and its golden test = a bug in the demo (or
stale paper text — in which case the paper is fixed first, then the test,
then the demo).
