# Norne Primer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One linear CLI script `examples/norne_primer.py` that walks the full VE lifecycle on Norne (tuple → Alg.1 → Alg.2/Lemma 2 → Alg.4/Theorem 1 → run → compare → golden self-check).

**Architecture:** Single narrative script, 7 numbered acts, each a function `act_N()` called from `main()`; shared data tables at module top; `--pause` waits for Enter between acts; self-check failures raise `SystemExit(1)`. Only public hyppo APIs (`hyppo.lattice_constructor`, `hyppo.coa`, `hyppo.runner`, `hyppo.comparison`); no GUI internals, no external data.

**Tech Stack:** Python ≥3.11, hyppo (this repo), networkx, argparse. No new deps.

**Conventions:** English comments/output; ruff + mypy must stay green (`uvx ruff check examples/norne_primer.py`, `uvx ruff format examples/norne_primer.py`); paper numbering H1–H16 printed, historical code names (H11, H12b, GRP, H19…) kept as internal IDs with the mapping from docs/gui_demo_spec.md (H11→H9, H12→H10, H12b→H11, H13→H12, H14→H13, H15→H14, GRP→H15, H19→H16).

**Source-of-truth references (read before coding):** `docs/gui_demo_spec.md` (demos 1,2,4,5), `tests/test_golden_claims.py` (`NORNE_GOLDEN_EDGES`, `_is_valid_plan`), `examples/norne_alg1_lattice.py` (Hyp/Workflow pattern), `hyppo/runner/_base.py:74` (`Runner.execute` signature).

---

### Task 1: Script skeleton + Act 1 (VE tuple)

**Files:**
- Create: `examples/norne_primer.py`

- [ ] **Step 1: Create the file with header, CLI, data tables, act 1**

```python
"""Norne primer: an end-to-end walkthrough of one virtual experiment.

Runs the complete lifecycle of the Norne HybridCRM virtual experiment
(Definition 1 tuple -> Algorithm 1 lattice -> Algorithm 2 incremental
update -> Algorithm 4 recompute plan + Theorem 1 -> execution ->
hypothesis comparison -> golden self-check).

Usage:
    python examples/norne_primer.py [--pause]

The script is self-checking: it exits 0 and prints ``PRIMER OK`` only if
every act reproduces the golden values pinned by tests/test_golden_claims.py
(16 nodes, 18 edges, DAG depth 10, Lemma 2 equivalence, Theorem 1).
"""
from __future__ import annotations

import argparse

import networkx as nx

from hyppo.coa._base import Equation, Structure
from hyppo.coa.graph import HypothesisGraph
from hyppo.comparison.compare import (
    compute_aic,
    gaussian_log_likelihood,
    sign_test,
)
from hyppo.lattice_constructor._base import HypothesisLattice
from hyppo.runner import Runner

# --- Norne HybridCRM data (paper [SVD], fig. 3; same case as the GUI demo) ---
# (code_name, paper_name, formula, meaning)
HYPS: list[tuple[str, str, str, str]] = [
    ("H1", "H1", "I_agg = w_ij * I_j", "aggregated injection"),
    ("H2", "H2", "q_f = a_f*q_f_prev + b_f*I_agg", "fast CRM channel"),
    ("H3", "H3", "q_s = a_s*q_s_prev + b_s*I_agg", "slow CRM channel"),
    ("H4", "H4", "q_c = w_f*q_f + (1-w_f)*q_s", "channel mixing"),
    ("H5", "H5", "q_liq_phys = J*q_c + q_prim", "physics liquid rate"),
    ("H6", "H6", "q_prim = q_prev*exp(-dt*taup)", "primary decline"),
    ("H7", "H7", "l_ml = MLP(x_hist)", "ML liquid correction"),
    ("H8", "H8", "l = g*q_liq_phys + (1-g)*l_ml", "LPR fusion"),
    ("H11", "H9", "Sw = Sw_prev + (Winj - l)*dt/Vp", "material balance"),
    ("H12", "H10", "krw = ((Sw-Swc)/(1-Swc-Sor))**nw", "Corey krw"),
    ("H12b", "H11", "kro = ((1-Sw-Sor)/(1-Swc-Sor))**no", "Corey kro"),
    ("H13", "H12", "fw = 1/(1 + kro*muw/(krw*muo))", "fractional flow"),
    ("H14", "H13", "o_p = 1 - fw", "physics watercut"),
    ("H15", "H14", "o = gw*o_p + (1-gw)*o_m", "watercut fusion"),
    ("GRP", "H15", "J = J0 + dJ_grp", "frac job (GTM) modulation"),
    ("H19", "H16", "q_oil = l * o", "oil-rate forecast"),
]

PAPER = {code: paper for code, paper, _, _ in HYPS}

# Workflow tasks (Section 3.1: groups executed as one stage each).
TASKS: list[list[str]] = [
    ["H1"], ["H2", "H3"], ["H4"], ["H5", "H6"], ["H7"], ["H8"],
    ["H11"], ["H12", "H12b"], ["H13"], ["H14", "H15"], ["H19"], ["GRP"],
]

# Ontology O: the domain vocabulary the OWL layer reasons over.
ONTOLOGY_CLASSES = [
    ("Reservoir", "the field under study"),
    ("Well", "producer or injector"),
    ("Hypothesis", "a falsifiable statement with an equation structure"),
    ("Model", "a computable realisation of a hypothesis"),
]
ONTOLOGY_PROPERTIES = [
    ("derived_by", "Hypothesis -> Hypothesis, transitive, acyclic (rule 5)"),
    ("injects_into", "Injector -> Producer"),
    ("is_implemented_by_model", "Hypothesis -> Model"),
]

# Configuration space C: 13 binary + 3 ternary axes (GUI demo, constraint C1).
N_BINARY_AXES = 13
N_TERNARY_AXES = 3

PAUSE = False


class Hyp:
    """Plain hypothesis: a name plus an equation structure (Definition 1, H)."""

    def __init__(self, name: str, formula: str) -> None:
        self.name = name
        self.structure = Structure([Equation(formula=formula)])

    def __repr__(self) -> str:
        return self.name


class Workflow:
    """Minimal workflow W: ordered groups of hypotheses (tasks)."""

    def __init__(self, tasks: list[list[str]], hyp_map: dict[str, Hyp]) -> None:
        self._tasks = [[hyp_map[h] for h in task] for task in tasks]

    def get_tasks(self) -> list[list[Hyp]]:
        return self._tasks


HYP_OBJS: dict[str, Hyp] = {code: Hyp(code, f) for code, _, f, _ in HYPS}
ALL_HYPS: list[Hyp] = [HYP_OBJS[code] for code, _, _, _ in HYPS]
WF = Workflow(TASKS, HYP_OBJS)


def _pause() -> None:
    if PAUSE:
        input("\n-- press Enter to continue --")


def _act(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\nACT {n}. {title}\n{'=' * 72}")


def act_1_tuple() -> None:
    """Print every element of the VE tuple <O, H, M, R, W, C> (Definition 1)."""
    _act(1, "The virtual experiment tuple <O, H, M, R, W, C> (Definition 1)")
    print("\nO — domain ontology (classes and properties):")
    for name, doc in ONTOLOGY_CLASSES:
        print(f"  class    {name:<12} {doc}")
    for name, doc in ONTOLOGY_PROPERTIES:
        print(f"  property {name:<24} {doc}")
    print(f"\nH — {len(HYPS)} hypotheses (paper numbering H1-H16):")
    for code, paper, formula, meaning in HYPS:
        tag = f"{paper} ({code})" if paper != code else paper
        print(f"  {tag:<12} {formula:<42} # {meaning}")
    print("\nM and R — models and the hypothesis->model mapping:")
    print("  every hypothesis is implemented by one model callable;")
    print("  in this primer models are synthetic fits returning r2/aic metrics.")
    print(f"\nW — workflow: {len(TASKS)} tasks (stages), e.g. "
          f"t2 = {{H2, H3}} runs both CRM channels in one stage.")
    n_configs = 2 ** N_BINARY_AXES * 3 ** N_TERNARY_AXES
    print(f"\nC — configuration space: {N_BINARY_AXES} binary + "
          f"{N_TERNARY_AXES} ternary axes, |C| = {n_configs} "
          "(constraint C1 prunes incompatible branch combinations).")
    _pause()


def main() -> None:
    global PAUSE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pause", action="store_true",
                        help="wait for Enter between acts")
    PAUSE = parser.parse_args().pause
    act_1_tuple()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify act 1 output**

Run: `.venv/Scripts/python examples/norne_primer.py`
Expected: header `ACT 1.`, 4 classes, 3 properties, 16 hypothesis lines, `|C| = 221184`. Exit 0.

- [ ] **Step 3: Lint**

Run: `uvx ruff check examples/norne_primer.py && uvx ruff format examples/norne_primer.py && uvx mypy examples/norne_primer.py --ignore-missing-imports`
Expected: 0 findings (format may rewrite — rerun check after).

- [ ] **Step 4: Commit**

```bash
git add examples/norne_primer.py
git commit -m "feat(examples): norne primer act 1 — VE tuple walkthrough"
```

---

### Task 2: Act 2 — Algorithm 1 (lattice)

**Files:**
- Modify: `examples/norne_primer.py` (add `act_2_lattice`, call from `main`)

- [ ] **Step 1: Add act 2 after `act_1_tuple`**

```python
def act_2_algorithm1() -> tuple[HypothesisLattice, "nx.DiGraph"]:
    """Algorithm 1: build the hypothesis lattice from equations + workflow."""
    _act(2, "Algorithm 1 — automatic hypothesis-graph construction")
    lattice = HypothesisLattice(ALL_HYPS, WF)
    g = lattice.lattice
    print("\nDerived_by edges (h_i -> h_j means h_j consumes the output of h_i):")
    for u, v in sorted(g.edges(), key=lambda e: (PAPER[str(e[0])], PAPER[str(e[1])])):
        print(f"  {PAPER[str(u)]:<4} -> {PAPER[str(v)]:<4}   "
              f"(output of {u} appears in the equation of {v})")
    depth = nx.dag_longest_path_length(g)
    print(f"\nNodes: {g.number_of_nodes()}   Edges: {g.number_of_edges()}   "
          f"DAG: {nx.is_directed_acyclic_graph(g)}   Depth: {depth}")
    print("Golden (paper [SVD], fig. 3): 16 nodes, 18 edges, depth 10.")
    _pause()
    return lattice, g
```

In `main()` replace the body after argument parsing with:

```python
    act_1_tuple()
    lattice, g = act_2_algorithm1()
```

Note: `HypothesisLattice` node objects are `Hyp` instances; `str(node)` is the code name via `__repr__` — that is why edge printing wraps nodes in `str()`. If the actual node type is the hypothesis object, `PAPER[str(u)]` works because `__repr__` returns `self.name`; verify on first run and, if nodes are plain strings, drop the `str()` calls.

- [ ] **Step 2: Run and verify golden numbers**

Run: `.venv/Scripts/python examples/norne_primer.py`
Expected: `Nodes: 16   Edges: 18   DAG: True   Depth: 10`.

- [ ] **Step 3: Lint + commit**

```bash
uvx ruff check examples/norne_primer.py
git add examples/norne_primer.py
git commit -m "feat(examples): norne primer act 2 — Algorithm 1 lattice"
```

---

### Task 3: Act 3 — Algorithm 2 (Lemma 2)

**Files:**
- Modify: `examples/norne_primer.py`

- [ ] **Step 1: Add act 3**

```python
def act_3_algorithm2(g_full: "nx.DiGraph") -> None:
    """Algorithm 2: incremental add is equivalent to a full rebuild (Lemma 2)."""
    _act(3, "Algorithm 2 — incremental addition (Lemma 2)")
    partial = HypothesisLattice(ALL_HYPS[:-1], WF)   # without H19 (paper H16)
    before = partial.lattice.number_of_edges()
    partial.add_hypothesis(HYP_OBJS["H19"])          # incremental, O(|H|) merges
    after_edges = {(str(u), str(v)) for u, v in partial.lattice.edges()}
    full_edges = {(str(u), str(v)) for u, v in g_full.edges()}
    new = sorted(after_edges - {(str(u), str(v))
                                for u, v in HypothesisLattice(ALL_HYPS[:-1], WF)
                                .lattice.edges()})
    print(f"\nLattice without H16: {before} edges.")
    print(f"add_hypothesis(H16) added edges: "
          f"{[f'{PAPER[u]}->{PAPER[v]}' for u, v in new]}")
    equal = after_edges == full_edges
    print(f"incremental == full rebuild: {equal}")
    print("Golden: True; new edges H8->H16, H14->H16 "
          "(liquid and watercut branches merge into the oil forecast).")
    if not equal:
        raise SystemExit("LEMMA 2 CHECK FAILED")
    _pause()
```

Call from `main()`:

```python
    act_3_algorithm2(g)
```

Note on expected new edges: gui_demo_spec demo 2 names them `H8→H19, H15→H19` in CODE names, i.e. paper `H8→H16, H14→H16`. Trust the printed run; if the actual pair differs, fix the printed "Golden:" line to match `NORNE_GOLDEN_EDGES` in tests/test_golden_claims.py — never the code.

- [ ] **Step 2: Run, verify `incremental == full rebuild: True`, lint, commit**

```bash
.venv/Scripts/python examples/norne_primer.py
uvx ruff check examples/norne_primer.py
git add examples/norne_primer.py
git commit -m "feat(examples): norne primer act 3 — Algorithm 2, Lemma 2"
```

---

### Task 4: Act 4 — Algorithm 4 + Theorem 1

**Files:**
- Modify: `examples/norne_primer.py`

- [ ] **Step 1: Add act 4**

```python
def act_4_plan_theorem1(g: "nx.DiGraph") -> set[str]:
    """Algorithm 4: cascade recompute plan; Theorem 1: correct + minimal."""
    _act(4, "Algorithm 4 — recompute plan; Theorem 1 — correctness/minimality")
    codes = [code for code, _, _, _ in HYPS]
    idx = {c: i for i, c in enumerate(codes)}
    edges = [(idx[str(u)], idx[str(v)]) for u, v in g.edges()]
    hg = HypothesisGraph.from_edges(len(codes), edges)

    changed = "H8"                       # scenario: the LPR fusion was re-fit
    cached = set(range(len(codes))) - {idx[changed]}
    plan = hg.plan(cached)
    plan_names = sorted(PAPER[codes[i]] for i in plan)
    print(f"\nScenario: {PAPER[changed]} changed -> plan P_ne = {plan_names}")
    print("Cascade: the whole liquid-fusion downstream (material balance,")
    print("watercut chain, oil forecast) is invalidated; upstream is not.")

    # Theorem 1, part 1 — correctness (independent reachability oracle).
    succ: dict[int, set[int]] = {i: set() for i in range(len(codes))}
    for u, v in edges:
        succ[u].add(v)

    def is_valid(p: set[int]) -> bool:
        if not set(range(len(codes))).difference(cached) <= p:
            return False
        return all(succ[x] <= p for x in p)

    correct = is_valid(plan)
    # Theorem 1, part 2 — subset-minimality: dropping ANY element breaks it.
    minimal = all(not is_valid(plan - {x}) for x in plan)
    print(f"\nTheorem 1: plan is correct: {correct};  "
          f"subset-minimal (dropping any element breaks correctness): {minimal}")
    if not (correct and minimal):
        raise SystemExit("THEOREM 1 CHECK FAILED")
    _pause()
    return {codes[i] for i in plan}
```

Call from `main()`:

```python
    p_ne = act_4_plan_theorem1(g)
```

- [ ] **Step 2: Run, verify `correct: True` and `minimal: True`, lint, commit**

```bash
.venv/Scripts/python examples/norne_primer.py
uvx ruff check examples/norne_primer.py
git add examples/norne_primer.py
git commit -m "feat(examples): norne primer act 4 — Algorithm 4 plan, Theorem 1"
```

---

### Task 5: Act 5 — execution (Runner)

**Files:**
- Modify: `examples/norne_primer.py`

- [ ] **Step 1: Add act 5**

```python
def act_5_run(g: "nx.DiGraph", p_ne: set[str]) -> None:
    """Execute the plan with the Runner (topological order, retries, statuses)."""
    _act(5, "Execution — Runner walks P_ne in topological order")
    # Synthetic model fits: deterministic metrics per hypothesis.
    r2_table = {code: 0.75 + 0.01 * i for i, (code, _, _, _) in enumerate(HYPS)}
    r2_table["H7"] = 0.65        # the pure-ML hypothesis under-performs alone

    def make_model(code: str):
        def model(config: dict) -> dict:
            return {"r2": r2_table[code], "aic": 100.0 - 10.0 * r2_table[code]}
        return model

    models = {code: make_model(code) for code, _, _, _ in HYPS}
    all_codes = {code for code, _, _, _ in HYPS}
    plan = {"p_ne": set(p_ne), "p_e": all_codes - set(p_ne)}
    runner = Runner()
    results = runner.execute(
        plan,
        models,
        lattice_edges=[(str(u), str(v)) for u, v in g.edges()],
        competes={"H5": {"H7"}, "H7": {"H5"}},
    )
    print(f"\nExecuted {len(results)} hypotheses "
          f"(P_ne recomputed: {len(plan['p_ne'])}, cached P_e skipped: "
          f"{len(plan['p_e'])} — no repository attached, so only P_ne runs).")
    for code in sorted(results, key=lambda c: PAPER[c]):
        r = results[code]
        print(f"  {PAPER[code]:<4} status={r['status']:<8} "
              f"epistemic={r.get('epistemic_status', '-'):<10} "
              f"metrics={r.get('metrics', {})}")
    _pause()
```

Call from `main()`:

```python
    act_5_run(g, p_ne)
```

- [ ] **Step 2: Run, verify every P_ne hypothesis prints `status=SUCCESS`, lint, commit**

```bash
.venv/Scripts/python examples/norne_primer.py
uvx ruff check examples/norne_primer.py
git add examples/norne_primer.py
git commit -m "feat(examples): norne primer act 5 — Runner execution"
```

---

### Task 6: Acts 6-7 — comparison + golden self-check, README

**Files:**
- Modify: `examples/norne_primer.py`
- Modify: `examples/README.md` (add primer section)

- [ ] **Step 1: Add acts 6 and 7, finish `main`**

```python
def act_6_compare() -> None:
    """Compare two competing hypotheses (Definitions 9-11)."""
    _act(6, "Comparison — physics (H5) vs ML (H7) liquid-rate hypotheses")
    import numpy as np

    rng = np.random.default_rng(42)
    y_true = rng.normal(100.0, 10.0, size=50)
    pred_phys = y_true + rng.normal(0.0, 3.0, size=50)   # tighter residuals
    pred_ml = y_true + rng.normal(0.0, 6.0, size=50)
    err_phys = list(np.abs(y_true - pred_phys))
    err_ml = list(np.abs(y_true - pred_ml))
    p = sign_test(err_phys, err_ml)
    aic_phys = compute_aic(3, gaussian_log_likelihood(y_true, pred_phys))
    aic_ml = compute_aic(12, gaussian_log_likelihood(y_true, pred_ml))
    print(f"\nSign test (|err_H5| vs |err_H7|): p = {p:.4f} "
          "(small p -> H5 errors systematically smaller)")
    print(f"AIC: H5 (3 params) = {aic_phys:.1f}   H7 (12 params) = {aic_ml:.1f}")
    print("Verdict: H5 preferred on both criteria; H8 fuses the two, which is")
    print("why the paper keeps BOTH in the lattice instead of discarding H7.")
    _pause()


def act_7_selfcheck(g: "nx.DiGraph") -> None:
    """Assert golden values from tests/test_golden_claims.py."""
    _act(7, "Self-check against golden claims")
    ok = (g.number_of_nodes() == 16 and g.number_of_edges() == 18
          and nx.is_directed_acyclic_graph(g)
          and nx.dag_longest_path_length(g) == 10)
    print(f"\n16 nodes / 18 edges / DAG / depth 10: {ok}")
    if not ok:
        raise SystemExit("GOLDEN SELF-CHECK FAILED")
    print("\nPRIMER OK")
```

`main()` final body:

```python
    act_1_tuple()
    lattice, g = act_2_algorithm1()
    act_3_algorithm2(g)
    p_ne = act_4_plan_theorem1(g)
    act_5_run(g, p_ne)
    act_6_compare()
    act_7_selfcheck(g)
```

(The `lattice` variable is intentionally kept: it demonstrates the object API in act 2 and keeps OWL individuals alive for the whole run; prefix with `_` if ruff flags it unused: `_lattice, g = ...`.)

- [ ] **Step 2: Full run, both modes**

Run: `.venv/Scripts/python examples/norne_primer.py`
Expected: 7 act headers, final line `PRIMER OK`, exit 0.
Run: `echo "" | .venv/Scripts/python examples/norne_primer.py --pause | tail -1` — still completes (pauses read EOF).

- [ ] **Step 3: Add README section**

In `examples/README.md`, add under the script index:

```markdown
### norne_primer.py — end-to-end walkthrough

The full virtual-experiment lifecycle on the Norne HybridCRM case in one
readable script: the Definition 1 tuple <O,H,M,R,W,C>, Algorithm 1 (lattice:
16 nodes / 18 edges / depth 10), Algorithm 2 incremental update (Lemma 2),
Algorithm 4 recompute plan with a Theorem 1 correctness/minimality check,
Runner execution with epistemic statuses, and statistical comparison of
competing hypotheses. Self-checks against the golden values pinned by
`tests/test_golden_claims.py`.

    python examples/norne_primer.py            # full narrative
    python examples/norne_primer.py --pause    # step through act by act

Expected tail: `PRIMER OK` (exit 0).
```

- [ ] **Step 4: Final checks**

```bash
uvx ruff check examples/norne_primer.py
uvx ruff format --check examples/norne_primer.py
uvx mypy examples/norne_primer.py --ignore-missing-imports
.venv/Scripts/python -m pytest tests/test_golden_claims.py -q
```
Expected: all clean; golden 48 passed (primer must not disturb anything — it is read-only w.r.t. the package).

- [ ] **Step 5: Commit**

```bash
git add examples/norne_primer.py examples/README.md
git commit -m "feat(examples): norne primer acts 6-7 — comparison, golden self-check, README"
```
