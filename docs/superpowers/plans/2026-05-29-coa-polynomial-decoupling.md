# COA Polynomial Causal Core + owlready Decoupling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the exponential, owlready-coupled, latex2sympy-dependent causal-ordering code in `hyppo/coa/_base.py` with a pure-Python polynomial core (Dulmage–Mendelsohn decomposition) that runs on Python 3.13 with zero native dependencies.

**Architecture:** A pure stdlib core module `hyppo/coa/causal.py` implements completeness, perfect matching (Kuhn), strongly connected components (Tarjan), block decomposition, causal mapping, and transitive closure on plain data (equation = `frozenset[str]`). `hyppo/coa/_base.py` keeps `Equation`/`Structure` as thin plain-Python classes (no owlready `Thing`/`Artefact`) that parse formulas via `sympy.sympify` and delegate all algorithms to the core.

**Tech Stack:** Python 3.13, stdlib only for the core; `sympy` (already a core dep) for formula parsing in `Equation`; `pytest` for tests. No owlready2, no latex2sympy, no networkx in the algorithm path.

**Run all commands from** `F:\git-repos\wf\diss\hyppo-ref` **using its venvs.** The pure core + tests run on Python 3.13 via the default `.venv` (`.venv/Scripts/python.exe`). `sympy` must be importable there.

---

### Task 1: Pure core — variables, completeness, perfect matching

**Files:**
- Create: `hyppo/coa/causal.py`
- Test: `tests/test_causal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_causal.py
from hyppo.coa import causal


def test_variables_and_completeness():
    eqs = [frozenset({"x_1"}), frozenset({"x_1", "x_2"})]
    assert causal.variables(eqs) == {"x_1", "x_2"}
    assert causal.is_complete(eqs)
    assert not causal.is_complete([frozenset({"x_1", "x_2", "x_3"})])


def test_perfect_matching_assigns_contained_vars():
    eqs = [frozenset({"x_0", "x_1"}), frozenset({"x_1", "x_2"}),
           frozenset({"x_0", "x_2"})]  # 3-cycle, irreducible
    m = causal.perfect_matching(eqs)
    assert m is not None
    assert len(m) == 3
    for i, v in m.items():
        assert v in eqs[i]                 # every assignment is a contained var
    assert len(set(m.values())) == 3       # distinct vars


def test_perfect_matching_none_when_singular():
    # 2 equations referencing only 1 variable -> no perfect matching
    eqs = [frozenset({"x_0"}), frozenset({"x_0"})]
    assert causal.perfect_matching(eqs) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_causal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hyppo.coa.causal'`

- [ ] **Step 3: Implement the core foundation**

```python
# hyppo/coa/causal.py
"""Pure causal-ordering core for COA. stdlib only -- no owlready, sympy, networkx.

An *equation* is a set of variable names (``frozenset[str]``); a *structure* is a
list of equations. A structure is *complete* iff ``|equations| == |variables|``.
The Dulmage-Mendelsohn decomposition (a perfect matching plus the strongly
connected components of the matching-induced dependency digraph) yields the
irreducible ("minimal complete") blocks in polynomial time.
"""
from __future__ import annotations

from collections import defaultdict


def variables(equations):
    """All distinct variable names across the equations."""
    out = set()
    for eq in equations:
        out |= set(eq)
    return out


def is_complete(equations):
    """A structure is complete iff |equations| == |distinct variables|."""
    return len(equations) == len(variables(equations))


def perfect_matching(equations):
    """Match each equation to a distinct variable it contains (Kuhn's algorithm).

    Returns ``{eq_index: var}`` saturating every equation, or ``None`` if no such
    matching exists (the structure is structurally singular). Candidate variables
    are tried in sorted order, giving a deterministic, name-stable result.
    """
    cand = [sorted(eq) for eq in equations]
    match_var = {}  # var -> eq index

    def try_aug(i, seen):
        for v in cand[i]:
            if v not in seen:
                seen.add(v)
                if v not in match_var or try_aug(match_var[v], seen):
                    match_var[v] = i
                    return True
        return False

    for i in range(len(equations)):
        if not try_aug(i, set()):
            return None
    return {i: v for v, i in match_var.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_causal.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add hyppo/coa/causal.py tests/test_causal.py
git commit -m "feat(coa): pure core foundation -- variables, completeness, Kuhn matching"
```

---

### Task 2: SCC, block decomposition, minimal blocks (DM)

**Files:**
- Modify: `hyppo/coa/causal.py`
- Test: `tests/test_causal.py`

- [ ] **Step 1: Write failing tests (with exhaustive brute-force oracle)**

```python
# append to tests/test_causal.py
import random
from itertools import combinations


def _brute_minimal(eqs):
    """Reference: inclusion-minimal complete subsets of equations (exponential)."""
    n = len(eqs)
    complete = []
    for r in range(1, n + 1):
        for c in combinations(range(n), r):
            sub = [eqs[i] for i in c]
            if len(sub) == len(set().union(*sub)):
                complete.append(frozenset(c))
    return {cs for cs in complete if not any(o < cs for o in complete)}


def _random_complete(n_eq, rng):
    """n_eq equations over exactly n_eq vars with a built-in perfect matching."""
    eqs = []
    for i in range(n_eq):
        own = f"x_{i}"
        k = rng.randint(0, min(2, n_eq - 1))
        extras = rng.sample([j for j in range(n_eq) if j != i], k) if k else []
        eqs.append(frozenset({own, *(f"x_{j}" for j in extras)}))
    return eqs


def test_minimal_blocks_named_example():
    eqs = [frozenset({"x_1"}), frozenset({"x_2"}), frozenset({"x_3"}),
           frozenset({"x_1", "x_2", "x_3", "x_4", "x_5"}),
           frozenset({"x_1", "x_3", "x_4", "x_5"}),
           frozenset({"x_4", "x_6"}), frozenset({"x_5", "x_7"})]
    got = causal.minimal_blocks(eqs)
    assert set(got) == _brute_minimal(eqs)
    assert set(got) == {frozenset({0}), frozenset({1}), frozenset({2})}


def test_minimal_blocks_equiv_bruteforce_exhaustive():
    rng = random.Random(42)
    for _ in range(3000):
        n = rng.randint(1, 8)
        eqs = _random_complete(n, rng)
        if not causal.is_complete(eqs):
            continue
        assert set(causal.minimal_blocks(eqs)) == _brute_minimal(eqs)


def test_block_decomposition_partitions_all_equations():
    eqs = _random_complete(7, random.Random(1))
    blocks = causal.block_decomposition(eqs)
    covered = set()
    for b in blocks:
        assert not (covered & b)          # disjoint
        covered |= b
    assert covered == set(range(len(eqs)))  # covers every equation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_causal.py -k "block or minimal" -v`
Expected: FAIL with `AttributeError: module 'hyppo.coa.causal' has no attribute 'minimal_blocks'`

- [ ] **Step 3: Implement SCC + block decomposition + minimal blocks**

```python
# append to hyppo/coa/causal.py
def _dependency_graph(equations, matching):
    """Digraph over variables: edge u -> v means the equation matched to v also
    contains u, i.e. v depends on u. Every variable is present as a node."""
    adj = defaultdict(set)
    for v in variables(equations):
        adj[v]  # touch -> ensure isolated nodes exist
    for i, eq in enumerate(equations):
        v = matching[i]
        for u in eq:
            if u != v:
                adj[u].add(v)
    return adj


def strongly_connected_components(adj):
    """Tarjan's SCC. ``adj``: {node: set(successors)}. Returns list of frozensets."""
    index = {}
    low = {}
    on_stack = set()
    stack = []
    result = []
    counter = [0]

    def strongconnect(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj[v]:
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.add(w)
                if w == v:
                    break
            result.append(frozenset(comp))

    for v in list(adj.keys()):
        if v not in index:
            strongconnect(v)
    return result


def block_decomposition(equations):
    """Irreducible blocks as frozensets of equation indices. Returns None if the
    structure admits no perfect matching."""
    m = perfect_matching(equations)
    if m is None:
        return None
    var_to_eq = {v: i for i, v in m.items()}
    comps = strongly_connected_components(_dependency_graph(equations, m))
    return [frozenset(var_to_eq[v] for v in comp) for comp in comps]


def minimal_blocks(equations):
    """Inclusion-minimal complete subsets of equations (the self-contained
    irreducible blocks -- those that reference only their own matched variables).
    Returns a list of frozensets of equation indices, or None if unmatchable."""
    blocks = block_decomposition(equations)
    if blocks is None:
        return None
    m = perfect_matching(equations)
    out = []
    for b in blocks:
        own = {m[i] for i in b}
        referenced = set()
        for i in b:
            referenced |= set(equations[i])
        if referenced <= own:            # no external variable -> self-contained
            out.append(b)
    return out
```

**Note on recursion:** `strongconnect` recurses to depth ≤ number of variables in one structure (small by design — structures are bounded). No recursion-limit change needed for the tested sizes; if a future caller builds very deep single structures, convert to an explicit-stack Tarjan.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_causal.py -v`
Expected: all passed (incl. 3000-case exhaustive equivalence)

- [ ] **Step 5: Commit**

```bash
git add hyppo/coa/causal.py tests/test_causal.py
git commit -m "feat(coa): DM block decomposition + minimal blocks (== brute-force, poly)"
```

---

### Task 3: Causal mapping + transitive closure

**Files:**
- Modify: `hyppo/coa/causal.py`
- Test: `tests/test_causal.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_causal.py
def test_causal_mapping_valid():
    eqs = [frozenset({"x_0", "x_1"}), frozenset({"x_1", "x_2"}),
           frozenset({"x_0", "x_2"})]
    m = causal.causal_mapping(eqs)
    for i, v in m.items():
        assert v in eqs[i]
    assert len(set(m.values())) == 3


def test_transitive_closure_triangular():
    # x_0 solved alone; x_1 depends on x_0; x_2 depends on x_0,x_1
    eqs = [frozenset({"x_0"}), frozenset({"x_0", "x_1"}),
           frozenset({"x_0", "x_1", "x_2"})]
    tc = causal.transitive_closure(eqs)
    assert tc["x_0"] == {"x_1", "x_2"}   # x_0 reaches dependents
    assert tc["x_2"] == set()            # leaf depends-target reaches nothing
    assert "x_0" not in tc["x_0"]        # self excluded


def test_transitive_closure_excludes_self_in_cycle():
    eqs = [frozenset({"x_0", "x_1"}), frozenset({"x_1", "x_2"}),
           frozenset({"x_0", "x_2"})]   # 3-cycle
    tc = causal.transitive_closure(eqs)
    for v in ("x_0", "x_1", "x_2"):
        assert v not in tc[v]            # never include self
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_causal.py -k "causal_mapping or transitive" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'causal_mapping'`

- [ ] **Step 3: Implement**

```python
# append to hyppo/coa/causal.py
def causal_mapping(equations):
    """Full causal mapping eq_index -> variable (deterministic, sorted tie-break).
    None if the structure admits no perfect matching."""
    return perfect_matching(equations)


def transitive_closure(equations):
    """Reachability in the dependency digraph: {var: set(vars that depend on it,
    transitively)}. The variable itself is excluded from its own set. None if
    unmatchable."""
    m = perfect_matching(equations)
    if m is None:
        return None
    adj = _dependency_graph(equations, m)
    tc = {}
    for start in adj:
        seen = set()
        st = [start]
        while st:
            node = st.pop()
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    st.append(nb)
        seen.discard(start)
        tc[start] = seen
    return tc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_causal.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add hyppo/coa/causal.py tests/test_causal.py
git commit -m "feat(coa): causal mapping + transitive closure on pure core"
```

---

### Task 4: Stability stress test (10k structures, Python 3.13, no crash)

**Files:**
- Test: `tests/test_causal.py`

- [ ] **Step 1: Write the stress test**

```python
# append to tests/test_causal.py
def test_stress_10k_no_crash():
    """10000 random complete structures through the full core in one process.
    The pure core has no native deps, so this must complete without segfault."""
    rng = random.Random(2026)
    runs = 0
    for _ in range(10000):
        n = rng.randint(2, 12)
        eqs = _random_complete(n, rng)
        if not causal.is_complete(eqs):
            continue
        m = causal.causal_mapping(eqs)
        assert m is not None
        for i, v in m.items():                 # bug-fix invariant
            assert v in eqs[i]
        causal.minimal_blocks(eqs)
        causal.transitive_closure(eqs)
        runs += 1
    assert runs > 9000
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/test_causal.py::test_stress_10k_no_crash -v`
Expected: PASS, process exits 0 (no segfault, no nonsense TypeError)

- [ ] **Step 3: Commit**

```bash
git add tests/test_causal.py
git commit -m "test(coa): 10k stress test confirms pure core stability on 3.13"
```

---

### Task 5: Rewrite `coa/_base.py` — plain Equation/Structure delegating to core

**Files:**
- Modify (full rewrite): `hyppo/coa/_base.py`
- Test: `tests/test_coa.py` (Task 6 adapts it)

- [ ] **Step 1: Write the new `_base.py`**

Replace the entire contents of `hyppo/coa/_base.py` with:

```python
"""COA data types: Equation and Structure (plain Python, no owlready).

Formula parsing uses sympy.sympify (works on Python 3.13); all causal-ordering
algorithms delegate to the pure core in :mod:`hyppo.coa.causal`. Variables are
sympy Symbols on the public surface; the core works on their names internally.
"""
from __future__ import annotations

from sympy import sympify

from hyppo.coa import causal


class Equation:
    """A single equation, identified by its formula string, with the set of free
    variables it contains (sympy Symbols)."""

    def __init__(self, formula=None, vars=None):
        self.formula = formula
        if vars is not None:
            self.vars = sorted(vars, key=lambda s: s.name)
        elif formula is not None:
            self.vars = self._parse(formula)
        else:
            raise ValueError("Equation requires either 'formula' or 'vars'")
        self.equation = self._expr(formula) if formula is not None else None

    @staticmethod
    def _expr(formula):
        s = str(formula)
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            s = f"({lhs})-({rhs})"
        return sympify(s)

    @classmethod
    def _parse(cls, formula):
        return sorted(cls._expr(formula).free_symbols, key=lambda x: x.name)

    def get_vars(self):
        return self.vars


class Structure:
    """A set of equations over a set of variables (sympy Symbols)."""

    def __init__(self, equations, vars=None):
        self.equations = list(equations)
        allv = set()
        for eq in self.equations:
            allv |= set(eq.vars)
        self.vars = set(vars) if vars is not None else allv
        self._name2sym = {s.name: s for s in (allv | self.vars)}
        self._eqsets = [frozenset(s.name for s in eq.vars) for eq in self.equations]

    # ---- predicates -----------------------------------------------------
    def is_complete(self):
        return causal.is_complete(self._eqsets)

    def is_structure(self):
        """True iff every subset of equations references at least as many
        variables (Hall's condition) -- equivalently, a matching saturating all
        equations exists."""
        return causal.perfect_matching(self._eqsets) is not None

    def is_minimal(self):
        return self.is_complete() and not self.find_minimal_structures()

    # ---- decomposition --------------------------------------------------
    def find_minimal_structures(self):
        """Inclusion-minimal complete *proper* substructures, as Structure objects.
        Returns [] when the structure is itself irreducible (no proper minimal
        substructure) -- matching the historical contract."""
        if not self.is_complete():
            return []
        blocks = causal.minimal_blocks(self._eqsets) or []
        result = []
        for b in blocks:
            if len(b) == len(self.equations):
                continue  # the whole structure is not a *proper* substructure
            result.append(Structure([self.equations[i] for i in b]))
        return result

    def build_full_causal_mapping(self):
        """{equation.formula: variable Symbol} -- each equation's computed var."""
        if not self.is_complete():
            raise Exception("Structure is not complete")
        m = causal.causal_mapping(self._eqsets)
        if m is None:
            raise Exception("Structure admits no perfect matching")
        return {self.equations[i].formula: self._name2sym[v] for i, v in m.items()}

    def build_transitive_closure(self):
        """{variable Symbol: set of transitively dependent variable Symbols}."""
        tc = causal.transitive_closure(self._eqsets)
        if tc is None:
            return {}
        return {self._name2sym[k]: {self._name2sym[n] for n in deps}
                for k, deps in tc.items()}

    # ---- set operations -------------------------------------------------
    def union(self, other):
        return Structure(self.equations + list(other.equations))

    def difference(self, others):
        drop = set()
        for s in others:
            drop |= {id(e) for e in s.equations}
        return Structure([e for e in self.equations if id(e) not in drop])

    # ---- variable roles -------------------------------------------------
    def exogenous(self):
        exo = set()
        for eq in self.equations:
            if len(eq.vars) == 1:
                exo |= set(eq.vars)
        return exo

    def endogenous(self):
        return self.vars - self.exogenous()
```

**Removed deliberately:** `Thing`/`Artefact` inheritance, `with virtual_experiment_onto:`, `from latex2sympy import strToSympy`, `powerset`, recursive `find_minimal_structures`, `vars=`-substitution in `__init__`, `build_matrix`/`build_dcg`/`h_encode` (unused by tests, examples, and `lattice_constructor`; reintroduce later behind lazy numpy/graphviz imports only if a real caller needs them).

- [ ] **Step 2: Smoke-test the new module imports on 3.13**

Run: `.venv/Scripts/python.exe -c "from hyppo.coa._base import Structure, Equation; print('import OK')"`
Expected: `import OK` (no latex2sympy / owlready errors)

- [ ] **Step 3: Verify the dissertation 7-equation example end-to-end**

Run:
```bash
.venv/Scripts/python.exe -c "
from hyppo.coa._base import Structure, Equation
f=[r'f_1(x_1)=0',r'f_2(x_2)=0',r'f_3(x_3)=0',r'x_1+x_2+x_3+x_4+x_5=0',r'x_1 + 6*x_3+x_4+x_5=0',r'f_6(x_4, x_6)=0',r'f_7(x_5, x_7)=0']
s=Structure([Equation(formula=x) for x in f])
print('complete', s.is_complete())
print('minimal', sorted(sorted(v.name for v in m.vars) for m in s.find_minimal_structures()))
fcm=s.build_full_causal_mapping()
for k,v in fcm.items(): assert v in {e for e in Equation(formula=k).vars}
print('fcm valid, tc keys', len(s.build_transitive_closure()))
"
```
Expected: `complete True`, minimal blocks `[['x_1'], ['x_2'], ['x_3']]`, `fcm valid, tc keys 7`

- [ ] **Step 4: Commit**

```bash
git add hyppo/coa/_base.py
git commit -m "refactor(coa): plain Equation/Structure delegating to pure core; drop owlready+latex2sympy"
```

---

### Task 6: Adapt and un-skip existing tests

**Files:**
- Modify: `tests/test_coa.py`
- Run: `tests/test_lattice.py`

- [ ] **Step 1: Update `tests/test_coa.py` header**

The current module skips when `latex2sympy` is missing. Replace lines 1-10 (the docstring + `importorskip`) with a direct import (the module now needs only `sympy`):

```python
"""Tests for COA (causal ordering analysis). Pure Python + sympy; runs on 3.13."""
import pytest

from hyppo.coa._base import Structure, Equation
```

Then in each test body, replace `from hyppo.coa._base import ...` lines (they become redundant) — leave the rest of each test as-is. The `test_powerset` test (lines 13-21) references the removed `powerset` helper: **delete that test** (powerset is gone by design).

- [ ] **Step 2: Run the COA tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_coa.py -v`
Expected: all remaining tests pass (`test_equation_parses_vars`, `test_structure_is_complete`, `test_structure_not_complete`, `test_structure_is_structure`, `test_exogenous_endogenous`, `test_transitive_closure_returns_dict`, `test_find_minimal_structures`)

- [ ] **Step 3: Run the lattice tests (must be unaffected)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_lattice.py -v`
Expected: all pass (they use a duck-typed `FakeStructure`, independent of this change)

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: no regressions introduced by this change (pre-existing unrelated skips/failures, if any, unchanged)

- [ ] **Step 5: Commit**

```bash
git add tests/test_coa.py
git commit -m "test(coa): un-skip COA tests (run on 3.13), drop powerset test"
```

---

### Task 7: Drop the phantom dependency from `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:22`

- [ ] **Step 1: Edit the `coa` optional-dependencies extra**

Current line 22:
```toml
coa = ["latex2sympy2>=1.9", "graphviz>=0.20"]
```
Replace with (sympy is already a core dep; graphviz only needed if visualization is reintroduced):
```toml
coa = ["graphviz>=0.20"]
```

- [ ] **Step 2: Verify import without the coa extra installed**

Run: `.venv/Scripts/python.exe -c "import hyppo.coa.causal, hyppo.coa._base; print('coa imports without latex2sympy')"`
Expected: `coa imports without latex2sympy`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build(coa): drop phantom latex2sympy2 dependency from coa extra"
```

---

### Task 8: Re-measure the |H|-exponent on the real (now-runnable) library

**Files:**
- Create: `tests/test_algorithm1_scaling.py`

- [ ] **Step 1: Write the scaling test**

```python
# tests/test_algorithm1_scaling.py
"""Algorithm 1 (lattice build) scales polynomially in |H| with the polynomial
core. Runs the real library on Python 3.13; asserts the fitted log-log exponent
is well below the old exponential blow-up and consistent with ~quadratic."""
import math
import random
import time

from hyppo.coa._base import Structure, Equation


def _gen_struct(rng, n_eq=5, pool=20):
    av = [f"x_{k}" for k in range(pool)]
    chosen = rng.sample(av, n_eq)
    eqs = []
    for i in range(n_eq):
        extras = rng.sample([v for v in chosen if v != chosen[i]],
                            rng.randint(1, min(3, n_eq - 1)))
        eqs.append(Equation(formula="+".join([chosen[i], *extras]) + "=0"))
    return Structure(eqs)


def _build_cost(n_h, rng):
    hyps = [_gen_struct(rng) for _ in range(n_h)]
    edges = [(i, j) for i in range(n_h) for j in range(i + 1, n_h)
             if rng.random() < 0.3]
    adj = {i: set() for i in range(n_h)}
    for u, v in edges:
        adj[u].add(v)
    reach = {}
    for s in range(n_h):
        seen, st = set(), [s]
        while st:
            c = st.pop()
            for nb in adj[c]:
                if nb not in seen:
                    seen.add(nb)
                    st.append(nb)
        reach[s] = seen
    for i in range(n_h):
        for j in reach[i]:
            su = hyps[i].union(hyps[j])
            if su.is_complete():
                su.build_transitive_closure()


def test_algorithm1_exponent_is_polynomial():
    h_values = [10, 20, 30, 50, 70, 100, 150]
    means = []
    for n_h in h_values:
        ts = []
        for rep in range(10):
            rng = random.Random(42 + rep)
            t0 = time.perf_counter()
            _build_cost(n_h, rng)
            ts.append(time.perf_counter() - t0)
        means.append(sum(ts) / len(ts))
    lh = [math.log(h) for h in h_values]
    lt = [math.log(m) for m in means]
    n = len(lh)
    mx, my = sum(lh) / n, sum(lt) / n
    a = sum((x - mx) * (y - my) for x, y in zip(lh, lt)) / sum((x - mx) ** 2 for x in lh)
    print(f"fitted exponent a = {a:.3f}")
    assert a < 2.8, f"exponent {a:.3f} too high -- expected near-quadratic"
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/test_algorithm1_scaling.py -v -s`
Expected: PASS, printed `fitted exponent a ≈ 2.0–2.4` (no exponential)

- [ ] **Step 3: Commit**

```bash
git add tests/test_algorithm1_scaling.py
git commit -m "test(coa): Algorithm 1 scales polynomially in |H| on the real library"
```

---

### Task 9: Remove scratch artifacts

**Files:**
- Delete: `scripts/poly_minimal_verify.py`, `scripts/poly_exponent_partc.py`, `scripts/run_library_dm.py`, `scripts/latex2sympy.py`, `scripts/test_powerset_fix.py`, `scripts/test_fix_deterministic.py` (these live in the **diss** repo `F:\git-repos\wf\diss\scripts\`, not hyppo-ref)

- [ ] **Step 1: Confirm their findings are now covered by committed tests**

The equivalence (`poly_minimal_verify`), exponent (`poly_exponent_partc`), and fix (`test_*_fix`) checks are all reproduced by `tests/test_causal.py` and `tests/test_algorithm1_scaling.py`. The `latex2sympy.py` shim is obsolete (sympy parsing is now in `Equation`).

- [ ] **Step 2: Delete the scratch files**

```bash
rm F:/git-repos/wf/diss/scripts/poly_minimal_verify.py \
   F:/git-repos/wf/diss/scripts/poly_exponent_partc.py \
   F:/git-repos/wf/diss/scripts/run_library_dm.py \
   F:/git-repos/wf/diss/scripts/latex2sympy.py \
   F:/git-repos/wf/diss/scripts/test_powerset_fix.py \
   F:/git-repos/wf/diss/scripts/test_fix_deterministic.py
```

- [ ] **Step 3: (diss repo) Note in commit** — these are in the `diss` repo; commit there separately only if the user asks. Do not commit deletions to hyppo-ref.

---

## Definition of Done (verify all)

- [ ] `.venv/Scripts/python.exe -m pytest tests/test_causal.py tests/test_coa.py tests/test_lattice.py tests/test_algorithm1_scaling.py -v` — all green on **Python 3.13**
- [ ] `test_stress_10k_no_crash` passes (process exits 0, no segfault)
- [ ] `test_minimal_blocks_equiv_bruteforce_exhaustive` passes (DM == brute-force, 3000 cases)
- [ ] `test_algorithm1_exponent_is_polynomial` prints `a < 2.8`
- [ ] `import hyppo.coa._base` succeeds without latex2sympy/owlready
- [ ] Phantom `latex2sympy2` dependency removed from `pyproject.toml`
- [ ] Scratch artifacts removed
