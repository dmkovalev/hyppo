# Депонируемый листинг исходного текста программы для ЭВМ

**Название программы для ЭВМ:** Платформа управления виртуальными
экспериментами Hyppo (gedanken)

**Автор:** Ковалев Дмитрий

**Версия:** 1.1.0

**Дата:** 2026-07-09

**Язык программирования:** Python 3.11+

**Состав настоящего листинга:** полные исходные тексты 10 ключевых модулей,
реализующих причинно-структурное ядро платформы (Алгоритм 1 — построение
решётки гипотез, Алгоритм 2 — инкрементальное добавление, Алгоритм 3 — OWL
проверка корректной определённости, Алгоритм 4 и Теорема 1 — построение и
минимальность плана пересчёта, исполнение плана с эпистемическими
статусами гипотез), плюс сквозной пример полного жизненного цикла
виртуального эксперимента (`examples/norne_primer.py`). Далее приведена
сводная таблица объёма (файл — число строк) для остальных 66 модулей
пакета `hyppo`, не включённых в листинг целиком.

**Оценочный объём настоящего листинга:** ≈ 2 220 строк (≈ 55 строк/стр.) —
**≈ 41 страница**.

---

## Перечень депонируемых модулей (полный текст)

1. `hyppo/coa/_base.py` — типы Equation/Structure (COA)
2. `hyppo/coa/graph.py` — HypothesisGraph: Алгоритмы 1, 2, 4 (эталонная реализация над чистым ядром causal.py)
3. `hyppo/coa/causal.py` — чистое ядро причинного упорядочивания (Дулмедж-Мендельсон: паросочетание Хопкрофта-Карпа + SCC Тарьяна)
4. `hyppo/lattice_constructor/_base.py` — HypothesisLattice: Алгоритм 1/2 над OWL-гипотезами
5. `hyppo/planner/_base.py` — Алгоритм 4: построение минимального плана пересчёта (P_ne/P_e)
6. `hyppo/runner/_base.py` — исполнение плана, эпистемические статусы гипотез
7. `hyppo/core/_base.py` — OWL-онтология виртуального эксперимента (Определение 1, Теорема 1)
8. `hyppo/core/_epistemic.py` — эпистемический статус гипотезы (PROPOSED/SUPPORTED/REFUTED/SUPERSEDED)
9. `hyppo/manager/_base.py` — оркестрация полного жизненного цикла виртуального эксперимента
10. `examples/norne_primer.py` — сквозной пример: Norne HybridCRM, Алгоритмы 1–4, Теорема 1, золотые значения

---

=== hyppo/coa/_base.py ===

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
        if formula is not None and vars is not None:
            raise ValueError("Equation accepts 'formula' or 'vars', not both")
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
        return sympify(s, locals={})

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
        """True iff a matching saturating every equation exists (Hall's
        condition: every subset of equations references at least as many
        variables)."""
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
        """{variable Symbol: set of transitively dependent variable Symbols}.

        Returns {} when the structure is incomplete OR structurally singular
        (no perfect matching). This is deliberate: lattice construction calls
        this over many candidate unions, some legitimately unsolvable, and must
        not raise. (build_full_causal_mapping, a direct query, raises instead.)
        """
        tc = causal.transitive_closure(self._eqsets)
        if tc is None:
            return {}
        return {
            self._name2sym[k]: {self._name2sym[n] for n in deps}
            for k, deps in tc.items()
        }

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

---

=== hyppo/coa/graph.py ===

```python
"""Hypothesis-graph layer over the COA core -- Algorithms 1, 2 and 4 of the paper
"Планирование виртуальных экспериментов с повторным использованием вычисленных
фрагментов" (Kovalev, ИиП 2026).

A *hypothesis* is a structure: a ``list[frozenset[str]]`` of equations over
variable names. A :class:`HypothesisGraph` holds hypotheses connected by
``derived_by`` edges (a DAG) and is the single source of truth for the three
planning algorithms. All causal work delegates to :mod:`hyppo.coa.causal`
(the Dulmage-Mendelsohn decomposition: Kuhn matching + Tarjan SCC), so the
measured complexity is the library's, not a script's reimplementation.

Pure stdlib (imports only :mod:`hyppo.coa.causal`); safe to import without
sympy/owlready.
"""

from __future__ import annotations

from collections import defaultdict

from hyppo.coa import causal


class HypothesisGraph:
    """A DAG of hypotheses over the ``derived_by`` relation.

    Nodes are hypotheses (structures), identified by insertion index; edges
    ``i -> j`` mean *j is derived_by i*. The graph exposes the three algorithms
    of the paper:

    * :meth:`build` -- Algorithm 1 (build the hypothesis lattice),
    * :meth:`add_hypothesis` -- Algorithm 2 (incremental add),
    * :meth:`plan` -- Algorithm 4 (cascade-aware recompute planning).
    """

    def __init__(self) -> None:
        self._hyps: list[list[frozenset[str]]] = []
        self._edges: set[tuple[int, int]] = set()
        self._adj: dict[int, set[int]] = defaultdict(set)

    # ---- construction ---------------------------------------------------
    def add(self, equations) -> int:
        """Add a hypothesis (a structure) as a new node; return its index."""
        self._hyps.append([frozenset(eq) for eq in equations])
        return len(self._hyps) - 1

    def connect(self, i: int, j: int) -> None:
        """Add a ``derived_by`` edge ``i -> j`` (hypothesis *j* is derived from *i*)."""
        n = len(self._hyps)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError("hypothesis index out of range")
        self._edges.add((i, j))
        self._adj[i].add(j)

    def __len__(self) -> int:
        return len(self._hyps)

    @classmethod
    def from_edges(cls, n: int, edges) -> "HypothesisGraph":
        """Build a graph of ``n`` hypotheses (with empty structures -- planning uses
        only the topology) and the given ``derived_by`` edges. Convenience for
        cascade planning over a pre-existing adjacency/edge list."""
        g = cls()
        for _ in range(n):
            g.add([])
        for u, v in edges:
            g.connect(u, v)
        return g

    # ---- reachability ---------------------------------------------------
    def _reachable(self) -> dict[int, set[int]]:
        """For each node, the set of nodes reachable from it along ``derived_by``."""
        out: dict[int, set[int]] = {}
        for s in range(len(self._hyps)):
            seen: set[int] = set()
            st = [s]
            while st:
                nd = st.pop()
                for nb in self._adj[nd]:
                    if nb not in seen:
                        seen.add(nb)
                        st.append(nb)
            out[s] = seen
        return out

    # ---- Algorithm 1: build_lattice -------------------------------------
    def build(self) -> list[tuple[int, int]]:
        """Algorithm 1. For every reachable pair ``(i, j)`` whose union of
        structures is complete, compute its causal transitive closure via the DM
        core. Returns the lattice edges -- the pairs admitting a complete causal
        union. Cost is ``O(|H|^2 * s_max * v_max)`` (Lemma 1)."""
        reach = self._reachable()
        lattice: list[tuple[int, int]] = []
        for i in range(len(self._hyps)):
            for j in reach[i]:
                union = self._hyps[i] + self._hyps[j]
                if causal.is_complete(union):
                    causal.transitive_closure(union)
                    lattice.append((i, j))
        return lattice

    # ---- Algorithm 2: add_hypothesis ------------------------------------
    def add_hypothesis(self, equations) -> int:
        """Algorithm 2. Add a new hypothesis and incrementally compute its causal
        unions against the existing hypotheses -- ``O(|H| * s_max * v_max)``
        closures (Lemma 2), without rebuilding the whole lattice. Returns the new
        node index. Connect its ``derived_by`` edges afterwards via :meth:`connect`."""
        h_new = [frozenset(eq) for eq in equations]
        for i in range(len(self._hyps)):
            union = self._hyps[i] + h_new
            if causal.is_complete(union):
                causal.transitive_closure(union)
        return self.add(h_new)

    # ---- Algorithm 4: plan ----------------------------------------------
    def plan(self, cached) -> set[int]:
        """Algorithm 4. Given the indices of already-cached hypotheses, return the
        recompute set ``P_ne``: every non-cached node and, transitively, all of its
        ``derived_by`` dependents (the *cascade effect* -- an invalidated fragment
        forces recomputation of everything downstream). Processes nodes in
        topological order so each is settled before its dependents are reached."""
        cached = set(cached)
        n = len(self._hyps)
        indeg = {i: 0 for i in range(n)}
        for _, j in self._edges:
            indeg[j] += 1
        topo: list[int] = []
        q = [i for i in range(n) if indeg[i] == 0]
        while q:
            nd = q.pop()
            topo.append(nd)
            for nb in self._adj[nd]:
                indeg[nb] -= 1
                if indeg[nb] == 0:
                    q.append(nb)

        p_ne: set[int] = set()
        processed: set[int] = set()
        for h in topo:
            if h in processed:
                continue
            if h not in cached:
                p_ne.add(h)
                st = [h]
                while st:
                    nd = st.pop()
                    for dep in self._adj[nd]:
                        if dep not in p_ne:
                            p_ne.add(dep)
                            st.append(dep)
                processed |= p_ne
            else:
                processed.add(h)
        return p_ne
```

---

=== hyppo/coa/causal.py ===

```python
"""Pure causal-ordering core for COA. stdlib only -- no owlready, sympy, networkx.

An *equation* is a set of variable names (``frozenset[str]``); a *structure* is a
list of equations. A structure is *complete* iff ``|equations| == |variables|``.
The Dulmage-Mendelsohn decomposition (a perfect matching plus the strongly
connected components of the matching-induced dependency digraph) yields the
irreducible ("minimal complete") blocks in polynomial time.
"""

from __future__ import annotations

from collections import defaultdict, deque


def variables(equations: list[frozenset[str]]) -> set[str]:
    """All distinct variable names across the equations."""
    out = set()
    for eq in equations:
        out |= set(eq)
    return out


def is_complete(equations: list[frozenset[str]]) -> bool:
    """A structure is complete iff |equations| == |distinct variables|."""
    return len(equations) == len(variables(equations))


def perfect_matching(equations: list[frozenset[str]]) -> dict[int, str] | None:
    """Match each equation to a distinct variable it contains (Hopcroft-Karp).

    Returns ``{eq_index: var}`` saturating every equation, or ``None`` if no such
    matching exists (the structure is structurally singular). Candidate variables
    are tried in sorted order and free equations in index order, giving a
    deterministic, name-stable result. Returns an empty dict for an empty list.

    Uses the Hopcroft-Karp algorithm -- ``O(|E|*sqrt(|V|))`` -- which augments along
    all shortest alternating paths of a BFS phase before re-layering, matching the
    complexity stated for the causal-ordering construction. Both the BFS phase and
    the augmenting DFS are iterative (explicit stacks), so a long alternating path
    (e.g. a dependency chain of thousands of equations) never hits Python's
    recursion limit.
    """
    cand = [sorted(eq) for eq in equations]
    n = len(equations)
    match_eq: dict[int, str] = {}  # eq index -> var
    match_var: dict[str, int] = {}  # var -> eq index
    INF = float("inf")
    dist: dict[int, float] = {}

    def bfs() -> bool:
        """Layer free equations by alternating-path distance; return True if any
        augmenting path to a free variable exists."""
        q: deque[int] = deque()
        for i in range(n):
            if i not in match_eq:
                dist[i] = 0
                q.append(i)
            else:
                dist[i] = INF
        reachable_free = False
        while q:
            i = q.popleft()
            for v in cand[i]:
                w = match_var.get(v)
                if w is None:
                    reachable_free = True
                elif dist.get(w, INF) == INF:
                    dist[w] = dist[i] + 1
                    q.append(w)
        return reachable_free

    def augment(start: int) -> bool:
        """Iterative layered DFS: find one shortest augmenting path from free
        equation ``start`` and flip it. ``dist`` confines the search to BFS layers
        and dead ends are pruned via ``dist[i] = INF``."""
        stack = [(start, iter(cand[start]))]
        trail: list[tuple[int, str]] = []  # (eq, var) edges descended into
        while stack:
            i, it = stack[-1]
            descended = False
            for v in it:
                w = match_var.get(v)
                if w is None:  # free var -> flip the whole path
                    match_eq[i] = v
                    match_var[v] = i
                    for eq_p, var_p in reversed(trail):
                        match_eq[eq_p] = var_p
                        match_var[var_p] = eq_p
                    return True
                if dist.get(w, INF) == dist[i] + 1:
                    trail.append((i, v))
                    stack.append((w, iter(cand[w])))
                    descended = True
                    break
            if not descended:
                dist[i] = INF  # exhausted in this phase
                stack.pop()
                if trail:
                    trail.pop()
        return False

    while bfs():
        for i in range(n):
            if i not in match_eq:
                augment(i)

    if len(match_eq) != n:
        return None
    return dict(match_eq)


def _dependency_graph(
    equations: list[frozenset[str]], matching: dict[int, str]
) -> dict[str, set[str]]:
    """Digraph over variables: edge u -> v means the equation matched to v also
    contains u, i.e. v depends on u. Every variable is present as a node."""
    adj: dict[str, set[str]] = defaultdict(set)
    for v in variables(equations):
        adj.setdefault(v, set())  # ensure isolated nodes exist
    for i, eq in enumerate(equations):
        v = matching[i]
        for u in eq:
            if u != v:
                adj[u].add(v)
    return adj


def strongly_connected_components(adj: dict[str, set[str]]) -> list[frozenset[str]]:
    """Tarjan's SCC. ``adj``: {node: set(successors)}. Returns list of frozensets."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[frozenset[str]] = []
    counter = [0]

    def strongconnect(v: str) -> None:
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
            comp: set[str] = set()
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


def _block_decomposition_with_matching(equations):
    """Internal: blocks + the matching used to build them, sharing one
    perfect_matching call. Returns (blocks, matching) or None if unmatchable."""
    m = perfect_matching(equations)
    if m is None:
        return None
    var_to_eq = {v: i for i, v in m.items()}
    comps = strongly_connected_components(_dependency_graph(equations, m))
    blocks = [frozenset(var_to_eq[v] for v in comp) for comp in comps]
    return blocks, m


def block_decomposition(equations: list[frozenset[str]]) -> list[frozenset[int]] | None:
    """Irreducible blocks as frozensets of equation indices. None if unmatchable."""
    r = _block_decomposition_with_matching(equations)
    return None if r is None else r[0]


def minimal_blocks(equations: list[frozenset[str]]) -> list[frozenset[int]] | None:
    """Inclusion-minimal complete subsets of equations (self-contained irreducible
    blocks -- those that reference only their own matched variables). List of
    frozensets of equation indices, or None if unmatchable."""
    r = _block_decomposition_with_matching(equations)
    if r is None:
        return None
    blocks, m = r
    out = []
    for b in blocks:
        own = {m[i] for i in b}
        referenced = set()
        for i in b:
            referenced |= set(equations[i])
        if referenced <= own:
            out.append(b)
    return out


def causal_mapping(equations: list[frozenset[str]]) -> dict[int, str] | None:
    """Full causal mapping eq_index -> variable (deterministic, sorted tie-break).
    None if the structure admits no perfect matching."""
    return perfect_matching(equations)


def transitive_closure(equations: list[frozenset[str]]) -> dict[str, set[str]] | None:
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

---

=== hyppo/lattice_constructor/_base.py ===

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from hyppo.core._base import virtual_experiment_onto

if TYPE_CHECKING:
    pass


with virtual_experiment_onto:

    class HypothesisLattice:
        def __init__(self, hypotheses, workflow):
            self.hypotheses = hypotheses
            self.workflow = workflow
            self.lattice = self.build_lattice()

        def build_lattice(self):
            """Algorithm 1: Build hypothesis lattice from hypothesis structures.

            An edge (h_i, h_j) is derived when the output variable of h_i
            appears among the input variables of h_j's equations (i.e. h_j is
            derived_by h_i); see :meth:`_build_hypothesis_var_mapping`.
            """
            lattice = nx.DiGraph()
            # check if all hypotheses are in workflow
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")

            for dep in self._build_hypothesis_var_mapping():
                lattice.add_edge(dep[0], dep[1])

            return lattice

        def add_hypothesis(self, hypothesis):
            """Algorithm 2: Add a hypothesis to an existing lattice.

            Registers the new hypothesis, then rebuilds variable-level
            transitive closure with it, and maps to hypothesis-level edges.
            """
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")
            if hypothesis not in self.hypotheses:
                self.hypotheses.append(hypothesis)
            self.lattice.add_node(hypothesis)

            for dep in self._build_hypothesis_var_mapping():
                self.lattice.add_edge(dep[0], dep[1])

        def derived_by(self, hypothesis):
            """Return hypotheses that are derived by the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            return set(self.lattice.predecessors(hypothesis))

        def _build_hypothesis_var_mapping(self):
            """Derive hypothesis-level edges from equation variables.

            Edge (h_i, h_j) added iff output variable of h_i appears among
            input variables of h_j (h_j depends on h_i).
            """
            dependencies = []
            for h_i in self.hypotheses:
                if not hasattr(h_i, "structure") or not h_i.structure.equations:
                    continue
                i_out = self._output_variable(h_i)
                if i_out is None:
                    continue
                for h_j in self.hypotheses:
                    if h_i is h_j:
                        continue
                    if not hasattr(h_j, "structure"):
                        continue
                    j_out = self._output_variable(h_j)
                    for eq in h_j.structure.equations:
                        if len(eq.vars) <= 1:
                            continue
                        for v in eq.vars:
                            if v != j_out and v == i_out:
                                dependencies.append((h_i, h_j))
            return dependencies

        @staticmethod
        def _output_variable(hypothesis):
            """Return the output variable of a hypothesis.

            If an equation was built from a formula string with an explicit
            left-hand side (``out = f(...)``), the output is the LHS symbol.
            Otherwise fall back to the exogenous-variable heuristic (first
            variable not declared by a single-variable equation).
            """
            from sympy import sympify

            for eq in hypothesis.structure.equations:
                formula = getattr(eq, "formula", None)
                if formula is not None and "=" in str(formula):
                    lhs = sympify(str(formula).split("=", 1)[0], locals={})
                    syms = sorted(lhs.free_symbols, key=lambda s: s.name)
                    if len(syms) == 1:
                        return syms[0]
            exo_vars = set()
            for e in hypothesis.structure.equations:
                if len(e.vars) == 1:
                    exo_vars.add(e.vars[0])
            for eq in hypothesis.structure.equations:
                if len(eq.vars) > 1:
                    candidates = [v for v in eq.vars if v not in exo_vars]
                    return candidates[0] if candidates else eq.vars[0]
            return None

        def competes(self, hypothesis):
            """Return hypotheses that compete with the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            # Competing hypotheses share predecessors without a direct relationship
            predecessors = set(self.lattice.predecessors(hypothesis))
            competitors = set()
            for h in self.hypotheses:
                if h != hypothesis and h not in self.derived_by(hypothesis):
                    h_predecessors = set(self.lattice.predecessors(h))
                    if predecessors & h_predecessors:  # If they share any predecessors
                        competitors.add(h)
            return competitors

        def impacts(self, hypothesis):
            """Return hypotheses that are impacted by the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            # Impacted hypotheses are those that are reachable from this hypothesis
            return set(nx.descendants(self.lattice, hypothesis))

        def remove_hypothesis(self, hypothesis):
            """Remove a hypothesis from the lattice."""
            if hypothesis in self.hypotheses:
                self.hypotheses.remove(hypothesis)
                self.lattice.remove_node(hypothesis)

        def _is_correct(self):
            """Check if all hypotheses are present in the workflow."""
            workflow_hypotheses = set()
            tasks = self.workflow.get_tasks()
            for task in tasks:
                workflow_hypotheses.update(task)

            return all(h in workflow_hypotheses for h in self.hypotheses)
```

---

=== hyppo/planner/_base.py ===

```python
"""Планировщик виртуальных экспериментов.

Реализация Алгоритма 4 из диссертации: «Построение плана виртуального эксперимента».
Алгоритм принимает конфигурацию эксперимента и решетку гипотез, определяет,
какие гипотезы требуют пересчета моделей, а какие могут быть взяты из кеша
репозитория метаинформации.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Set

import networkx as nx

if TYPE_CHECKING:
    from hyppo.lattice_constructor._base import HypothesisLattice
    from hyppo.storage._base import Database

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Результат планирования виртуального эксперимента.

    Attributes:
        needs_execution: Pne -- множество гипотез, модели которых требуют пересчета.
        cached: Pe -- множество гипотез, для которых можно использовать
            ранее вычисленные результаты.
    """

    needs_execution: Set = field(default_factory=set)
    cached: Set = field(default_factory=set)


def _find_nearest_lattice(
    lattice: "HypothesisLattice",
    db: "Database",
) -> Optional["HypothesisLattice"]:
    """Поиск ближайшей ранее вычисленной решетки в репозитории метаинформации.

    Ближайшей считается решетка, имеющая наибольшее пересечение
    множества гипотез с текущей решеткой.

    Args:
        lattice: Текущая решетка гипотез.
        db: Экземпляр базы данных (репозиторий метаинформации).

    Returns:
        Ближайшая решетка из репозитория или None, если репозиторий пуст.
    """
    stored_objects = db.load_all(storage="lattices")
    if not stored_objects:
        return None

    current_hypotheses = set(lattice.hypotheses)
    best_lattice = None
    best_overlap = -1

    for pickled in stored_objects:
        if pickled is None:
            continue
        candidate = pickled.obj
        candidate_hypotheses = set(candidate.hypotheses)
        overlap = len(current_hypotheses & candidate_hypotheses)
        if overlap > best_overlap:
            best_overlap = overlap
            best_lattice = candidate

    return best_lattice


def _cfg_dict(configuration) -> dict:
    """Привести конфигурацию к dict для ключа общего кэша (config_hash).
    Согласовано с runner, который пишет результаты по (hypothesis_id, config)."""
    if isinstance(configuration, dict):
        return configuration
    params = getattr(configuration, "parameters", None)
    if params is not None:
        return {"parameters": [str(p) for p in params]}
    return {"repr": str(configuration)}


def _config_for(hypothesis, configuration, per_hypothesis_configs) -> object:
    """Разрешить конфигурацию для конкретной гипотезы.

    Если задан ``per_hypothesis_configs`` (отображение id гипотезы → конфигурация),
    возвращается индивидуальная конфигурация гипотезы (при отсутствии — общий
    ``configuration`` как значение по умолчанию). Иначе — общий ``configuration``
    для всех гипотез (прежнее поведение)."""
    if per_hypothesis_configs is None:
        return configuration
    name = getattr(hypothesis, "name", str(hypothesis))
    return per_hypothesis_configs.get(name, configuration)


def _has_cached_result(
    hypothesis,
    configuration,
    db: "Database",
) -> bool:
    """Проверка наличия кешированного результата для гипотезы при данной конфигурации.

    Для каждого набора параметров cm из конфигурации C и для каждой модели m,
    реализующей гипотезу h, проверяется, существует ли вычисленный результат
    h -> m(cm) в репозитории метаинформации.

    Args:
        hypothesis: Проверяемая гипотеза.
        configuration: Конфигурация эксперимента (набор параметров).
        db: Экземпляр базы данных.

    Returns:
        True, если для ВСЕХ комбинаций параметров и моделей существует
        кешированный результат; False -- если хотя бы для одной комбинации
        результат отсутствует.
    """
    # Быстрый путь: репозиторий с ключом (гипотеза, конфигурация) — ОБЩИЙ кэш
    # planner↔runner (SharedCache / MetadataRepository). Так планировщик видит
    # результаты, записанные раннером, в одном SQLite. Legacy Database (только
    # load/save) идёт по старому пути ниже.
    if hasattr(db, "has_result") and hasattr(db, "load_result"):
        return db.has_result(
            getattr(hypothesis, "name", str(hypothesis)), _cfg_dict(configuration)
        )

    # Получаем модели, реализующие гипотезу.
    # В онтологии is_implemented_by_model -- ObjectProperty, возвращающее список
    # связанных объектов Model. Owlready2 делает его вызываемым для получения
    # инстансов, поэтому пробуем вызвать как метод; если атрибут уже является
    # списком, используем его напрямую.
    models_attr = getattr(hypothesis, "is_implemented_by_model", None)
    if callable(models_attr):
        models = models_attr()
    else:
        models = models_attr
    if not models:
        return False

    # Получаем наборы параметров из конфигурации.
    # Configuration -- самостоятельная сущность; параметры хранятся как список
    # внутри неё (например, configuration.parameters).  Если такого атрибута нет,
    # используем саму конфигурацию как единственный набор параметров.
    param_sets = getattr(configuration, "parameters", None)
    if not param_sets:
        param_sets = [configuration]

    for cm in param_sets:
        for model in models:
            # Формируем ключ кеша: идентификатор гипотезы + модели + параметров
            cache_key = (
                f"{getattr(hypothesis, 'name', str(hypothesis))}"
                f"__{getattr(model, 'name', str(model))}"
                f"__{str(cm)}"
            )
            result = db.load(cache_key, storage="results")
            if result is None:
                return False

    return True


def _get_cached_r2(
    hypothesis,
    configuration,
    db: "Database",
) -> Optional[float]:
    """Извлечь R² из кешированного результата для гипотезы (если есть).

    Returns:
        R² (float) или None, если результат не найден или R² не записан.
    """
    # Быстрый путь: общий кэш planner↔runner (см. _has_cached_result).
    if hasattr(db, "has_result") and hasattr(db, "load_result"):
        rec = db.load_result(
            getattr(hypothesis, "name", str(hypothesis)), _cfg_dict(configuration)
        )
        if rec:
            return rec.get("metrics", {}).get("r2")
        return None

    models_attr = getattr(hypothesis, "is_implemented_by_model", None)
    if callable(models_attr):
        models = models_attr()
    else:
        models = models_attr
    if not models:
        return None

    param_sets = getattr(configuration, "parameters", None)
    if not param_sets:
        param_sets = [configuration]

    # Возвращаем R² первого найденного результата
    for cm in param_sets:
        for model in models:
            cache_key = (
                f"{getattr(hypothesis, 'name', str(hypothesis))}"
                f"__{getattr(model, 'name', str(model))}"
                f"__{str(cm)}"
            )
            result = db.load(cache_key, storage="results")
            if result is not None and hasattr(result, "obj"):
                obj = result.obj
                if isinstance(obj, dict) and "r2" in obj:
                    return obj["r2"]
    return None


def build_optimal_plan(
    configuration,
    lattice: "HypothesisLattice",
    db: "Database",
    r2_threshold: float = 0.7,
    *,
    per_hypothesis_configs: "Mapping | None" = None,
) -> ExecutionPlan:
    """Построение оптимального плана виртуального эксперимента (Алгоритм 4).

    Алгоритм обходит решетку гипотез сверху вниз (начиная с вершин
    без входящих ребер). Для каждой гипотезы проверяется наличие
    кешированного результата в репозитории метаинформации:

    - Если результата нет -- гипотеза и все зависимые от неё гипотезы
      добавляются в множество Pne (требуют пересчета).
    - Если результат есть -- гипотеза добавляется в множество Pe
      (повторное вычисление не требуется).
    - Отсечение по R² (Раздел 3.1.6.2): если R² кешированного результата
      гипотезы ниже порога r2_threshold, гипотеза и все зависимые от неё
      исключаются из плана (не попадают ни в Pne, ни в Pe).

    Это позволяет значительно сократить время исполнения виртуального
    эксперимента за счет повторного использования ранее вычисленных
    результатов и отсечения низкокачественных ветвей.

    Args:
        configuration: Конфигурация эксперимента (C) -- набор параметров.
        lattice: Решетка гипотез (L) -- объект HypothesisLattice с графом
            зависимостей между гипотезами.
        db: Экземпляр базы данных (репозиторий метаинформации) для поиска
            ранее вычисленных результатов.
        r2_threshold: Минимальный порог R² для включения гипотезы в план.
            Гипотезы с R² < порога и все их потомки исключаются (Раздел 3.1.6.2).
            По умолчанию 0.7.
        per_hypothesis_configs: Необязательное отображение id гипотезы →
            конфигурация. Если задано, кэш каждой гипотезы проверяется по её
            собственной конфигурации (при отсутствии — по общей ``configuration``).
            Это позволяет инвалидировать кэш ОТДЕЛЬНОЙ гипотезы: смена её
            конфигурации даёт промах только по ней и пересчёт её замыкания вниз.

    Returns:
        ExecutionPlan с двумя множествами:
            - needs_execution (Pne): гипотезы, требующие пересчета моделей.
            - cached (Pe): гипотезы с доступными кешированными результатами.
    """
    plan = ExecutionPlan()

    # Шаг 1: Поиск ближайшей ранее вычисленной решетки в репозитории
    nearest_lattice = _find_nearest_lattice(lattice, db)

    # Строим рабочий граф на основе ТЕКУЩЕЙ решетки (чтобы гарантировать
    # полноту: Pne U Pe = V(L)).  Ближайшая решетка используется только
    # для обогащения кеша -- гипотезы, присутствующие в nearest_lattice,
    # с большей вероятностью имеют кешированные результаты, но сам обход
    # всегда идёт по текущей решетке.
    graph: nx.DiGraph = lattice.lattice

    # Множество гипотез из ближайшей решетки (для возможной дополнительной
    # логики; на данный момент кеш-проверка через _has_cached_result уже
    # обращается к репозиторию напрямую).
    _nearest_hypotheses: Set = set()
    if nearest_lattice is not None:
        _nearest_hypotheses = set(nearest_lattice.hypotheses)

    # Шаг 2: Инициализация множеств
    # Pne -- гипотезы, требующие пересчета
    # Pe  -- гипотезы, не требующие пересчета
    # F   -- обработанные вершины
    # pruned -- гипотезы, отсеченные по R² (Раздел 3.1.6.2)
    pne: Set = set()
    pe: Set = set()
    finished: Set = set()
    pruned: Set = set()

    # Шаг 3: Множество необработанных вершин V(L)
    remaining = set(graph.nodes())

    # Пустая решетка -- пустой план
    if not remaining:
        return plan

    # Шаг 4: Обход решетки сверху вниз
    # Начинаем с гипотез, не имеющих входящих ребер (корневые гипотезы)
    while remaining:
        # Выбираем вершины без входящих ребер среди необработанных
        # (топологический порядок: сначала обрабатываем «верхние» гипотезы)
        roots = [
            h
            for h in remaining
            if all(pred in finished for pred in graph.predecessors(h))
        ]

        if not roots:
            raise ValueError(
                "Hypothesis graph contains a cycle — "
                "precondition of Algorithm 4 violated "
                "(Theorem 1 requires a DAG). "
                "Run check_consistency() before build_optimal_plan()."
            )

        for h in roots:
            if h in finished:
                continue

            # Отсечение по R² (Раздел 3.1.6.2): если предок был отсечен,
            # текущая гипотеза тоже отсекается.
            if h in pruned:
                finished.add(h)
                continue

            # Конфигурация именно этой гипотезы (по-гипотезная или общая)
            cfg_h = _config_for(h, configuration, per_hypothesis_configs)

            # Проверяем наличие кешированного результата для данной конфигурации
            if not _has_cached_result(h, cfg_h, db):
                # Результата нет: помечаем гипотезу и все зависимые (потомки в графе)
                # как требующие пересчета
                dependents = nx.descendants(graph, h)
                pne.add(h)
                pne.update(dependents)
                finished.add(h)
                finished.update(dependents)
            else:
                # Результат есть: проверяем R² перед включением в Pe
                r2 = _get_cached_r2(h, cfg_h, db)
                if r2 is not None and r2 < r2_threshold:
                    # R² ниже порога: отсекаем гипотезу и все зависимые
                    dependents = nx.descendants(graph, h)
                    pruned.add(h)
                    pruned.update(dependents)
                    finished.add(h)
                    finished.update(dependents)
                else:
                    # Повторное вычисление не требуется
                    pe.add(h)
                    finished.add(h)

        # Обновляем множество необработанных вершин
        remaining -= finished

    plan.needs_execution = pne
    plan.cached = pe

    return plan


class Planner:
    """Объектная обёртка над build_optimal_plan для удобства использования.

    Пример::

        planner = Planner(db=my_db, r2_threshold=0.8)
        plan = planner.plan(configuration, lattice)
    """

    def __init__(self, db: "Database", r2_threshold: float = 0.7) -> None:
        self.db = db
        self.r2_threshold = r2_threshold

    def plan(
        self,
        configuration,
        lattice: "HypothesisLattice",
        r2_threshold: float | None = None,
        per_hypothesis_configs: "Mapping | None" = None,
    ) -> ExecutionPlan:
        """Построить план виртуального эксперимента.

        Args:
            configuration: Конфигурация эксперимента.
            lattice: Решетка гипотез.
            r2_threshold: Порог R² для отсечения (переопределяет значение из __init__).
            per_hypothesis_configs: Необязательное отображение id гипотезы →
                конфигурация (см. build_optimal_plan).

        Returns:
            ExecutionPlan с множествами needs_execution и cached.
        """
        threshold = r2_threshold if r2_threshold is not None else self.r2_threshold
        return build_optimal_plan(
            configuration,
            lattice,
            self.db,
            r2_threshold=threshold,
            per_hypothesis_configs=per_hypothesis_configs,
        )
```

---

=== hyppo/runner/_base.py ===

```python
"""VirtualExperimentRunner — executes virtual experiments according to plan.

Implements the runner described in Section 3.1.7 of the dissertation.
Executes models in topological order, retries on failure (k=3),
cascades SKIPPED status to dependent hypotheses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from hyppo.core._epistemic import EpistemicStatus, evaluate_status

logger = logging.getLogger(__name__)


class Status(Enum):
    """Execution status of a single hypothesis run.

    Distinct from :class:`hyppo.core._epistemic.EpistemicStatus` (the
    scientific verdict) — this tracks whether the model function ran at all.
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class RunResult:
    """Outcome of executing one hypothesis's model.

    Attributes:
        hypothesis_id: ID of the hypothesis that was executed.
        status: Execution outcome (SUCCESS/FAILED/SKIPPED).
        metrics: Metrics returned by the model function (e.g. r2, aic).
        error: Error message if the run failed, else None.
        epistemic_status: Scientific verdict assigned after execution
            (Section 2); defaults to PROPOSED until evaluated.
    """

    hypothesis_id: str
    status: Status
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    epistemic_status: EpistemicStatus = EpistemicStatus.PROPOSED


class Runner:
    """Executes virtual experiment plan (Section 3.1.7).

    For each hypothesis in P_ne (topological order):
    - Call model function with config
    - Retry up to max_retries times on failure
    - If still fails: mark FAILED, cascade SKIPPED to dependents
    - Save results to repository
    """

    def __init__(self, repository=None, max_retries: int = 3) -> None:
        """Initialize the runner.

        Args:
            repository: Optional result-cache object exposing
                ``has_result``/``load_result``/``save_result``; None disables
                caching of P_e results.
            max_retries: Number of attempts before a hypothesis is marked FAILED.
        """
        self.repository = repository
        self.max_retries = max_retries

    def execute(
        self,
        plan: dict,  # {"p_ne": set[str], "p_e": set[str]}
        models: dict[str, Callable],
        configs: dict[str, dict] | None = None,
        lattice_edges: list[tuple[str, str]] | None = None,
        competes: dict[str, set[str]] | None = None,
        theta_sup: float = 0.7,
        theta_aic: float = 10.0,
    ) -> dict[str, dict]:
        """Execute plan and return results.

        Args:
            plan: {"p_ne": IDs to compute, "p_e": IDs to load from cache}
            models: {hypothesis_id: callable(config) -> {"r2": float, "aic": ...}}
            configs: {hypothesis_id: config_dict}
            lattice_edges: list of (parent, child) derived_by edges
            competes: {hypothesis_id: set of competing hypothesis IDs} -- used to
                assign the SUPERSEDED epistemic status via Delta AIC.
            theta_sup: R^2 support threshold for SUPPORTED/REFUTED (default 0.7).
            theta_aic: Delta AIC threshold for SUPERSEDED (default 10).

        Returns:
            {hypothesis_id: {"status": str, "metrics": dict, "epistemic_status": str}}
        """
        results: dict[str, dict] = {}
        p_ne = plan.get("p_ne", set())
        p_e = plan.get("p_e", set())
        configs = configs or {}
        failed_ancestors: set[str] = set()

        # Build dependency graph for cascade
        dependents: dict[str, set[str]] = {}
        if lattice_edges:
            for parent, child in lattice_edges:
                dependents.setdefault(parent, set()).add(child)

        # Load cached results for P_e
        for h_id in p_e:
            if self.repository and self.repository.has_result(
                h_id, configs.get(h_id, {})
            ):
                cached = self.repository.load_result(h_id, configs.get(h_id, {}))
                results[h_id] = {
                    "status": Status.SUCCESS.value,
                    "metrics": cached.get("metrics", {}),
                }
            else:
                results[h_id] = {"status": Status.SUCCESS.value, "metrics": {}}

        # Execute P_ne in order (caller should provide topological order)
        for h_id in p_ne:
            # Check if ancestor failed
            if h_id in failed_ancestors:
                results[h_id] = {"status": Status.SKIPPED.value, "metrics": {}}
                # Cascade SKIPPED to dependents
                self._cascade_skip(h_id, dependents, failed_ancestors)
                logger.info(f"Skipped {h_id} (ancestor failed)")
                continue

            # Try to execute with retries
            config = configs.get(h_id, {})
            model_fn = models.get(h_id)
            if model_fn is None:
                results[h_id] = {
                    "status": Status.FAILED.value,
                    "metrics": {},
                    "error": "No model function",
                }
                self._cascade_skip(h_id, dependents, failed_ancestors)
                continue

            success = False
            last_error = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    metrics = model_fn(config)
                    results[h_id] = {"status": Status.SUCCESS.value, "metrics": metrics}
                    # Save to repository
                    if self.repository:
                        self.repository.save_result(
                            h_id, config, metrics, Status.SUCCESS.value
                        )
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries} failed for {h_id}: {e}"
                    )

            if not success:
                results[h_id] = {
                    "status": Status.FAILED.value,
                    "metrics": {},
                    "error": last_error,
                }
                self._cascade_skip(h_id, dependents, failed_ancestors)
                logger.error(f"Failed {h_id} after {self.max_retries} attempts")

        self._assign_epistemic_status(results, competes or {}, theta_sup, theta_aic)
        return results

    def _assign_epistemic_status(
        self,
        results: dict[str, dict],
        competes: dict[str, set[str]],
        theta_sup: float,
        theta_aic: float,
    ) -> None:
        """Write ``epistemic_status`` for every result (Section 2). A successfully
        evaluated hypothesis is SUPPORTED/REFUTED by its R^2, or SUPERSEDED if a
        competitor's AIC beats it by more than ``theta_aic``; anything not evaluated
        (no R^2, or FAILED/SKIPPED) stays PROPOSED."""

        def aic_of(hid: str) -> float | None:
            r = results.get(hid)
            if r and r["status"] == Status.SUCCESS.value:
                return r["metrics"].get("aic")
            return None

        for h_id, res in results.items():
            if res["status"] != Status.SUCCESS.value:
                res["epistemic_status"] = EpistemicStatus.PROPOSED.value
                continue
            r2 = res["metrics"].get("r2")
            own_aic = res["metrics"].get("aic")
            competitor_aics = [
                a for c in competes.get(h_id, set()) if (a := aic_of(c)) is not None
            ]
            best = min(competitor_aics) if competitor_aics else None
            status = evaluate_status(
                r2, own_aic, best, theta_sup=theta_sup, theta_aic=theta_aic
            )
            res["epistemic_status"] = status.value

    def _cascade_skip(
        self, h_id: str, dependents: dict[str, set[str]], failed_set: set[str]
    ) -> None:
        """Recursively mark all dependents as needing skip."""
        for dep in dependents.get(h_id, set()):
            failed_set.add(dep)
            self._cascade_skip(dep, dependents, failed_set)
```

---

=== hyppo/core/_base.py ===

```python
"""OWL ontology for the virtual-experiment domain (Definition 1, Chapter 2).

Defines the ``virtual_experiment_onto`` OWL ontology (owlready2) that carries
the formal artefacts of a virtual experiment: hypotheses, models, the
workflow, and the causal structure (equations/variables) that a hypothesis
is built from. ``derived_by``/``impacts`` encode the lattice edges consumed
by :class:`hyppo.lattice_constructor._base.HypothesisLattice` (Algorithm 1).
"""

import datetime

from owlready2 import (
    AllDisjoint,
    DataProperty,
    FunctionalProperty,
    ObjectProperty,
    SymmetricProperty,
    Thing,
    TransitiveProperty,
    get_ontology,
)

virtual_experiment_onto = get_ontology(
    "http://synthesis.ipi.ac.ru/virtual_experiment.owl"
)
hcp_brain_onto = get_ontology("http://synthesis.ipi.ac.ru/hcp_brain_onto.owl")
virtual_experiment_onto.imported_ontologies.append(hcp_brain_onto)

with virtual_experiment_onto:
    # define base class and its properties
    class Artefact(Thing):
        """Root OWL class for every documented artefact (Definition 1).

        Base class for ``Hypothesis``, ``Model``, ``VirtualExperiment`` and
        the structural classes below; carries the mandatory identity/
        provenance properties (id, name, description, authors, timestamps).
        """

    # class Specification(Thing): pass
    class has_for_id(Artefact >> int, DataProperty, FunctionalProperty):
        """Functional data property: unique integer identifier of an artefact."""

        python_name = "id"

    class has_for_name(Artefact >> str, DataProperty, FunctionalProperty):
        """Functional data property: human-readable name of an artefact."""

        python_name = "name"

    class has_for_description(Artefact >> str, DataProperty, FunctionalProperty):
        """Functional data property: free-text description of an artefact."""

        python_name = "description"

    class has_for_authors(Artefact >> str, DataProperty):
        """Non-functional data property: one or more author names (min 1)."""

        python_name = "authors"

    class has_for_createdate(
        Artefact >> datetime.datetime, DataProperty, FunctionalProperty
    ):
        """Functional data property: artefact creation timestamp."""

        python_name = "create_date"

    class has_for_lastupdate(
        Artefact >> datetime.datetime, DataProperty, FunctionalProperty
    ):
        """Functional data property: artefact last-modification timestamp."""

        python_name = "last_update"

    # class has_for_specification(Artefact >> Specification): pass

    Artefact.is_a.extend(
        [
            has_for_authors.min(1),
            has_for_name.exactly(1),
            has_for_description.exactly(1),
            has_for_id.exactly(1),
            has_for_lastupdate.exactly(1),
            has_for_createdate.exactly(1),
        ]
    )

    class Hypothesis(Artefact):
        """OWL individual for a hypothesis h in H (Definition 1).

        Node type of the hypothesis lattice built by Algorithm 1
        (:class:`hyppo.lattice_constructor._base.HypothesisLattice`); linked
        to its implementing ``Model`` via ``is_implemented_by_model`` and to
        competing/derived hypotheses via ``competes``/``derived_by``.
        """

    class Model(Artefact):
        """OWL individual for the model that implements a ``Hypothesis``.

        Paired 1:1 with its hypothesis through the mutually-inverse,
        functional properties ``is_implemented_by_model`` / ``refers_to_hypothesis``
        (see the Theorem 1 axiomatic support note below).
        """

    # class Mapping(Artefact): pass
    # class Relation(Artefact): pass

    # TODO probability > 0.0 and < 1.0
    class has_for_probability(Hypothesis >> float, DataProperty, FunctionalProperty):
        """Functional data property: prior/posterior probability of a hypothesis."""

        python_name = "probability"

    class is_implemented_by_model(Hypothesis >> Model):
        """Object property: hypothesis -> the model implementing it (some Model)."""

        class_property_type = ["some"]

    class refers_to_hypothesis(ObjectProperty):
        """Object property: model -> the hypothesis it implements (inverse of
        ``is_implemented_by_model``); both are declared functional below for
        the Theorem 1 uniqueness proof (paired 1:1 correspondence)."""

        domain = [Model]
        range = [Hypothesis]
        inverse_property = is_implemented_by_model
        class_property_type = ["only"]

    class competes(Hypothesis >> Hypothesis, SymmetricProperty):
        """Symmetric object property: two hypotheses compete over the same
        phenomenon (used by :func:`hyppo.core._epistemic.evaluate_status` via
        the runner's Delta-AIC SUPERSEDED check)."""

    class derived_by(Hypothesis >> Hypothesis, TransitiveProperty):
        """Transitive object property: lattice edge h_j derived_by h_i, i.e.
        h_j depends on h_i's output (Definition 1); the edge set is computed
        by Algorithm 1 and incrementally maintained by Algorithm 2."""

    # Note: AsymmetricProperty and IrreflexiveProperty removed because
    # OWL 2 DL simplicity constraint forbids them on transitive properties.
    # Acyclicity is enforced by Algorithm 3 (consistency check), not OWL axioms.
    class impacts(ObjectProperty, TransitiveProperty):
        """Transitive object property, inverse of ``derived_by``: h_i impacts
        h_j means h_j is derived_by h_i. Acyclicity of this relation (no
        hypothesis impacts itself transitively) is enforced procedurally by
        Algorithm 3's consistency check, not by an OWL axiom (see note above)."""

        domain = [Hypothesis]
        range = [Hypothesis]
        inverse_property = derived_by

    class VirtualExperiment(Artefact):
        """OWL individual for a virtual experiment: bundles a set of
        hypotheses, their models, a ``Workflow`` and a ``Configuration``."""

    class Configuration(Artefact):
        """OWL individual holding the run configuration of a virtual experiment."""

    class Workflow(Artefact):
        """OWL individual for the task DAG of a virtual experiment; the
        Python-level counterpart with execution semantics is
        :class:`hyppo.core._workflow.Workflow`."""

    # class Task(Thing): pass

    class has_for_hypothesis(VirtualExperiment >> Hypothesis):
        """Object property: virtual experiment -> its hypotheses (some Hypothesis)."""

        class_property_type = ["some"]

    class has_for_model(VirtualExperiment >> Model):
        """Object property: virtual experiment -> its models (some Model)."""

        class_property_type = ["some"]

    class has_for_workflow(VirtualExperiment >> Workflow):
        """Object property: virtual experiment -> its task workflow (only Workflow)."""

        class_property_type = ["only"]

    class has_for_configuration(VirtualExperiment >> Configuration):
        """Object property: virtual experiment -> its run configuration
        (some Configuration)."""

        class_property_type = ["some"]

    # class has_for_task(Workflow >> Task): class_property_type = ["some"]

    class Structure(Artefact):
        """OWL counterpart of a causal-ordering structure: a set of equations
        over a set of variables, as manipulated by the pure-Python core
        :class:`hyppo.coa._base.Structure` (Algorithm 1 input)."""

    class FullStructure(Structure):
        """A ``Structure`` that is complete (Hall's condition holds): every
        equation can be matched to a distinct output variable, so a causal
        mapping (``FullCausalMapping``) can be derived from it."""

    class Equation(Thing):
        """OWL counterpart of a single causal-ordering equation, mirroring
        :class:`hyppo.coa._base.Equation` (formula + its free variables)."""

    class Variable(Thing):
        """OWL individual for a variable appearing in an ``Equation`` /
        ``Structure`` (sympy Symbol on the pure-Python side)."""

    class has_for_varname(Variable >> str, DataProperty, FunctionalProperty):
        """Functional data property: the symbolic name of a ``Variable``."""

        python_name = "name"

    class FullCausalMapping(Artefact):
        """The causal mapping (assignment equation -> output variable)
        derived from a ``FullStructure`` by the causal ordering algorithm."""

    class has_for_fcm(FullStructure >> FullCausalMapping):
        """Object property: a ``FullStructure`` -> its derived
        ``FullCausalMapping`` (only)."""

        class_property_type = ["only"]

    class has_for_structure(Hypothesis >> Structure):
        """Object property: a ``Hypothesis`` -> the ``Structure`` of
        equations it is built from (only)."""

        class_property_type = ["only"]

    class DependencySet(Artefact):
        """Set of variable/hypothesis dependencies derived from a
        ``FullStructure``'s causal mapping."""

    class has_for_dependecy_set(FullStructure >> DependencySet):
        """Object property: a ``FullStructure`` -> its ``DependencySet``
        (only)."""

        class_property_type = ["only"]

    class TransitiveClosure(DependencySet):
        """A ``DependencySet`` closed under transitivity (all indirect
        dependencies made explicit)."""

    class ResearchLattice(Artefact):
        """OWL counterpart of the hypothesis lattice built by Algorithm 1
        (:class:`hyppo.lattice_constructor._base.HypothesisLattice`): the set
        of hypotheses under study, linked via ``has_for_lattice_hypothesis``."""

    class has_for_lattice_hypothesis(ResearchLattice >> Hypothesis):
        """Object property: a ``ResearchLattice`` -> its member hypotheses
        (some Hypothesis)."""

        class_property_type = ["some"]

    # class has_for_vars(Equation >> Variable, DataProperty):
    #     class_property_type = ["some"]
    #     python_name = "vars"

    # class has_for_equation(Structure >> Equation, DataProperty):
    #     python_name = "equation"

    # class has_for_structure_variable(Structure >> Variable):
    #     class_property_type = ["some"]
    #     python_name = "vars"

    AllDisjoint([VirtualExperiment, Configuration, Workflow, Hypothesis, Model])

    # ── Theorem 1 axiomatic support (iip2026_planning.tex §3) ──────────────
    # Adds the three OWL axioms required for the C2 source-of-inconsistency
    # bijection in the consistency-check correctness proof:
    #   (i)  is_implemented_by_model is Functional → uniqueness of m
    #        (each hypothesis is implemented by at most one model);
    #   (ii) refers_to_hypothesis is Functional → uniqueness of h per model
    #        (paired-functional design across the inverse property);
    #   (iii) Hypothesis ⊑ ∃is_implemented_by_model.Model → existence of m
    #        (every Hypothesis has at least one implementing model).
    # Together these give the ∃! m ∈ Model. R(m) = h required by C2.
    # NOTE on UNA: OWL 2 DL does not assume Unique Name Assumption, so
    # FunctionalProperty alone only unifies (m1 ≡ m2) rather than yielding
    # owl:Nothing. Concrete VE instantiations must add AllDifferent on the
    # set of Model individuals to enforce C2-uniqueness inconsistency
    # detection by HermiT (see paper §2, Theorem 1 proof).
    is_implemented_by_model.is_a.append(FunctionalProperty)
    refers_to_hypothesis.is_a.append(FunctionalProperty)
    Hypothesis.is_a.append(is_implemented_by_model.some(Model))


if __name__ == "__main__":
    virtual_experiment_onto = get_ontology(
        "http://synthesis.ipi.ac.ru/virtual_experiment.owl"
    )
    print(list(virtual_experiment_onto.classes()))
    virtual_experiment_onto.save("ve.owl")
    art = Artefact("123")
    art.has_for_author = [123]
    print(has_for_authors.range)
    print(art.name)
```

---

=== hyppo/core/_epistemic.py ===

```python
"""Epistemic status of a hypothesis (Section 2, part2.tex:570-588).

A hypothesis carries an *epistemic status* distinct from its *execution* status
(SUCCESS/FAILED/SKIPPED, see :mod:`hyppo.runner`). The epistemic status records the
scientific verdict after a hypothesis has been evaluated against data:

* ``PROPOSED``   -- created, not yet evaluated;
* ``SUPPORTED``  -- quality metric clears the threshold (R^2 >= theta_sup);
* ``REFUTED``    -- quality metric below the threshold (R^2 < theta_sup);
* ``SUPERSEDED`` -- a competing hypothesis is decisively better
  (Delta AIC = AIC(h) - AIC(h') > theta_aic), regardless of R^2.

The transition is a pure function so it can be tested in isolation; the
:class:`hyppo.runner.Runner` is the only place that gathers the inputs (R^2, AIC,
competitors) and writes the resulting status. Defaults follow the dissertation:
``theta_sup = 0.7`` (Burnham-Anderson conservative threshold) and ``theta_aic = 10``
(Delta AIC > 10 == no empirical support for the worse model).
"""

from __future__ import annotations

from enum import Enum


class EpistemicStatus(Enum):
    """Scientific verdict on a hypothesis after evaluation against data.

    See the module docstring for the meaning of each member and the
    thresholds (``theta_sup``, ``theta_aic``) used to derive them.
    """

    PROPOSED = "PROPOSED"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    SUPERSEDED = "SUPERSEDED"


def evaluate_status(
    r2: float | None,
    own_aic: float | None = None,
    best_competitor_aic: float | None = None,
    *,
    theta_sup: float = 0.7,
    theta_aic: float = 10.0,
) -> EpistemicStatus:
    """Return the epistemic status of a hypothesis from its evaluation metrics.

    Args:
        r2: coefficient of determination on the test data; ``None`` if the
            hypothesis has not been evaluated yet.
        own_aic: AIC of this hypothesis (needed only for the SUPERSEDED check).
        best_competitor_aic: smallest AIC among competing hypotheses (``competes``),
            or ``None`` if there are no evaluated competitors.
        theta_sup: R^2 support threshold (default 0.7).
        theta_aic: Delta AIC threshold for being superseded (default 10).

    Precedence (part2.tex:584): SUPERSEDED dominates -- a hypothesis decisively
    beaten by a competitor is superseded even if its own R^2 clears the threshold.

    Returns:
        EpistemicStatus: ``PROPOSED`` if ``r2`` is ``None``; else ``SUPERSEDED``
        if decisively beaten by a competitor; else ``SUPPORTED`` if
        ``r2 >= theta_sup``; else ``REFUTED``.
    """
    if r2 is None:
        return EpistemicStatus.PROPOSED
    if (
        own_aic is not None
        and best_competitor_aic is not None
        and own_aic - best_competitor_aic > theta_aic
    ):
        return EpistemicStatus.SUPERSEDED
    if r2 >= theta_sup:
        return EpistemicStatus.SUPPORTED
    return EpistemicStatus.REFUTED
```

---

=== hyppo/manager/_base.py ===

```python
"""VirtualExperimentManager — orchestrates the full lifecycle (Section 3.1.2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import networkx as nx

from hyppo.metadata_repository import MetadataRepository
from hyppo.runner import Runner


class Manager:
    """Orchestrates virtual experiment lifecycle (Section 3.1.2).

    Four stages:
    1. Initialization — create experiment, save to repository
    2. Build lattice — construct hypothesis lattice from structures
    3. Planning — determine P_ne/P_e with caching
    4. Execution — run models, save results
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        r2_threshold: float = 0.7,
        max_retries: int = 3,
    ) -> None:
        self.repository = MetadataRepository(db_path=db_path)
        self.r2_threshold = r2_threshold
        self.runner = Runner(repository=self.repository, max_retries=max_retries)

    def orchestrate(
        self,
        hypotheses: list[str],
        workflow_edges: list[tuple[str, str]],
        models: dict[str, Callable],
        config: dict[str, dict] | None = None,
        structures: dict | None = None,
    ) -> dict[str, dict]:
        """Run full virtual experiment lifecycle.

        Args:
            hypotheses: list of hypothesis IDs
            workflow_edges: DAG edges (parent, child)
            models: {hypothesis_id: callable(config) -> {"r2": float, ...}}
            config: {hypothesis_id: param_dict}
            structures: hypothesis structures for lattice construction (optional)

        Returns:
            {hypothesis_id: {"status": str, "metrics": dict}}
        """
        config = config or {h: {} for h in hypotheses}

        # Stage 1: Initialize
        # Stage 2: Build lattice (Algorithm 1). When causal structures are
        # supplied, run HypothesisGraph.build() to derive the lattice edges from
        # complete causal unions over the workflow-reachable pairs; otherwise fall
        # back to the workflow DAG itself.
        lattice_edges = list(workflow_edges)
        if structures:
            from hyppo.coa import HypothesisGraph

            idx = {h: i for i, h in enumerate(hypotheses)}
            graph = HypothesisGraph()
            for h in hypotheses:
                graph.add(structures.get(h, []))
            for u, v in workflow_edges:
                graph.connect(idx[u], idx[v])
            lattice_edges = [(hypotheses[i], hypotheses[j]) for i, j in graph.build()]

        lattice = nx.DiGraph()
        lattice.add_nodes_from(hypotheses)
        lattice.add_edges_from(lattice_edges)

        # Stage 3: Plan — determine P_ne (needs execution) and P_e (cached)
        p_ne: list[str] = []
        p_e: set[str] = set()

        topo_order = list(nx.topological_sort(lattice))
        for h in topo_order:
            if self.repository.has_result(h, config.get(h, {})):
                cached = self.repository.load_result(h, config.get(h, {}))
                if cached is not None:
                    r2 = cached.get("metrics", {}).get("r2")
                    if r2 is not None and r2 < self.r2_threshold:
                        # Prune low-quality hypothesis and descendants
                        continue
                p_e.add(h)
            else:
                p_ne.append(h)

        # Stage 4: Execute
        results = self.runner.execute(
            plan={"p_ne": p_ne, "p_e": p_e},
            models=models,
            configs=config,
            lattice_edges=lattice_edges,
        )

        return results

    def close(self) -> None:
        """Close the underlying repository connection."""
        self.repository.close()
```

---

=== examples/norne_primer.py ===

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
from collections.abc import Callable

import networkx as nx
import numpy as np

from hyppo.coa._base import Equation, Structure
from hyppo.coa.graph import HypothesisGraph
from hyppo.comparison.compare import compute_aic, gaussian_log_likelihood, sign_test
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
    ["H1"],
    ["H2", "H3"],
    ["H4"],
    ["H5", "H6"],
    ["H7"],
    ["H8"],
    ["H11"],
    ["H12", "H12b"],
    ["H13"],
    ["H14", "H15"],
    ["H19"],
    ["GRP"],
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
        try:
            input("\n-- press Enter to continue --")
        except EOFError:
            pass


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
    print(
        f"\nW — workflow: {len(TASKS)} tasks (stages), e.g. "
        f"t2 = {{H2, H3}} runs both CRM channels in one stage."
    )
    n_configs = 2**N_BINARY_AXES * 3**N_TERNARY_AXES
    print(
        f"\nC — configuration space: {N_BINARY_AXES} binary + "
        f"{N_TERNARY_AXES} ternary axes, |C| = {n_configs} "
        "(constraint C1 prunes incompatible branch combinations)."
    )
    _pause()


def act_2_algorithm1() -> tuple[HypothesisLattice, nx.DiGraph]:
    """Algorithm 1: build the hypothesis lattice from equations + workflow."""
    _act(2, "Algorithm 1 — automatic hypothesis-graph construction")
    lattice = HypothesisLattice(ALL_HYPS, WF)
    g = lattice.lattice
    print("\nDerived_by edges (h_i -> h_j means h_j consumes the output of h_i):")
    edges = sorted(
        g.edges(),
        key=lambda e: (int(PAPER[str(e[0])][1:]), int(PAPER[str(e[1])][1:])),
    )
    for u, v in edges:
        print(
            f"  {PAPER[str(u)]:<4} -> {PAPER[str(v)]:<4}   "
            f"(output of {PAPER[str(u)]} appears in the equation of {PAPER[str(v)]})"
        )
    depth = nx.dag_longest_path_length(g)
    print(
        f"\nNodes: {g.number_of_nodes()}   Edges: {g.number_of_edges()}   "
        f"DAG: {nx.is_directed_acyclic_graph(g)}   Depth: {depth}"
    )
    print("Golden (paper [SVD], fig. 3): 16 nodes, 18 edges, depth 10.")
    _pause()
    return lattice, g


def act_3_algorithm2(g_full: nx.DiGraph) -> None:
    """Algorithm 2: incremental add is equivalent to a full rebuild (Lemma 2)."""
    _act(3, "Algorithm 2 — incremental addition (Lemma 2)")
    partial = HypothesisLattice(ALL_HYPS[:-1], WF)  # without H19 (paper H16)
    before = partial.lattice.number_of_edges()
    before_edges = {(str(u), str(v)) for u, v in partial.lattice.edges()}
    partial.add_hypothesis(HYP_OBJS["H19"])  # incremental, O(|H|) merges
    after_edges = {(str(u), str(v)) for u, v in partial.lattice.edges()}
    full_edges = {(str(u), str(v)) for u, v in g_full.edges()}
    new = sorted(after_edges - before_edges)
    print(f"\nLattice without H16: {before} edges.")
    print(
        "add_hypothesis(H16) added edges: "
        f"{[f'{PAPER[u]}->{PAPER[v]}' for u, v in new]}"
    )
    equal = after_edges == full_edges
    print(f"incremental == full rebuild: {equal}")
    print(
        "Golden: True; new edges H8->H16, H14->H16 "
        "(liquid and watercut branches merge into the oil forecast)."
    )
    if not equal:
        raise SystemExit("LEMMA 2 CHECK FAILED")
    _pause()


def act_4_plan_theorem1(g: nx.DiGraph) -> set[str]:
    """Algorithm 4: cascade recompute plan; Theorem 1: correct + minimal."""
    _act(4, "Algorithm 4 — recompute plan; Theorem 1 — correctness/minimality")
    codes = [code for code, _, _, _ in HYPS]
    idx = {c: i for i, c in enumerate(codes)}
    edges = [(idx[str(u)], idx[str(v)]) for u, v in g.edges()]
    hg = HypothesisGraph.from_edges(len(codes), edges)

    changed = "H8"  # scenario: the LPR fusion was re-fit
    cached = set(range(len(codes))) - {idx[changed]}
    plan = hg.plan(cached)
    plan_names = sorted((PAPER[codes[i]] for i in plan), key=lambda p: int(p[1:]))
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
    print(
        f"\nTheorem 1: plan is correct: {correct};  "
        f"subset-minimal (dropping any element breaks correctness): {minimal}"
    )
    if not (correct and minimal):
        raise SystemExit("THEOREM 1 CHECK FAILED")
    _pause()
    return {codes[i] for i in plan}


def act_5_run(g: nx.DiGraph, p_ne: set[str]) -> None:
    """Execute the plan with the Runner (topological order, retries, statuses)."""
    _act(5, "Execution — Runner walks P_ne in topological order")
    # Synthetic model fits: deterministic metrics per hypothesis.
    r2_table = {code: 0.75 + 0.01 * i for i, (code, _, _, _) in enumerate(HYPS)}
    r2_table["H7"] = 0.65  # the pure-ML hypothesis under-performs alone

    def make_model(code: str) -> Callable[[dict], dict]:
        def model(config: dict) -> dict:
            return {
                "r2": round(r2_table[code], 4),
                "aic": round(100.0 - 10.0 * r2_table[code], 1),
            }

        return model

    models = {code: make_model(code) for code, _, _, _ in HYPS}
    all_codes = {code for code, _, _, _ in HYPS}
    # Runner expects P_ne already in topological order (see Runner.execute).
    p_ne_ordered = [str(n) for n in nx.topological_sort(g) if str(n) in p_ne]
    plan = {"p_ne": p_ne_ordered, "p_e": all_codes - set(p_ne)}
    runner = Runner()
    results = runner.execute(
        plan,
        models,
        lattice_edges=[(str(u), str(v)) for u, v in g.edges()],
        competes={"H5": {"H7"}, "H7": {"H5"}},
    )
    print(
        f"\nExecuted {len(results)} hypotheses: P_ne ({len(plan['p_ne'])}) "
        f"recomputed with real metrics; P_e ({len(plan['p_e'])}) returned as "
        "vacuous SUCCESS with empty metrics — no repository attached, so "
        "cached results cannot be loaded."
    )
    print(
        "P_ne rows are SUPPORTED because their synthetic r2 (>= 0.75) exceeds "
        "theta_sup = 0.7; P_e rows stay PROPOSED because empty metrics mean "
        "'not evaluated' — membership in P_ne does not imply support."
    )
    for code in sorted(results, key=lambda c: int(PAPER[c][1:])):
        r = results[code]
        print(
            f"  {PAPER[code]:<4} status={r['status']:<8} "
            f"epistemic={r.get('epistemic_status', '-'):<10} "
            f"metrics={r.get('metrics', {})}"
        )
    _pause()


def act_6_compare() -> None:
    """Compare two competing hypotheses (Definitions 9-11)."""
    _act(6, "Comparison — physics (H5) vs ML (H7) liquid-rate hypotheses")
    rng = np.random.default_rng(42)
    y_true = rng.normal(100.0, 10.0, size=50)
    pred_phys = y_true + rng.normal(0.0, 3.0, size=50)  # tighter residuals
    pred_ml = y_true + rng.normal(0.0, 6.0, size=50)
    err_phys = list(np.abs(y_true - pred_phys))
    err_ml = list(np.abs(y_true - pred_ml))
    p = sign_test(err_phys, err_ml)
    n_h5_better = sum(a < b for a, b in zip(err_phys, err_ml, strict=True))
    # H5: ~3 physics params (J, a, b); H7: toy MLP ~12 weights
    aic_phys = compute_aic(3, gaussian_log_likelihood(y_true, pred_phys))
    aic_ml = compute_aic(12, gaussian_log_likelihood(y_true, pred_ml))
    print(
        f"\nH5 errors smaller in {n_h5_better}/{len(err_phys)} pairs; "
        f"sign test p = {p:.4f} (small p -> the difference is systematic)"
    )
    print(f"AIC: H5 (3 params) = {aic_phys:.1f}   H7 (12 params) = {aic_ml:.1f}")
    if not (p < 0.05 and aic_phys < aic_ml):
        raise SystemExit("ACT 6 VERDICT CHECK FAILED")
    print("Verdict: H5 preferred on both criteria; H8 fuses the two, which is")
    print("why the paper keeps BOTH in the lattice instead of discarding H7.")
    _pause()


def act_7_selfcheck(g: nx.DiGraph) -> None:
    """Assert golden values from tests/test_golden_claims.py."""
    _act(7, "Self-check against golden claims")
    ok = (
        g.number_of_nodes() == 16
        and g.number_of_edges() == 18
        and nx.is_directed_acyclic_graph(g)
        and nx.dag_longest_path_length(g) == 10
    )
    print(f"\n16 nodes / 18 edges / DAG / depth 10: {ok}")
    if not ok:
        raise SystemExit("GOLDEN SELF-CHECK FAILED")
    print("\nPRIMER OK")


def main() -> None:
    global PAUSE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pause", action="store_true", help="wait for Enter between acts"
    )
    PAUSE = parser.parse_args().pause
    act_1_tuple()
    _lattice, g = act_2_algorithm1()
    act_3_algorithm2(g)
    p_ne = act_4_plan_theorem1(g)
    act_5_run(g, p_ne)
    act_6_compare()
    act_7_selfcheck(g)


if __name__ == "__main__":
    main()
```

---

## Сводная таблица объёма остальных модулей пакета `hyppo`

Ниже перечислены все прочие исходные файлы пакета `hyppo` (не включённые
выше целиком), с числом строк каждого. Полные тексты депонированы в
составе дистрибутива и репозитория проекта; здесь приведён их объём для
подтверждения общего объёма программы.

| Файл | Строк |
|---|---|
| `hyppo/__init__.py` | 7 |
| `hyppo/actions/__init__.py` | 7 |
| `hyppo/actions/diff.py` | 159 |
| `hyppo/actions/registry.py` | 67 |
| `hyppo/actions/types.py` | 19 |
| `hyppo/actions/version.py` | 281 |
| `hyppo/actions/virtual_experiment.py` | 197 |
| `hyppo/adapters/__init__.py` | 5 |
| `hyppo/adapters/wfopt_adapter.py` | 423 |
| `hyppo/coa/__init__.py` | 13 |
| `hyppo/comparison/__init__.py` | 23 |
| `hyppo/comparison/compare.py` | 187 |
| `hyppo/core/__init__.py` | 11 |
| `hyppo/core/_hypothesis.py` | 280 |
| `hyppo/core/_workflow.py` | 68 |
| `hyppo/generator/__init__.py` | 3 |
| `hyppo/generator/_generator.py` | 67 |
| `hyppo/generator/_gp.py` | 3 |
| `hyppo/gui/__init__.py` | 3 |
| `hyppo/gui/api/__init__.py` | 0 |
| `hyppo/gui/api/comparison.py` | 23 |
| `hyppo/gui/api/graph.py` | 15 |
| `hyppo/gui/api/hypotheses.py` | 32 |
| `hyppo/gui/api/plan.py` | 15 |
| `hyppo/gui/api/projects.py` | 36 |
| `hyppo/gui/api/real.py` | 28 |
| `hyppo/gui/api/runs.py` | 28 |
| `hyppo/gui/api/ve.py` | 27 |
| `hyppo/gui/app.py` | 72 |
| `hyppo/gui/cli.py` | 24 |
| `hyppo/gui/demo.py` | 237 |
| `hyppo/gui/projects.py` | 97 |
| `hyppo/gui/schemas.py` | 26 |
| `hyppo/gui/services.py` | 88 |
| `hyppo/gui/ws.py` | 25 |
| `hyppo/lattice_constructor/__init__.py` | 0 |
| `hyppo/manager/__init__.py` | 3 |
| `hyppo/mcp/__init__.py` | 1 |
| `hyppo/mcp/__main__.py` | 6 |
| `hyppo/mcp/cli.py` | 44 |
| `hyppo/mcp/resources.py` | 41 |
| `hyppo/mcp/server.py` | 58 |
| `hyppo/mcp/tools.py` | 65 |
| `hyppo/metadata_repository/__init__.py` | 4 |
| `hyppo/metadata_repository/metadata_repository.py` | 133 |
| `hyppo/metadata_repository/shared_cache.py` | 61 |
| `hyppo/ontology/__init__.py` | 174 |
| `hyppo/ontology/consistency.py` | 237 |
| `hyppo/ontology/core_rules.py` | 182 |
| `hyppo/ontology/lifecycle.py` | 163 |
| `hyppo/ontology/markers.py` | 512 |
| `hyppo/ontology/model_compatibility.py` | 103 |
| `hyppo/ontology/multi_experiment.py` | 37 |
| `hyppo/ontology/oil_constraints.py` | 118 |
| `hyppo/ontology/provenance.py` | 114 |
| `hyppo/ontology/quality_gates.py` | 63 |
| `hyppo/ontology/quality_gates_qa.py` | 50 |
| `hyppo/ontology/workflow_validation.py` | 63 |
| `hyppo/personas/__init__.py` | 1 |
| `hyppo/planner/__init__.py` | 3 |
| `hyppo/runner/__init__.py` | 3 |
| `hyppo/storage/__init__.py` | 3 |
| `hyppo/storage/_base.py` | 219 |
| `hyppo/versioning/__init__.py` | 13 |
| `hyppo/versioning/_db.py` | 65 |
| `hyppo/versioning/version_store.py` | 185 |

**Итого прочих модулей:** 66 файлов, 5 320 строк.

**Итого по пакету `hyppo` (полные тексты выше + сводная таблица):**
75 файлов, 7 068 строк Python-кода (≈ 250 КБ); плюс сквозной пример
`examples/norne_primer.py` (352 строки) и статические ассеты
веб-интерфейса `hyppo/gui/static/` (3 файла, ≈ 232 КБ).
