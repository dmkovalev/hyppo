"""Hypothesis-graph layer over the COA core -- Algorithms 1, 2 and 4 of the
dissertation chapter on planning virtual experiments with reuse of previously
computed fragments (D. Yu. Kovalev).

A *hypothesis* is a structure: a ``list[frozenset[str]]`` of equations over
variable names. A :class:`HypothesisGraph` holds hypotheses connected by
``derived_by`` edges (a DAG) and is the single source of truth for the three
planning algorithms. All causal work delegates to :mod:`hyppo.coa.causal`
(the Dulmage-Mendelsohn decomposition: Hopcroft-Karp matching + Tarjan SCC), so the
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


def plan_cascade(
    nodes: list[str],
    edges: list[tuple[str, str]],
    cached: set[str],
) -> set[str]:
    """String-id façade over :meth:`HypothesisGraph.plan` (Algorithm 4).

    Pure two-way topological cascade: the recompute set ``P_ne`` is every
    non-cached node together with all of its ``derived_by`` descendants. No R²
    filtering (the R²-aware three-way prune lives in the planner/manager layer,
    which cannot be expressed by this two-way core). Maps string ids to the
    integer core, delegates to the single golden-pinned ``plan`` implementation,
    and maps the result back to strings.
    """
    index = {name: i for i, name in enumerate(nodes)}
    int_edges = [(index[u], index[v]) for u, v in edges]
    int_cached = {index[h] for h in cached if h in index}
    p_ne_int = HypothesisGraph.from_edges(len(nodes), int_edges).plan(int_cached)
    return {nodes[i] for i in p_ne_int}
