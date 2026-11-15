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

    * :meth:`build` -- Algorithm 1 (build the hypothesis graph),
    * :meth:`add_hypothesis` -- Algorithm 2 (incremental add),
    * :meth:`plan` -- Algorithm 4 (cascade-aware recompute planning).
    """

    def __init__(self) -> None:
        self._hyps: list[list[frozenset[str]]] = []
        self._edges: set[tuple[int, int]] = set()
        self._adj: dict[int, set[int]] = defaultdict(set)
        # Per-hypothesis signature computed once at insertion (condition У0):
        # outputs Out(h) = variables matched by the causal mapping phi over the
        # (exogenised) structure, inputs In(h) = the remaining variables. Having
        # Out/In per hypothesis lets Algorithm 1 decide every edge by the cheap
        # test Out(h_i) & In(h_j) != empty (bridge to causal ordering,
        # Theorem thm:build:II) instead of a per-pair transitive closure.
        self._out: list[set[str]] = []
        self._in: list[set[str]] = []
        # Pairs found by the last :meth:`build` that violate condition У3:
        # output collisions (competes, a legitimate rivalry resolved by
        # re-evaluation; the derivability edge is still built if one rival reads
        # the other's output) and counter-supplies (order conflicts, a
        # specification error; no edge).
        self.competes_pairs: list[tuple[int, int]] = []
        self.order_conflicts: list[tuple[int, int]] = []

    # ---- construction ---------------------------------------------------
    def _signature(
        self, eqs: list[frozenset[str]], inputs=None
    ) -> tuple[set[str], set[str]]:
        """Out(h)/In(h) and the У0 check -- one Hopcroft-Karp matching, O(s * sqrt(v)).

        With declared inputs (condition У2, the model's signature) the split is
        the dissertation's: ``In(h)`` as declared, ``Out(h) = V_h \\ In(h)``,
        independent of the matching; the matching only certifies У0
        (``|E_h| + |In(h)| = |V_h|`` and every equation maps to a distinct
        output). Without declared inputs the outputs fall back to the variables
        saturated by a deterministic matching -- adequate for complete structures
        (``|E| = |V|``, then ``In = ∅``), but the choice is matching-dependent
        when ``|E| < |V|``, so callers should declare inputs in that case."""
        allv = causal.variables(eqs)
        if inputs is None:
            m = causal.perfect_matching(eqs)
            out = set(m.values()) if m else set()
            return out, allv - out
        inp = set(inputs)
        if not inp <= allv:
            raise ValueError(f"declared inputs not in the structure: {sorted(inp - allv)}")
        out = allv - inp
        if len(eqs) != len(out):
            raise ValueError("condition У0 violated: |E| + |In| != |V|")
        if eqs and causal.perfect_matching([eq - inp for eq in eqs]) is None:
            raise ValueError("condition У0 violated: no complete causal mapping onto Out(h)")
        return out, inp

    def add(self, equations, inputs=None) -> int:
        """Add a hypothesis (a structure) as a new node; return its index.

        ``inputs`` -- the declared input variables In(h) of the implementing
        model (condition У2). Computes the signature Out(h)/In(h) once and checks
        condition У0 with a single Hopcroft-Karp matching, so building the graph
        never needs a per-pair causal closure."""
        eqs = [frozenset(eq) for eq in equations]
        out, inp = self._signature(eqs, inputs)
        self._hyps.append(eqs)
        self._out.append(out)
        self._in.append(inp)
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

    # ---- Algorithm 1: build_graph -------------------------------------
    def build(self) -> list[tuple[int, int]]:
        """Algorithm 1 (build_graph), faithful to Lemma 1.

        Two phases, matching the complexity proof:

        * У0 phase -- the signature Out(h)/In(h) of every hypothesis is already
          materialised (one Hopcroft-Karp matching each, done in :meth:`add`),
          costing ``O(|H| * s_max * sqrt(v_max))`` overall;
        * pair phase -- reachability in the workflow is precomputed once
          (``O(|V(W)|*(|V(W)|+|E(W)|))``); then for every reachable ordered pair
          the derivability edge is decided by the cheap set test
          ``Out(h_i) & In(h_j) != empty`` in ``O(v_max)``, with NO per-pair
          transitive closure (bridge to causal ordering, Theorem thm:build:II).

        Total ``O(|H|^2 * v_max + |H| * s_max * sqrt(v_max) +
        |V(W)|*(|V(W)|+|E(W)|))`` (Lemma 1)."""
        reach = self._reachable()
        hyp_graph: list[tuple[int, int]] = []
        self.competes_pairs = []
        self.order_conflicts = []
        ins = list(self._in)  # snapshot; the pair loop reads In(h_j) once per pair
        for i in range(len(self._hyps)):
            out_i, in_i = self._out[i], ins[i]
            for j in reach[i]:
                if j == i:
                    continue
                if self._pair(i, j, out_i, in_i, self._in[j]):
                    hyp_graph.append((i, j))
        self.hyp_graph = set(hyp_graph)
        return hyp_graph

    def _pair(self, i, j, out_i, in_i, in_j) -> bool:
        """Decide the ordered pair (i, j), tau(i) < tau(j), by definition def:derived.

        An output collision is recorded as competes and does NOT suppress the
        edge (the rivals may still depend on each other through another
        variable); a counter-supply is recorded as an order conflict and gives
        no edge. Otherwise the edge exists iff ``Out(i) & In(j) != empty``.
        O(v_max)."""
        out_j = self._out[j]
        collision = bool(out_i & out_j)
        counter = bool(in_i & out_j)
        if collision:
            self.competes_pairs.append((i, j))
        if counter:
            self.order_conflicts.append((i, j))
        return not counter and bool(out_i & in_j)

    def _reach_from(self, s: int, adj) -> set[int]:
        seen: set[int] = set()
        st = [s]
        while st:
            nd = st.pop()
            for nb in adj.get(nd, ()):
                if nb not in seen:
                    seen.add(nb)
                    st.append(nb)
        return seen

    # ---- Algorithm 2: add_hypothesis ------------------------------------
    def add_hypothesis(self, equations, inputs=None, preds=(), succs=()) -> int:
        """Algorithm 2 (incremental add), faithful to Lemma 2 and Theorem thm:build (III).

        Adds the hypothesis as a new node with its workflow edges ``preds -> new``
        and ``new -> succs``, checks У0 with one Hopcroft-Karp matching
        (``O(s_max * sqrt(v_max))``), finds the nodes related to the new one by
        reachability in either direction with two traversals
        (``O(|V(W)| + |E(W)|)``), and decides only the pairs involving the new
        hypothesis by the same rule as :meth:`build` (``O(|H| * v_max)``). The
        new derivability edges are added to :attr:`graph` and returned in
        :attr:`last_added`.

        Precondition of thm:build (III): reachability between the existing nodes
        is preserved. If the new node opens a path ``p -> new -> s`` between
        previously unrelated nodes, the incremental result would be incomplete,
        so the graph is rebuilt in full (and :attr:`last_add_was_rebuild` is set).
        The precondition check costs O(min(|preds|, |succs|) * (|V|+|E|));
        the subsequent incremental step has the bound above. Thus the entire
        call, excluding sorting the returned edges, is linear on sparse workflows
        only when that minimum is bounded (and no full rebuild is needed).
        Returns the new node index."""
        preds, succs = tuple(set(preds)), tuple(set(succs))
        n = len(self._hyps)
        if any(not 0 <= v < n for v in (*preds, *succs)):
            raise IndexError("hypothesis index out of range")
        if not hasattr(self, "hyp_graph"):
            self.build()
        radj: dict = {}
        for u, v in self._edges:
            radj.setdefault(v, set()).add(u)
        # A new path through the inserted vertex must not introduce a cycle.
        seen = set(succs)
        stack = list(succs)
        while stack:
            for v in self._adj.get(stack.pop(), ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if seen.intersection(preds):
            raise ValueError("insertion would create a workflow cycle")
        new_paths = False
        if preds and succs:
            # Traverse from the smaller boundary, reversing edges when needed.
            starts, targets, adj = (preds, succs, self._adj) if len(preds) <= len(succs) else (succs, preds, radj)
            for node in starts:
                r = self._reach_from(node, adj)
                if any(target not in r for target in targets):
                    new_paths = True
                    break
        idx = self.add(equations, inputs)
        for pnode in preds:
            self.connect(pnode, idx)
        for snode in succs:
            self.connect(idx, snode)
        self.last_add_was_rebuild = new_paths
        if new_paths:
            before = set(self.hyp_graph)
            self.build()
            self.last_added = sorted(self.hyp_graph - before)
            return idx
        radj: dict = {}
        for u, v in self._edges:
            radj.setdefault(v, set()).add(u)
        ancestors = self._reach_from(idx, radj)
        descendants = self._reach_from(idx, self._adj)
        out_n, in_n = self._out[idx], self._in[idx]
        added = []
        for a in ancestors:
            if self._pair(a, idx, self._out[a], self._in[a], in_n):
                added.append((a, idx))
        for b in descendants:
            if self._pair(idx, b, out_n, in_n, self._in[b]):
                added.append((idx, b))
        self.hyp_graph |= set(added)
        self.last_added = sorted(added)
        return idx

    # ---- Algorithm 4: plan ----------------------------------------------
    def plan(self, cached) -> set[int]:
        """Algorithm 4. Given the indices of already-cached hypotheses, return the
        recompute set ``P_ne``: every non-cached node and, transitively, all of its
        ``derived_by`` dependents (the *cascade effect* -- an invalidated fragment
        forces recomputation of everything downstream). Processes nodes in
        topological order so each is settled before its dependents are reached.
        Each vertex enters the cascade at most once; total work is O(V + E)
        with constant-time set operations."""
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
        for h in topo:
            if h in p_ne:
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
