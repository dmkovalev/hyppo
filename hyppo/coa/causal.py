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
