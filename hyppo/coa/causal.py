"""Pure causal-ordering core for COA. stdlib only -- no owlready, sympy, networkx.

An *equation* is a set of variable names (``frozenset[str]``); a *structure* is a
list of equations. A structure is *complete* iff ``|equations| == |variables|``.
The Dulmage-Mendelsohn decomposition (a perfect matching plus the strongly
connected components of the matching-induced dependency digraph) yields the
irreducible ("minimal complete") blocks in polynomial time.
"""
from __future__ import annotations

from collections import defaultdict


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
    """Match each equation to a distinct variable it contains (Kuhn's algorithm).

    Returns ``{eq_index: var}`` saturating every equation, or ``None`` if no such
    matching exists (the structure is structurally singular). Candidate variables
    are tried in sorted order, giving a deterministic, name-stable result.
    Returns an empty dict (not None) for an empty equation list.
    """
    cand = [sorted(eq) for eq in equations]
    match_var = {}  # var -> eq index

    def try_aug(i: int, seen: set[str]) -> bool:
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


def strongly_connected_components(
    adj: dict[str, set[str]]
) -> list[frozenset[str]]:
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
