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
