"""Causal refinement of the safe recomputation plan (theorem 5).

A hypothesis ``h`` is a system of equations ``structure: equation -> frozenset(variables)``
with declared exogenous inputs ``In(h)``; ``Out(h) = Vars(h) \\ In(h)``. Hypotheses are
connected by the graph of hypotheses (edges ``supplier -> consumer``); with output
collisions every producer among the predecessors supplies the input ("variant A").

Parameter identity. A *parameter* of ``h`` is a pair ``(h, name)`` with ``name`` in
``In(h)`` not produced by any predecessor of ``h``. Inputs that are produced by a
predecessor are *data inputs*: their influence reaches ``h`` through the supplier's
output variable.

Influence graph ``D``. Nodes:

* ``("p", h, name)`` -- parameter ``name`` of ``h``;
* ``("in", h, x)``   -- data input ``x`` of ``h``;
* ``("v", h, x)``    -- output variable ``x`` of ``h``.

Inside ``h`` the causal ordering is computed from a perfect matching ``phi`` of the
exogenized structure (Hopcroft--Karp, :func:`hyppo.coa.causal.perfect_matching`): for
every equation ``g`` and every ``u in Vars(g) \\ {phi(g)}`` there is an edge
``u -> phi(g)``. A cyclic block (strongly connected component of this digraph,
i.e. a Dulmage--Mendelsohn block) is therefore strongly connected in ``D``, so every
variable of the block is reachable from every input and parameter of the block --
exactly the block rule "every variable of a block depends on all inputs and
parameters of the block". Reachability does not depend on the chosen perfect
matching (phi-invariance). Between hypotheses: ``("v", a, x) -> ("in", h, x)`` for
every edge ``(a, h)`` with ``x in Out(a) & In(h)``.

The trace ``Pi(h)`` is the set of parameters from which some output of ``h`` is
reachable in ``D``. ``N_caus`` = hypotheses whose trace meets the changed parameters,
the changed hypotheses themselves, and the stale-marked hypotheses with all their
descendants in the graph of hypotheses.
"""
from __future__ import annotations

import networkx as nx

from hyppo.coa import causal

Structure = dict[str, frozenset[str]]  # equation -> variables

__all__ = ["Structure", "influence_graph", "traces", "affected_variables", "n_caus"]


def _matching(structure: Structure, inputs: set[str]) -> dict[str, str]:
    """Perfect matching ``equation -> output variable`` of the exogenized structure.

    Inputs are exogenized by trivial equations ``{x}``; the structure must be complete
    (``|E| = |Out|``) and admit a perfect matching, otherwise ``ValueError``.
    """
    names = sorted(structure)
    eqs = [frozenset(structure[n]) for n in names]
    allvars = causal.variables(eqs) | set(inputs)
    outs = allvars - set(inputs)
    if len(eqs) != len(outs):
        raise ValueError(f"structure is not complete: {len(eqs)} equations, {len(outs)} outputs")
    full = eqs + [frozenset([x]) for x in sorted(inputs)]
    m = causal.perfect_matching(full)
    if m is None or len(m) != len(full):
        raise ValueError("structure has no perfect matching (structurally singular)")
    phi = {names[i]: m[i] for i in range(len(names))}
    assert set(phi.values()) == outs
    return phi


def _causal_digraph(structure: Structure, phi: dict[str, str]) -> nx.DiGraph:
    g = nx.DiGraph()
    for e, vs in structure.items():
        g.add_nodes_from(vs)
        for u in vs:
            if u != phi[e]:
                g.add_edge(u, phi[e])
    return g


def _outputs(structure: Structure, inputs: set[str]) -> set[str]:
    return causal.variables([frozenset(v) for v in structure.values()]) - set(inputs)


def _predecessors(hyps, edges) -> dict[str, set[str]]:
    preds: dict[str, set[str]] = {h: set() for h in hyps}
    for a, b in edges:
        preds.setdefault(b, set()).add(a)
        preds.setdefault(a, set())
    return preds


def influence_graph(structures: dict[str, Structure], inputs: dict[str, set[str]],
                    edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Influence graph ``D`` over parameter, data-input and output-variable nodes.

    Node kinds: ``("p", h, name)`` parameter, ``("in", h, x)`` data input (supplied by a
    predecessor), ``("v", h, x)`` output variable. Variant A: every predecessor producing
    ``x`` supplies the input ``x`` of the consumer (edge ``("v", a, x) -> ("in", h, x)``).
    Raises ``ValueError`` if ``edges`` mention unknown hypotheses or are not acyclic.
    """
    for a, b in edges:
        if a not in structures or b not in structures:
            raise ValueError(f"edge ({a!r}, {b!r}) mentions a hypothesis not in structures")
    if not nx.is_directed_acyclic_graph(nx.DiGraph(list(edges))):
        raise ValueError("edges of the graph of hypotheses must form a DAG")
    preds = _predecessors(structures, edges)
    outs = {h: _outputs(s, inputs.get(h, set())) for h, s in structures.items()}
    D = nx.DiGraph()
    for h, s in structures.items():
        ins = set(inputs.get(h, set()))
        suppliers = {x: sorted(a for a in preds[h] if x in outs.get(a, set())) for x in ins}

        def node(u, h=h, ins=ins, suppliers=suppliers):
            if u not in ins:
                return ("v", h, u)
            return ("in", h, u) if suppliers[u] else ("p", h, u)

        for x in sorted(ins):
            D.add_node(node(x))
            for a in suppliers[x]:
                D.add_edge(("v", a, x), ("in", h, x))
        for v in sorted(outs[h]):
            D.add_node(("v", h, v))
        phi = _matching(s, ins)
        for u, w in _causal_digraph(s, phi).edges:
            D.add_edge(node(u), node(w))
    return D


def traces(D: nx.DiGraph, hypotheses: list[str]) -> dict[str, set]:
    """``Pi(h)``: returns ``dict h -> set of (h', name)`` parameter pairs from which some
    output of ``h`` is reachable in ``D``."""
    out_nodes: dict[str, list] = {h: [] for h in hypotheses}
    for n in D.nodes:
        if n[0] == "v" and n[1] in out_nodes:
            out_nodes[n[1]].append(n)
    res: dict[str, set] = {}
    for h in hypotheses:
        anc: set = set()
        for n in out_nodes[h]:
            anc |= nx.ancestors(D, n)
        res[h] = {(n[1], n[2]) for n in anc if n[0] == "p"}
    return res


def affected_variables(structure: Structure, edited_equations: set[str], inputs: set[str]) -> set[str]:
    """Variables affected by an edit of ``edited_equations``: forward closure in the causal
    digraph of ``structure`` from the outputs matched to the edited equations.

    The closure is computed with the matching ``phi`` of the OLD structure. For an edit
    ``S -> S'`` that introduces new variables the caller adds ``V' \ V`` and intersects the
    result with ``V'`` (theorem on affected variables). Parameter changes are not handled
    here; they go through :func:`n_caus`.
    """
    unknown = set(edited_equations) - set(structure)
    if unknown:
        raise ValueError(f"unknown equations: {sorted(unknown)}")
    phi = _matching(structure, set(inputs))
    g = _causal_digraph(structure, phi)
    seeds = {phi[e] for e in edited_equations}
    res = set(seeds)
    for s in seeds:
        res |= nx.descendants(g, s)
    return res - set(inputs)


def n_caus(structures: dict[str, Structure], inputs: dict[str, set[str]], edges: list[tuple[str, str]],
           changed_params: set, stale: set[str]) -> set[str]:
    """Minimal safe plan under structural knowledge.

    ``changed_params`` is a set of ``(h, name)`` pairs. A parameter is an input of ``h`` not
    supplied by a predecessor; a global exogenous ``x`` read by several hypotheses must be
    passed as every pair ``(h, x)``. ``stale`` is the set of hypotheses ALREADY marked
    ``StaleHypothesis``, i.e. the marked set ``H_dep(U)`` of revisions ``U`` (descendants of
    revised hypotheses). Each marked hypothesis is recomputed together with its descendants.
    A revised hypothesis ``u`` itself is NOT stale (its own result stays valid): callers pass
    ``descendants(U)``, not ``U``. Unknown names in ``stale`` raise ``ValueError``.
    """
    unknown = set(stale) - set(structures)
    if unknown:
        raise ValueError(f"stale hypotheses not in structures: {sorted(unknown)}")
    D = influence_graph(structures, inputs, edges)
    for h, name in changed_params:
        if ("p", h, name) not in D:
            raise ValueError(f"({h!r}, {name!r}) is not a parameter (unknown or supplied by a predecessor)")
    hyps = list(structures)
    tr = traces(D, hyps)
    changed = set(changed_params)
    res = {h for h in hyps if tr[h] & changed} | {h for h, _ in changed}
    G = nx.DiGraph()
    G.add_nodes_from(hyps)
    G.add_edges_from(edges)
    for h in stale:
        res.add(h)
        if h in G:
            res |= nx.descendants(G, h)
    return res
