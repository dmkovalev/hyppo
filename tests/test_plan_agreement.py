"""Cross-layer agreement for Algorithm 4 cascade planning.

Proves the four planning surfaces stay in lockstep:

- the pure two-way core (`coa.graph.plan_cascade`, str façade over the
  golden-pinned `HypothesisGraph.plan`);
- the GUI preview (`gui.services.plan_preview`, delegates to plan_cascade);
- the R2-aware three-way planner (`planner.build_optimal_plan`);
- the R2-aware three-way manager (`Manager._partition`).

Where no low-R2 pruning happens, all four agree on P_ne (three-way identity).
Where a cached result is below threshold, planner and manager prune the node
and its whole subtree identically (the GUI/core layers have no R2 and are not
compared there, by design).
"""

import networkx as nx

from hyppo.coa.graph import plan_cascade
from hyppo.gui.services import plan_preview
from hyppo.manager import Manager
from hyppo.metadata_repository import MetadataRepository, SharedCache
from hyppo.planner._base import build_optimal_plan


class _Lat:
    """Minimal HypothesisLattice-like wrapper around a DiGraph (str nodes)."""

    def __init__(self, graph: nx.DiGraph):
        self.lattice = graph
        self.hypotheses = list(graph.nodes())


def _lattice(nodes: list[str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def _planner_pne(nodes, edges, cache_content, tmp_path_factory=None):
    g = _lattice(nodes, edges)
    cache = SharedCache(":memory:")
    try:
        for h, metrics in cache_content.items():
            cache.save_result(h, {}, metrics)
        plan = build_optimal_plan(
            None,
            _Lat(g),
            cache,
            r2_threshold=0.7,
            per_hypothesis_configs={n: {} for n in nodes},
        )
        return set(plan.needs_execution), set(plan.cached)
    finally:
        cache.close()


def _manager_pne(nodes, edges, cache_content, db_path):
    mgr = Manager(db_path=db_path, r2_threshold=0.7)
    try:
        for h, metrics in cache_content.items():
            mgr.repository.save_result(h, {}, metrics)
        p_ne, p_e = mgr._partition(_lattice(nodes, edges), {n: {} for n in nodes})
        return set(p_ne), set(p_e)
    finally:
        mgr.close()


def _gui_pne(nodes, edges, cache_content, db_path):
    repo = MetadataRepository(db_path=db_path)
    try:
        for h, metrics in cache_content.items():
            repo.save_result(h, {}, metrics)
    finally:
        repo.close()
    ve = {
        "hypotheses": [{"id": n} for n in nodes],
        "workflow_edges": [list(e) for e in edges],
    }
    return set(plan_preview(ve, db_path)["p_ne"])


def test_three_way_identity_no_prune(tmp_path):
    """No low-R2 anywhere: core/gui/planner/manager return identical P_ne."""
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c"), ("a", "d")]
    # 'a' cached with good R2; b, c, d miss -> cascade recompute.
    cache_content = {"a": {"r2": 0.9}}

    oracle = plan_cascade(nodes, edges, set(cache_content))
    assert oracle == {"b", "c", "d"}

    gui = _gui_pne(nodes, edges, cache_content, str(tmp_path / "gui.db"))
    planner_pne, planner_pe = _planner_pne(nodes, edges, cache_content)
    manager_pne, manager_pe = _manager_pne(
        nodes, edges, cache_content, tmp_path / "mgr.db"
    )

    assert oracle == gui == planner_pne == manager_pne
    assert planner_pe == manager_pe == {"a"}


def test_low_r2_prune_planner_manager_identity(tmp_path):
    """Low-R2 node prunes itself and its subtree from BOTH sets, identically
    in planner and manager."""
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c"), ("b", "d")]
    # a good; b below threshold -> b, c, d pruned entirely.
    cache_content = {
        "a": {"r2": 0.9},
        "b": {"r2": 0.4},
        "c": {"r2": 0.95},
        "d": {"r2": 0.95},
    }

    planner_pne, planner_pe = _planner_pne(nodes, edges, cache_content)
    manager_pne, manager_pe = _manager_pne(
        nodes, edges, cache_content, tmp_path / "mgr.db"
    )

    planner_pruned = set(nodes) - planner_pne - planner_pe
    manager_pruned = set(nodes) - manager_pne - manager_pe

    assert planner_pruned == manager_pruned == {"b", "c", "d"}
    assert planner_pe == manager_pe == {"a"}
    assert planner_pne == manager_pne == set()


def test_plan_cascade_matches_reachability_oracle():
    """plan_cascade == non-cached nodes ∪ their nx.descendants (two-way core)."""
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("b", "c"), ("d", "e")]
    cached = {"a", "d"}

    result = plan_cascade(nodes, edges, cached)

    g = _lattice(nodes, edges)
    non_cached = set(nodes) - cached
    oracle = set(non_cached)
    for n in non_cached:
        oracle |= nx.descendants(g, n)

    assert result == oracle == {"b", "c", "e"}
