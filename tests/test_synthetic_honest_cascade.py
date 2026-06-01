"""Топология 19-узлового графа HybridCRM и физические каскады."""
from hyppo.actions.diff import derived_by_closure
from experiments.chapter4.synthetic_honest import (
    HYBRIDCRM_19_EDGES, HYBRIDCRM_19_NODES,
)


def test_graph_has_19_nodes_21_edges():
    assert len(HYBRIDCRM_19_NODES) == 19
    assert len(HYBRIDCRM_19_EDGES) == 21


def test_water_breakthrough_cascade_h12():
    # Прорыв воды = рост фракционного потока f_w (Баклея–Леверетта, H12).
    desc = derived_by_closure(HYBRIDCRM_19_EDGES, ["H12"])
    assert set(desc) == {"H14", "H16", "H17", "H18", "H19"}
    assert len(desc) == 5  # 5 потомков


def test_connectivity_change_cascade_h1():
    # Изменение связности скважин (ГТМ) = корневая гипотеза H1.
    desc = derived_by_closure(HYBRIDCRM_19_EDGES, ["H1"])
    assert len(desc) == 10  # 10 потомков — совпадает с синопсисом
    assert "H19" in desc and "H10" in desc


def test_longest_path_is_7():
    import functools
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in HYBRIDCRM_19_EDGES:
        adj[a].append(b)

    @functools.lru_cache(None)
    def lp(n):
        return 0 if not adj[n] else 1 + max(lp(c) for c in adj[n])
    assert max(lp(n) for n in HYBRIDCRM_19_NODES) == 7
