import networkx as nx
import pytest

from examples.research.safe_recompute import common


def test_crm_graph_matches_dissertation():
    g = common.crm_graph()
    assert g.number_of_nodes() == 16 and g.number_of_edges() == 18
    assert nx.is_directed_acyclic_graph(g)
    assert nx.dag_longest_path_length(g) == 10


def test_nfcore_corpus_has_26_dags():
    gs = common.nfcore_graphs()
    assert len(gs) == 26
    assert all(nx.is_directed_acyclic_graph(g) for g in gs.values())


def test_wfcommons_corpus_loads():
    gs = common.wfcommons_graphs(limit=5)
    assert 1 <= len(gs) <= 5
    assert all(nx.is_directed_acyclic_graph(g) and g.number_of_nodes() > 0 for g in gs.values())


def test_save_json_roundtrip(tmp_path):
    p = common.save_json({"a": 1}, tmp_path / "x.json")
    assert common.load_json(p) == {"a": 1}


def test_wfcommons_full_corpus_matches_e2():
    if not (common.PLANNING_CACHE / "wfcommons").exists():
        pytest.skip("WfCommons cache missing")
    gs = common.wfcommons_graphs()
    assert len(gs) == 171
    sweep = common.load_json(common.PLANNING_DATA / "wfcommons_per_workflow_sweep.json")["by_workflow"]
    assert sorted(g.number_of_nodes() for g in gs.values()) == sorted(v["n"] for v in sweep.values())
    with pytest.raises(ValueError):
        common.wfcommons_graphs(limit=0)


def test_crm_edges_equal_algorithm1_golden():
    """Единый источник: CRM_EDGES = рёбра, выведенные алгоритмом 1 (golden), в нумерации H1..H16."""
    from tests.test_golden_claims import NORNE_GOLDEN_EDGES

    to_diss = {"H11": "H9", "H12": "H10", "H12b": "H11", "H13": "H12", "H14": "H13",
               "H15": "H14", "H19": "H16", "GRP": "H15"}
    mapped = {(to_diss.get(a, a), to_diss.get(b, b)) for a, b in NORNE_GOLDEN_EDGES}
    assert set(common.CRM_EDGES) == mapped and len(common.CRM_EDGES) == len(mapped) == 18
