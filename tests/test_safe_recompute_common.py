import networkx as nx

from examples.research.safe_recompute import common


def test_crm_graph_matches_dissertation():
    g = common.crm_graph()
    assert g.number_of_nodes() == 16 and g.number_of_edges() == 17
    assert nx.is_directed_acyclic_graph(g)
    assert nx.dag_longest_path_length(g) == 11


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
