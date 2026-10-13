import networkx as nx
import pytest

from examples.research.safe_recompute import common


def test_crm_graph_matches_dissertation():
    g = common.crm_graph()
    assert g.number_of_nodes() == 16 and g.number_of_edges() == 19
    assert nx.is_directed_acyclic_graph(g)
    assert nx.dag_longest_path_length(g) == 10


def test_nfcore_corpus_has_26_dags():
    gs = common.nfcore_graphs()
    assert len(gs) == 26
    assert all(nx.is_directed_acyclic_graph(g) for g in gs.values())


needs_wfcommons = pytest.mark.skipif(not (common.PLANNING_CACHE / "wfcommons").exists(),
                                     reason="WfCommons cache missing (fetch_wfcommons)")


@needs_wfcommons
def test_wfcommons_corpus_loads():
    gs = common.wfcommons_graphs(limit=5)
    assert 1 <= len(gs) <= 5
    assert all(nx.is_directed_acyclic_graph(g) and g.number_of_nodes() > 0 for g in gs.values())


def test_save_json_roundtrip(tmp_path):
    p = common.save_json({"a": 1}, tmp_path / "x.json")
    assert common.load_json(p) == {"a": 1}


@needs_wfcommons
def test_wfcommons_full_corpus_matches_e2():
    gs = common.wfcommons_graphs()
    assert len(gs) == 171
    sweep = common.load_json(common.PLANNING_DATA / "wfcommons_per_workflow_sweep.json")["by_workflow"]
    assert sorted(g.number_of_nodes() for g in gs.values()) == sorted(v["n"] for v in sweep.values())
    with pytest.raises(ValueError):
        common.wfcommons_graphs(limit=0)


def test_wfcommons_manifest_pins_the_e2_corpus():
    m = common.wfcommons_manifest()
    assert len(m) == 171
    assert {tuple(k.split("/")) for k in m} == common._wfcommons_corpus()
    sweep = common.load_json(common.PLANNING_DATA / "wfcommons_per_workflow_sweep.json")["by_workflow"]
    assert sorted(v["n_nodes"] for v in m.values()) == sorted(v["n"] for v in sweep.values())


def test_wfcommons_graphs_rejects_changed_or_missing_trace(tmp_path):
    key, m = next(iter(common.wfcommons_manifest().items()))
    with pytest.raises(common.CorpusMismatch, match="missing"):
        common.wfcommons_graphs(limit=1, root=tmp_path)
    path = tmp_path / m["path"]
    path.parent.mkdir(parents=True)
    path.write_text('{"workflow": {"specification": {"tasks": []}}}', encoding="utf-8")
    with pytest.raises(common.CorpusMismatch, match="differs"):
        common.wfcommons_graphs(limit=1, root=tmp_path)


def test_crm_edges_equal_algorithm1_golden():
    """Единый источник: CRM_EDGES = рёбра, выведенные алгоритмом 1 (golden), в нумерации H1..H16."""
    from tests.test_golden_claims import NORNE_GOLDEN_EDGES

    to_diss = {"H11": "H9", "H12": "H10", "H12b": "H11", "H13": "H12", "H14": "H13",
               "H15": "H14", "H19": "H16", "GRP": "H15"}
    mapped = {(to_diss.get(a, a), to_diss.get(b, b)) for a, b in NORNE_GOLDEN_EDGES}
    assert set(common.CRM_EDGES) == mapped and len(common.CRM_EDGES) == len(mapped) == 19
