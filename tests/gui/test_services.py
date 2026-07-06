from hyppo.gui.services import build_graph, plan_preview

VE = {
    "hypotheses": [{"id": "a", "params": {}}, {"id": "b", "params": {}}, {"id": "c", "params": {}}],
    "workflow_edges": [["a", "b"], ["b", "c"]],
}


def test_build_graph():
    g = build_graph(VE)
    assert set(g["nodes"]) == {"a", "b", "c"}
    assert ["a", "b"] in g["edges"] and ["b", "c"] in g["edges"]


def test_plan_preview_all_recompute_when_empty(tmp_path):
    # empty repository → nothing cached → all in P_ne
    plan = plan_preview(VE, db_path=str(tmp_path / "m.db"))
    assert set(plan["p_ne"]) == {"a", "b", "c"}
    assert plan["p_e"] == []
