from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def _setup(tmp_path):
    c = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    pid = c.post("/api/projects", json={"name": "d"}).json()["id"]
    c.put(f"/api/projects/{pid}/hypotheses", json={
        "hypotheses": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "workflow_edges": [["a", "b"], ["b", "c"]],
    })
    return c, pid


def test_graph_and_plan(tmp_path):
    c, pid = _setup(tmp_path)
    g = c.get(f"/api/projects/{pid}/graph").json()
    assert set(g["nodes"]) == {"a", "b", "c"}
    p = c.get(f"/api/projects/{pid}/plan").json()
    assert set(p["p_ne"]) == {"a", "b", "c"}
