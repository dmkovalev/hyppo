# tests/gui/test_api_comparison.py
from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def test_comparison(tmp_path):
    c = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    pid = c.post("/api/projects", json={"name": "d"}).json()["id"]
    c.put(f"/api/projects/{pid}/hypotheses", json={
        "hypotheses": [{"id": "a"}, {"id": "b"}],
        "workflow_edges": [["a", "b"]],
    })
    c.post(f"/api/projects/{pid}/runs")
    r = c.get(f"/api/projects/{pid}/comparison")
    assert r.status_code == 200
    rows = r.json()["rows"]
    ids = {row["hypothesis"] for row in rows}
    assert ids == {"a", "b"}
    assert all("r2" in row for row in rows)
