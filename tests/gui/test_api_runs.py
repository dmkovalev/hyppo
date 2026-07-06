from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def _setup(tmp_path):
    c = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    pid = c.post("/api/projects", json={"name": "d"}).json()["id"]
    c.put(f"/api/projects/{pid}/hypotheses", json={
        "hypotheses": [{"id": "a"}, {"id": "b"}],
        "workflow_edges": [["a", "b"]],
    })
    return c, pid


def test_run_creates_iteration_with_results(tmp_path):
    c, pid = _setup(tmp_path)
    r = c.post(f"/api/projects/{pid}/runs")
    assert r.status_code == 201
    body = r.json()
    assert body["iteration"] == 1
    assert set(body["results"].keys()) == {"a", "b"}

    hist = c.get(f"/api/projects/{pid}/runs").json()
    assert hist[0]["iteration"] == 1
    # second run reuses cache → reused > 0
    c.post(f"/api/projects/{pid}/runs")
    hist2 = c.get(f"/api/projects/{pid}/runs").json()
    assert hist2[0]["iteration"] == 2
    assert hist2[0]["reused"] >= 1
