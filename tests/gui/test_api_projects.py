from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def _client(tmp_path):
    return TestClient(create_app(db_path=str(tmp_path / "g.db")))


def test_crud(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/projects", json={"name": "demo", "description": "d"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert c.get("/api/projects").json()[0]["name"] == "demo"
    assert c.get(f"/api/projects/{pid}").json()["id"] == pid
    assert c.delete(f"/api/projects/{pid}").status_code == 204
    assert c.get(f"/api/projects/{pid}").status_code == 404
