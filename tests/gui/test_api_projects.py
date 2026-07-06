from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def _client(tmp_path):
    return TestClient(create_app(db_path=str(tmp_path / "g.db")))


def test_crud(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/projects", json={"name": "demo", "description": "d"})
    assert r.status_code == 201
    pid = r.json()["id"]
    # A seeded demo project may also be present, so look up by id rather
    # than assuming the created project is first in the list.
    names = {p["id"]: p["name"] for p in c.get("/api/projects").json()}
    assert names[pid] == "demo"
    assert c.get(f"/api/projects/{pid}").json()["id"] == pid
    assert c.delete(f"/api/projects/{pid}").status_code == 204
    assert c.get(f"/api/projects/{pid}").status_code == 404
