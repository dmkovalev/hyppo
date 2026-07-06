from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def test_demo_seeded(tmp_path):
    c = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    names = [p["name"] for p in c.get("/api/projects").json()]
    assert "norne-brugge" in names
    pid = next(p["id"] for p in c.get("/api/projects").json()
               if p["name"] == "norne-brugge")
    ve = c.get(f"/api/projects/{pid}/hypotheses").json()
    assert ve["config_space_size"] >= 2
    assert len(ve["hypotheses"]) >= 2


def test_demo_seeded_once(tmp_path):
    db = str(tmp_path / "g.db")
    TestClient(create_app(db_path=db))
    c2 = TestClient(create_app(db_path=db))
    names = [p["name"] for p in c2.get("/api/projects").json()]
    assert names.count("norne-brugge") == 1
