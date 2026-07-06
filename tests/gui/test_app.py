from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def test_health():
    client = TestClient(create_app(db_path=":memory:"))
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
