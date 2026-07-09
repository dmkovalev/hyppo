import json

from fastapi.testclient import TestClient

from hyppo.gui.api import real as real_api
from hyppo.gui.app import create_app


def test_env_override_respected(tmp_path, monkeypatch):
    data_path = tmp_path / "custom_real_data.json"
    data_path.write_text(json.dumps({"nodes": ["x"]}), encoding="utf-8")
    monkeypatch.setenv("HYPPO_REAL_DATA", str(data_path))
    monkeypatch.setattr(real_api, "_CACHE", None)

    client = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    response = client.get("/api/real")

    assert response.status_code == 200
    assert response.json() == {"nodes": ["x"]}
