from pathlib import Path

from fastapi.testclient import TestClient

from hyppo.gui.app import create_app


def test_spa_served_when_built(tmp_path):
    dist = Path(__file__).resolve().parents[2] / "webui" / "dist"
    client = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    if dist.exists():
        assert client.get("/").status_code == 200
    else:
        # API still healthy without a build
        assert client.get("/api/health").status_code == 200
