from pathlib import Path

from fastapi.testclient import TestClient

from hyppo.gui.app import create_app


def test_spa_served_when_built(tmp_path):
    static_dir = Path(__file__).resolve().parents[2] / "hyppo" / "gui" / "static"
    client = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    if static_dir.exists():
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    else:
        # API still healthy without a build
        assert client.get("/api/health").status_code == 200


def test_warns_when_static_assets_missing(tmp_path, caplog):
    missing_dir = tmp_path / "no-static-here"
    with caplog.at_level("WARNING", logger="hyppo.gui.app"):
        create_app(db_path=str(tmp_path / "g.db"), static_dir=missing_dir)
    assert any(
        "SPA static assets not found" in record.message for record in caplog.records
    )


def test_serves_html_when_static_dir_provided(tmp_path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>hyppo</html>", encoding="utf-8")
    client = TestClient(
        create_app(db_path=str(tmp_path / "g.db"), static_dir=static_dir)
    )
    response = client.get("/")
    assert response.status_code == 200
    assert "hyppo" in response.text
