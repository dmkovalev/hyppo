from fastapi.testclient import TestClient
from hyppo.gui.app import create_app


def test_define_and_read_ve(tmp_path):
    c = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    pid = c.post("/api/projects", json={"name": "d"}).json()["id"]
    ve = {
        "hypotheses": [
            {"id": "h_atlas", "params": {"atlas": ["HO", "AAL", "S100", "S200"]}},
            {"id": "h_conn", "params": {"kind": ["pearson", "partial", "cov", "tangent"]}},
            {"id": "h_group", "params": {"grp": ["m", "f", "mix"]}},
        ],
        "workflow_edges": [["h_atlas", "h_conn"], ["h_conn", "h_group"]],
    }
    r = c.put(f"/api/projects/{pid}/hypotheses", json=ve)
    assert r.status_code == 200
    body = c.get(f"/api/projects/{pid}/hypotheses").json()
    assert body["config_space_size"] == 48  # 4*4*3
    assert len(body["hypotheses"]) == 3
