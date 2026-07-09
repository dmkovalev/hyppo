from fastapi.testclient import TestClient

from hyppo.gui.app import create_app


def test_ws_emits_progress(tmp_path):
    c = TestClient(create_app(db_path=str(tmp_path / "g.db")))
    pid = c.post("/api/projects", json={"name": "d"}).json()["id"]
    c.put(
        f"/api/projects/{pid}/hypotheses",
        json={
            "hypotheses": [{"id": "a"}, {"id": "b"}],
            "workflow_edges": [["a", "b"]],
        },
    )
    with c.websocket_connect(f"/api/projects/{pid}/runs/ws") as ws:
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m.get("event") == "done":
                break
    kinds = [m["event"] for m in msgs]
    assert kinds.count("progress") == 2  # one per hypothesis
    assert kinds[-1] == "done"
