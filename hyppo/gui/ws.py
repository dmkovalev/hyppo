import json

from fastapi import APIRouter, WebSocket

from hyppo.gui.services import run_iteration

router = APIRouter()


@router.websocket("/api/projects/{pid}/runs/ws")
async def run_ws(websocket: WebSocket, pid: str) -> None:
    await websocket.accept()
    store = websocket.app.state.projects
    raw = store.load_ve(pid)
    if raw is None:
        await websocket.send_json({"event": "error", "detail": "VE not defined"})
        await websocket.close()
        return
    ve = json.loads(raw)
    for h in ve["hypotheses"]:
        await websocket.send_json({"event": "progress", "hypothesis": h["id"]})
    outcome = run_iteration(ve, db_path=websocket.app.state.db_path)
    it = store.add_iteration(pid, json.dumps(outcome))
    await websocket.send_json({"event": "done", "iteration": it, **outcome})
    await websocket.close()
