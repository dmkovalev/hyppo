import json
from fastapi import APIRouter, HTTPException, Request
from hyppo.gui.services import run_iteration, iteration_history

router = APIRouter(prefix="/api/projects/{pid}/runs", tags=["runs"])


@router.post("", status_code=201)
def create_run(pid: str, req: Request) -> dict:
    store = req.app.state.projects
    raw = store.load_ve(pid)
    if raw is None:
        raise HTTPException(404, "VE not defined")
    outcome = run_iteration(json.loads(raw), db_path=req.app.state.db_path)
    it = store.add_iteration(pid, json.dumps(outcome))
    record = {"iteration": it, **outcome}
    return record


@router.get("")
def history(pid: str, req: Request) -> list[dict]:
    store = req.app.state.projects
    hist = iteration_history(store, pid)
    # attach descending iteration numbers
    total = len(hist)
    return [{"iteration": total - i, **h} for i, h in enumerate(hist)]
