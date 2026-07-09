import json

from fastapi import APIRouter, HTTPException, Request

from hyppo.gui.services import plan_preview

router = APIRouter(prefix="/api/projects/{pid}/plan", tags=["plan"])


@router.get("")
def plan(pid: str, req: Request) -> dict:
    raw = req.app.state.projects.load_ve(pid)
    if raw is None:
        raise HTTPException(404, "VE not defined")
    return plan_preview(json.loads(raw), db_path=req.app.state.db_path)
