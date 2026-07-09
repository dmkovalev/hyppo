import json

from fastapi import APIRouter, HTTPException, Request

from hyppo.gui.services import build_graph

router = APIRouter(prefix="/api/projects/{pid}/graph", tags=["graph"])


@router.get("")
def graph(pid: str, req: Request) -> dict:
    raw = req.app.state.projects.load_ve(pid)
    if raw is None:
        raise HTTPException(404, "VE not defined")
    return build_graph(json.loads(raw))
