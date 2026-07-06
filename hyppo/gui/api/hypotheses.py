import json
from math import prod
from fastapi import APIRouter, HTTPException, Request
from hyppo.gui.schemas import VEDefinition, VEView

router = APIRouter(prefix="/api/projects/{pid}/hypotheses", tags=["hypotheses"])


def _config_space_size(ve: VEDefinition) -> int:
    sizes = [max(1, len(v)) for h in ve.hypotheses for v in h.params.values()]
    return prod(sizes) if sizes else 0


@router.put("", response_model=VEView)
def define(pid: str, body: VEDefinition, req: Request) -> VEView:
    store = req.app.state.projects
    if store.get(pid) is None:
        raise HTTPException(404, "project not found")
    store.save_ve(pid, body.model_dump_json())
    return VEView(**body.model_dump(), config_space_size=_config_space_size(body))


@router.get("", response_model=VEView)
def read(pid: str, req: Request) -> VEView:
    store = req.app.state.projects
    raw = store.load_ve(pid)
    if raw is None:
        raise HTTPException(404, "VE not defined")
    ve = VEDefinition(**json.loads(raw))
    return VEView(**ve.model_dump(), config_space_size=_config_space_size(ve))
