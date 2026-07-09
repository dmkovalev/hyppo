import json
from math import prod

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/projects/{pid}/ve", tags=["ve"])


def _config_space_size(ve: dict) -> int:
    sizes = [
        max(1, len(v))
        for h in ve.get("hypotheses", [])
        for v in h.get("params", {}).values()
    ]
    return prod(sizes) if sizes else 0


@router.get("")
def get_ve(pid: str, req: Request) -> dict:
    """Return the full virtual-experiment tuple <O, H, M, R, W, C> plus the
    computed configuration-space size. Serves the VE inspector screen."""
    raw = req.app.state.projects.load_ve(pid)
    if raw is None:
        raise HTTPException(404, "VE not defined")
    ve = json.loads(raw)
    ve["config_space_size"] = _config_space_size(ve)
    return ve
