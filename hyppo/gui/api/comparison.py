# hyppo/gui/api/comparison.py
from fastapi import APIRouter, HTTPException, Request
from hyppo.gui.services import iteration_history

router = APIRouter(prefix="/api/projects/{pid}/comparison", tags=["comparison"])


@router.get("")
def comparison(pid: str, req: Request) -> dict:
    hist = iteration_history(req.app.state.projects, pid)
    if not hist:
        raise HTTPException(404, "no iterations yet")
    latest = hist[0]
    rows = [
        {"hypothesis": hid,
         "r2": res.get("metrics", {}).get("r2"),
         "status": res.get("status")}
        for hid, res in latest["results"].items()
    ]
    return {"rows": rows}
