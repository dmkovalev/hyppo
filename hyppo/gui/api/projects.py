from fastapi import APIRouter, HTTPException, Request, Response
from hyppo.gui.schemas import Project, ProjectCreate

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _store(req: Request):
    return req.app.state.projects


@router.post("", status_code=201, response_model=Project)
def create(body: ProjectCreate, req: Request) -> Project:
    pid = _store(req).create(body.name, body.description)
    return Project(id=pid, name=body.name, description=body.description)


@router.get("", response_model=list[Project])
def list_projects(req: Request) -> list[Project]:
    return [Project(**p) for p in _store(req).list()]


@router.get("/{pid}", response_model=Project)
def get(pid: str, req: Request) -> Project:
    p = _store(req).get(pid)
    if p is None:
        raise HTTPException(404, "project not found")
    return Project(**p)


@router.delete("/{pid}", status_code=204)
def delete(pid: str, req: Request) -> Response:
    if _store(req).get(pid) is None:
        raise HTTPException(404, "project not found")
    _store(req).delete(pid)
    return Response(status_code=204)
