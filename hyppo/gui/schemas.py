from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class Project(BaseModel):
    id: str
    name: str
    description: str = ""


class Hypothesis(BaseModel):
    id: str
    params: dict[str, list[str]] = {}


class VEDefinition(BaseModel):
    hypotheses: list[Hypothesis]
    workflow_edges: list[list[str]] = []


class VEView(VEDefinition):
    config_space_size: int
