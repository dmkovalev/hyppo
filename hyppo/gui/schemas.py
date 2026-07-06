from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class Project(BaseModel):
    id: str
    name: str
    description: str = ""
