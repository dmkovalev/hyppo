"""Expose hyppo persona prompts as MCP resources."""

from __future__ import annotations

from importlib import resources as imres

from mcp.server import Server
from mcp.types import Resource

_FILE_RESOURCES: list[tuple[str, str, str, str]] = [
    (
        "hyppo://personas/lattice_steward.md",
        "Lattice Steward agent persona prompt",
        "text/markdown",
        "hyppo.personas/lattice_steward.md",
    ),
]


def _load_text(pkg_path: str) -> str:
    pkg, _, name = pkg_path.partition("/")
    return imres.files(pkg).joinpath(name).read_text(encoding="utf-8")


def register_resources(server: Server) -> dict:
    @server.list_resources()
    async def _list() -> list[Resource]:
        return [
            Resource(uri=uri, name=name, mimeType=mime, description=name)
            for (uri, name, mime, _path) in _FILE_RESOURCES
        ]

    @server.read_resource()
    async def _read(uri: str) -> str:
        uri = str(uri)
        for u, _n, _m, path in _FILE_RESOURCES:
            if u == uri:
                return _load_text(path)
        raise ValueError(f"unknown resource: {uri!r}")

    return {"resources/list": _list, "resources/read": _read}
