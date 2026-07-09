"""MCP server factory and transport runners for hyppo."""

from __future__ import annotations

from mcp.server import Server

from hyppo.mcp.resources import register_resources
from hyppo.mcp.tools import register_tools


def create_server() -> Server:
    server = Server("hyppo")
    register_tools(server)
    register_resources(server)
    return server


async def run_stdio() -> None:
    """Run as a stdio MCP server."""
    from mcp.server.stdio import stdio_server

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_http(host: str, port: int) -> None:
    """Run as a streamable HTTP MCP server (sibling to neqsim/wfonto)."""
    from contextlib import asynccontextmanager

    import uvicorn
    from mcp.server.fastmcp.server import StreamableHTTPASGIApp
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Route

    mcp_server = create_server()
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        json_response=True,
        stateless=True,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette):  # type: ignore[type-arg]
        async with session_manager.run():
            yield

    asgi_app = StreamableHTTPASGIApp(session_manager)
    starlette_app = Starlette(
        routes=[Route("/mcp", endpoint=asgi_app, methods=["POST"])],
        lifespan=lifespan,
    )
    uvicorn.run(starlette_app, host=host, port=port, log_level="info")
