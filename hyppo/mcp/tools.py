"""Bridge hyppo.actions.ACTION_REGISTRY to MCP tool primitives."""

from __future__ import annotations

import inspect
import json
import logging

from mcp.server import Server
from mcp.types import TextContent, Tool

import hyppo.actions  # noqa: F401 — eager-imports populate ACTION_REGISTRY
from hyppo.actions.registry import ACTION_REGISTRY, ActionSpec

audit_log = logging.getLogger("hyppo.mcp.audit")


def _action_to_mcp_tool(spec: ActionSpec) -> Tool:
    role_str = "|".join(sorted(r.value for r in spec.allowed_roles))
    annotation = f"[trust={spec.trust.value} roles={role_str}] "
    return Tool(
        name=spec.kind,
        description=annotation + (spec.docstring or f"Action {spec.kind}"),
        inputSchema=spec.inputs.model_json_schema(),
    )


def register_tools(server: Server) -> dict:
    @server.list_tools()
    async def _list() -> list[Tool]:
        return [_action_to_mcp_tool(s) for s in ACTION_REGISTRY.values()]

    @server.call_tool()
    async def _call(name: str, arguments: dict) -> list[TextContent]:
        spec = ACTION_REGISTRY.get(name)
        if spec is None:
            raise ValueError(f"unknown tool: {name!r}")
        payload = spec.inputs.model_validate(arguments)
        if spec.requires_audit:
            audit_log.info(
                "mcp_tool_call kind=%s trust=%s args=%s",
                name,
                spec.trust.value,
                str(payload.model_dump())[:200],
            )
        try:
            result = spec.fn(payload)
            if inspect.isawaitable(result):
                result = await result
        except NotImplementedError as exc:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "not_implemented",
                            "kind": name,
                            "detail": str(exc),
                        }
                    ),
                )
            ]
        return [TextContent(type="text", text=result.model_dump_json(indent=2))]

    return {"tools/list": _list, "tools/call": _call}
