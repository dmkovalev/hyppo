"""Smoke tests for hyppo.mcp server factory + tool wiring."""
import pytest


EXPECTED_TOOLS = {
    "BuildVirtualExperiment",
    "GetHypothesisLattice",
    "DiffHypothesisStates",
    "RegisterHypothesisVersion",
    "GetHypothesisVersion",
    "ListVersionsForHypothesis",
    "ResolveStaleRuns",
    "MarkRunWithVersion",
}


def test_hyppo_mcp_package_importable():
    import hyppo.mcp  # noqa: F401


def test_mcp_sdk_is_available():
    import mcp.server  # noqa: F401


def test_create_server_named_hyppo():
    from hyppo.mcp.server import create_server
    server = create_server()
    assert getattr(server, "name", None) == "hyppo"


async def test_list_tools_returns_all_eight():
    from hyppo.mcp.server import create_server
    from hyppo.mcp.tools import register_tools

    server = create_server()
    handlers = register_tools(server)
    tools = await handlers["tools/list"]()
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS, f"missing/extra: {names ^ EXPECTED_TOOLS}"


async def test_tools_have_trust_role_annotation():
    from hyppo.mcp.server import create_server
    from hyppo.mcp.tools import register_tools

    handlers = register_tools(create_server())
    tools = await handlers["tools/list"]()
    for t in tools:
        assert t.description.startswith("[trust="), (
            f"Tool {t.name!r} description missing [trust=...] prefix: "
            f"{t.description[:80]!r}"
        )
