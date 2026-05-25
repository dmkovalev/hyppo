"""Smoke tests for hyppo.mcp server factory + tool wiring."""
import importlib

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


@pytest.fixture(autouse=True)
def _repopulate_action_registry():
    """Defend against test-suite-wide registry pollution from
    `test_action_registry.py` (which calls clear_registry() then registers
    OrderA/OrderB fixtures). Reloads the action side-effect modules so
    ACTION_REGISTRY contains the 8 production tools regardless of ordering."""
    from hyppo.actions.registry import ACTION_REGISTRY, clear_registry

    clear_registry()
    import hyppo.actions.diff
    import hyppo.actions.version
    import hyppo.actions.virtual_experiment

    importlib.reload(hyppo.actions.diff)
    importlib.reload(hyppo.actions.version)
    importlib.reload(hyppo.actions.virtual_experiment)
    yield


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
