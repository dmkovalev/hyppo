"""Unit tests for hyppo.mcp.resources list/read."""
import pytest

from hyppo.mcp.server import create_server
from hyppo.mcp.resources import register_resources


@pytest.fixture
def handlers():
    return register_resources(create_server())


async def test_list_resources_includes_lattice_steward(handlers):
    resources = await handlers["resources/list"]()
    uris = {str(r.uri) for r in resources}
    assert "hyppo://personas/lattice_steward.md" in uris


async def test_read_persona_returns_markdown(handlers):
    text = await handlers["resources/read"]("hyppo://personas/lattice_steward.md")
    assert text.strip().startswith("# Lattice Steward")
    assert "BuildVirtualExperiment" in text


async def test_read_unknown_uri_raises(handlers):
    with pytest.raises(ValueError, match="unknown resource"):
        await handlers["resources/read"]("hyppo://bogus")
