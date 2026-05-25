"""Shared test fixtures for Hyppo test suite.

Forces pytest-asyncio into `auto` mode without modifying pyproject.toml.
Required because hyppo-mcp T5+ introduces async write-action tests that
use bare `async def test_*` (same convention as wfonto).
"""


def pytest_configure(config):
    """Set asyncio_mode='auto' on the live config before pytest-asyncio
    inspects it. pytest-asyncio reads `--asyncio-mode` from options OR
    `asyncio_mode` from ini; we patch the option side since the ini value
    is locked behind pyproject.toml (out of scope for hyppo/actions/ work).
    """
    current = getattr(config.option, "asyncio_mode", None)
    if current in (None, "strict"):
        config.option.asyncio_mode = "auto"
