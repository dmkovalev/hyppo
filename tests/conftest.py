"""Shared test fixtures for Hyppo test suite."""
import pytest
import pytest_asyncio


# ── owlready2 world isolation via test ordering ─────────────────────────────
# owlready2 stores all OWL individuals in a single global SQLite world.
# Tests that create OWL individuals (test_owl_reasoning, test_oil_adapter)
# must run BEFORE tests that populate the adapter cache
# (test_action_build_virtual_experiment) to avoid world pollution.
#
# On Unix: pytest-forked would give true process isolation.
# On Windows: we reorder so OWL-heavy files run first in a clean world.

_OWL_FIRST_FILES = ("test_owl_reasoning.py", "test_oil_adapter.py")


def pytest_collection_modifyitems(items):
    """Reorder: OWL-heavy tests first, then everything else."""
    owl_items = []
    other_items = []
    for item in items:
        if item.fspath.basename in _OWL_FIRST_FILES:
            owl_items.append(item)
        else:
            other_items.append(item)
    items[:] = owl_items + other_items


# ── wfdb aiosqlite fixture ──────────────────────────────────────────────────

@pytest_asyncio.fixture
async def wfdb_session(monkeypatch, tmp_path):
    """Per-test aiosqlite database with all ORM tables provisioned."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from wfdb.base import Base

    db_path = tmp_path / "hyppo_test.sqlite"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    yield db_url
