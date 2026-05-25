"""Shared test fixtures for Hyppo test suite."""
import pytest_asyncio


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
