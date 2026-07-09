"""Out-of-box versioning scenario: no DATABASE_URL, fresh cwd, no pre-existing
schema. Reproduces the audit's live probe (insert then read in separate
calls used to fail with "no such table")."""

from __future__ import annotations

import pytest

from hyppo.versioning import version_store
from hyppo.versioning._db import get_engine, reset_engines


@pytest.fixture(autouse=True)
async def _reset_engine_cache():
    """Isolate the module-global engine cache across tests in this file."""
    await reset_engines()
    yield
    await reset_engines()


async def test_insert_then_find_latest_out_of_box(monkeypatch, tmp_path):
    """No DATABASE_URL set: default file engine auto-creates schema and a
    second, independent call sees the row inserted by the first."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    await version_store.insert_hypothesis_version(
        version_id="v1",
        hypothesis_kind="h_boot",
        content_sha256="a" * 64,
        snapshot_json={"x": 1},
        model_id=None,
        supersedes=None,
        created_by="test",
    )
    latest = await version_store.find_latest_active("h_boot")
    assert latest == "v1"


async def test_default_db_file_created(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    await version_store.insert_hypothesis_version(
        version_id="v1",
        hypothesis_kind="h_boot",
        content_sha256="b" * 64,
        snapshot_json={},
        model_id=None,
        supersedes=None,
        created_by="test",
    )
    assert (tmp_path / "hyppo_versions.db").exists()


async def test_get_engine_memoizes_per_url(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    engine1 = get_engine()
    engine2 = get_engine()
    assert engine1 is engine2


async def test_get_engine_distinct_per_url(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    engine_a = get_engine("sqlite+aiosqlite:///a.db")
    engine_b = get_engine("sqlite+aiosqlite:///b.db")
    assert engine_a is not engine_b
