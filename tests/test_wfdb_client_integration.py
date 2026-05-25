"""Integration tests: wfdb_client against real aiosqlite."""
import asyncio

import pytest
from sqlalchemy.exc import IntegrityError

from hyppo.mcp import wfdb_client


async def test_insert_and_find_latest(wfdb_session):
    await wfdb_client.insert_hypothesis_version(
        version_id="v1", hypothesis_kind="h_CRM",
        content_sha256="a" * 64, snapshot_json={"x": 1},
        model_id=None, supersedes=None, created_by="test",
    )
    latest = await wfdb_client.find_latest_active("h_CRM")
    assert latest == "v1"


async def test_select_version_by_id_hit(wfdb_session):
    await wfdb_client.insert_hypothesis_version(
        version_id="v2", hypothesis_kind="h_ML",
        content_sha256="b" * 64, snapshot_json={"y": 2},
        model_id=None, supersedes=None, created_by="test",
    )
    row = await wfdb_client.select_version_by_id("v2")
    assert row is not None
    assert row["hypothesis_kind"] == "h_ML"
    assert row["snapshot_json"] == {"y": 2}


async def test_select_version_by_id_miss(wfdb_session):
    row = await wfdb_client.select_version_by_id("nonexistent")
    assert row is None


async def test_select_versions_by_kind_ordered(wfdb_session):
    await wfdb_client.insert_hypothesis_version(
        version_id="v1", hypothesis_kind="h_CRM",
        content_sha256="c" * 64, snapshot_json={},
        model_id=None, supersedes=None, created_by="test",
    )
    await asyncio.sleep(0.05)
    await wfdb_client.insert_hypothesis_version(
        version_id="v2", hypothesis_kind="h_CRM",
        content_sha256="d" * 64, snapshot_json={},
        model_id=None, supersedes="v1", created_by="test",
    )
    rows = await wfdb_client.select_versions_by_kind("h_CRM")
    assert [r["version_id"] for r in rows] == ["v2", "v1"]


async def test_select_superseding_versions(wfdb_session):
    await wfdb_client.insert_hypothesis_version(
        version_id="v1", hypothesis_kind="h_CRM",
        content_sha256="e" * 64, snapshot_json={},
        model_id=None, supersedes=None, created_by="test",
    )
    await asyncio.sleep(0.05)
    await wfdb_client.insert_hypothesis_version(
        version_id="v2", hypothesis_kind="h_CRM",
        content_sha256="f" * 64, snapshot_json={},
        model_id=None, supersedes="v1", created_by="test",
    )
    superseding = await wfdb_client.select_superseding_versions("v1")
    assert len(superseding) == 1
    assert superseding[0]["version_id"] == "v2"


async def test_upsert_run_link_insert_then_noop(wfdb_session):
    await wfdb_client.insert_hypothesis_version(
        version_id="v1", hypothesis_kind="h_CRM",
        content_sha256="g" * 64, snapshot_json={},
        model_id=None, supersedes=None, created_by="test",
    )
    inserted = await wfdb_client.upsert_run_link(
        run_id="run-1", hypothesis_kind="h_CRM", version_id="v1",
    )
    assert inserted is True
    inserted2 = await wfdb_client.upsert_run_link(
        run_id="run-1", hypothesis_kind="h_CRM", version_id="v1",
    )
    assert inserted2 is False


async def test_select_runs_for_version(wfdb_session):
    await wfdb_client.insert_hypothesis_version(
        version_id="v1", hypothesis_kind="h_CRM",
        content_sha256="h" * 64, snapshot_json={},
        model_id=None, supersedes=None, created_by="test",
    )
    await wfdb_client.upsert_run_link(
        run_id="run-A", hypothesis_kind="h_CRM", version_id="v1",
    )
    await wfdb_client.upsert_run_link(
        run_id="run-B", hypothesis_kind="h_CRM", version_id="v1",
    )
    runs = await wfdb_client.select_runs_for_version("v1")
    assert {r["run_id"] for r in runs} == {"run-A", "run-B"}


async def test_duplicate_kind_sha_raises(wfdb_session):
    await wfdb_client.insert_hypothesis_version(
        version_id="v1", hypothesis_kind="h_CRM",
        content_sha256="i" * 64, snapshot_json={},
        model_id=None, supersedes=None, created_by="test",
    )
    with pytest.raises(IntegrityError):
        await wfdb_client.insert_hypothesis_version(
            version_id="v2", hypothesis_kind="h_CRM",
            content_sha256="i" * 64, snapshot_json={},
            model_id=None, supersedes=None, created_by="test",
        )
