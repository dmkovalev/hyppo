"""Tests for hyppo.actions.version.resolve_stale_runs."""
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from hyppo.actions.version import (
    ResolveStaleRunsInput,
    ResolveStaleRunsOutput,
    RunRef,
    resolve_stale_runs,
)


async def test_runs_pinned_to_superseded_version_are_returned(monkeypatch):
    from hyppo.mcp import version_store

    monkeypatch.setattr(
        version_store, "select_version_by_id",
        AsyncMock(return_value={
            "version_id": "v1",
            "hypothesis_kind": "h_CRM",
            "content_sha256": "0" * 64,
            "model_id": None,
            "supersedes": None,
            "snapshot_json": {},
            "created_at": datetime(2026, 1, 1),
            "created_by": "hyppo-mcp",
        }),
    )
    monkeypatch.setattr(
        version_store, "select_superseding_versions",
        AsyncMock(return_value=[
            {"version_id": "v2", "created_at": datetime(2026, 2, 1)},
            {"version_id": "v3", "created_at": datetime(2026, 3, 1)},
        ]),
    )
    monkeypatch.setattr(
        version_store, "select_runs_for_version",
        AsyncMock(return_value=[
            {"run_id": "run-A", "hypothesis_kind": "h_CRM", "version_id": "v1"},
            {"run_id": "run-B", "hypothesis_kind": "h_CRM", "version_id": "v1"},
        ]),
    )

    out: ResolveStaleRunsOutput = await resolve_stale_runs(
        ResolveStaleRunsInput(version_id="v1")
    )
    run_ids = {r.run_id for r in out.runs}
    assert run_ids == {"run-A", "run-B"}


async def test_no_superseding_versions_returns_empty(monkeypatch):
    from hyppo.mcp import version_store
    monkeypatch.setattr(
        version_store, "select_version_by_id",
        AsyncMock(return_value={"version_id": "vlatest",
                                 "hypothesis_kind": "h_CRM",
                                 "content_sha256": "0"*64,
                                 "model_id": None, "supersedes": None,
                                 "snapshot_json": {},
                                 "created_at": datetime(2026, 4, 1),
                                 "created_by": "hyppo-mcp"}),
    )
    monkeypatch.setattr(
        version_store, "select_superseding_versions", AsyncMock(return_value=[]),
    )
    # If nothing supersedes this version, no runs are stale relative to it,
    # so we should not even consult select_runs_for_version.
    runs_for_version = AsyncMock(return_value=[])
    monkeypatch.setattr(version_store, "select_runs_for_version", runs_for_version)

    out = await resolve_stale_runs(ResolveStaleRunsInput(version_id="vlatest"))
    assert out.runs == []
    runs_for_version.assert_not_awaited()


async def test_unknown_version_raises(monkeypatch):
    from hyppo.mcp import version_store
    monkeypatch.setattr(
        version_store, "select_version_by_id", AsyncMock(return_value=None),
    )
    with pytest.raises(RuntimeError, match="not found"):
        await resolve_stale_runs(ResolveStaleRunsInput(version_id="zzz"))
