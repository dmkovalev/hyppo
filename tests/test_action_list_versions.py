"""Tests for hyppo.actions.version.list_versions_for_hypothesis."""
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from hyppo.actions.version import (
    HypothesisVersionList,
    ListVersionsForHypothesisInput,
    list_versions_for_hypothesis,
)


def _row(version_id: str, created_at: datetime, kind: str = "h_CRM") -> dict:
    return {
        "version_id": version_id,
        "hypothesis_kind": kind,
        "content_sha256": "0" * 64,
        "model_id": None,
        "supersedes": None,
        "snapshot_json": {},
        "created_at": created_at,
        "created_by": "hyppo-mcp",
    }


async def test_list_returns_descending_order(monkeypatch):
    rows = [
        _row("v3", datetime(2026, 5, 25, 12, 0, 0)),
        _row("v2", datetime(2026, 5, 20, 12, 0, 0)),
        _row("v1", datetime(2026, 5, 10, 12, 0, 0)),
    ]
    from hyppo.mcp import version_store
    monkeypatch.setattr(
        version_store, "select_versions_by_kind", AsyncMock(return_value=rows),
    )
    out: HypothesisVersionList = await list_versions_for_hypothesis(
        ListVersionsForHypothesisInput(hypothesis_kind="h_CRM")
    )
    ids = [r.version_id for r in out.records]
    assert ids == ["v3", "v2", "v1"]


async def test_list_empty_is_ok(monkeypatch):
    from hyppo.mcp import version_store
    monkeypatch.setattr(
        version_store, "select_versions_by_kind", AsyncMock(return_value=[]),
    )
    out = await list_versions_for_hypothesis(
        ListVersionsForHypothesisInput(hypothesis_kind="h_ML")
    )
    assert out.records == []


async def test_list_rejects_unknown_kind():
    with pytest.raises(ValueError, match="h_BOGUS"):
        await list_versions_for_hypothesis(
            ListVersionsForHypothesisInput(hypothesis_kind="h_BOGUS")
        )
