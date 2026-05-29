"""Tests for hyppo.actions.version.get_hypothesis_version."""
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from hyppo.actions.version import (
    GetHypothesisVersionInput,
    HypothesisVersionRecord,
    get_hypothesis_version,
)


@pytest.fixture
def mock_row():
    return {
        "version_id": "abc-uuid",
        "hypothesis_kind": "h_CRM",
        "content_sha256": "0" * 64,
        "model_id": None,
        "supersedes": None,
        "snapshot_json": {"USE_DUAL_TAU_CRM": True},
        "created_at": datetime(2026, 5, 25, 12, 0, 0),
        "created_by": "hyppo-mcp",
    }


async def test_get_hit_returns_record(monkeypatch, mock_row):
    from hyppo.mcp import version_store
    monkeypatch.setattr(
        version_store, "select_version_by_id", AsyncMock(return_value=mock_row),
    )
    out: HypothesisVersionRecord = await get_hypothesis_version(
        GetHypothesisVersionInput(version_id="abc-uuid")
    )
    assert out.version_id == "abc-uuid"
    assert out.hypothesis_kind == "h_CRM"


async def test_get_miss_raises(monkeypatch):
    from hyppo.mcp import version_store
    monkeypatch.setattr(
        version_store, "select_version_by_id", AsyncMock(return_value=None),
    )
    with pytest.raises(RuntimeError, match="not found"):
        await get_hypothesis_version(GetHypothesisVersionInput(version_id="zzz"))
