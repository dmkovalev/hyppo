"""Tests for hyppo.actions.version.mark_run_with_version."""
from unittest.mock import AsyncMock

import pytest

from hyppo.actions.version import (
    MarkRunWithVersionInput,
    MarkRunWithVersionOutput,
    mark_run_with_version,
)


async def test_mark_writes_one_row_per_kind(monkeypatch):
    from hyppo.versioning import version_store
    upsert = AsyncMock(return_value=True)
    monkeypatch.setattr(version_store, "upsert_run_link", upsert)
    out: MarkRunWithVersionOutput = await mark_run_with_version(
        MarkRunWithVersionInput(
            run_id="run-xxx",
            version_ids={"h_CRM": "v1", "h_ML": "v2"},
        )
    )
    assert out.run_id == "run-xxx"
    assert out.n_links_written == 2
    assert upsert.await_count == 2


async def test_mark_idempotent_returns_zero(monkeypatch):
    """Second call with identical inputs upserts no new rows."""
    from hyppo.versioning import version_store
    monkeypatch.setattr(version_store, "upsert_run_link", AsyncMock(return_value=False))
    out = await mark_run_with_version(
        MarkRunWithVersionInput(
            run_id="run-xxx",
            version_ids={"h_CRM": "v1", "h_ML": "v2"},
        )
    )
    assert out.n_links_written == 0


async def test_mark_rejects_empty_version_ids():
    with pytest.raises(ValueError, match="empty"):
        await mark_run_with_version(
            MarkRunWithVersionInput(run_id="r", version_ids={})
        )


async def test_mark_rejects_unknown_kind():
    with pytest.raises(ValueError, match="h_BOGUS"):
        await mark_run_with_version(
            MarkRunWithVersionInput(
                run_id="r", version_ids={"h_BOGUS": "v1"},
            )
        )
