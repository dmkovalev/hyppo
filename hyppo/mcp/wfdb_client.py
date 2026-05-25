"""Async wfdb helpers for hyppo MCP write actions.

Function bodies raise NotImplementedError until sub-project B
('wfdb-migration') creates the hypothesis_version + hypothesis_run_link
tables. Hyppo-MCP tests `monkeypatch` these symbols with AsyncMock — at
no point does the test suite call the real bodies.

Once sub-project B lands, each body becomes a small async function
calling SQLAlchemy via wfdb.engine.get_engine / get_session_factory.
"""
from __future__ import annotations

import os
from typing import Any


def database_url() -> str | None:
    """Return DATABASE_URL env or None. Bridge clients check this exists."""
    return os.environ.get("DATABASE_URL")


async def insert_hypothesis_version(
    *,
    version_id: str,
    hypothesis_kind: str,
    content_sha256: str,
    snapshot_json: dict[str, Any],
    model_id: str | None,
    supersedes: str | None,
    created_by: str,
) -> None:
    """INSERT row into hypothesis_version. Raise IntegrityError if
    UNIQUE(kind, content_sha256) violated."""
    raise NotImplementedError("blocked on sub-project B (wfdb-migration)")


async def find_latest_active(hypothesis_kind: str) -> str | None:
    """Return version_id of the most recently created row for `kind`,
    or None if no rows exist yet."""
    raise NotImplementedError("blocked on sub-project B (wfdb-migration)")


async def select_version_by_id(version_id: str) -> dict[str, Any] | None:
    """Return one hypothesis_version row as dict, or None if missing."""
    raise NotImplementedError("blocked on sub-project B (wfdb-migration)")


async def select_versions_by_kind(hypothesis_kind: str) -> list[dict[str, Any]]:
    """Return all rows for `kind` ordered by created_at DESC."""
    raise NotImplementedError("blocked on sub-project B (wfdb-migration)")


async def select_superseding_versions(version_id: str) -> list[dict[str, Any]]:
    """Return rows where created_at > version_id.created_at AND
    hypothesis_kind = version_id.hypothesis_kind. These are the
    versions that supersede `version_id`."""
    raise NotImplementedError("blocked on sub-project B (wfdb-migration)")


async def select_runs_for_version(version_id: str) -> list[dict[str, Any]]:
    """Return rows from hypothesis_run_link where version_id=?."""
    raise NotImplementedError("blocked on sub-project B (wfdb-migration)")


async def upsert_run_link(
    *,
    run_id: str,
    hypothesis_kind: str,
    version_id: str,
) -> bool:
    """UPSERT on (run_id, hypothesis_kind). Return True if a row was
    inserted, False if an existing row already matched."""
    raise NotImplementedError("blocked on sub-project B (wfdb-migration)")
