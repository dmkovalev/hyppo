"""Async wfdb helpers for hyppo MCP write actions.

Real implementations using wfdb.get_engine / get_session_factory.
Each function opens an async session, performs the query, and commits
where needed.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from wfdb import get_engine, get_session_factory
from wfdb.models.hypothesis_version import HypothesisVersion
from wfdb.models.hypothesis_run_link import HypothesisRunLink


def database_url() -> str | None:
    """Return DATABASE_URL env or None. Bridge clients check this exists."""
    return os.environ.get("DATABASE_URL")


async def _get_session():
    engine = get_engine()
    factory = get_session_factory(engine)
    return factory()


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
    async with await _get_session() as session:
        row = HypothesisVersion(
            version_id=version_id,
            hypothesis_kind=hypothesis_kind,
            content_sha256=content_sha256,
            snapshot_json=snapshot_json,
            model_id=model_id,
            supersedes=supersedes,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
        )
        session.add(row)
        await session.commit()


async def find_latest_active(hypothesis_kind: str) -> str | None:
    """Return version_id of the most recently created row for `kind`,
    or None if no rows exist yet."""
    async with await _get_session() as session:
        stmt = (
            select(HypothesisVersion.version_id)
            .where(HypothesisVersion.hypothesis_kind == hypothesis_kind)
            .order_by(HypothesisVersion.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def select_version_by_id(version_id: str) -> dict[str, Any] | None:
    """Return one hypothesis_version row as dict, or None if missing."""
    async with await _get_session() as session:
        row = await session.get(HypothesisVersion, version_id)
        if row is None:
            return None
        return {
            "version_id": row.version_id,
            "hypothesis_kind": row.hypothesis_kind,
            "content_sha256": row.content_sha256,
            "model_id": row.model_id,
            "supersedes": row.supersedes,
            "snapshot_json": row.snapshot_json,
            "created_at": row.created_at,
            "created_by": row.created_by,
        }


async def select_versions_by_kind(hypothesis_kind: str) -> list[dict[str, Any]]:
    """Return all rows for `kind` ordered by created_at DESC."""
    async with await _get_session() as session:
        stmt = (
            select(HypothesisVersion)
            .where(HypothesisVersion.hypothesis_kind == hypothesis_kind)
            .order_by(HypothesisVersion.created_at.desc())
        )
        result = await session.execute(stmt)
        return [
            {
                "version_id": r.version_id,
                "hypothesis_kind": r.hypothesis_kind,
                "content_sha256": r.content_sha256,
                "model_id": r.model_id,
                "supersedes": r.supersedes,
                "snapshot_json": r.snapshot_json,
                "created_at": r.created_at,
                "created_by": r.created_by,
            }
            for r in result.scalars().all()
        ]


async def select_superseding_versions(version_id: str) -> list[dict[str, Any]]:
    """Return rows where created_at > version_id.created_at AND
    hypothesis_kind = version_id.hypothesis_kind."""
    async with await _get_session() as session:
        base = await session.get(HypothesisVersion, version_id)
        if base is None:
            return []
        stmt = (
            select(HypothesisVersion)
            .where(
                HypothesisVersion.hypothesis_kind == base.hypothesis_kind,
                HypothesisVersion.created_at > base.created_at,
            )
            .order_by(HypothesisVersion.created_at.asc())
        )
        result = await session.execute(stmt)
        return [
            {"version_id": r.version_id, "created_at": r.created_at}
            for r in result.scalars().all()
        ]


async def select_runs_for_version(version_id: str) -> list[dict[str, Any]]:
    """Return rows from hypothesis_run_link where version_id=?."""
    async with await _get_session() as session:
        stmt = select(HypothesisRunLink).where(
            HypothesisRunLink.version_id == version_id,
        )
        result = await session.execute(stmt)
        return [
            {
                "run_id": r.run_id,
                "hypothesis_kind": r.hypothesis_kind,
                "version_id": r.version_id,
            }
            for r in result.scalars().all()
        ]


async def upsert_run_link(
    *,
    run_id: str,
    hypothesis_kind: str,
    version_id: str,
) -> bool:
    """UPSERT on (run_id, hypothesis_kind). Return True if a row was
    inserted, False if an existing row already matched."""
    async with await _get_session() as session:
        existing = await session.get(
            HypothesisRunLink, (run_id, hypothesis_kind),
        )
        if existing is not None:
            if existing.version_id == version_id:
                return False
            existing.version_id = version_id
            existing.marked_at = datetime.now(timezone.utc)
            await session.commit()
            return False
        link = HypothesisRunLink(
            run_id=run_id,
            hypothesis_kind=hypothesis_kind,
            version_id=version_id,
            marked_at=datetime.now(timezone.utc),
        )
        session.add(link)
        await session.commit()
        return True
