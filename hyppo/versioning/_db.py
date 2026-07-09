"""Self-contained persistence for hypothesis versioning / run provenance.

Models and the engine are native to hyppo, so the reference library has no
dependency on any external production ORM. Async SQLAlchemy 2.0 over the
``DATABASE_URL`` engine (aiosqlite in tests, Postgres in deployment).
"""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, UniqueConstraint
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class HypothesisVersion(Base):
    """An immutable, content-addressed snapshot of a hypothesis (kind + content
    hash). ``UNIQUE(hypothesis_kind, content_sha256)`` makes re-registering the
    same content idempotent (the INSERT raises IntegrityError)."""

    __tablename__ = "hypothesis_version"
    __table_args__ = (
        UniqueConstraint("hypothesis_kind", "content_sha256", name="uq_kind_sha"),
    )

    version_id: Mapped[str] = mapped_column(String, primary_key=True)
    hypothesis_kind: Mapped[str] = mapped_column(String, index=True)
    content_sha256: Mapped[str] = mapped_column(String)
    snapshot_json: Mapped[dict] = mapped_column(JSON)
    model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    supersedes: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String)


class HypothesisRunLink(Base):
    """Binds a run to the hypothesis version it used, keyed by
    ``(run_id, hypothesis_kind)`` so a run pins exactly one version per kind."""

    __tablename__ = "hypothesis_run_link"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    hypothesis_kind: Mapped[str] = mapped_column(String, primary_key=True)
    version_id: Mapped[str] = mapped_column(String)
    marked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def get_engine(url: str | None = None) -> AsyncEngine:
    """Async engine from ``url`` or ``DATABASE_URL`` (in-memory aiosqlite default)."""
    url = url or os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    return create_async_engine(url, echo=False)


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
