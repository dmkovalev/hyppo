"""Hypothesis-version persistence: content-addressed snapshots and run provenance.

Owns ``version_store`` (async query functions) and ``_db`` (SQLAlchemy models
and engine). Kept independent of ``hyppo.mcp`` / ``hyppo.actions`` to avoid an
import cycle: ``hyppo.actions`` depends on this package, not the other way
around.
"""
from __future__ import annotations

from hyppo.versioning import _db, version_store

__all__ = ["_db", "version_store"]
