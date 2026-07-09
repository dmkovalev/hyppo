"""Tests for hyppo.actions.version.register_hypothesis_version."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from hyppo.actions.version import (
    HypothesisVersionRecord,
    RegisterHypothesisVersionInput,
    register_hypothesis_version,
)


@pytest.fixture
def mock_store(monkeypatch):
    from hyppo.versioning import version_store
    monkeypatch.setattr(version_store, "insert_hypothesis_version", AsyncMock())
    monkeypatch.setattr(version_store, "find_latest_active", AsyncMock(return_value=None))
    return version_store


async def test_register_happy_path_no_prior(mock_store):
    out: HypothesisVersionRecord = await register_hypothesis_version(
        RegisterHypothesisVersionInput(
            hypothesis_kind="h_CRM",
            snapshot_json={"USE_DUAL_TAU_CRM": True, "CRM_KNN": 5},
            model_id="sha256:abc",
        )
    )
    assert out.hypothesis_kind == "h_CRM"
    assert out.supersedes is None
    assert len(out.content_sha256) == 64
    assert out.version_id  # non-empty UUID
    mock_store.insert_hypothesis_version.assert_awaited_once()
    call_kwargs = mock_store.insert_hypothesis_version.await_args.kwargs
    assert call_kwargs["hypothesis_kind"] == "h_CRM"
    assert call_kwargs["model_id"] == "sha256:abc"
    assert call_kwargs["supersedes"] is None


async def test_register_supersedes_prior_version(monkeypatch):
    from hyppo.versioning import version_store
    monkeypatch.setattr(version_store, "insert_hypothesis_version", AsyncMock())
    monkeypatch.setattr(
        version_store, "find_latest_active",
        AsyncMock(return_value="prior-uuid-xxxx"),
    )
    out = await register_hypothesis_version(
        RegisterHypothesisVersionInput(
            hypothesis_kind="h_ML",
            snapshot_json={"HIDDEN_DIM": 64},
            model_id=None,
        )
    )
    assert out.supersedes == "prior-uuid-xxxx"


async def test_register_duplicate_raises(monkeypatch):
    from sqlalchemy.exc import IntegrityError
    from hyppo.versioning import version_store

    monkeypatch.setattr(
        version_store, "insert_hypothesis_version",
        AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("dup"))),
    )
    monkeypatch.setattr(version_store, "find_latest_active", AsyncMock(return_value=None))
    with pytest.raises(RuntimeError, match="already registered"):
        await register_hypothesis_version(
            RegisterHypothesisVersionInput(
                hypothesis_kind="h_BL",
                snapshot_json={"BACKPERIOD": 24},
                model_id=None,
            )
        )


async def test_register_rejects_unknown_kind(monkeypatch):
    with pytest.raises(ValueError, match="h_BOGUS"):
        await register_hypothesis_version(
            RegisterHypothesisVersionInput(
                hypothesis_kind="h_BOGUS",
                snapshot_json={},
                model_id=None,
            )
        )


def test_content_sha_is_deterministic():
    """Same snapshot dict → same sha across runs (canonicalisation is stable)."""
    from hyppo.actions.version import _canonical_sha256
    a = _canonical_sha256({"x": 1, "y": [2, 3], "z": {"k": "v"}})
    b = _canonical_sha256({"z": {"k": "v"}, "y": [2, 3], "x": 1})
    assert a == b, "key order must not affect the hash"
