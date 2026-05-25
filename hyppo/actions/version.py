"""Hypothesis-version actions: register, get, list, mark, resolve.

T5 implements `register_hypothesis_version`. T6-T8 append more actions
to this same file. All DB I/O goes through hyppo.mcp.wfdb_client.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from hyppo.actions.registry import action
from hyppo.actions.types import AgentRole, TrustLevel
from hyppo.actions.virtual_experiment import ALL_HYPOTHESIS_KINDS
from hyppo.mcp import wfdb_client

logger = logging.getLogger(__name__)

HypothesisKind = Literal["h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL", "h_WCT"]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """SHA256 over JSON with sort_keys=True so structurally-equal dicts hash equal."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RegisterHypothesisVersionInput(BaseModel):
    model_config = {"frozen": True}

    hypothesis_kind: str = Field(
        description="Must be one of h_CRM, h_ML, h_LPR, h_MB, h_BL, h_WCT.",
    )
    snapshot_json: dict[str, Any] = Field(
        description="Hyperparam dict for this hypothesis (subset of default_space axes).",
    )
    model_id: str | None = Field(
        default=None,
        description="Optional sha256 of the wfopt ModelRecord this version came from.",
    )


class HypothesisVersionRecord(BaseModel):
    model_config = {"frozen": True}

    version_id: str
    hypothesis_kind: str
    content_sha256: str = Field(min_length=64, max_length=64)
    model_id: str | None
    supersedes: str | None
    snapshot_json: dict[str, Any]
    created_at: datetime
    created_by: str


@action(
    kind="RegisterHypothesisVersion",
    trust=TrustLevel.STAGING,
    inputs=RegisterHypothesisVersionInput,
    outputs=HypothesisVersionRecord,
    allowed_roles={AgentRole.Coordinator, AgentRole.ReservoirEngineer},
    requires_audit=True,
)
async def register_hypothesis_version(
    payload: RegisterHypothesisVersionInput,
) -> HypothesisVersionRecord:
    """Insert a new hypothesis_version row; auto-link supersedes."""
    if payload.hypothesis_kind not in ALL_HYPOTHESIS_KINDS:
        raise ValueError(
            f"hypothesis_kind={payload.hypothesis_kind!r} not in "
            f"{ALL_HYPOTHESIS_KINDS}"
        )

    content_sha256 = _canonical_sha256(payload.snapshot_json)
    supersedes = await wfdb_client.find_latest_active(payload.hypothesis_kind)
    version_id = str(uuid4())
    created_at = datetime.utcnow()
    created_by = "hyppo-mcp"

    try:
        await wfdb_client.insert_hypothesis_version(
            version_id=version_id,
            hypothesis_kind=payload.hypothesis_kind,
            content_sha256=content_sha256,
            snapshot_json=payload.snapshot_json,
            model_id=payload.model_id,
            supersedes=supersedes,
            created_by=created_by,
        )
    except IntegrityError as exc:
        raise RuntimeError(
            f"version already registered for kind={payload.hypothesis_kind} "
            f"sha={content_sha256[:16]}..."
        ) from exc

    return HypothesisVersionRecord(
        version_id=version_id,
        hypothesis_kind=payload.hypothesis_kind,
        content_sha256=content_sha256,
        model_id=payload.model_id,
        supersedes=supersedes,
        snapshot_json=payload.snapshot_json,
        created_at=created_at,
        created_by=created_by,
    )
