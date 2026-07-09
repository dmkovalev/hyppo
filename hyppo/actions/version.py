"""Hypothesis-version actions: register, get, list, mark, resolve.

T5 implements `register_hypothesis_version`. T6-T8 append more actions
to this same file. All DB I/O goes through hyppo.versioning.version_store.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from hyppo.actions.registry import action
from hyppo.actions.types import AgentRole, TrustLevel
from hyppo.actions.virtual_experiment import ALL_HYPOTHESIS_KINDS
from hyppo.versioning import version_store

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
        description="Hyperparam dict for this hypothesis (subset of default_space "
        "axes)."
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
            f"hypothesis_kind={payload.hypothesis_kind!r} not in {ALL_HYPOTHESIS_KINDS}"
        )

    content_sha256 = _canonical_sha256(payload.snapshot_json)
    supersedes = await version_store.find_latest_active(payload.hypothesis_kind)
    version_id = str(uuid4())
    created_at = datetime.now(UTC)
    created_by = "hyppo-mcp"

    try:
        await version_store.insert_hypothesis_version(
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


class GetHypothesisVersionInput(BaseModel):
    model_config = {"frozen": True}

    version_id: str


@action(
    kind="GetHypothesisVersion",
    trust=TrustLevel.SAFE,
    inputs=GetHypothesisVersionInput,
    outputs=HypothesisVersionRecord,
    allowed_roles={
        AgentRole.Coordinator,
        AgentRole.ReservoirEngineer,
        AgentRole.Auditor,
        AgentRole.Geologist,
        AgentRole.ProductionEngineer,
        AgentRole.Economist,
    },
)
async def get_hypothesis_version(
    payload: GetHypothesisVersionInput,
) -> HypothesisVersionRecord:
    """Read one hypothesis_version row by version_id."""
    row = await version_store.select_version_by_id(payload.version_id)
    if row is None:
        raise RuntimeError(f"version not found: {payload.version_id!r}")
    return HypothesisVersionRecord(**row)


class ListVersionsForHypothesisInput(BaseModel):
    model_config = {"frozen": True}

    hypothesis_kind: str


class HypothesisVersionList(BaseModel):
    model_config = {"frozen": True}

    records: list[HypothesisVersionRecord]


@action(
    kind="ListVersionsForHypothesis",
    trust=TrustLevel.SAFE,
    inputs=ListVersionsForHypothesisInput,
    outputs=HypothesisVersionList,
    allowed_roles={
        AgentRole.Coordinator,
        AgentRole.ReservoirEngineer,
        AgentRole.Auditor,
        AgentRole.Geologist,
        AgentRole.ProductionEngineer,
        AgentRole.Economist,
    },
)
async def list_versions_for_hypothesis(
    payload: ListVersionsForHypothesisInput,
) -> HypothesisVersionList:
    """Return all hypothesis_version rows for `hypothesis_kind`, newest first."""
    if payload.hypothesis_kind not in ALL_HYPOTHESIS_KINDS:
        raise ValueError(
            f"hypothesis_kind={payload.hypothesis_kind!r} not in {ALL_HYPOTHESIS_KINDS}"
        )
    rows = await version_store.select_versions_by_kind(payload.hypothesis_kind)
    return HypothesisVersionList(records=[HypothesisVersionRecord(**r) for r in rows])


class MarkRunWithVersionInput(BaseModel):
    model_config = {"frozen": True}

    run_id: str = Field(min_length=1)
    version_ids: dict[str, str] = Field(
        description="hypothesis_kind -> version_id. Must be non-empty.",
    )


class MarkRunWithVersionOutput(BaseModel):
    model_config = {"frozen": True}

    run_id: str
    n_links_written: int = Field(ge=0)


@action(
    kind="MarkRunWithVersion",
    trust=TrustLevel.STAGING,
    inputs=MarkRunWithVersionInput,
    outputs=MarkRunWithVersionOutput,
    allowed_roles={AgentRole.Coordinator, AgentRole.ReservoirEngineer},
    requires_audit=True,
)
async def mark_run_with_version(
    payload: MarkRunWithVersionInput,
) -> MarkRunWithVersionOutput:
    """Pin a run to its hypothesis-version set (idempotent UPSERT per kind)."""
    if not payload.version_ids:
        raise ValueError("version_ids is empty — supply at least one kind")
    bad = [k for k in payload.version_ids if k not in ALL_HYPOTHESIS_KINDS]
    if bad:
        raise ValueError(f"unknown hypothesis_kinds in version_ids: {bad}")

    n_written = 0
    for kind, version_id in payload.version_ids.items():
        inserted = await version_store.upsert_run_link(
            run_id=payload.run_id,
            hypothesis_kind=kind,
            version_id=version_id,
        )
        if inserted:
            n_written += 1

    return MarkRunWithVersionOutput(
        run_id=payload.run_id,
        n_links_written=n_written,
    )


class RunRef(BaseModel):
    model_config = {"frozen": True}

    run_id: str
    hypothesis_kind: str
    version_id: str


class ResolveStaleRunsInput(BaseModel):
    model_config = {"frozen": True}

    version_id: str


class ResolveStaleRunsOutput(BaseModel):
    model_config = {"frozen": True}

    runs: list[RunRef]


@action(
    kind="ResolveStaleRuns",
    trust=TrustLevel.SAFE,
    inputs=ResolveStaleRunsInput,
    outputs=ResolveStaleRunsOutput,
    allowed_roles={
        AgentRole.Coordinator,
        AgentRole.ReservoirEngineer,
        AgentRole.Auditor,
        AgentRole.Geologist,
        AgentRole.ProductionEngineer,
        AgentRole.Economist,
    },
)
async def resolve_stale_runs(payload: ResolveStaleRunsInput) -> ResolveStaleRunsOutput:
    """Find runs pinned to `version_id` after a newer version of the same
    hypothesis_kind has been registered."""
    version_row = await version_store.select_version_by_id(payload.version_id)
    if version_row is None:
        raise RuntimeError(f"version not found: {payload.version_id!r}")

    superseding = await version_store.select_superseding_versions(payload.version_id)
    if not superseding:
        return ResolveStaleRunsOutput(runs=[])

    rows = await version_store.select_runs_for_version(payload.version_id)
    return ResolveStaleRunsOutput(
        runs=[RunRef(**r) for r in rows],
    )
