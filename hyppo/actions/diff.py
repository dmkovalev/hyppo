"""DiffHypothesisStates action + derived_by_closure helper.

The closure helper is exported (not private) because Task 8
(ResolveStaleRuns) reuses it.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from hyppo.actions.registry import action
from hyppo.actions.types import AgentRole, TrustLevel
from hyppo.actions.virtual_experiment import ALL_HYPOTHESIS_KINDS


class HypothesisSnapshot(BaseModel):
    """One side of a diff: which hypotheses are active + their hyperparams."""
    active_hypotheses: list[str] = Field(
        description="Subset of canonical kinds present in this snapshot.",
    )
    hyperparams: dict[str, dict[str, Any]] = Field(
        description="kind -> {axis_name: value}. Axes outside default_space "
                    "are accepted but flagged in tests.",
    )


class DiffHypothesisStatesInput(BaseModel):
    snapshot_a: HypothesisSnapshot
    snapshot_b: HypothesisSnapshot
    base_snapshot: HypothesisSnapshot | None = Field(
        default=None,
        description="If supplied, lattice edges come from this snapshot. "
                    "Otherwise the default oil_waterflood lattice is used.",
    )


class HypothesisDiff(BaseModel):
    changed_hypotheses: list[str]
    hyperparam_diff: dict[str, dict[str, list]]  # kind -> axis -> [val_a, val_b]
    stale_cascade: list[str]


def _validate_kinds(snap: HypothesisSnapshot, label: str) -> None:
    bad = [k for k in snap.active_hypotheses if k not in ALL_HYPOTHESIS_KINDS]
    if bad:
        raise ValueError(f"snapshot_{label}: unknown hypothesis kinds {bad}")


def derived_by_closure(
    edges: list[tuple[str, str]],
    seed_kinds: list[str],
) -> list[str]:
    """Forward-traverse `derived_by` edges (src ⇒ dst, "dst is derived by src").

    Returns the closure of dependants reachable from `seed_kinds`, excluding
    the seeds themselves, in BFS order.
    """
    adj: dict[str, list[str]] = {}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)
    seen: set[str] = set(seed_kinds)
    order: list[str] = []
    queue: deque[str] = deque(seed_kinds)
    while queue:
        node = queue.popleft()
        for child in adj.get(node, []):
            if child not in seen:
                seen.add(child)
                order.append(child)
                queue.append(child)
    return order


def _default_oil_edges() -> list[tuple[str, str]]:
    return [
        ("h_CRM", "h_LPR"),
        ("h_ML",  "h_LPR"),
        ("h_LPR", "h_MB"),
        ("h_MB",  "h_BL"),
        ("h_BL",  "h_WCT"),
        ("h_ML",  "h_WCT"),
    ]


@action(
    kind="DiffHypothesisStates",
    trust=TrustLevel.SAFE,
    inputs=DiffHypothesisStatesInput,
    outputs=HypothesisDiff,
    allowed_roles={AgentRole.Coordinator, AgentRole.ReservoirEngineer,
                   AgentRole.Auditor, AgentRole.Geologist,
                   AgentRole.ProductionEngineer, AgentRole.Economist},
)
def diff_hypothesis_states(payload: DiffHypothesisStatesInput) -> HypothesisDiff:
    """Compute semantic diff between two hypothesis snapshots.

    Three components:
    - changed_hypotheses: kinds with different active state OR different
      hyperparams between A and B.
    - hyperparam_diff: per-kind, per-axis [value_a, value_b] for shared kinds.
    - stale_cascade: downstream closure (via derived_by) of changed_hypotheses.
    """
    _validate_kinds(payload.snapshot_a, "a")
    _validate_kinds(payload.snapshot_b, "b")

    active_a = set(payload.snapshot_a.active_hypotheses)
    active_b = set(payload.snapshot_b.active_hypotheses)
    activity_changes = active_a.symmetric_difference(active_b)

    hp_diff: dict[str, dict[str, list]] = {}
    for kind in active_a & active_b:
        params_a = payload.snapshot_a.hyperparams.get(kind, {})
        params_b = payload.snapshot_b.hyperparams.get(kind, {})
        axes = set(params_a.keys()) | set(params_b.keys())
        per_axis: dict[str, list] = {}
        for axis in axes:
            va = params_a.get(axis)
            vb = params_b.get(axis)
            if va != vb:
                per_axis[axis] = [va, vb]
        if per_axis:
            hp_diff[kind] = per_axis

    changed = sorted(activity_changes | set(hp_diff.keys()))
    edges = _default_oil_edges()  # base_snapshot extension point — out of MVP scope
    cascade = derived_by_closure(edges, changed)

    return HypothesisDiff(
        changed_hypotheses=changed,
        hyperparam_diff=hp_diff,
        stale_cascade=cascade,
    )
