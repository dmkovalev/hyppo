"""Tests for hyppo.actions.diff.diff_hypothesis_states.

Lattice (derived_by, src -> dst):
    h_CRM -> h_LPR
    h_ML  -> h_LPR
    h_LPR -> h_MB
    h_MB  -> h_BL
    h_BL  -> h_WCT
    h_ML  -> h_WCT

Cascade truth-table (kind changed → kinds that become stale):
    h_CRM ⇒ {h_LPR, h_MB, h_BL, h_WCT}
    h_ML  ⇒ {h_LPR, h_MB, h_BL, h_WCT}
    h_LPR ⇒ {h_MB, h_BL, h_WCT}
    h_MB  ⇒ {h_BL, h_WCT}
    h_BL  ⇒ {h_WCT}
    h_WCT ⇒ set()
"""

import pytest

from hyppo.actions.diff import (
    DiffHypothesisStatesInput,
    HypothesisDiff,
    HypothesisSnapshot,
    derived_by_closure,
    diff_hypothesis_states,
)


def _default_snapshot(overrides: dict[str, dict] | None = None) -> HypothesisSnapshot:
    """Build a snapshot where every kind is active with empty hyperparams,
    optionally overridden per kind via `overrides`."""
    base = {k: {} for k in ("h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL", "h_WCT")}
    if overrides:
        for k, params in overrides.items():
            base[k] = params
    return HypothesisSnapshot(active_hypotheses=list(base.keys()), hyperparams=base)


def test_identical_snapshots_produce_empty_diff():
    a = _default_snapshot()
    b = _default_snapshot()
    out: HypothesisDiff = diff_hypothesis_states(
        DiffHypothesisStatesInput(snapshot_a=a, snapshot_b=b)
    )
    assert out.changed_hypotheses == []
    assert out.hyperparam_diff == {}
    assert out.stale_cascade == []


@pytest.mark.parametrize(
    "changed_kind,expected_cascade",
    [
        ("h_CRM", {"h_LPR", "h_MB", "h_BL", "h_WCT"}),
        ("h_ML", {"h_LPR", "h_MB", "h_BL", "h_WCT"}),
        ("h_LPR", {"h_MB", "h_BL", "h_WCT"}),
        ("h_MB", {"h_BL", "h_WCT"}),
        ("h_BL", {"h_WCT"}),
        ("h_WCT", set()),
    ],
)
def test_cascade_truth_table(changed_kind, expected_cascade):
    a = _default_snapshot()
    b = _default_snapshot(overrides={changed_kind: {"USE_DUAL_TAU_CRM": True}})
    out = diff_hypothesis_states(DiffHypothesisStatesInput(snapshot_a=a, snapshot_b=b))
    assert changed_kind in out.changed_hypotheses
    assert set(out.stale_cascade) == expected_cascade


def test_hyperparam_only_change_populates_hyperparam_diff():
    a = _default_snapshot(overrides={"h_BL": {"BACKPERIOD": 24}})
    b = _default_snapshot(overrides={"h_BL": {"BACKPERIOD": 36}})
    out = diff_hypothesis_states(DiffHypothesisStatesInput(snapshot_a=a, snapshot_b=b))
    assert out.hyperparam_diff == {"h_BL": {"BACKPERIOD": [24, 36]}}
    assert "h_BL" in out.changed_hypotheses
    assert set(out.stale_cascade) == {"h_WCT"}


def test_inactive_kind_in_b_is_treated_as_change():
    a = _default_snapshot()
    b_kinds = ["h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL"]  # h_WCT off
    b = HypothesisSnapshot(
        active_hypotheses=b_kinds, hyperparams={k: {} for k in b_kinds}
    )
    out = diff_hypothesis_states(DiffHypothesisStatesInput(snapshot_a=a, snapshot_b=b))
    assert "h_WCT" in out.changed_hypotheses
    # leaf node: no downstream cascade
    assert "h_WCT" not in out.stale_cascade


def test_invalid_kind_in_snapshot_raises():
    a = _default_snapshot()
    bad_kinds = ["h_CRM", "h_BOGUS"]
    b = HypothesisSnapshot(
        active_hypotheses=bad_kinds,
        hyperparams={"h_CRM": {}, "h_BOGUS": {}},
    )
    with pytest.raises(ValueError, match="h_BOGUS"):
        diff_hypothesis_states(DiffHypothesisStatesInput(snapshot_a=a, snapshot_b=b))


def test_derived_by_closure_isolated():
    edges = [
        ("h_CRM", "h_LPR"),
        ("h_ML", "h_LPR"),
        ("h_LPR", "h_MB"),
        ("h_MB", "h_BL"),
        ("h_BL", "h_WCT"),
        ("h_ML", "h_WCT"),
    ]
    assert derived_by_closure(edges, ["h_CRM"]) == ["h_LPR", "h_MB", "h_BL", "h_WCT"]
    assert derived_by_closure(edges, ["h_BL"]) == ["h_WCT"]
    assert derived_by_closure(edges, ["h_WCT"]) == []
