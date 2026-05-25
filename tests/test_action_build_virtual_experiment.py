"""Tests for hyppo.actions.virtual_experiment.build_virtual_experiment."""
import pytest

from hyppo.actions.virtual_experiment import (
    BuildVirtualExperimentInput,
    HypothesisRef,
    VirtualExperimentSnapshot,
    build_virtual_experiment,
)


EXPECTED_KINDS = {"h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL", "h_WCT"}

EXPECTED_EDGES = {
    ("h_CRM", "h_LPR"),
    ("h_ML", "h_LPR"),
    ("h_LPR", "h_MB"),
    ("h_MB", "h_BL"),
    ("h_BL", "h_WCT"),
    ("h_ML", "h_WCT"),
}


def test_build_default_domain_returns_six_hypotheses():
    snap = build_virtual_experiment(BuildVirtualExperimentInput())
    assert isinstance(snap, VirtualExperimentSnapshot)
    kinds = {h.kind for h in snap.hypotheses}
    assert kinds == EXPECTED_KINDS, f"got {kinds}"


def test_build_returns_six_derived_by_edges():
    snap = build_virtual_experiment(BuildVirtualExperimentInput())
    edge_set = {(e.from_, e.to) for e in snap.edges}
    assert edge_set == EXPECTED_EDGES, f"got {edge_set}"


def test_build_returns_configuration_space_with_17_axes():
    snap = build_virtual_experiment(BuildVirtualExperimentInput())
    assert len(snap.configuration_space) == 17


def test_build_each_hypothesis_has_classification_chain():
    snap = build_virtual_experiment(BuildVirtualExperimentInput())
    h_crm: HypothesisRef = next(h for h in snap.hypotheses if h.kind == "h_CRM")
    # 'PhysicsModel' must appear in the model-implementation chain of h_CRM.
    assert "PhysicsModel" in h_crm.model_classes, f"got {h_crm.model_classes}"


def test_build_unknown_domain_raises():
    with pytest.raises(ValueError, match="domain"):
        build_virtual_experiment(BuildVirtualExperimentInput(domain="refining"))
