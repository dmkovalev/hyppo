"""Tests for hyppo.actions.virtual_experiment.get_hypothesis_lattice."""

import pytest

from hyppo.actions.virtual_experiment import (
    GetHypothesisLatticeInput,
    LatticeGraph,
    get_hypothesis_lattice,
)


def test_lattice_has_six_nodes_and_six_edges():
    out = get_hypothesis_lattice(GetHypothesisLatticeInput())
    assert isinstance(out, LatticeGraph)
    assert len(out.nodes) == 6
    assert len(out.edges) == 6


def test_lattice_nodes_are_canonical_kinds():
    out = get_hypothesis_lattice(GetHypothesisLatticeInput())
    assert set(out.nodes) == {"h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL", "h_WCT"}


def test_lattice_unknown_domain_raises():
    with pytest.raises(ValueError):
        get_hypothesis_lattice(GetHypothesisLatticeInput(domain="refining"))
