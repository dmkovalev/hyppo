"""Tests for hyppo.actions.virtual_experiment.get_hypothesis_graph."""

import pytest

from hyppo.actions.virtual_experiment import (
    GetGraphConstructorInput,
    GraphPayload,
    get_hypothesis_graph,
)


def test_graph_has_six_nodes_and_six_edges():
    out = get_hypothesis_graph(GetGraphConstructorInput())
    assert isinstance(out, GraphPayload)
    assert len(out.nodes) == 6
    assert len(out.edges) == 6


def test_graph_nodes_are_canonical_kinds():
    out = get_hypothesis_graph(GetGraphConstructorInput())
    assert set(out.nodes) == {"h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL", "h_WCT"}


def test_graph_unknown_domain_raises():
    with pytest.raises(ValueError):
        get_hypothesis_graph(GetGraphConstructorInput(domain="refining"))
