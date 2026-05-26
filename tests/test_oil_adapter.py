"""Tests for the WFOpt oil waterflood adapter.

Structural tests (no Pellet required) verify hypothesis creation, lattice
topology, parameter mapping, and cascade invalidation.  Reasoning tests
(marked ``needs_pellet``) verify OWL auto-classification via Pellet.

See Section 4.10 of dissertation.
"""

from __future__ import annotations

import pytest

# ── Guard: Pellet availability ───────────────────────────────────────────────
import os
import shutil

try:
    from owlready2 import sync_reasoner_hermit, default_world

    _OWL_AVAILABLE = True
except ImportError:
    _OWL_AVAILABLE = False

# Ensure Java is on PATH
_JAVA_DIRS = [
    r"C:\Program Files\Eclipse Adoptium\jre-17.0.18.8-hotspot\bin",
    r"C:\Program Files\Eclipse Adoptium\jre-21.0.10.7-hotspot\bin",
]
for jdir in _JAVA_DIRS:
    if os.path.isdir(jdir) and jdir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = jdir + os.pathsep + os.environ.get("PATH", "")
        break

_REASONER_AVAILABLE = False
if _OWL_AVAILABLE and shutil.which("java") is not None:
    try:
        sync_reasoner_hermit(infer_property_values=False)
        _REASONER_AVAILABLE = True
    except Exception:
        pass

needs_pellet = pytest.mark.skipif(
    not _REASONER_AVAILABLE,
    reason="owlready2 + Pellet (Java 17) required",
)

# ── Imports ──────────────────────────────────────────────────────────────────
from hyppo.core._base import (
    Hypothesis,
    virtual_experiment_onto as onto,
)
from hyppo.ontology.core_rules import (
    InvalidHypothesis,
    PhysicsHypothesis,
    DataDrivenHypothesis,
    HybridHypothesis,
    StaleHypothesis,
)
from hyppo.adapters.wfopt_adapter import (
    OilFieldOntology,
    Well,
    Injector,
    Producer,
    ReservoirParameter,
    ConnectivityFraction,
    TimeConstant,
    HYPOTHESIS_PARAM_MAP,
    CONFIGURATION_SPACE,
    build_oil_virtual_experiment,
    run_oil_experiment_demo,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def ve_data():
    """Build a fresh virtual experiment for each test."""
    data = build_oil_virtual_experiment()
    yield data
    # Cleanup OWL individuals to avoid ABox pollution
    from owlready2 import destroy_entity
    for ind in list(onto.individuals()):
        destroy_entity(ind)


# ============================================================================
# Structural tests (no Pellet needed)
# ============================================================================

class TestBuildOilExperiment:
    """Tests for build_oil_virtual_experiment()."""

    def test_build_oil_experiment_creates_6_hypotheses(self, ve_data):
        """VE must contain exactly 6 hypotheses."""
        assert len(ve_data["hypotheses"]) == 6

    def test_hypothesis_names(self, ve_data):
        """All six expected hypothesis names must be present."""
        expected = {"h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL", "h_WCT"}
        assert set(ve_data["hypotheses_map"].keys()) == expected

    def test_lattice_has_6_edges(self, ve_data):
        """Lattice DAG must have exactly 6 directed edges."""
        lattice = ve_data["lattice"]
        assert lattice.number_of_edges() == 6

    def test_lattice_has_6_nodes(self, ve_data):
        """Lattice DAG must have exactly 6 nodes."""
        lattice = ve_data["lattice"]
        assert lattice.number_of_nodes() == 6

    def test_lattice_edges_match_derived_by(self, ve_data):
        """Each lattice edge must correspond to a derived_by OWL relation."""
        expected_edges = {
            ("h_CRM", "h_LPR"),
            ("h_ML", "h_LPR"),
            ("h_LPR", "h_MB"),
            ("h_MB", "h_BL"),
            ("h_BL", "h_WCT"),
            ("h_ML", "h_WCT"),
        }
        actual_edges = set(ve_data["lattice"].edges())
        assert actual_edges == expected_edges

    def test_lattice_is_dag(self, ve_data):
        """Lattice must be a directed acyclic graph."""
        import networkx as nx
        assert nx.is_directed_acyclic_graph(ve_data["lattice"])

    def test_configuration_space_has_17_axes(self, ve_data):
        """Configuration space mirrors the 17 axes from default_space.yaml."""
        assert len(ve_data["configuration_space"]) == 17

    def test_all_params_mapped(self, ve_data):
        """Every param in HYPOTHESIS_PARAM_MAP must exist in configuration space."""
        all_params = set()
        for params in HYPOTHESIS_PARAM_MAP.values():
            all_params.update(params)
        config_keys = set(ve_data["configuration_space"].keys())
        assert all_params.issubset(config_keys)

    def test_models_attached(self, ve_data):
        """Each hypothesis must have a model attached (FunctionalProperty: single value)."""
        for h in ve_data["hypotheses"]:
            assert h.is_implemented_by_model is not None

    def test_virtual_experiment_has_ontology(self, ve_data):
        """The VE individual must have an OilFieldOntology attached."""
        ve = ve_data["virtual_experiment"]
        assert len(ve.has_for_ontology) == 1
        assert isinstance(ve.has_for_ontology[0], OilFieldOntology)


class TestHypothesisClassificationStructural:
    """Structural (non-Pellet) classification checks via model type."""

    def test_h_CRM_has_physics_model(self, ve_data):
        """h_CRM must be implemented by a PhysicsModel."""
        from hyppo.ontology.core_rules import PhysicsModel
        h = ve_data["hypotheses_map"]["h_CRM"]
        assert isinstance(h.is_implemented_by_model, PhysicsModel)

    def test_h_ML_has_datadriven_model(self, ve_data):
        """h_ML must be implemented by a DataDrivenModel."""
        from hyppo.ontology.core_rules import DataDrivenModel
        h = ve_data["hypotheses_map"]["h_ML"]
        assert isinstance(h.is_implemented_by_model, DataDrivenModel)

    def test_h_LPR_has_hybrid_model(self, ve_data):
        """h_LPR must be implemented by a HybridModel."""
        from hyppo.ontology.core_rules import HybridModel
        h = ve_data["hypotheses_map"]["h_LPR"]
        assert isinstance(h.is_implemented_by_model, HybridModel)


class TestCascadeInvalidation:
    """Test cascade invalidation through derived_by chains."""

    def test_cascade_invalidation(self, ve_data):
        """Marking h_CRM invalid must cascade to h_LPR, h_MB, h_BL, h_WCT."""
        hyps = ve_data["hypotheses_map"]
        h_crm = hyps["h_CRM"]

        # Mark h_CRM as invalid
        h_crm.is_a.append(InvalidHypothesis)

        # Walk the derived_by chain structurally
        stale: set[str] = set()
        queue = ["h_CRM"]
        visited: set[str] = set()
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for name, h in hyps.items():
                if name in visited:
                    continue
                deps = h.derived_by if h.derived_by else []
                if any(d is hyps[cur] for d in deps):
                    stale.add(name)
                    queue.append(name)

        # h_LPR depends on h_CRM -> stale
        # h_MB depends on h_LPR -> stale
        # h_BL depends on h_MB -> stale
        # h_WCT depends on h_BL and h_ML -> stale (via h_BL)
        assert "h_LPR" in stale
        assert "h_MB" in stale
        assert "h_BL" in stale
        assert "h_WCT" in stale
        # h_ML should NOT be stale (independent of h_CRM)
        assert "h_ML" not in stale

    def test_cascade_does_not_affect_ml(self, ve_data):
        """Invalidating h_CRM must not affect h_ML."""
        hyps = ve_data["hypotheses_map"]
        h_crm = hyps["h_CRM"]
        h_ml = hyps["h_ML"]

        h_crm.is_a.append(InvalidHypothesis)

        # h_ML has no derived_by pointing to h_CRM
        deps = h_ml.derived_by if h_ml.derived_by else []
        assert h_crm not in deps


class TestOilConstraintsValidation:
    """Test Pydantic validators from oil_constraints.py."""

    def test_fractional_flow_valid(self):
        from hyppo.ontology.oil_constraints import FractionalFlowParams
        p = FractionalFlowParams(f_ij=[0.3, 0.4, 0.2])
        assert abs(sum(p.f_ij) - 0.9) < 1e-9

    def test_fractional_flow_exceeds_one(self):
        from hyppo.ontology.oil_constraints import FractionalFlowParams
        with pytest.raises(ValueError, match="exceeds 1.0"):
            FractionalFlowParams(f_ij=[0.6, 0.5])

    def test_timescale_valid(self):
        from hyppo.ontology.oil_constraints import TimeScaleParams
        p = TimeScaleParams(tau_fast=1.0, tau_slow=10.0)
        assert p.tau_fast < p.tau_slow

    def test_timescale_fast_ge_slow_fails(self):
        from hyppo.ontology.oil_constraints import TimeScaleParams
        with pytest.raises(ValueError, match="tau_fast"):
            TimeScaleParams(tau_fast=10.0, tau_slow=5.0)

    def test_saturation_valid(self):
        from hyppo.ontology.oil_constraints import SaturationParams
        p = SaturationParams(s_o=0.3, s_w=0.7)
        assert abs(p.s_o + p.s_w - 1.0) < 1e-9

    def test_saturation_invalid(self):
        from hyppo.ontology.oil_constraints import SaturationParams
        with pytest.raises(ValueError, match="expected 1.0"):
            SaturationParams(s_o=0.5, s_w=0.6)

    def test_corey_positive(self):
        from hyppo.ontology.oil_constraints import CoreyExponentParams
        p = CoreyExponentParams(n_oil=2.0, n_water=3.0)
        assert p.n_oil > 0

    def test_corey_negative_fails(self):
        from hyppo.ontology.oil_constraints import CoreyExponentParams
        with pytest.raises(ValueError, match="must be > 0"):
            CoreyExponentParams(n_oil=-1.0, n_water=2.0)


class TestOilDomainOntologyClasses:
    """Test domain-specific OWL classes."""

    def test_well_hierarchy(self):
        assert issubclass(Injector, Well)
        assert issubclass(Producer, Well)

    def test_reservoir_parameter_hierarchy(self):
        assert issubclass(ConnectivityFraction, ReservoirParameter)
        assert issubclass(TimeConstant, ReservoirParameter)

    def test_oil_field_ontology_is_domain_ontology(self):
        from hyppo.ontology.core_rules import DomainOntology
        assert issubclass(OilFieldOntology, DomainOntology)


# ============================================================================
# Reasoning tests (require Pellet)
# ============================================================================

@needs_pellet
class TestPelletClassification:
    """OWL reasoner-based classification tests."""

    def test_hypothesis_classification_physics(self, ve_data):
        """Pellet must classify h_CRM as PhysicsHypothesis."""
        sync_reasoner_hermit(infer_property_values=True)
        h = ve_data["hypotheses_map"]["h_CRM"]
        assert PhysicsHypothesis in h.is_a

    def test_hypothesis_classification_datadriven(self, ve_data):
        """Pellet must classify h_ML as DataDrivenHypothesis."""
        sync_reasoner_hermit(infer_property_values=True)
        h = ve_data["hypotheses_map"]["h_ML"]
        assert DataDrivenHypothesis in h.is_a

    def test_hypothesis_classification_hybrid(self, ve_data):
        """Pellet must classify h_LPR as HybridHypothesis."""
        sync_reasoner_hermit(infer_property_values=True)
        h = ve_data["hypotheses_map"]["h_LPR"]
        assert HybridHypothesis in h.is_a

    def test_pellet_cascade_invalidation(self, ve_data):
        """Pellet must infer StaleHypothesis for h_LPR when h_CRM is invalid."""
        hyps = ve_data["hypotheses_map"]
        hyps["h_CRM"].is_a.append(InvalidHypothesis)
        sync_reasoner_hermit(infer_property_values=True)
        assert StaleHypothesis in hyps["h_LPR"].is_a


class TestDemoRun:
    """Test the demo function runs without errors."""

    def test_run_oil_experiment_demo(self):
        """Demo must complete and return expected keys."""
        # Cleanup first
        from owlready2 import destroy_entity
        for ind in list(onto.individuals()):
            destroy_entity(ind)

        result = run_oil_experiment_demo()
        assert "ve" in result
        assert "classifications" in result
        assert "stale_after_invalidation" in result
        assert "provenance" in result
        assert len(result["stale_after_invalidation"]) >= 4

        # Cleanup
        for ind in list(onto.individuals()):
            destroy_entity(ind)
