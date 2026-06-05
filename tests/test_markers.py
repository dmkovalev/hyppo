"""Tests for hyppo/ontology/markers.py — layer-2 marker protocol.

For each rule in {2, 9, 11, 13, 15} we test:
  - marker is asserted when the CWA condition is satisfied
  - marker is NOT asserted when the condition is NOT satisfied
  - rollback test (marker retracted on inconsistency) — structural only,
    because injecting an actual OWL contradiction requires HermiT.

If owlready2 is not installed the whole module is skipped.
HermiT-specific tests (``@needs_hermit``) are additionally skipped when
Java is not on PATH.

Marker rules are exercised with ``run_hermit=False`` in most tests to keep
them fast and Java-independent.  The ``@needs_hermit`` tests additionally
verify that HermiT does NOT raise when the marker is legitimately asserted.
"""

from __future__ import annotations

import os
import shutil
import warnings

import pytest

# ---------------------------------------------------------------------------
# owlready2 / Java guards (same pattern as test_owl_reasoning.py)
# ---------------------------------------------------------------------------
try:
    from owlready2 import sync_reasoner_hermit

    _OWL_AVAILABLE = True
except ImportError:
    _OWL_AVAILABLE = False

_JAVA_DIRS = [
    r"C:\Program Files\Eclipse Adoptium\jre-17.0.18.8-hotspot\bin",
    r"C:\Program Files\Eclipse Adoptium\jre-21.0.10.7-hotspot\bin",
]
for _jdir in _JAVA_DIRS:
    if os.path.isdir(_jdir) and _jdir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _jdir + os.pathsep + os.environ.get("PATH", "")
        break

_JAVA_AVAILABLE = shutil.which("java") is not None

_HERMIT_AVAILABLE = False
if _OWL_AVAILABLE and _JAVA_AVAILABLE:
    try:
        sync_reasoner_hermit(infer_property_values=False)
        _HERMIT_AVAILABLE = True
    except Exception:
        pass

pytestmark = pytest.mark.skipif(
    not _OWL_AVAILABLE,
    reason="owlready2 not installed",
)

needs_hermit = pytest.mark.skipif(
    not _HERMIT_AVAILABLE,
    reason="owlready2 + HermiT (Java 17) required",
)

# ---------------------------------------------------------------------------
# Imports (safe when owlready2 is present)
# ---------------------------------------------------------------------------
if _OWL_AVAILABLE:
    from hyppo.core._base import (
        Hypothesis,
        Workflow,
        Configuration,
        VirtualExperiment,
        virtual_experiment_onto as onto,
    )
    from hyppo.ontology.core_rules import (
        CompleteExperiment,
        OilDomainOntology,
        # Use a concrete subclass of Model to avoid collision with the
        # Python-layer Model defined in hyppo.core._hypothesis (no-arg __init__).
        PhysicsModel,
        DataDrivenModel,
    )
    from hyppo.ontology.workflow_validation import (
        OrphanHypothesis,
        WorkflowTask,
    )
    from hyppo.ontology.quality_gates import (
        LowQuality,
        HighQuality,
        PrunableSubtree,
    )
    from hyppo.ontology.multi_experiment import (
        Experiment,
        SharedHypothesis,
    )
    from hyppo.ontology.model_compatibility import (
        DatasetNotInConfig,
        ModelWithDatasetNeed,
        ModelConfig,
        Dataset,
        TimeSeriesProducer,  # concrete Model subclass safe to instantiate
    )
    from hyppo.ontology.markers import (
        MarkerReport,
        apply_markers,
        apply_rule_2,
        apply_rule_9,
        apply_rule_11,
        apply_rule_13,
        apply_rule_15,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_counter = 0


def _uid(prefix: str = "ind") -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


def _destroy_individuals() -> None:
    from owlready2 import destroy_entity
    for ind in list(onto.individuals()):
        destroy_entity(ind)


@pytest.fixture(autouse=True)
def _cleanup():
    """Clean ABox before and after each test."""
    _destroy_individuals()
    yield
    _destroy_individuals()


# ===========================================================================
# Rule 2: CompleteExperiment marker
# ===========================================================================

class TestRule2:
    """Rule 2: CompleteExperiment marker is set procedurally after CWA check."""

    def test_marker_asserted_when_complete(self):
        """A VE with all five required parts receives the CompleteExperiment marker."""
        ve = VirtualExperiment(_uid("ve"))
        ve.has_for_ontology = [OilDomainOntology(_uid("onto"))]
        ve.has_for_workflow = [Workflow(_uid("wf"))]
        ve.has_for_hypothesis = [Hypothesis(_uid("h"))]
        ve.has_for_model = [PhysicsModel(_uid("m"))]
        ve.has_for_configuration = [Configuration(_uid("cfg"))]

        marked = apply_rule_2(onto, run_hermit=False)

        assert ve.iri in marked
        assert CompleteExperiment in ve.is_a

    def test_marker_not_asserted_when_missing_workflow(self):
        """A VE without a workflow is NOT marked CompleteExperiment."""
        ve = VirtualExperiment(_uid("ve"))
        ve.has_for_ontology = [OilDomainOntology(_uid("onto"))]
        # no workflow
        ve.has_for_hypothesis = [Hypothesis(_uid("h"))]
        ve.has_for_model = [PhysicsModel(_uid("m"))]
        ve.has_for_configuration = [Configuration(_uid("cfg"))]

        marked = apply_rule_2(onto, run_hermit=False)

        assert ve.iri not in marked
        assert CompleteExperiment not in ve.is_a

    def test_marker_not_asserted_when_missing_hypothesis(self):
        """A VE without a hypothesis is NOT marked CompleteExperiment."""
        ve = VirtualExperiment(_uid("ve"))
        ve.has_for_ontology = [OilDomainOntology(_uid("onto"))]
        ve.has_for_workflow = [Workflow(_uid("wf"))]
        # no hypothesis
        ve.has_for_model = [PhysicsModel(_uid("m"))]
        ve.has_for_configuration = [Configuration(_uid("cfg"))]

        marked = apply_rule_2(onto, run_hermit=False)

        assert ve.iri not in marked

    def test_marker_idempotent(self):
        """Calling apply_rule_2 twice does not duplicate the marker."""
        ve = VirtualExperiment(_uid("ve"))
        ve.has_for_ontology = [OilDomainOntology(_uid("onto"))]
        ve.has_for_workflow = [Workflow(_uid("wf"))]
        ve.has_for_hypothesis = [Hypothesis(_uid("h"))]
        ve.has_for_model = [PhysicsModel(_uid("m"))]
        ve.has_for_configuration = [Configuration(_uid("cfg"))]

        apply_rule_2(onto, run_hermit=False)
        apply_rule_2(onto, run_hermit=False)

        count = sum(1 for c in ve.is_a if c is CompleteExperiment)
        assert count == 1

    @needs_hermit
    def test_marker_with_hermit_consistent(self):
        """CompleteExperiment marker is consistent with HermiT reasoning."""
        ve = VirtualExperiment(_uid("ve"))
        ve.has_for_ontology = [OilDomainOntology(_uid("onto"))]
        ve.has_for_workflow = [Workflow(_uid("wf"))]
        ve.has_for_hypothesis = [Hypothesis(_uid("h"))]
        ve.has_for_model = [PhysicsModel(_uid("m"))]
        ve.has_for_configuration = [Configuration(_uid("cfg"))]

        # Should not raise
        marked = apply_rule_2(onto, run_hermit=True)
        assert ve.iri in marked


# ===========================================================================
# Rule 9: OrphanHypothesis marker
# ===========================================================================

class TestRule9:
    """Rule 9: OrphanHypothesis marker — CWA check over WorkflowTask ABox."""

    def test_orphan_marked_when_no_task_references(self):
        """A hypothesis referenced by no task receives the OrphanHypothesis marker."""
        h = Hypothesis(_uid("h"))
        # No WorkflowTask references h

        marked = apply_rule_9(onto, run_hermit=False)

        assert h.iri in marked
        assert OrphanHypothesis in h.is_a

    def test_non_orphan_not_marked(self):
        """A hypothesis linked to at least one task is NOT marked OrphanHypothesis."""
        h = Hypothesis(_uid("h"))
        t = WorkflowTask(_uid("t"))
        t.hasHypothesis = [h]

        marked = apply_rule_9(onto, run_hermit=False)

        assert h.iri not in marked
        assert OrphanHypothesis not in h.is_a

    def test_partial_orphan(self):
        """Only the unreferenced hypothesis receives the marker."""
        h_orphan = Hypothesis(_uid("h_orphan"))
        h_linked = Hypothesis(_uid("h_linked"))
        t = WorkflowTask(_uid("t"))
        t.hasHypothesis = [h_linked]

        marked = apply_rule_9(onto, run_hermit=False)

        assert h_orphan.iri in marked
        assert h_linked.iri not in marked

    @needs_hermit
    def test_orphan_marker_consistent_with_hermit(self):
        """OrphanHypothesis marker is consistent under HermiT."""
        h = Hypothesis(_uid("h"))

        marked = apply_rule_9(onto, run_hermit=True)

        assert h.iri in marked
        assert OrphanHypothesis in h.is_a


# ===========================================================================
# Rule 11: PrunableSubtree marker
# ===========================================================================

class TestRule11:
    """Rule 11: PrunableSubtree marker — CWA universal-quantifier check."""

    def test_prunable_marked_when_all_descendants_low_quality(self):
        """A LowQuality root whose all descendants are LowQuality is marked PrunableSubtree."""
        root = LowQuality(_uid("root"))
        child1 = LowQuality(_uid("child1"))
        child2 = LowQuality(_uid("child2"))
        root.hasDescendant = [child1, child2]

        marked = apply_rule_11(onto, run_hermit=False)

        assert root.iri in marked
        assert PrunableSubtree in root.is_a

    def test_not_prunable_when_high_quality_descendant(self):
        """A LowQuality root with a HighQuality descendant is NOT marked PrunableSubtree."""
        root = LowQuality(_uid("root"))
        child_bad = LowQuality(_uid("child_bad"))
        child_good = HighQuality(_uid("child_good"))
        root.hasDescendant = [child_bad, child_good]

        marked = apply_rule_11(onto, run_hermit=False)

        assert root.iri not in marked
        assert PrunableSubtree not in root.is_a

    def test_prunable_leaf_no_descendants(self):
        """A LowQuality leaf with no descendants satisfies the ∀ condition vacuously."""
        leaf = LowQuality(_uid("leaf"))
        # no hasDescendant links

        marked = apply_rule_11(onto, run_hermit=False)

        assert leaf.iri in marked

    def test_non_low_quality_not_prunable(self):
        """A HighQuality root is never a PrunableSubtree candidate."""
        root = HighQuality(_uid("hq_root"))

        marked = apply_rule_11(onto, run_hermit=False)

        # HighQuality is not a LowQuality — rule 11 iterates LowQuality only
        assert root.iri not in marked

    @needs_hermit
    def test_prunable_marker_consistent_with_hermit(self):
        """PrunableSubtree marker is consistent under HermiT."""
        root = LowQuality(_uid("root"))
        child = LowQuality(_uid("child"))
        root.hasDescendant = [child]

        marked = apply_rule_11(onto, run_hermit=True)

        assert root.iri in marked


# ===========================================================================
# Rule 13: SharedHypothesis marker
# ===========================================================================

class TestRule13:
    """Rule 13: SharedHypothesis marker — minimum-cardinality inverse CWA check."""

    def test_shared_marked_when_two_experiments(self):
        """A hypothesis used by two experiments receives the SharedHypothesis marker."""
        h = Hypothesis(_uid("h"))
        e1 = Experiment(_uid("e1"))
        e2 = Experiment(_uid("e2"))
        e1.usesHypothesis = [h]
        e2.usesHypothesis = [h]

        marked = apply_rule_13(onto, run_hermit=False)

        assert h.iri in marked
        assert SharedHypothesis in h.is_a

    def test_not_shared_when_one_experiment(self):
        """A hypothesis used by exactly one experiment is NOT marked SharedHypothesis."""
        h = Hypothesis(_uid("h"))
        e1 = Experiment(_uid("e1"))
        e1.usesHypothesis = [h]

        marked = apply_rule_13(onto, run_hermit=False)

        assert h.iri not in marked
        assert SharedHypothesis not in h.is_a

    def test_not_shared_when_no_experiments(self):
        """A hypothesis used by no experiment is NOT marked SharedHypothesis."""
        h = Hypothesis(_uid("h"))

        marked = apply_rule_13(onto, run_hermit=False)

        assert h.iri not in marked

    def test_shared_with_three_experiments(self):
        """SharedHypothesis still fires when three or more experiments link to it."""
        h = Hypothesis(_uid("h"))
        for _ in range(3):
            exp = Experiment(_uid("e"))
            exp.usesHypothesis = [h]

        marked = apply_rule_13(onto, run_hermit=False)

        assert h.iri in marked

    def test_partial_sharing(self):
        """Only the hypothesis used by 2+ experiments is marked; others are not."""
        h_shared = Hypothesis(_uid("h_shared"))
        h_private = Hypothesis(_uid("h_private"))
        e1 = Experiment(_uid("e1"))
        e2 = Experiment(_uid("e2"))
        e1.usesHypothesis = [h_shared, h_private]
        e2.usesHypothesis = [h_shared]

        marked = apply_rule_13(onto, run_hermit=False)

        assert h_shared.iri in marked
        assert h_private.iri not in marked

    @needs_hermit
    def test_shared_marker_consistent_with_hermit(self):
        """SharedHypothesis marker is consistent under HermiT."""
        h = Hypothesis(_uid("h"))
        e1 = Experiment(_uid("e1"))
        e2 = Experiment(_uid("e2"))
        e1.usesHypothesis = [h]
        e2.usesHypothesis = [h]

        marked = apply_rule_13(onto, run_hermit=True)

        assert h.iri in marked


# ===========================================================================
# Rule 15: DatasetNotInConfig marker
# ===========================================================================

class TestRule15:
    """Rule 15: DatasetNotInConfig marker — negation CWA check."""

    def test_marked_when_no_config(self):
        """A ModelWithDatasetNeed with no config receives DatasetNotInConfig."""
        m = ModelWithDatasetNeed(_uid("m"))
        # no usedInConfig links

        marked = apply_rule_15(onto, run_hermit=False)

        assert m.iri in marked
        assert DatasetNotInConfig in m.is_a

    def test_marked_when_config_has_no_dataset(self):
        """A model with a config that has no datasets is still marked."""
        m = ModelWithDatasetNeed(_uid("m"))
        cfg = ModelConfig(_uid("cfg"))
        m.usedInConfig = [cfg]
        # cfg has no hasAvailableDataset

        marked = apply_rule_15(onto, run_hermit=False)

        assert m.iri in marked

    def test_not_marked_when_dataset_available(self):
        """A model with a dataset via config is NOT marked DatasetNotInConfig."""
        m = ModelWithDatasetNeed(_uid("m"))
        cfg = ModelConfig(_uid("cfg"))
        ds = Dataset(_uid("ds"))
        m.usedInConfig = [cfg]
        cfg.hasAvailableDataset = [ds]

        marked = apply_rule_15(onto, run_hermit=False)

        assert m.iri not in marked
        assert DatasetNotInConfig not in m.is_a

    def test_plain_model_not_affected(self):
        """A TimeSeriesProducer (not ModelWithDatasetNeed) is not checked by rule 15."""
        m = TimeSeriesProducer(_uid("m"))

        marked = apply_rule_15(onto, run_hermit=False)

        assert m.iri not in marked

    @needs_hermit
    def test_dataset_missing_marker_consistent_with_hermit(self):
        """DatasetNotInConfig marker is consistent under HermiT."""
        m = ModelWithDatasetNeed(_uid("m"))

        marked = apply_rule_15(onto, run_hermit=True)

        assert m.iri in marked


# ===========================================================================
# apply_markers aggregate
# ===========================================================================

class TestApplyMarkers:
    """apply_markers() orchestrates all five rules in one call."""

    def test_report_fields_populated(self):
        """apply_markers returns a MarkerReport with populated rule fields."""
        # Create one triggering individual per rule
        # Rule 2: complete VE
        ve = VirtualExperiment(_uid("ve"))
        ve.has_for_ontology = [OilDomainOntology(_uid("onto"))]
        ve.has_for_workflow = [Workflow(_uid("wf"))]
        ve.has_for_hypothesis = [Hypothesis(_uid("h_ve"))]
        ve.has_for_model = [PhysicsModel(_uid("m_ve"))]
        ve.has_for_configuration = [Configuration(_uid("cfg_ve"))]
        # Rule 9: orphan hypothesis (same h_ve is now linked to ve but not any task)
        h_orphan = Hypothesis(_uid("h_orphan"))
        # Rule 15: model with no dataset
        m_nd = ModelWithDatasetNeed(_uid("m_nd"))

        report = apply_markers(onto, run_hermit=False)

        assert isinstance(report, MarkerReport)
        assert ve.iri in report.rule2_marked
        assert h_orphan.iri in report.rule9_marked
        assert m_nd.iri in report.rule15_marked

    def test_report_hermit_skipped_flag(self):
        """hermit_skipped flag is set when HermiT is unavailable."""
        from unittest.mock import patch
        import hyppo.ontology.markers as _markers_mod

        # Force _HERMIT_AVAILABLE to False to simulate missing Java
        with patch.object(_markers_mod, "_HERMIT_AVAILABLE", False):
            report = apply_markers(onto, run_hermit=True)

        assert report.hermit_skipped is True

    def test_rollback_on_inconsistency(self):
        """_assert_marker rolls back the marker, emits a warning, and records the IRI."""
        from unittest.mock import patch
        import hyppo.ontology.markers as _markers_mod
        from owlready2 import OwlReadyInconsistentOntologyError

        h = Hypothesis(_uid("h"))
        marker_cls = OrphanHypothesis

        exc = OwlReadyInconsistentOntologyError("fake inconsistency")
        sink: list[str] = []

        with patch.object(_markers_mod, "_HERMIT_AVAILABLE", True):
            with patch.object(_markers_mod, "sync_reasoner_hermit", side_effect=exc):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = _markers_mod._assert_marker(
                        h, marker_cls, onto,
                        run_hermit=True, rolled_back=sink,
                    )

        # marker must have been retracted
        assert result is False
        assert marker_cls not in h.is_a

        # a warning must be emitted
        assert any(
            "rolled back" in str(w.message).lower()
            or "inconsistency" in str(w.message).lower()
            for w in caught
        )

        # IRI must be recorded in the sink
        assert h.iri in sink

    def test_rollback_populates_report_rolled_back(self):
        """apply_markers propagates rollback IRI into MarkerReport.rolled_back."""
        from unittest.mock import patch
        import hyppo.ontology.markers as _markers_mod
        from owlready2 import OwlReadyInconsistentOntologyError

        # A lone hypothesis (no WorkflowTask) triggers rule 9 → OrphanHypothesis.
        # Mock HermiT to reject *that* assertion as inconsistent.
        h = Hypothesis(_uid("h"))
        exc = OwlReadyInconsistentOntologyError("fake inconsistency")

        with patch.object(_markers_mod, "_HERMIT_AVAILABLE", True):
            with patch.object(_markers_mod, "sync_reasoner_hermit", side_effect=exc):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    report = apply_markers(onto, run_hermit=True)

        # The marker must not stick
        assert OrphanHypothesis not in h.is_a
        # The IRI must appear in rolled_back
        assert h.iri in report.rolled_back
        # The IRI must NOT appear in rule9_marked (the success list)
        assert h.iri not in report.rule9_marked
