"""OWL 2 DL core rules (1-6).

Rule 1 -- Auto-classification of hypothesis kinds via ``equivalent_to``:
    PhysicsHypothesis, DataDrivenHypothesis, HybridHypothesis.
Rule 2 -- Experiment completeness via cardinality restrictions.
Rule 3 -- Mandatory dependency for MaterialBalanceHypothesis.
Rule 4 -- Cascade invalidation: ValidHypothesis / StaleHypothesis /
    UncomputedHypothesis with AllDisjoint.
Rule 5 -- derived_by is also IrreflexiveProperty.
Rule 6 -- Cross-domain ontology hierarchy.
"""

from __future__ import annotations

from types import SimpleNamespace

from owlready2 import (
    AllDisjoint,
    ObjectProperty,
)

from hyppo.core._base import (
    Artefact,
    Configuration,
    Hypothesis,
    Model,
    VirtualExperiment,
    Workflow,
    derived_by,
    has_for_configuration,
    has_for_hypothesis,
    has_for_model,
    has_for_workflow,
    is_implemented_by_model,
    virtual_experiment_onto,
)

__all__ = [
    "PhysicsModel",
    "DataDrivenModel",
    "HybridModel",
    "PhysicsHypothesis",
    "DataDrivenHypothesis",
    "HybridHypothesis",
    "has_for_ontology",
    "CompleteExperiment",
    "PredictionSourceHypothesis",
    "MaterialBalanceHypothesis",
    "has_dependency",
    "InvalidHypothesis",
    "ValidHypothesis",
    "StaleHypothesis",
    "UncomputedHypothesis",
    "DomainOntology",
    "OilDomainOntology",
    "NeuroDomainOntology",
]


# ---------------------------------------------------------------------------
# All new classes live inside the shared ontology namespace
# ---------------------------------------------------------------------------
def define_rules(onto, ns):
    """Declare this module's OWL rules in ``onto`` using base entities
    from ``ns``; register the created classes back onto ``ns``."""
    with onto:
        # ── Rule 1: Auto-classification ───────────────────────────────────────
        class PhysicsModel(ns.Model):
            """A model built exclusively on physics-based equations."""

        class DataDrivenModel(ns.Model):
            """A purely data-driven / ML model."""

        class HybridModel(ns.Model):
            """A model combining physics and data-driven components."""

        AllDisjoint([PhysicsModel, DataDrivenModel, HybridModel])

        class PhysicsHypothesis(ns.Hypothesis):
            """ns.Hypothesis implemented by at least one physics-based model.

            Note: ``only(PhysicsModel)`` removed because under OWA the reasoner
            cannot verify the closure axiom without explicit enumeration of all
            model links.  The ``only`` check is delegated to Python validation.
            """

            equivalent_to = [
                ns.Hypothesis & ns.is_implemented_by_model.some(PhysicsModel)
            ]

        class DataDrivenHypothesis(ns.Hypothesis):
            """ns.Hypothesis implemented by at least one data-driven model.

            Note: ``only(DataDrivenModel)`` removed -- see PhysicsHypothesis.
            """

            equivalent_to = [
                ns.Hypothesis & ns.is_implemented_by_model.some(DataDrivenModel)
            ]

        class HybridHypothesis(ns.Hypothesis):
            """ns.Hypothesis implemented by at least one hybrid model."""

            equivalent_to = [
                ns.Hypothesis & ns.is_implemented_by_model.some(HybridModel)
            ]

        # ── Rule 2: Experiment completeness ───────────────────────────────────
        class DomainOntology(ns.Artefact):
            """Abstract domain ontology artefact (see Rule 6)."""

        class has_for_ontology(ObjectProperty):
            """Associates a VirtualExperiment with its domain ontology."""

            domain = [ns.VirtualExperiment]
            range = [DomainOntology]

        class CompleteExperiment(ns.VirtualExperiment):
            """A fully-specified experiment satisfying all cardinality gates.

            Requires at least 1 ontology, at least 1 workflow, at least 1
            hypothesis, at least 1 model, and at least 1 configuration.

            Note: ``exactly(1, ...)`` replaced with ``some(...)`` because under
            OWA the reasoner cannot verify exact cardinality without closure
            axioms.  Exact-count validation is delegated to Python layer.
            """

            equivalent_to = [
                ns.VirtualExperiment
                & has_for_ontology.some(DomainOntology)
                & ns.has_for_workflow.some(ns.Workflow)
                & ns.has_for_hypothesis.some(ns.Hypothesis)
                & ns.has_for_model.some(ns.Model)
                & ns.has_for_configuration.some(ns.Configuration)
            ]

        # ── Rule 3: Mandatory dependency ─────────────────────────────────────
        class has_dependency(ns.Hypothesis >> ns.Hypothesis, ObjectProperty):
            """Explicit dependency between hypotheses (non-transitive)."""

        class PredictionSourceHypothesis(ns.Hypothesis):
            """A hypothesis that serves as a prediction source."""

        class MaterialBalanceHypothesis(ns.Hypothesis):
            """Material-balance hypothesis; must depend on a prediction source."""

            is_a = [has_dependency.some(PredictionSourceHypothesis)]

        # ── Rule 4: Cascade invalidation ──────────────────────────────────────
        class InvalidHypothesis(ns.Hypothesis):
            """Marker class: hypothesis whose evidence has been invalidated."""

        class ValidHypothesis(ns.Hypothesis):
            """A hypothesis with no invalid dependencies (via transitive ns.derived_by).

            Note: the original ``Not(InvalidHypothesis) & ns.derived_by.only(...)``
            definition requires CWA.  Under OWA the reasoner cannot verify the
            absence of invalid ancestors.  The class is retained as a positive
            marker for manual assertion; negation-based validation is delegated
            to the Python layer.
            """

        class StaleHypothesis(ns.Hypothesis):
            """A hypothesis that transitively depends on an invalid one.

            Uses ns.derived_by (TransitiveProperty) so that staleness propagates
            through the entire dependency chain — the formal analogue of
            Algorithm 4's cascade recomputation.
            """

            equivalent_to = [ns.Hypothesis & ns.derived_by.some(InvalidHypothesis)]

        class UncomputedHypothesis(ns.Hypothesis):
            """A hypothesis not yet associated with any model run.

            Note: ``Not(ns.is_implemented_by_model.some(ns.Model))`` requires CWA to
            infer absence.  Under OWA the reasoner cannot conclude that no model
            exists unless explicit closure axioms are added.  The negation-based
            check is delegated to Python validation; the OWL definition retains
            the positive marker class for manual assertion.
            """

        AllDisjoint([ValidHypothesis, StaleHypothesis, UncomputedHypothesis])

        # ── Rule 5: acyclicity of ns.derived_by ────────────────────────────────
        # OWL 2 DL simplicity constraint forbids IrreflexiveProperty and
        # AsymmetricProperty on transitive roles.  Acyclicity is enforced by
        # Algorithm 3 (procedural DAG check) rather than OWL axioms.
        # TransitiveProperty on ns.derived_by is sufficient for cascade reasoning.

        # ── Rule 6: Cross-domain ontology hierarchy ───────────────────────────
        class OilDomainOntology(DomainOntology):
            """Oil reservoir engineering domain ontology."""

        class NeuroDomainOntology(DomainOntology):
            """Neuroscience / brain-imaging domain ontology."""

        AllDisjoint([OilDomainOntology, NeuroDomainOntology])

    ns.PhysicsModel = PhysicsModel
    ns.DataDrivenModel = DataDrivenModel
    ns.HybridModel = HybridModel
    ns.PhysicsHypothesis = PhysicsHypothesis
    ns.DataDrivenHypothesis = DataDrivenHypothesis
    ns.HybridHypothesis = HybridHypothesis
    ns.DomainOntology = DomainOntology
    ns.has_for_ontology = has_for_ontology
    ns.CompleteExperiment = CompleteExperiment
    ns.has_dependency = has_dependency
    ns.PredictionSourceHypothesis = PredictionSourceHypothesis
    ns.MaterialBalanceHypothesis = MaterialBalanceHypothesis
    ns.InvalidHypothesis = InvalidHypothesis
    ns.ValidHypothesis = ValidHypothesis
    ns.StaleHypothesis = StaleHypothesis
    ns.UncomputedHypothesis = UncomputedHypothesis
    ns.OilDomainOntology = OilDomainOntology
    ns.NeuroDomainOntology = NeuroDomainOntology
    return ns


_ns = SimpleNamespace(
    Artefact=Artefact,
    Configuration=Configuration,
    Hypothesis=Hypothesis,
    Model=Model,
    VirtualExperiment=VirtualExperiment,
    Workflow=Workflow,
    derived_by=derived_by,
    has_for_configuration=has_for_configuration,
    has_for_hypothesis=has_for_hypothesis,
    has_for_model=has_for_model,
    has_for_workflow=has_for_workflow,
    is_implemented_by_model=is_implemented_by_model,
)
define_rules(virtual_experiment_onto, _ns)

PhysicsModel = _ns.PhysicsModel
DataDrivenModel = _ns.DataDrivenModel
HybridModel = _ns.HybridModel
PhysicsHypothesis = _ns.PhysicsHypothesis
DataDrivenHypothesis = _ns.DataDrivenHypothesis
HybridHypothesis = _ns.HybridHypothesis
DomainOntology = _ns.DomainOntology
has_for_ontology = _ns.has_for_ontology
CompleteExperiment = _ns.CompleteExperiment
has_dependency = _ns.has_dependency
PredictionSourceHypothesis = _ns.PredictionSourceHypothesis
MaterialBalanceHypothesis = _ns.MaterialBalanceHypothesis
InvalidHypothesis = _ns.InvalidHypothesis
ValidHypothesis = _ns.ValidHypothesis
StaleHypothesis = _ns.StaleHypothesis
UncomputedHypothesis = _ns.UncomputedHypothesis
OilDomainOntology = _ns.OilDomainOntology
NeuroDomainOntology = _ns.NeuroDomainOntology
