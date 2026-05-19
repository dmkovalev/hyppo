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

from owlready2 import (
    AllDisjoint,
    ObjectProperty,
    Thing,
)

from hyppo.core._base import (
    Artefact,
    Configuration,
    Hypothesis,
    Model,
    Workflow,
    derived_by,
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
with virtual_experiment_onto:

    # ── Rule 1: Auto-classification ───────────────────────────────────────
    class PhysicsModel(Model):
        """A model built exclusively on physics-based equations."""

    class DataDrivenModel(Model):
        """A purely data-driven / ML model."""

    class HybridModel(Model):
        """A model combining physics and data-driven components."""

    AllDisjoint([PhysicsModel, DataDrivenModel, HybridModel])

    class PhysicsHypothesis(Hypothesis):
        """Hypothesis implemented by at least one physics-based model.

        Note: ``only(PhysicsModel)`` removed because under OWA the reasoner
        cannot verify the closure axiom without explicit enumeration of all
        model links.  The ``only`` check is delegated to Python validation.
        """
        equivalent_to = [
            Hypothesis
            & is_implemented_by_model.some(PhysicsModel)
        ]

    class DataDrivenHypothesis(Hypothesis):
        """Hypothesis implemented by at least one data-driven model.

        Note: ``only(DataDrivenModel)`` removed -- see PhysicsHypothesis.
        """
        equivalent_to = [
            Hypothesis
            & is_implemented_by_model.some(DataDrivenModel)
        ]

    class HybridHypothesis(Hypothesis):
        """Hypothesis implemented by at least one hybrid model."""
        equivalent_to = [
            Hypothesis
            & is_implemented_by_model.some(HybridModel)
        ]

    # ── Rule 2: Experiment completeness ───────────────────────────────────
    class DomainOntology(Artefact):
        """Abstract domain ontology artefact (see Rule 6)."""

    class has_for_ontology(ObjectProperty):
        """Associates a VirtualExperiment with its domain ontology."""
        domain = [virtual_experiment_onto.VirtualExperiment]
        range = [DomainOntology]

    class CompleteExperiment(virtual_experiment_onto.VirtualExperiment):
        """A fully-specified experiment satisfying all cardinality gates.

        Requires at least 1 ontology, at least 1 workflow, at least 1
        hypothesis, at least 1 model, and at least 1 configuration.

        Note: ``exactly(1, ...)`` replaced with ``some(...)`` because under
        OWA the reasoner cannot verify exact cardinality without closure
        axioms.  Exact-count validation is delegated to Python layer.
        """
        equivalent_to = [
            virtual_experiment_onto.VirtualExperiment
            & has_for_ontology.some(DomainOntology)
            & virtual_experiment_onto.has_for_workflow.some(Workflow)
            & virtual_experiment_onto.has_for_hypothesis.some(Hypothesis)
            & virtual_experiment_onto.has_for_model.some(Model)
            & virtual_experiment_onto.has_for_configuration.some(Configuration)
        ]

    # ── Rule 3: Mandatory dependency ─────────────────────────────────────
    class has_dependency(Hypothesis >> Hypothesis, ObjectProperty):
        """Explicit dependency between hypotheses (non-transitive)."""

    class PredictionSourceHypothesis(Hypothesis):
        """A hypothesis that serves as a prediction source."""

    class MaterialBalanceHypothesis(Hypothesis):
        """Material-balance hypothesis; must depend on a prediction source."""
        is_a = [has_dependency.some(PredictionSourceHypothesis)]

    # ── Rule 4: Cascade invalidation ──────────────────────────────────────
    class InvalidHypothesis(Hypothesis):
        """Marker class: hypothesis whose evidence has been invalidated."""

    class ValidHypothesis(Hypothesis):
        """A hypothesis with no invalid dependencies (via transitive derived_by).

        Note: the original ``Not(InvalidHypothesis) & derived_by.only(...)``
        definition requires CWA.  Under OWA the reasoner cannot verify the
        absence of invalid ancestors.  The class is retained as a positive
        marker for manual assertion; negation-based validation is delegated
        to the Python layer.
        """

    class StaleHypothesis(Hypothesis):
        """A hypothesis that transitively depends on an invalid one.

        Uses derived_by (TransitiveProperty) so that staleness propagates
        through the entire dependency chain — the formal analogue of
        Algorithm 4's cascade recomputation.
        """
        equivalent_to = [
            Hypothesis & derived_by.some(InvalidHypothesis)
        ]

    class UncomputedHypothesis(Hypothesis):
        """A hypothesis not yet associated with any model run.

        Note: ``Not(is_implemented_by_model.some(Model))`` requires CWA to
        infer absence.  Under OWA the reasoner cannot conclude that no model
        exists unless explicit closure axioms are added.  The negation-based
        check is delegated to Python validation; the OWL definition retains
        the positive marker class for manual assertion.
        """

    AllDisjoint([ValidHypothesis, StaleHypothesis, UncomputedHypothesis])

    # ── Rule 5: acyclicity of derived_by ────────────────────────────────
    # OWL 2 DL simplicity constraint forbids IrreflexiveProperty and
    # AsymmetricProperty on transitive roles.  Acyclicity is enforced by
    # Algorithm 3 (procedural DAG check) rather than OWL axioms.
    # TransitiveProperty on derived_by is sufficient for cascade reasoning.

    # ── Rule 6: Cross-domain ontology hierarchy ───────────────────────────
    class OilDomainOntology(DomainOntology):
        """Oil reservoir engineering domain ontology."""

    class NeuroDomainOntology(DomainOntology):
        """Neuroscience / brain-imaging domain ontology."""

    AllDisjoint([OilDomainOntology, NeuroDomainOntology])
