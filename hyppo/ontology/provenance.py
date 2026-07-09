"""OWL 2 DL provenance rules (7-8).

Rule 7 -- DerivedStaleRun: property chains detect experiment runs that
    transitively depend on a hypothesis with stale ancestors.
Rule 8 -- ObsoleteVersion: a hypothesis version superseded by a newer one.
"""

from __future__ import annotations

from types import SimpleNamespace

from owlready2 import (
    ObjectProperty,
    PropertyChain,
    Thing,
    TransitiveProperty,
)

from hyppo.core._base import (
    Hypothesis,
    derived_by,
    virtual_experiment_onto,
)
from hyppo.ontology.core_rules import InvalidHypothesis

__all__ = [
    "HypothesisVersion",
    "ExperimentRun",
    "uses_hypothesis_version",
    "version_of",
    "uses_hypothesis",
    "run_depends_on_hypothesis",
    "HypothesisWithStaleAncestor",
    "StaleRun",
    "DerivedStaleRun",
    "superseded_by",
    "ObsoleteVersion",
]


def define_rules(onto, ns):
    """Declare this module's OWL rules in ``onto`` using base entities
    from ``ns``; register the created classes back onto ``ns``."""
    with onto:
        # ── Supporting classes ─────────────────────────────────────────────────
        class HypothesisVersion(Thing):
            """A concrete versioned snapshot of a ns.Hypothesis."""

        class ExperimentRun(Thing):
            """A single execution of a virtual experiment."""

        # ── Properties ─────────────────────────────────────────────────────────
        class uses_hypothesis_version(
            ExperimentRun >> HypothesisVersion, ObjectProperty
        ):
            """Links an experiment run to the hypothesis version it used."""

        class version_of(HypothesisVersion >> ns.Hypothesis, ObjectProperty):
            """Links a versioned snapshot back to its parent hypothesis."""

        class uses_hypothesis(ExperimentRun >> ns.Hypothesis, ObjectProperty):
            """Derived: the hypothesis behind a run's version.

            Inferred via PropertyChain(uses_hypothesis_version, version_of).
            """

        # Chain: uses_hypothesis_version o version_of -> uses_hypothesis
        uses_hypothesis.property_chain.append(
            PropertyChain([uses_hypothesis_version, version_of])
        )

        class run_depends_on_hypothesis(ExperimentRun >> ns.Hypothesis, ObjectProperty):
            """Derived: transitive hypothesis dependency of a run.

            Inferred via PropertyChain(uses_hypothesis, ns.derived_by).
            """

        # Chain: uses_hypothesis o ns.derived_by -> run_depends_on_hypothesis
        run_depends_on_hypothesis.property_chain.append(
            PropertyChain([uses_hypothesis, ns.derived_by])
        )

        # ── Rule 7: DerivedStaleRun ────────────────────────────────────────────
        class HypothesisWithStaleAncestor(ns.Hypothesis):
            """ns.Hypothesis with at least one stale ancestor via ns.derived_by."""

            equivalent_to = [ns.Hypothesis & ns.derived_by.some(ns.InvalidHypothesis)]

        class StaleRun(ExperimentRun):
            """An experiment run explicitly marked as stale."""

        class DerivedStaleRun(ExperimentRun):
            """A run that is implicitly stale because it uses a hypothesis
            with a stale ancestor.

            Note: ``Not(StaleRun)`` removed because negation requires CWA.
            Under OWA the positive existential restriction is sufficient to
            detect runs affected by stale hypotheses.  Distinguishing
            DerivedStaleRun from explicitly-flagged StaleRun is delegated
            to Python validation.
            """

            equivalent_to = [
                ExperimentRun & uses_hypothesis.some(HypothesisWithStaleAncestor)
            ]

        # ── Rule 8: ObsoleteVersion ────────────────────────────────────────────
        class superseded_by(
            HypothesisVersion >> HypothesisVersion, ObjectProperty, TransitiveProperty
        ):
            """Links an older version to its successor.

            TransitiveProperty ensures that if v1 superseded_by v2 and
            v2 superseded_by v3, then v1 is also inferred as obsolete
            relative to v3 — enabling chain detection of arbitrary depth.
            """

        class ObsoleteVersion(HypothesisVersion):
            """A hypothesis version that has been superseded."""

            equivalent_to = [HypothesisVersion & superseded_by.some(HypothesisVersion)]

    ns.HypothesisVersion = HypothesisVersion
    ns.ExperimentRun = ExperimentRun
    ns.uses_hypothesis_version = uses_hypothesis_version
    ns.version_of = version_of
    ns.uses_hypothesis = uses_hypothesis
    ns.run_depends_on_hypothesis = run_depends_on_hypothesis
    ns.HypothesisWithStaleAncestor = HypothesisWithStaleAncestor
    ns.StaleRun = StaleRun
    ns.DerivedStaleRun = DerivedStaleRun
    ns.superseded_by = superseded_by
    ns.ObsoleteVersion = ObsoleteVersion
    return ns


_ns = SimpleNamespace(
    Hypothesis=Hypothesis,
    InvalidHypothesis=InvalidHypothesis,
    derived_by=derived_by,
)
define_rules(virtual_experiment_onto, _ns)

HypothesisVersion = _ns.HypothesisVersion
ExperimentRun = _ns.ExperimentRun
uses_hypothesis_version = _ns.uses_hypothesis_version
version_of = _ns.version_of
uses_hypothesis = _ns.uses_hypothesis
run_depends_on_hypothesis = _ns.run_depends_on_hypothesis
HypothesisWithStaleAncestor = _ns.HypothesisWithStaleAncestor
StaleRun = _ns.StaleRun
DerivedStaleRun = _ns.DerivedStaleRun
superseded_by = _ns.superseded_by
ObsoleteVersion = _ns.ObsoleteVersion
