"""OWL 2 DL multi-experiment rule (13).

Rule 13 -- SharedHypothesis: a hypothesis used by two or more experiments,
    enabling cross-experiment analysis and reuse tracking.
"""

from __future__ import annotations

from types import SimpleNamespace

from owlready2 import ObjectProperty, Thing

from hyppo.core._base import Hypothesis, virtual_experiment_onto

__all__ = [
    "Experiment",
    "usesHypothesis",
    "SharedHypothesis",
]


def define_rules(onto, ns):
    """Declare this module's OWL rules in ``onto`` using base entities
    from ``ns``; register the created classes back onto ``ns``."""
    with onto:
        # ── Supporting classes / properties ─────────────────────────────────────
        class Experiment(Thing):
            """A named experiment container (may differ from VirtualExperiment
            when tracking cross-project reuse)."""

        class usesHypothesis(Experiment >> ns.Hypothesis, ObjectProperty):
            """Links an experiment to the hypotheses it relies on."""

        # ── Rule 13: SharedHypothesis ─────────────────────────────────────────
        class SharedHypothesis(ns.Hypothesis):
            """A hypothesis referenced by at least two distinct experiments.

            Note: ``Inverse(usesHypothesis).min(2, Experiment)`` should work
            under OWA in principle, but HermiT does not always infer minimum
            cardinality on inverse properties for individuals.  The class is
            retained as a positive marker; shared-hypothesis detection (count
            of referencing experiments >= 2) is delegated to Python validation.
            """

    ns.Experiment = Experiment
    ns.usesHypothesis = usesHypothesis
    ns.SharedHypothesis = SharedHypothesis
    return ns


_ns = SimpleNamespace(
    Hypothesis=Hypothesis,
)
define_rules(virtual_experiment_onto, _ns)

Experiment = _ns.Experiment
usesHypothesis = _ns.usesHypothesis
SharedHypothesis = _ns.SharedHypothesis
