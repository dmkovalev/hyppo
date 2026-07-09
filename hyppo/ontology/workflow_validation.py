"""OWL 2 DL workflow validation rules (9-10).

Rule 9  -- OrphanHypothesis: a hypothesis not referenced by any workflow task.
Rule 10 -- ConflictingTask: a workflow task whose hypotheses compete.
"""

from __future__ import annotations

from types import SimpleNamespace

from owlready2 import (
    ObjectProperty,
    Thing,
)

from hyppo.core._base import (
    Hypothesis,
    competes,
    virtual_experiment_onto,
)

__all__ = [
    "WorkflowTask",
    "hasHypothesis",
    "OrphanHypothesis",
    "ConflictFreeTask",
    "ConflictingTask",
]


def define_rules(onto, ns):
    """Declare this module's OWL rules in ``onto`` using base entities
    from ``ns``; register the created classes back onto ``ns``."""
    with onto:
        # ── Supporting classes / properties ─────────────────────────────────────
        class WorkflowTask(Thing):
            """A single task within an experiment workflow."""

        class hasHypothesis(WorkflowTask >> ns.Hypothesis, ObjectProperty):
            """Associates a workflow task with the hypotheses it tests."""

        # ── Rule 9: OrphanHypothesis ──────────────────────────────────────────
        class OrphanHypothesis(ns.Hypothesis):
            """A hypothesis not referenced by any workflow task.

            Note: ``Not(Inverse(hasHypothesis).some(WorkflowTask))`` requires
            CWA — under OWA the reasoner cannot infer absence of a task link.
            The class is retained as a positive marker; orphan detection is
            delegated to Python validation.
            """

        # ── Rule 10: ConflictingTask ──────────────────────────────────────────
        class ConflictFreeTask(WorkflowTask):
            """A task whose hypotheses have no ``ns.competes`` relations.

            Note: ``hasHypothesis.only(ns.competes.only(Nothing))`` requires CWA
            to verify the universal restriction.  Under OWA the class is
            retained as a positive marker; conflict-free verification is
            delegated to Python validation.
            """

        class ConflictingTask(WorkflowTask):
            """A task with competing hypotheses.

            Formally: WorkflowTask AND
            hasHypothesis SOME (competes SOME Hypothesis).
            This positive existential restriction works under OWA.
            """

            equivalent_to = [
                WorkflowTask & hasHypothesis.some(ns.competes.some(ns.Hypothesis))
            ]

    ns.WorkflowTask = WorkflowTask
    ns.hasHypothesis = hasHypothesis
    ns.OrphanHypothesis = OrphanHypothesis
    ns.ConflictFreeTask = ConflictFreeTask
    ns.ConflictingTask = ConflictingTask
    return ns


_ns = SimpleNamespace(
    Hypothesis=Hypothesis,
    competes=competes,
)
define_rules(virtual_experiment_onto, _ns)

WorkflowTask = _ns.WorkflowTask
hasHypothesis = _ns.hasHypothesis
OrphanHypothesis = _ns.OrphanHypothesis
ConflictFreeTask = _ns.ConflictFreeTask
ConflictingTask = _ns.ConflictingTask
