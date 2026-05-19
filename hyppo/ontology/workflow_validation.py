"""OWL 2 DL workflow validation rules (9-10).

Rule 9  -- OrphanHypothesis: a hypothesis not referenced by any workflow task.
Rule 10 -- ConflictingTask: a workflow task whose hypotheses compete.
"""

from __future__ import annotations

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

with virtual_experiment_onto:

    # ── Supporting classes / properties ─────────────────────────────────────
    class WorkflowTask(Thing):
        """A single task within an experiment workflow."""

    class hasHypothesis(WorkflowTask >> Hypothesis, ObjectProperty):
        """Associates a workflow task with the hypotheses it tests."""

    # ── Rule 9: OrphanHypothesis ──────────────────────────────────────────
    class OrphanHypothesis(Hypothesis):
        """A hypothesis not referenced by any workflow task.

        Note: ``Not(Inverse(hasHypothesis).some(WorkflowTask))`` requires
        CWA — under OWA the reasoner cannot infer absence of a task link.
        The class is retained as a positive marker; orphan detection is
        delegated to Python validation.
        """

    # ── Rule 10: ConflictingTask ──────────────────────────────────────────
    class ConflictFreeTask(WorkflowTask):
        """A task whose hypotheses have no ``competes`` relations.

        Note: ``hasHypothesis.only(competes.only(Nothing))`` requires CWA
        to verify the universal restriction.  Under OWA the class is
        retained as a positive marker; conflict-free verification is
        delegated to Python validation.
        """

    class ConflictingTask(WorkflowTask):
        """A task with competing hypotheses.

        Formally: WorkflowTask AND hasHypothesis SOME (competes SOME Hypothesis).
        This positive existential restriction works under OWA.
        """
        equivalent_to = [
            WorkflowTask & hasHypothesis.some(competes.some(Hypothesis))
        ]
