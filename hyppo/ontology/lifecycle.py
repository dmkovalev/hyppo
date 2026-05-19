"""OWL 2 DL lifecycle rule (16).

Rule 16 -- BlockingDeprecation: a deprecated hypothesis that still has active
    dependants, preventing safe removal.  Also defines a disjoint-union of
    lifecycle states (Draft / Active / Deprecated / Archived).
"""

from __future__ import annotations

from owlready2 import AllDisjoint, Inverse, ObjectProperty

from hyppo.core._base import Hypothesis, derived_by, virtual_experiment_onto

__all__ = [
    "DraftHypothesis",
    "ActiveHypothesis",
    "DeprecatedHypothesis",
    "ArchivedHypothesis",
    "BlockingDeprecation",
]

with virtual_experiment_onto:

    # ── Lifecycle states ──────────────────────────────────────────────────
    class DraftHypothesis(Hypothesis):
        """Hypothesis in draft / work-in-progress state."""

    class ActiveHypothesis(Hypothesis):
        """Hypothesis actively used in experiments."""

    class DeprecatedHypothesis(Hypothesis):
        """Hypothesis marked for retirement."""

    class ArchivedHypothesis(Hypothesis):
        """Hypothesis permanently archived (read-only)."""

    # Pairwise disjointness of lifecycle states.
    # In owlready2, full DisjointUnion (partition) is expressed via
    # AllDisjoint + explicit union in equivalent_to on the parent class.
    AllDisjoint([DraftHypothesis, ActiveHypothesis,
                 DeprecatedHypothesis, ArchivedHypothesis])

    # Covering axiom: every Hypothesis belongs to exactly one state.
    # Hypothesis ≡ Draft ⊔ Active ⊔ Deprecated ⊔ Archived
    Hypothesis.equivalent_to.append(
        DraftHypothesis | ActiveHypothesis
        | DeprecatedHypothesis | ArchivedHypothesis
    )

    # ── Rule 16: BlockingDeprecation ──────────────────────────────────────
    class BlockingDeprecation(DeprecatedHypothesis):
        """A deprecated hypothesis that cannot be safely removed because
        at least one *active* hypothesis depends on it.

        Formally: DeprecatedHypothesis AND inverse(derived_by) SOME ActiveHypothesis.
        """
        equivalent_to = [
            DeprecatedHypothesis
            & Inverse(derived_by).some(ActiveHypothesis)
        ]
