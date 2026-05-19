"""OWL 2 DL quality-gate rules (11-12).

Rule 11 -- PrunableSubtree: a low-quality node whose entire subtree is also
    low-quality (safe to prune).
Rule 12 -- PromisingRoute: any node that has at least one high-quality
    ancestor (worth exploring further).
"""

from __future__ import annotations

from owlready2 import (
    ObjectProperty,
    Thing,
    TransitiveProperty,
)

from hyppo.core._base import Hypothesis, virtual_experiment_onto

__all__ = [
    "LowQuality",
    "HighQuality",
    "hasDescendant",
    "hasAncestor",
    "PrunableSubtree",
    "PromisingRoute",
]

with virtual_experiment_onto:

    # ── Quality markers ────────────────────────────────────────────────────
    class LowQuality(Hypothesis):
        """Hypothesis flagged as low-quality by a quality gate."""

    class HighQuality(Hypothesis):
        """Hypothesis flagged as high-quality by a quality gate."""

    # ── Hierarchy navigation properties ────────────────────────────────────
    class hasDescendant(Hypothesis >> Hypothesis, ObjectProperty, TransitiveProperty):
        """Transitive descendant relation in the hypothesis lattice."""

    class hasAncestor(Hypothesis >> Hypothesis, ObjectProperty, TransitiveProperty):
        """Transitive ancestor relation in the hypothesis lattice."""
        inverse_property = hasDescendant

    # ── Rule 11: PrunableSubtree ──────────────────────────────────────────
    class PrunableSubtree(LowQuality):
        """A low-quality hypothesis whose *every* descendant is also
        low-quality -- the entire subtree can be pruned.

        Note: ``hasDescendant.only(LowQuality)`` requires CWA to verify
        the universal restriction.  Under OWA the reasoner cannot confirm
        that all descendants are LowQuality.  The class is retained as a
        positive marker; prunable-subtree detection is delegated to Python
        validation.
        """

    # ── Rule 12: PromisingRoute ───────────────────────────────────────────
    class PromisingRoute(Hypothesis):
        """A hypothesis reachable from a high-quality ancestor.

        Formally: hasAncestor SOME HighQuality.
        """
        equivalent_to = [
            Hypothesis & hasAncestor.some(HighQuality)
        ]
