"""OWL 2 DL lifecycle rules (16-17, 30-31).

Rule 16 -- BlockingDeprecation: a deprecated hypothesis that still has active
    dependants, preventing safe removal.  Also defines a disjoint-union of
    lifecycle states (Draft / Active / Deprecated / Archived).
Rule 17 -- ConflictingHypothesis: an active hypothesis that competes with
    another active hypothesis -- a hypothesis-level conflict signalling that a
    resolution decision is required (diagnostic marker, not an inconsistency).
Rule 30 -- FreshHypothesis (marker): closes the repair cycle — after
    recomputation, a hypothesis transitions Stale → Fresh (procedurally asserted).
Rule 31 -- PreferredHypothesis: an active hypothesis whose competitor has been
    refuted (Invalid) — auto-inferred from competes + lifecycle states.
"""

from __future__ import annotations

from owlready2 import AllDisjoint, Inverse

from hyppo.core._base import (
    Hypothesis,
    competes,
    derived_by,
    virtual_experiment_onto,
)
from hyppo.ontology.core_rules import InvalidHypothesis

__all__ = [
    "DraftHypothesis",
    "ActiveHypothesis",
    "DeprecatedHypothesis",
    "ArchivedHypothesis",
    "BlockingDeprecation",
    "ConflictingHypothesis",
    "FreshHypothesis",
    "PreferredHypothesis",
    "refresh_hypothesis",
    "apply_pydantic_to_ontology",
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
    AllDisjoint(
        [DraftHypothesis, ActiveHypothesis, DeprecatedHypothesis, ArchivedHypothesis]
    )

    # Covering axiom: every Hypothesis belongs to exactly one state.
    Hypothesis.equivalent_to.append(
        DraftHypothesis | ActiveHypothesis | DeprecatedHypothesis | ArchivedHypothesis
    )

    # ── Rule 16: BlockingDeprecation ──────────────────────────────────────
    class BlockingDeprecation(DeprecatedHypothesis):
        """A deprecated hypothesis that cannot be safely removed because
        at least one *active* hypothesis depends on it.

        Formally: DeprecatedHypothesis AND inverse(derived_by) SOME ActiveHypothesis.
        """

        equivalent_to = [
            DeprecatedHypothesis & Inverse(derived_by).some(ActiveHypothesis)
        ]

    # ── Rule 17: ConflictingHypothesis ────────────────────────────────────
    class ConflictingHypothesis(ActiveHypothesis):
        """An active hypothesis that competes with another active hypothesis.

        Formally: ActiveHypothesis AND competes SOME ActiveHypothesis.
        """

        equivalent_to = [ActiveHypothesis & competes.some(ActiveHypothesis)]

    # ── Rule 30: FreshHypothesis (marker, closes repair cycle) ───────────
    class FreshHypothesis(Hypothesis):
        """A hypothesis whose result has been recomputed and is up-to-date.

        Closes the repair cycle: Plan → Recompute → remove Stale marker
        → assert Fresh marker.  Procedurally asserted (CWA): the reasoner
        cannot infer "not Stale" under OWA, so Fresh is a positive marker
        that replaces Stale after successful recomputation.

        Lifecycle dimension: Freshness (orthogonal to Existence/Validity).
        """

    # ── Rule 31: PreferredHypothesis (competes resolution) ───────────────
    class PreferredHypothesis(ActiveHypothesis):
        """An active hypothesis confirmed by refutation of its competitor.

        If h_A competes with h_B, and h_B is Invalid (refuted), then h_A
        is Confirmed — the surviving hypothesis in a two-hypothesis contest.

        Formally: ActiveHypothesis AND competes SOME InvalidHypothesis.
        Auto-inferred by HermiT (positive existential, DL-compatible).
        """

        equivalent_to = [ActiveHypothesis & competes.some(InvalidHypothesis)]


# ── Procedural bridges (Python → ontology) ──────────────────────────────


def refresh_hypothesis(hypothesis, ontology=None):
    """Close the repair cycle: Stale → Fresh after recomputation.

    Removes StaleHypothesis from the individual's types and asserts
    FreshHypothesis, marking the result as up-to-date.  This is the
    formal counterpart to the planner executing the recalculation plan.
    """
    from hyppo.ontology.core_rules import StaleHypothesis

    # Remove stale marker (if present)
    if StaleHypothesis in hypothesis.is_a:
        hypothesis.is_a.remove(StaleHypothesis)
    # Assert fresh marker
    if FreshHypothesis not in hypothesis.is_a:
        hypothesis.is_a.append(FreshHypothesis)


def apply_pydantic_to_ontology(individual, params, ontology=None):
    """Bridge: run Pydantic validation; on failure, mark InvalidHypothesis.

    Connects Layer 3 (Pydantic) to Layer 1 (DL): when a physical invariant
    is violated, the hypothesis is marked InvalidHypothesis in the ABox,
    enabling the DL reasoner to propagate the consequence via cascade (rule 4).
    """
    from pydantic import ValidationError

    try:
        # params is a dict like {"f_ij": [...]} → validated by Pydantic model
        if "f_ij" in params:
            from hyppo.ontology.oil_constraints import FractionalFlowParams

            FractionalFlowParams(f_ij=params["f_ij"])
        if "tau_fast" in params and "tau_slow" in params:
            from hyppo.ontology.oil_constraints import TimeScaleParams

            TimeScaleParams(tau_fast=params["tau_fast"], tau_slow=params["tau_slow"])
        if "s_o" in params and "s_w" in params:
            from hyppo.ontology.oil_constraints import SaturationParams

            SaturationParams(s_o=params["s_o"], s_w=params["s_w"])
        if "n_oil" in params and "n_water" in params:
            from hyppo.ontology.oil_constraints import CoreyExponentParams

            CoreyExponentParams(n_oil=params["n_oil"], n_water=params["n_water"])
        return True  # all constraints passed
    except ValidationError:
        # Bridge: write InvalidHypothesis to ABox (not just Python exception)
        if InvalidHypothesis not in individual.is_a:
            individual.is_a.append(InvalidHypothesis)
        return False
