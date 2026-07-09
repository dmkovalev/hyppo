"""Core domain package: the virtual-experiment OWL ontology (Definition 1)
and the epistemic-status transition function (Section 2)."""

from hyppo.core._base import virtual_experiment_onto
from hyppo.core._epistemic import EpistemicStatus, evaluate_status

__all__ = [
    "virtual_experiment_onto",
    "EpistemicStatus",
    "evaluate_status",
]
