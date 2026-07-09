"""Core domain package: the virtual-experiment OWL ontology (Definition 1)
and the epistemic-status transition function (Section 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hyppo.core._base import virtual_experiment_onto
from hyppo.core._epistemic import EpistemicStatus, evaluate_status
from hyppo.core._types import Metrics

if TYPE_CHECKING:
    from types import SimpleNamespace

    from owlready2 import Ontology, World

__all__ = [
    "virtual_experiment_onto",
    "EpistemicStatus",
    "evaluate_status",
    "Metrics",
    "create_ve_world",
]

_VE_IRI = "http://synthesis.ipi.ac.ru/virtual_experiment.owl"


def create_ve_world(
    iri: str = _VE_IRI, world: World | None = None
) -> tuple[World, Ontology, SimpleNamespace]:
    """Compose the full virtual-experiment schema into an isolated owlready2 World.

    Declares, against a fresh ``World`` (or the one passed in), the base schema
    (:func:`hyppo.core._base.define_ve_schema`), all OWL rules (the seven
    ``hyppo.ontology`` rule modules) and the oil-waterflood domain classes
    (:func:`hyppo.adapters.norne_adapter.define_oil_schema`), threading a single
    ``SimpleNamespace`` so cross-module class references (InvalidHypothesis,
    OilDomainOntology, ...) resolve to this world's entities.

    Returns ``(world, ontology, ns)``. The module-global default-world schema is
    built by calling the same factory functions at import time and is unaffected.
    """
    from owlready2 import World as _World

    from hyppo.adapters.norne_adapter import define_oil_schema
    from hyppo.core._base import define_ve_schema
    from hyppo.ontology import (
        core_rules,
        lifecycle,
        model_compatibility,
        multi_experiment,
        provenance,
        quality_gates,
        workflow_validation,
    )

    if world is None:
        world = _World()
    onto = world.get_ontology(iri)

    ns = define_ve_schema(onto)
    # core_rules first: provenance/lifecycle/oil depend on its classes via ns.
    core_rules.define_rules(onto, ns)
    provenance.define_rules(onto, ns)
    multi_experiment.define_rules(onto, ns)
    workflow_validation.define_rules(onto, ns)
    quality_gates.define_rules(onto, ns)
    model_compatibility.define_rules(onto, ns)
    lifecycle.define_rules(onto, ns)
    define_oil_schema(onto, ns)
    return world, onto, ns
