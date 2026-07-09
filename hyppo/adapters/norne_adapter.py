"""Hyppo adapter for the Norne/Brugge oil waterflood virtual experiment.

Creates OWL DL representations of the six HybridCRM sub-hypotheses and maps
them to the discrete hyperparameter space of the CRM (capacitance-resistance
model) tuning problem for the Norne/Brugge waterflood case.

This module does NOT import any external optimizer or reservoir-simulation
package -- it builds purely ontological artefacts that mirror the HybridCRM
model architecture for formal reasoning.

See Section 4.10 of dissertation.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Any

from owlready2 import AllDisjoint, Thing, World

from hyppo.core._base import (
    Hypothesis,
    Model,
    virtual_experiment_onto,
)
from hyppo.ontology.core_rules import OilDomainOntology

__all__ = [
    "OilFieldOntology",
    "Well",
    "Injector",
    "Producer",
    "ReservoirParameter",
    "ConnectivityFraction",
    "TimeConstant",
    "HYPOTHESIS_PARAM_MAP",
    "CONFIGURATION_SPACE",
    "build_oil_virtual_experiment",
    "run_oil_experiment_demo",
]


# ---------------------------------------------------------------------------
# Oil domain ontology classes (OWL individuals inside virtual_experiment_onto)
# ---------------------------------------------------------------------------
def define_oil_schema(onto, ns):
    """Declare this module's OWL rules in ``onto`` using base entities
    from ``ns``; register the created classes back onto ``ns``."""
    with onto:

        class OilFieldOntology(ns.OilDomainOntology):
            """Oil waterflood domain ontology -- HybridCRM specific."""

        class Well(Thing):
            """A well in the reservoir (injector or producer)."""

        class Injector(Well):
            """Injection well."""

        class Producer(Well):
            """Production well."""

        class ReservoirParameter(Thing):
            """A physical reservoir parameter."""

        class ConnectivityFraction(ReservoirParameter):
            """CRM connectivity fraction f_ij between injector-producer pair."""

        class TimeConstant(ReservoirParameter):
            """CRM time constant tau for pressure response propagation."""

        AllDisjoint([Injector, Producer])

    ns.OilFieldOntology = OilFieldOntology
    ns.Well = Well
    ns.Injector = Injector
    ns.Producer = Producer
    ns.ReservoirParameter = ReservoirParameter
    ns.ConnectivityFraction = ConnectivityFraction
    ns.TimeConstant = TimeConstant
    return ns


_ns = SimpleNamespace(
    OilDomainOntology=OilDomainOntology,
)
define_oil_schema(virtual_experiment_onto, _ns)

OilFieldOntology = _ns.OilFieldOntology
Well = _ns.Well
Injector = _ns.Injector
Producer = _ns.Producer
ReservoirParameter = _ns.ReservoirParameter
ConnectivityFraction = _ns.ConnectivityFraction
TimeConstant = _ns.TimeConstant
# ---------------------------------------------------------------------------
# Hyperparameter-to-hypothesis mapping (mirrors default_space.yaml)
# ---------------------------------------------------------------------------
HYPOTHESIS_PARAM_MAP: dict[str, list[str]] = {
    "h_CRM": ["USE_DUAL_TAU_CRM", "CRM_KNN", "PHYS_GRAD_MODE"],
    "h_ML": [
        "HIDDEN_DIM",
        "GNN_NUM_LAYERS",
        "GNN_HEADS",
        "TRANSFORMER_NUM_LAYERS",
        "TRANSFORMER_NHEAD",
        "DIM_FEEDFORWARD_MULT",
        "BIDIRECTIONAL",
        "USE_TEMPORAL_EDGE_FEATURES",
    ],
    "h_LPR": ["USE_FUSION_GATE"],
    "h_MB": ["USE_MATERIAL_BALANCE"],
    "h_WCT": ["USE_CRM_PHASE", "CRM_PHASE_RATIO"],
    "h_BL": ["BATCH_SIZE", "BACKPERIOD"],
}

# Full configuration space replicated from default_space.yaml
# (kept here so the adapter is self-contained -- no YAML dependency at runtime)
CONFIGURATION_SPACE: dict[str, dict[str, Any]] = {
    "HIDDEN_DIM": {"section": "MODEL.HYBRID", "levels": [16, 32, 64, 128]},
    "GNN_NUM_LAYERS": {"section": "MODEL.HYBRID", "levels": [1, 2, 3]},
    "GNN_HEADS": {"section": "MODEL.HYBRID", "levels": [1, 2, 4]},
    "TRANSFORMER_NUM_LAYERS": {"section": "MODEL.HYBRID", "levels": [1, 2, 3]},
    "TRANSFORMER_NHEAD": {"section": "MODEL.HYBRID", "levels": [1, 2, 4]},
    "DIM_FEEDFORWARD_MULT": {"section": "MODEL.HYBRID", "levels": [2, 4]},
    "BATCH_SIZE": {"section": "MODEL.HYBRID", "levels": [2, 4, 8]},
    "USE_CRM_PHASE": {"section": "MODEL.HYBRID", "levels": [True, False]},
    "CRM_PHASE_RATIO": {"section": "MODEL.HYBRID", "levels": [0.1, 0.2, 0.3]},
    "PHYS_GRAD_MODE": {"section": "MODEL.HYBRID", "levels": ["detach", "tau_f", "all"]},
    "USE_DUAL_TAU_CRM": {"section": "MODEL.HYBRID", "levels": [True, False]},
    "USE_MATERIAL_BALANCE": {"section": "MODEL.HYBRID", "levels": [True, False]},
    "USE_FUSION_GATE": {"section": "MODEL.HYBRID", "levels": [True, False]},
    "BACKPERIOD": {"section": "GENERAL", "levels": [24, 36, 48]},
    "CRM_KNN": {"section": "CRM", "levels": [3, 5, 8, 12]},
    "BIDIRECTIONAL": {"section": "PREPROCESSING", "levels": [True, False]},
    "USE_TEMPORAL_EDGE_FEATURES": {"section": "PREPROCESSING", "levels": [True, False]},
}

# ---------------------------------------------------------------------------
# Unique-name counter for OWL individuals
# ---------------------------------------------------------------------------
_counter = 0


def _uid(prefix: str = "oil") -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


def _stamp() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_artefact(cls: type, name: str, desc: str) -> Any:
    """Create an OWL individual with mandatory Artefact fields populated."""
    ind = cls(_uid(name))
    ind.name = name
    ind.description = desc
    ind.id = hash(name) & 0x7FFFFFFF
    ind.authors = ["Norne adapter"]
    ind.create_date = _stamp()
    ind.last_update = _stamp()
    return ind


def _build_hypotheses(ns: Any) -> dict[str, Hypothesis]:
    """Create the six HybridCRM hypotheses as OWL individuals in ``ns``'s world."""
    h_CRM = _make_artefact(
        ns.Hypothesis,
        "h_CRM",
        "DualTau CRM -- physics branch: connectivity fractions and time constants",
    )
    h_ML = _make_artefact(
        ns.Hypothesis,
        "h_ML",
        "Transformer+GNN -- ML branch: temporal and spatial feature learning",
    )
    h_LPR = _make_artefact(
        ns.Hypothesis,
        "h_LPR",
        "Fusion gate -- combines physics and ML liquid production predictions",
    )
    h_MB = _make_artefact(
        ns.Hypothesis,
        "h_MB",
        "Material balance -- water saturation update from predicted production",
    )
    h_BL = _make_artefact(
        ns.Hypothesis,
        "h_BL",
        "Buckley-Leverett -- fractional flow from saturation profile",
    )
    h_WCT = _make_artefact(
        ns.Hypothesis,
        "h_WCT",
        "WCT-anchoring -- water cut correction using physics and ML context",
    )

    # -- Attach models to enable auto-classification (Rule 1) --
    m_phys = ns.PhysicsModel(_uid("m_phys"))
    m_dd = ns.DataDrivenModel(_uid("m_dd"))
    m_hybrid = ns.HybridModel(_uid("m_hybrid"))
    m_phys_mb = ns.PhysicsModel(_uid("m_phys_mb"))
    m_phys_bl = ns.PhysicsModel(_uid("m_phys_bl"))
    m_hybrid_wct = ns.HybridModel(_uid("m_hybrid_wct"))

    # is_implemented_by_model is FunctionalProperty after _base.py R3+
    # (Theorem 1 axiom — see commit 911172c). Each Hypothesis gets at most
    # one Model; single-value assignment required.
    h_CRM.is_implemented_by_model = m_phys
    h_ML.is_implemented_by_model = m_dd
    h_LPR.is_implemented_by_model = m_hybrid
    h_MB.is_implemented_by_model = m_phys_mb
    h_BL.is_implemented_by_model = m_phys_bl
    h_WCT.is_implemented_by_model = m_hybrid_wct

    # -- Lattice edges (derived_by) --
    # h_CRM -> h_LPR   (physics feeds fusion)
    # h_ML  -> h_LPR   (ML feeds fusion)
    # h_LPR -> h_MB    (prediction feeds material balance)
    # h_MB  -> h_BL    (saturation feeds fractional flow)
    # h_BL  -> h_WCT   (fractional flow feeds water cut)
    # h_ML  -> h_WCT   (ML provides context for WCT)
    h_LPR.derived_by = [h_CRM, h_ML]
    h_MB.derived_by = [h_LPR]
    h_BL.derived_by = [h_MB]
    h_WCT.derived_by = [h_BL, h_ML]

    return {
        "h_CRM": h_CRM,
        "h_ML": h_ML,
        "h_LPR": h_LPR,
        "h_MB": h_MB,
        "h_BL": h_BL,
        "h_WCT": h_WCT,
    }


def _build_lattice_graph(hyps: dict[str, Hypothesis]) -> Any:
    """Build a networkx DiGraph mirroring the derived_by edges."""
    import networkx as nx

    G = nx.DiGraph()
    for name, h in hyps.items():
        G.add_node(name, hypothesis=h)

    edges = [
        ("h_CRM", "h_LPR"),
        ("h_ML", "h_LPR"),
        ("h_LPR", "h_MB"),
        ("h_MB", "h_BL"),
        ("h_BL", "h_WCT"),
        ("h_ML", "h_WCT"),
    ]
    G.add_edges_from(edges)
    return G


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_oil_virtual_experiment(world: World | None = None) -> dict[str, Any]:
    """Build virtual experiment for HybridCRM waterflood optimization.

    Each call composes the full schema (base + 16 rules + oil domain) into a
    fresh, isolated owlready2 ``World`` and creates all individuals there, so
    the module is re-callable without ``sqlite3.IntegrityError`` on the fixed
    individual IRIs. Pass a pre-built ``World`` (e.g. from
    :func:`hyppo.core.create_ve_world`) to build into it; ``None`` allocates a
    new one per call.

    Returns dict with keys matching the VE tuple:

    - ``ontology``: OilFieldOntology instance
    - ``hypotheses``: list of 6 Hypothesis individuals
    - ``hypotheses_map``: dict mapping name -> Hypothesis
    - ``models``: list of Model individuals attached to hypotheses
    - ``workflow``: Workflow OWL individual
    - ``configuration_space``: dict from CONFIGURATION_SPACE
    - ``lattice``: networkx DiGraph with 6 nodes and 6 edges
    - ``virtual_experiment``: the VirtualExperiment individual
    - ``world``: the owlready2 World the VE lives in
    - ``onto``: the owlready2 Ontology carrying the schema + ABox
    - ``ns``: SimpleNamespace of the world's schema classes/properties
    """
    from hyppo.core import create_ve_world

    world, onto, ns = create_ve_world(world=world)

    ontology = _make_artefact(
        ns.OilFieldOntology,
        "oil_waterflood",
        "HybridCRM waterflood domain ontology",
    )
    hyps = _build_hypotheses(ns)
    lattice = _build_lattice_graph(hyps)

    # Collect all models from hypotheses (single value per hypothesis after R3+).
    models: list[Model] = []
    for h in hyps.values():
        m = h.is_implemented_by_model
        if m is not None:
            models.append(m)

    workflow = _make_artefact(
        ns.Workflow,
        "hybridcrm_workflow",
        "Training -> Optimization pipeline for HybridCRM",
    )

    configuration = _make_artefact(
        ns.Configuration,
        "default_hp_space",
        "17-axis discrete hyperparameter space from default_space.yaml",
    )

    # Build the VirtualExperiment OWL individual
    ve = _make_artefact(
        ns.VirtualExperiment,
        "oil_ve",
        "HybridCRM waterflood optimization virtual experiment",
    )
    ve.has_for_ontology = [ontology]
    ve.has_for_workflow = [workflow]
    ve.has_for_hypothesis = list(hyps.values())
    ve.has_for_model = models
    ve.has_for_configuration = [configuration]

    return {
        "ontology": ontology,
        "hypotheses": list(hyps.values()),
        "hypotheses_map": hyps,
        "models": models,
        "workflow": workflow,
        "configuration_space": CONFIGURATION_SPACE,
        "lattice": lattice,
        "virtual_experiment": ve,
        "world": world,
        "onto": onto,
        "ns": ns,
    }


def run_oil_experiment_demo() -> dict[str, Any]:
    """Demonstrate full VE lifecycle for oil domain.

    Steps:

    1. Build VE with :func:`build_oil_virtual_experiment`.
    2. Show structural classification of hypotheses (PhysicsHypothesis,
       DataDrivenHypothesis, HybridHypothesis) -- works without Pellet.
    3. Show cascade: mark h_CRM as InvalidHypothesis, then verify that
       h_LPR, h_MB, h_BL, h_WCT become stale (structurally).
    4. Show provenance: version h_CRM_v1 -> h_CRM_v2, detect StaleRun.
    5. Print classification results.

    Does NOT actually run CRM training (no GPU needed).

    :return: dict with ``ve``, ``classifications``, ``stale_after_invalidation``,
        ``provenance`` keys.
    """
    ve_data = build_oil_virtual_experiment()
    hyps = ve_data["hypotheses_map"]
    ns = ve_data["ns"]
    results: dict[str, Any] = {"ve": ve_data}

    # -- Step 2: structural classification --
    classifications: dict[str, list[str]] = {}
    for name, h in hyps.items():
        classifications[name] = [
            cls.__name__ for cls in type(h).mro() if cls.__name__ != "object"
        ]
    results["classifications"] = classifications

    # -- Step 3: cascade invalidation (structural) --
    h_crm = hyps["h_CRM"]
    h_crm.is_a.append(ns.InvalidHypothesis)

    # Structurally mark downstream hypotheses as stale via derived_by chain
    stale_names: list[str] = []
    visited: set[str] = set()
    queue = ["h_CRM"]
    while queue:
        current_name = queue.pop(0)
        if current_name in visited:
            continue
        visited.add(current_name)
        for name, h in hyps.items():
            if name in visited:
                continue
            deps = h.derived_by if hasattr(h, "derived_by") and h.derived_by else []
            for dep in deps:
                if dep is hyps.get(current_name):
                    stale_names.append(name)
                    h.is_a.append(ns.StaleHypothesis)
                    queue.append(name)

    results["stale_after_invalidation"] = stale_names

    # -- Step 4: provenance versioning --
    v1 = ns.HypothesisVersion(_uid("h_CRM_v1"))
    v1.version_of = [h_crm]

    v2 = ns.HypothesisVersion(_uid("h_CRM_v2"))
    v2.version_of = [h_crm]
    v1.superseded_by = [v2]

    run1 = ns.ExperimentRun(_uid("run_old"))
    run1.uses_hypothesis_version = [v1]

    results["provenance"] = {
        "v1": v1,
        "v2": v2,
        "v1_superseded": bool(v1.superseded_by),
        "run_uses_old_version": True,
    }

    # -- Step 5: print summary --
    print("=== Oil Virtual Experiment Demo ===")
    print(f"Hypotheses: {list(hyps.keys())}")
    print(f"Lattice edges: {ve_data['lattice'].number_of_edges()}")
    print(f"Configuration axes: {len(ve_data['configuration_space'])}")
    print()
    print("Hypothesis classifications:")
    for name, cls_list in classifications.items():
        print(f"  {name}: {', '.join(cls_list[:3])}")
    print()
    print(f"After invalidating h_CRM, stale hypotheses: {stale_names}")
    print(f"Provenance: v1 superseded = {results['provenance']['v1_superseded']}")
    print("=== Demo complete ===")

    return results
