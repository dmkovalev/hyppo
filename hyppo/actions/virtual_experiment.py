"""Read-only virtual-experiment introspection actions.

Wraps hyppo.adapters.wfopt_adapter so the dissertation reference code
stays untouched. Two actions:
- BuildVirtualExperiment: full snapshot (hypotheses + edges + config space)
- GetHypothesisLattice: graph-only subset (nodes + edges)

Both are SAFE: no DB I/O, deterministic, side-effect free except for
the owlready2 individuals created inside `wfopt_adapter`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from hyppo.actions.registry import action
from hyppo.actions.types import AgentRole, TrustLevel

Domain = Literal["oil_waterflood"]

ALL_HYPOTHESIS_KINDS = ("h_CRM", "h_ML", "h_LPR", "h_MB", "h_BL", "h_WCT")


class BuildVirtualExperimentInput(BaseModel):
    domain: Domain = Field(
        default="oil_waterflood",
        description="Single-domain MVP. Future: 'production_optimization', 'refining'.",
    )


class HypothesisRef(BaseModel):
    kind: str = Field(description="One of h_CRM, h_ML, h_LPR, h_MB, h_BL, h_WCT.")
    description: str
    model_classes: list[str] = Field(
        description="MRO of the OWL model class implementing this hypothesis "
                    "(e.g. ['HybridModel', 'Model', 'Thing']).",
    )
    hyperparam_axes: list[str] = Field(
        description="default_space.yaml axes that toggle inside this hypothesis.",
    )


class LatticeEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class ConfigurationAxis(BaseModel):
    name: str
    section: str
    levels: list


class VirtualExperimentSnapshot(BaseModel):
    domain: Domain
    hypotheses: list[HypothesisRef]
    edges: list[LatticeEdge]
    configuration_space: list[ConfigurationAxis]


class GetHypothesisLatticeInput(BaseModel):
    domain: Domain = "oil_waterflood"


class LatticeGraph(BaseModel):
    domain: Domain
    nodes: list[str]
    edges: list[LatticeEdge]


_OIL_SNAPSHOT_CACHE: VirtualExperimentSnapshot | None = None


def _build_oil_snapshot() -> VirtualExperimentSnapshot:
    """Single source of truth — call wfopt_adapter once, project to Pydantic.

    Memoised at module scope because `wfopt_adapter.build_oil_virtual_experiment`
    creates OWL individuals with fixed IRIs in the default world; a second call
    in the same process would raise `sqlite3.IntegrityError`. Reuse is safe —
    the output is deterministic.
    """
    global _OIL_SNAPSHOT_CACHE
    if _OIL_SNAPSHOT_CACHE is not None:
        return _OIL_SNAPSHOT_CACHE

    from hyppo.adapters.wfopt_adapter import (
        HYPOTHESIS_PARAM_MAP,
        CONFIGURATION_SPACE,
        build_oil_virtual_experiment,
    )

    ve = build_oil_virtual_experiment()
    hyps = []
    for kind, h in ve["hypotheses_map"].items():
        # is_implemented_by_model is FunctionalProperty (R3+); single value or None.
        model = h.is_implemented_by_model
        model_classes: list[str] = []
        if model is not None:
            for cls in type(model).mro():
                if cls.__name__ != "object":
                    model_classes.append(cls.__name__)
        hyps.append(HypothesisRef(
            kind=kind,
            description=h.description or "",
            model_classes=model_classes,
            hyperparam_axes=HYPOTHESIS_PARAM_MAP.get(kind, []),
        ))

    nx_graph = ve["lattice"]
    edges = [
        LatticeEdge(**{"from": src, "to": dst})
        for src, dst in nx_graph.edges()
    ]
    cfg = [
        ConfigurationAxis(name=name, section=spec["section"], levels=spec["levels"])
        for name, spec in CONFIGURATION_SPACE.items()
    ]
    _OIL_SNAPSHOT_CACHE = VirtualExperimentSnapshot(
        domain="oil_waterflood",
        hypotheses=hyps,
        edges=edges,
        configuration_space=cfg,
    )
    return _OIL_SNAPSHOT_CACHE


@action(
    kind="BuildVirtualExperiment",
    trust=TrustLevel.SAFE,
    inputs=BuildVirtualExperimentInput,
    outputs=VirtualExperimentSnapshot,
    allowed_roles={AgentRole.Coordinator, AgentRole.ReservoirEngineer,
                   AgentRole.Auditor, AgentRole.Geologist,
                   AgentRole.ProductionEngineer, AgentRole.Economist},
)
def build_virtual_experiment(
    payload: BuildVirtualExperimentInput,
) -> VirtualExperimentSnapshot:
    """Return the current VirtualExperiment snapshot (hypotheses + lattice + config)."""
    if payload.domain != "oil_waterflood":
        raise ValueError(f"domain={payload.domain!r} not supported in MVP")
    return _build_oil_snapshot()


@action(
    kind="GetHypothesisLattice",
    trust=TrustLevel.SAFE,
    inputs=GetHypothesisLatticeInput,
    outputs=LatticeGraph,
    allowed_roles={AgentRole.Coordinator, AgentRole.ReservoirEngineer,
                   AgentRole.Auditor, AgentRole.Geologist,
                   AgentRole.ProductionEngineer, AgentRole.Economist},
)
def get_hypothesis_lattice(payload: GetHypothesisLatticeInput) -> LatticeGraph:
    """Return only the derived_by graph (nodes + edges) — cheap, for UI."""
    if payload.domain != "oil_waterflood":
        raise ValueError(f"domain={payload.domain!r} not supported in MVP")
    snap = _build_oil_snapshot()
    return LatticeGraph(
        domain=snap.domain,
        nodes=[h.kind for h in snap.hypotheses],
        edges=snap.edges,
    )
