"""Two-stage consistency check for virtual experiments (Algorithm 1 of
`iip2026_planning.tex`).

Stage A — semantic: OWL 2 DL classification via HermiT detects
    C1 (ontology completeness via Concept closure axiom + AllDifferent),
    C2 (realizability via FunctionalProperty is_implemented_by_model
        + existential restriction + AllDifferent on Model individuals),
    C6 (parameter type compatibility via rdfs:range).
    Note: preprocessing axioms (Concept/Model closure, AllDifferent)
    are the obligation of VE assembly, not auto-injected here.

Stage B — structural: procedural checks for
    C3 (acyclicity of the hypothesis graph via Kahn's topo-sort),
    C4 (causal order via artefact-set intersection on each edge),
    C5 (configuration finiteness via direct enumeration of Q_i).

Public API: ``check_consistency(ve, ontology, lattice)`` returns
``ConsistencyResult`` with status flag and violation details.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping

try:
    from owlready2 import (
        OwlReadyInconsistentOntologyError,
        sync_reasoner_hermit,
    )

    _HERMIT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment guard
    _HERMIT_AVAILABLE = False

    class OwlReadyInconsistentOntologyError(Exception):  # type: ignore
        pass


log = logging.getLogger(__name__)


class Status:
    OK = "OK"
    C1_VIOLATED = "C1: ontology incompleteness"
    C2_VIOLATED = "C2: realizability violation"
    C3_VIOLATED = "C3: hypothesis graph has cycle"
    C4_VIOLATED = "C4: causal order violation"
    C5_VIOLATED = "C5: configuration space infinite"
    C6_VIOLATED = "C6: parameter type incompatibility"
    C7_VIOLATED = "C7: domain grounding violation"


@dataclass
class ConsistencyResult:
    ok: bool
    status: str
    details: dict = field(default_factory=dict)


def check_consistency(
    ve,
    ontology,
    lattice: Mapping[int, Iterable[int]],
    *,
    run_hermit: bool = True,
    artefacts: Mapping[int, dict] | None = None,
    configurations: Iterable[Iterable] | None = None,
    domain_terms: set | None = None,
    hypothesis_vars: Mapping[int, Iterable[str]] | None = None,
) -> ConsistencyResult:
    """Two-stage consistency check matching Algorithm 1 of the paper.

    Parameters
    ----------
    ve
        VirtualExperiment instance (carries Hypothesis/Model individuals).
    ontology
        owlready2 ontology object — passed to HermiT for Stage A.
    lattice
        Hypothesis graph as adjacency: ``{h_id: {dependent_h_ids, ...}}``.
        Edges are interpreted parent -> child (impacts-direction in code,
        corresponds to ``\\derived`` in the paper).
    run_hermit
        If False, Stage A is skipped (useful for unit tests without Java).
    artefacts
        Optional ``{h_id: {"in": set[str], "out": set[str]}}`` for C4 check.
        If omitted, C4 is reported as ``skipped``.
    configurations
        Optional iterable of finite parameter domains ``Q_1, ..., Q_d`` for C5.
        If omitted, C5 is reported as ``skipped`` (assumed True by construction).
    """
    details: dict = {}

    if run_hermit:
        if not _HERMIT_AVAILABLE:
            return ConsistencyResult(
                False,
                "stage_a: owlready2 not installed",
                {"stage_a": "skipped"},
            )
        try:
            with ontology:
                sync_reasoner_hermit(infer_property_values=False)
            details["stage_a"] = "passed"
        except OwlReadyInconsistentOntologyError as exc:
            return ConsistencyResult(
                False,
                _classify_dl_inconsistency(exc),
                {"stage_a": "inconsistent", "exception": str(exc)},
            )
    else:
        details["stage_a"] = "skipped"

    cycle = _find_cycle_via_kahn(lattice)
    if cycle is not None:
        return ConsistencyResult(
            False,
            Status.C3_VIOLATED,
            {**details, "cycle_witness": cycle},
        )
    details["c3"] = "passed"

    if artefacts is not None:
        offending = _find_c4_violation(lattice, artefacts)
        if offending is not None:
            return ConsistencyResult(
                False,
                Status.C4_VIOLATED,
                {**details, "c4_edge": offending},
            )
        details["c4"] = "passed"
    else:
        details["c4"] = "skipped"

    if configurations is not None:
        # C5: explicit declaration required. Absence of `finite=True` /
        # `is_finite=True` is treated as a violation (matches paper's
        # «проверка явного объявления каждого Q_i как конечного»).
        infinite = [
            i
            for i, q in enumerate(configurations)
            if not (
                getattr(q, "finite", False) is True
                or getattr(q, "is_finite", False) is True
            )
        ]
        if infinite:
            return ConsistencyResult(
                False,
                Status.C5_VIOLATED,
                {**details, "c5": f"failed (infinite Q_i at {infinite})"},
            )
        d = sum(1 for _ in configurations)
        details["c5"] = f"passed ({d} finite parameter domains)"
    else:
        details["c5"] = "skipped"

    if domain_terms is not None and hypothesis_vars is not None:
        # C7: доменное грунтирование. Каждая свободная переменная уравнения
        # гипотезы обязана быть объявленным термином предметной онтологии.
        # Выражается SHACL-формой (sh:in по словарю домена); здесь — в замкнутом
        # мире (как C3–C5), поскольку OWA-рассуждатель отсутствие термина не ловит.
        offending = _find_c7_violation(hypothesis_vars, set(domain_terms))
        if offending is not None:
            h_id, ungrounded = offending
            return ConsistencyResult(
                False,
                Status.C7_VIOLATED,
                {**details, "c7_hypothesis": h_id, "c7_ungrounded": sorted(ungrounded)},
            )
        details["c7"] = "passed (все переменные грунтированы в домене)"
    else:
        details["c7"] = "skipped"

    return ConsistencyResult(True, Status.OK, details)


def _classify_dl_inconsistency(exc: Exception) -> str:
    """Heuristic mapping HermiT exception text -> C1/C2/C6 code."""
    msg = str(exc).lower()
    if "disjoint" in msg or "owl:nothing" in msg or "alldifferent" in msg:
        return Status.C1_VIOLATED
    if "functional" in msg or "is_implemented_by_model" in msg:
        return Status.C2_VIOLATED
    if "range" in msg or "rdfs:range" in msg or "type" in msg:
        return Status.C6_VIOLATED
    return Status.C1_VIOLATED


def _find_cycle_via_kahn(lattice: Mapping[int, Iterable[int]]) -> list | None:
    """Kahn's topological sort; returns None if acyclic, else a witness list."""
    in_deg: dict[int, int] = {}
    for u, succ in lattice.items():
        in_deg.setdefault(u, 0)
        for v in succ:
            in_deg[v] = in_deg.get(v, 0) + 1
    queue = deque(u for u, d in in_deg.items() if d == 0)
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in lattice.get(u, ()):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    if visited == len(in_deg):
        return None
    return [v for v, d in in_deg.items() if d > 0]


def _find_c7_violation(
    hypothesis_vars: Mapping[int, Iterable[str]],
    domain_terms: set,
) -> tuple[int, set] | None:
    """C7: вернуть первую гипотезу, у которой есть переменные, не объявленные
    в словаре предметной онтологии (негрунтированные), иначе None."""
    for h_id, variables in hypothesis_vars.items():
        ungrounded = {v for v in variables if v not in domain_terms}
        if ungrounded:
            return (h_id, ungrounded)
    return None


def _find_c4_violation(
    lattice: Mapping[int, Iterable[int]],
    artefacts: Mapping[int, dict],
) -> tuple[int, int] | None:
    for u, succ in lattice.items():
        for v in succ:
            out_u = set(artefacts.get(u, {}).get("out", ()))
            in_v = set(artefacts.get(v, {}).get("in", ()))
            if not (out_u & in_v):
                return (u, v)
    return None
