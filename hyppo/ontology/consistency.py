"""Two-stage consistency check for virtual experiments (Algorithm 1 of
the planning chapter of the dissertation).

Stage A — semantic: the OWL 2 RL closure (owlrl + rdflib) detects
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

Paper conditions 4 and 5 (§2.4) — the procedural stage of the integrity theorem:
    condition 4 (``_check_condition4_projection``): pi_t(V) != empty for every
        task, via CSP backtracking (the NP-hard admissibility test);
    condition 5 (``_check_condition5_causal_order``, Algorithm ``consistence``):
        the hypothesis dependencies respect the workflow order — a topological
        traversal of W accumulating each task's ancestor hypotheses rejects any
        dependency on a hypothesis of a non-ancestor (later) task.

Public API: ``check_consistency(ve, ontology, lattice)`` returns
``ConsistencyResult`` with status flag and violation details.

Stage A engines
---------------
The default (``stage_a_engine="auto"``) materialises the OWL 2 RL closure with a
pure-Python engine (owlrl + rdflib) and checks consistency (clash detection) over
it — no external DL reasoner is required, matching the RL + IC-semantics design of
the integrity theorem (§2.4): under the ⊥-locality assumption the RL closure is
sufficient. A full OWL 2 DL reasoner is available only opt-in
(``stage_a_engine="hermit"``, requires Java) and is not used by default. The
recognising rules behind C1 (disjointness / owl:Nothing), C2 (functional-property
clash under AllDifferent) and C6 (rdfs:range typing) are in the RL profile;
conditions 1-3 as IC and other CWA-only obligations live in the marker layer
regardless of engine.
"""

from __future__ import annotations

import logging
import subprocess
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


try:
    from owlready2 import OwlReadyJavaError
except ImportError:  # pragma: no cover - environment guard

    class OwlReadyJavaError(Exception):  # type: ignore
        pass


try:
    import owlrl  # noqa: F401
    import rdflib  # noqa: F401

    _OWLRL_AVAILABLE = True
except ImportError:  # pragma: no cover - environment guard
    _OWLRL_AVAILABLE = False


# Exceptions that signal "Java/HermiT is unavailable or failed to launch" rather
# than a genuine ontology inconsistency. owlready2's ``sync_reasoner_hermit``
# shells out to the ``java`` executable via ``subprocess.check_output``; when the
# binary is absent that call raises ``FileNotFoundError`` (a subclass of
# ``OSError``) *outside* owlready2's own ``CalledProcessError`` guard. A present
# but failing JVM surfaces as ``OwlReadyJavaError`` / ``CalledProcessError``.
_JAVA_UNAVAILABLE_EXC = (
    OSError,  # includes FileNotFoundError when `java` is not on PATH
    subprocess.CalledProcessError,
    OwlReadyJavaError,
)

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
    # Paper conditions 4 and 5 (§2.4) — the procedural (structural) stage of the
    # integrity theorem, distinct from the internal C1-C7 structural checks above.
    COND4_VIOLATED = "condition 4: empty task projection pi_t(V) = empty"
    COND5_VIOLATED = "condition 5: causal order violation (Algorithm consistence)"


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
    stage_a_engine: str = "auto",
    artefacts: Mapping[int, dict] | None = None,
    configurations: Iterable[Iterable] | None = None,
    domain_terms: set | None = None,
    hypothesis_vars: Mapping[int, Iterable[str]] | None = None,
    task_hypotheses: Mapping[str, Iterable[int]] | None = None,
    workflow_edges: Iterable[tuple[str, str]] | None = None,
    param_domains: Mapping[int, Iterable] | None = None,
    forbidden_combos: Iterable[Mapping[int, object]] | None = None,
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
        Deprecated alias for engine selection kept for backward compatibility.
        ``False`` skips Stage A (as before); ``True`` (default) defers to
        ``stage_a_engine``. Explicit ``stage_a_engine`` always wins.
    stage_a_engine
        ``"auto"`` (default): try HermiT, degrade to the owlrl OWL 2 RL closure
        when Java is unavailable. ``"hermit"``: force HermiT (no fallback).
        ``"owlrl"``: force the Java-free RL closure. The engine that actually ran
        is reported in ``result.details["stage_a_engine"]``.
    artefacts
        Optional ``{h_id: {"in": set[str], "out": set[str]}}`` for C4 check.
        If omitted, C4 is reported as ``skipped``.
    configurations
        Optional iterable of finite parameter domains ``Q_1, ..., Q_d`` for C5.
        If omitted, C5 is reported as ``skipped`` (assumed True by construction).
    """
    details: dict = {}

    # `run_hermit=False` is the historical "skip Stage A" switch; it only applies
    # when the caller did not request an explicit engine.
    engine = stage_a_engine
    if not run_hermit and stage_a_engine == "auto":
        engine = "skip"

    stage_a_status, stage_a_details = _run_stage_a(ontology, engine)
    details.update(stage_a_details)
    if stage_a_status is not None:
        return ConsistencyResult(False, stage_a_status, details)

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

    # Condition 4 (§2.4): non-empty task projections pi_t(V) = C \\ F.
    if task_hypotheses is not None and param_domains is not None:
        empty_task = _check_condition4_projection(
            task_hypotheses, param_domains, forbidden_combos or ()
        )
        if empty_task is not None:
            return ConsistencyResult(
                False, Status.COND4_VIOLATED, {**details, "cond4_task": empty_task}
            )
        details["cond4"] = "passed (проекции pi_t(V) непусты)"
    else:
        details["cond4"] = "skipped"

    # Condition 5 (§2.4, Algorithm consistence): causal order vs the workflow DAG.
    if task_hypotheses is not None and workflow_edges is not None:
        edges = list(workflow_edges)
        all_tasks = set(task_hypotheses) | {x for e in edges for x in e}
        viol = _check_condition5_causal_order(
            all_tasks, edges, task_hypotheses, lattice
        )
        if viol is not None:
            return ConsistencyResult(
                False, Status.COND5_VIOLATED, {**details, "cond5": viol}
            )
        details["cond5"] = "passed (причинный порядок согласован с потоком работ)"
    else:
        details["cond5"] = "skipped"

    return ConsistencyResult(True, Status.OK, details)


def _run_stage_a(ontology, engine: str) -> tuple[str | None, dict]:
    """Execute Stage A with the requested engine.

    Returns ``(status, details)`` where ``status`` is ``None`` when Stage A is
    consistent or skipped, or a ``Status.*`` code when a violation is detected.
    ``details`` always records which engine handled the stage.
    """
    if engine == "skip":
        return None, {"stage_a": "skipped", "stage_a_engine": "skipped"}

    if engine == "owlrl":
        return _stage_a_owlrl(ontology, "owlrl")

    if engine == "hermit":
        return _stage_a_hermit(ontology, fallback=False)

    if engine == "auto":
        # RL + IC-semantics design (§2.4): the OWL 2 RL closure is the default
        # engine — no external DL reasoner. A full DL reasoner is opt-in via
        # ``stage_a_engine="hermit"``.
        return _stage_a_owlrl(ontology, "owlrl")

    raise ValueError(
        f"unknown stage_a_engine {engine!r}; expected "
        "'auto', 'hermit', 'owlrl' (or run_hermit=False to skip)"
    )


def _stage_a_hermit(ontology, *, fallback: bool) -> tuple[str | None, dict]:
    """Run HermiT; on Java unavailability optionally degrade to owlrl."""
    if not _HERMIT_AVAILABLE:
        if fallback:
            return _stage_a_owlrl(ontology, "owlrl (limited: OWL 2 RL profile)")
        return (
            "stage_a: owlready2 not installed",
            {"stage_a": "skipped", "stage_a_engine": "hermit (unavailable)"},
        )
    try:
        with ontology:
            sync_reasoner_hermit(infer_property_values=False)
        return None, {"stage_a": "passed", "stage_a_engine": "hermit"}
    except OwlReadyInconsistentOntologyError as exc:
        return (
            _classify_dl_inconsistency(exc),
            {
                "stage_a": "inconsistent",
                "stage_a_engine": "hermit",
                "exception": str(exc),
            },
        )
    except _JAVA_UNAVAILABLE_EXC as exc:
        if fallback:
            log.warning(
                "HermiT/Java unavailable (%s); degrading Stage A to owlrl "
                "OWL 2 RL closure",
                exc.__class__.__name__,
            )
            return _stage_a_owlrl(ontology, "owlrl (limited: OWL 2 RL profile)")
        return (
            "stage_a: Java/HermiT unavailable",
            {
                "stage_a": "skipped",
                "stage_a_engine": "hermit (unavailable)",
                "exception": str(exc),
            },
        )


def _stage_a_owlrl(ontology, engine_marker: str) -> tuple[str | None, dict]:
    """Run the Java-free OWL 2 RL closure Stage A."""
    if not _OWLRL_AVAILABLE:
        return (
            "stage_a: owlrl/rdflib not installed",
            {"stage_a": "skipped", "stage_a_engine": "owlrl (unavailable)"},
        )
    world = getattr(ontology, "world", None) or ontology
    consistent, errors = _run_owlrl_stage_a(world)
    if consistent:
        return None, {"stage_a": "passed", "stage_a_engine": engine_marker}
    return (
        _classify_owlrl_inconsistency(errors),
        {
            "stage_a": "inconsistent",
            "stage_a_engine": engine_marker,
            "errors": list(errors),
        },
    )


def _run_owlrl_stage_a(world) -> tuple[bool, list[str]]:
    """Compute the OWL 2 RL closure of ``world`` and detect inconsistency.

    Bridges the owlready2 world to rdflib, copies triples into a fresh
    ``rdflib.Graph`` (the live bridge graph is left unmutated), expands every
    ``owl:AllDifferent`` into pairwise ``owl:differentFrom`` (working around the
    owlrl eq-diff2/3 off-by-one that drops all-different pairs), then runs the
    RL closure. Returns ``(consistent, error_messages)``; a non-empty error list
    means an inconsistency was derived.
    """
    import owlrl
    from rdflib import Graph

    bridge = world.as_rdflib_graph()
    graph = Graph()
    for triple in bridge:
        graph.add(triple)

    _expand_all_different(graph)

    sem = owlrl.OWLRL_Semantics(graph, axioms=True, daxioms=True, rdfs=False)
    sem.closure()
    errors = [str(e) for e in (getattr(sem, "error_messages", None) or [])]
    return (not errors), errors


def _expand_all_different(graph) -> int:
    """Expand ``owl:AllDifferent`` sets into pairwise ``owl:differentFrom``.

    owlrl 7.6.2's eq-diff2/eq-diff3 rules carry an off-by-one
    (``range(i + 1, len(zis) - 1)``) that fails to emit any pair for an
    ``AllDifferent`` list, so functional-clash contradictions go undetected. We
    materialise the pairs directly on the graph before closure. Returns the
    number of ``owl:differentFrom`` triples added.
    """
    from rdflib import OWL, RDF
    from rdflib.collection import Collection

    added = 0
    for adiff in graph.subjects(RDF.type, OWL.AllDifferent):
        members: list = []
        for pred in (OWL.distinctMembers, OWL.members):
            head = graph.value(adiff, pred)
            if head is not None:
                members.extend(list(Collection(graph, head)))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                graph.add((members[i], OWL.differentFrom, members[j]))
                added += 1
    return added


def _classify_owlrl_inconsistency(errors: Iterable[str]) -> str:
    """Map owlrl RL error messages to C1/C2/C6 status codes, conservatively."""
    blob = " ".join(errors).lower()
    if "sameas" in blob or "differentfrom" in blob:
        # functional-property clash forcing two distinct individuals equal
        return Status.C2_VIOLATED
    if "disjoint" in blob or "nothing" in blob:
        return Status.C1_VIOLATED
    if "range" in blob or "datatype" in blob or "type" in blob:
        return Status.C6_VIOLATED
    return Status.C1_VIOLATED


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


# --- Paper conditions 4 and 5 (§2.4): the procedural / structural stage --------

def _csp_satisfiable(hyps, domains, forbidden):
    """Backtracking search for one admissible assignment of finite ``domains`` to
    ``hyps`` avoiding every ``forbidden`` partial combination (a mapping
    ``{h_id: value}`` that must NOT be fully matched). Returns a satisfying
    assignment or ``None``. Realises the (NP-hard) admissibility test of
    condition 4 by exhaustive backtracking with forward pruning, adequate for the
    bounded configuration spaces of the target domains."""
    order = list(hyps)
    clauses = [dict(f) for f in forbidden if f]

    def conflicts(assign):
        return any(
            all(k in assign and assign[k] == v for k, v in clause.items())
            for clause in clauses
        )

    def backtrack(i, assign):
        if i == len(order):
            return dict(assign)
        h = order[i]
        for val in domains.get(h, ()):
            assign[h] = val
            if not conflicts(assign):
                sol = backtrack(i + 1, assign)
                if sol is not None:
                    return sol
            del assign[h]
        return None

    return backtrack(0, {})


def _check_condition4_projection(task_hypotheses, domains, forbidden):
    """Condition 4 (§2.4): ``pi_t(V) != empty`` for every task ``t``. For each task
    the admissible subspace restricted to the task's hypotheses is tested for
    non-emptiness by CSP backtracking (NP-hard in general). Returns the name of a
    task with an empty projection, or ``None`` if all projections are non-empty."""
    for task, hyps in task_hypotheses.items():
        hyps = list(hyps)
        local_dom = {h: list(domains.get(h, ())) for h in hyps}
        if any(len(local_dom[h]) == 0 for h in hyps):
            return task  # a hypothesis with an empty domain empties pi_t
        local_forbidden = [
            {k: v for k, v in clause.items() if k in hyps}
            for clause in (forbidden or ())
            if clause and all(k in hyps for k in clause)
        ]
        if _csp_satisfiable(hyps, local_dom, local_forbidden) is None:
            return task
    return None


def _check_condition5_causal_order(tasks, edges, task_hypotheses, lattice):
    """Condition 5 (§2.4, Algorithm ``consistence``): hypothesis dependencies must
    respect the workflow order — every producer of a hypothesis's input must lie
    in an ancestor task (or the same task). Topologically traverses the workflow
    accumulating ``A_t = union over predecessors t' of (A_{t'} ∪ hyp(t'))``; a
    dependency from a non-ancestor task is a violation. ``lattice`` is the
    derived-by adjacency ``{producer: {consumers}}`` (producers of ``h`` are its
    lattice predecessors). Returns a violation dict or ``None``."""
    succ = {t: set() for t in tasks}
    preds = {t: set() for t in tasks}
    for a, b in edges:
        succ.setdefault(a, set()).add(b)
        preds.setdefault(b, set()).add(a)
        succ.setdefault(b, set())
        preds.setdefault(a, set())

    indeg = {t: len(preds[t]) for t in succ}
    queue = deque(t for t, d in indeg.items() if d == 0)
    topo = []
    while queue:
        t = queue.popleft()
        topo.append(t)
        for nb in succ[t]:
            indeg[nb] -= 1
            if indeg[nb] == 0:
                queue.append(nb)
    if len(topo) != len(succ):
        return None  # cyclic workflow — handled by the acyclicity check

    producers: dict = {}
    for u, children in lattice.items():
        for v in children:
            producers.setdefault(v, set()).add(u)

    task_of = {h: t for t, hs in task_hypotheses.items() for h in hs}
    accum: dict = {}
    for t in topo:
        anc = set()
        for p in preds[t]:
            anc |= accum.get(p, set()) | set(task_hypotheses.get(p, ()))
        accum[t] = anc
        own = set(task_hypotheses.get(t, ()))
        for h in own:
            for dep in producers.get(h, ()):
                if dep not in anc and dep not in own:
                    return {
                        "task": t,
                        "hypothesis": h,
                        "dependency": dep,
                        "dependency_task": task_of.get(dep),
                    }
    return None
