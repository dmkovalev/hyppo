"""Shared machinery of the chapter-4 domain experiments (BGM, HCP, petro).

Everything here is the *invariant core* applied to a domain layer
(``hyppo.adapters.*``): Algorithm 1 (``GraphConstructor`` and, as an
independent cross-check, ``HypothesisGraph.build`` with declared inputs and the
У0 matching), the consistency check (``check_consistency``: OWL 2 RL Stage A,
acyclicity, C4, finiteness, C7 domain grounding, conditions 4 and 5) together
with the IC conditions 1-3 over the RL closure, the ABox of the experiment in
the shared ontology, the OWL reasoner and the marker layer, and the mapping of
inferred classes to the 19 rules of tab:owl_rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
from owlready2 import destroy_entity, sync_reasoner

import hyppo.ontology.core_rules as cr
import hyppo.ontology.lifecycle as lc
import hyppo.ontology.model_compatibility as mc
import hyppo.ontology.multi_experiment as me
import hyppo.ontology.provenance as pv
import hyppo.ontology.quality_gates as qg
import hyppo.ontology.workflow_validation as wv
from hyppo.adapters._spec import FormulaHypothesis, HypothesisSpec, TaskWorkflow
from hyppo.coa import HypothesisGraph
from hyppo.core._base import (
    Configuration,
    Variable,
    VirtualExperiment,
    Workflow,
    virtual_experiment_onto as onto,
)
from hyppo.graph_constructor._base import GraphConstructor
from hyppo.ontology.consistency import check_consistency, rl_materialize
from hyppo.ontology.markers import apply_markers

#: inferred (RL-materialised) class -> rule number of tab:owl_rules
RULE_OF_CLASS = {
    "PhysicsHypothesis": 1, "DataDrivenHypothesis": 1, "HybridHypothesis": 1,
    "CompleteExperiment": 2,
    "StaleHypothesis": 4,
    "HypothesisWithStaleAncestor": 7, "DerivedStaleRun": 7,
    "ObsoleteVersion": 8,
    "ConflictingTask": 10,
    "PromisingRoute": 12,
    "FormatMismatch": 14,
    "BlockingDeprecation": 16,
    "ConflictingHypothesis": 17,
    "PreferredHypothesis": 19,
}
#: procedural (IC) marker -> rule number
MARKER_RULE = {"CompleteExperiment": 2, "OrphanHypothesis": 9, "PrunableSubtree": 11,
               "SharedHypothesis": 13, "DatasetNotInConfig": 15, "FreshHypothesis": 18}
RULE_NAMES = {
    1: "classification by model type", 2: "tuple completeness (condition 1)",
    3: "dependency via derived_by", 4: "cascade invalidation", 5: "transitivity of derived_by",
    6: "domain ontology hierarchy/disjointness", 7: "hidden staleness of runs",
    8: "version tracking", 9: "orphan hypotheses", 10: "tasks with rival hypotheses",
    11: "prunable subtrees", 12: "promising routes", 13: "shared hypotheses",
    14: "data-format mismatch", 15: "dataset missing in configuration",
    16: "blocking deprecation", 17: "conflict of active rivals", 18: "fresh after recompute",
    19: "preferred surviving rival",
}
MODEL_CLASS = {"physics": cr.PhysicsModel, "data": cr.DataDrivenModel, "hybrid": cr.HybridModel}


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
@dataclass
class DerivedGraph:
    nodes: list[str]
    edges: list[tuple[str, str]]
    competes: list[tuple[str, str]]
    order_conflicts: list[tuple[str, str]]
    depth: int
    hypothesis_graph_agrees: bool
    signatures: dict[str, dict[str, list[str]]]
    graph: nx.DiGraph = field(repr=False)


def derive_graph(specs: list[HypothesisSpec], tasks: list[list[str]],
                 task_edges: list[tuple[int, int]]) -> DerivedGraph:
    """Run Algorithm 1 on the hypothesis structures.

    ``GraphConstructor.build_graph`` decides every pair with
    ``tau(h_a) < tau(h_b)`` by ``Out(h_a) ∩ In(h_b) != ∅`` and no counter-supply
    (definition def:derived). ``HypothesisGraph.build`` (signatures from the
    declared inputs, У0 by one Hopcroft-Karp matching per hypothesis) is run on
    the same input as an independent implementation; both edge sets must agree.
    """
    by = {s.name: s for s in specs}
    for s in specs:
        s.check_u0()
    order = [n for t in tasks for n in t]
    if sorted(order) != sorted(by):
        raise ValueError("every hypothesis must belong to exactly one task")
    fh = {n: FormulaHypothesis(by[n]) for n in order}
    wf = TaskWorkflow([[fh[n] for n in t] for t in tasks], task_edges)
    gc = GraphConstructor([fh[n] for n in order], wf)
    edges = sorted((u.name, v.name) for u, v in gc.hyp_graph.edges())
    competes = sorted({tuple(sorted((h.name, c.name)))
                       for h in gc.hypotheses for c in gc.competes(h)})
    conflicts = sorted((a.name, b.name) for a, b in gc.order_conflicts())

    hg = HypothesisGraph()
    idx = {n: hg.add(by[n].equation_vars(), inputs=by[n].inputs) for n in order}
    for a, b in task_edges:
        for u in tasks[a]:
            for v in tasks[b]:
                hg.connect(idx[u], idx[v])
    hg.build()
    hg_edges = sorted((order[i], order[j]) for i, j in hg.hyp_graph)

    G = nx.DiGraph()
    G.add_nodes_from(order)
    G.add_edges_from(edges)
    return DerivedGraph(
        nodes=order, edges=edges, competes=competes, order_conflicts=conflicts,
        depth=nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else -1,
        hypothesis_graph_agrees=hg_edges == edges,
        signatures={n: {"in": sorted(by[n].inputs), "out": sorted(by[n].outputs)} for n in order},
        graph=G,
    )


def edge_diff(derived, expected) -> dict[str, list]:
    d, e = set(map(tuple, derived)), set(map(tuple, expected))
    return {"common": sorted(d & e), "only_derived": sorted(d - e), "only_expected": sorted(e - d)}


def cascade(G: nx.DiGraph, changed: str) -> list[str]:
    """Algorithm 4 recompute set P_ne when only ``changed`` is not cached."""
    from hyppo.coa.graph import plan_cascade

    nodes = list(G.nodes)
    return sorted(plan_cascade(nodes, list(G.edges), set(nodes) - {changed}))


# ---------------------------------------------------------------------------
# Consistency (conditions 1-5, C3-C7, Stage A)
# ---------------------------------------------------------------------------
class FiniteDomain(list):
    finite = True


def structural_check(specs, tasks, task_edges, edges, variables, *, engine="auto"):
    """``check_consistency`` with every argument the experiment supplies."""
    by = {s.name: s for s in specs}
    order = [n for t in tasks for n in t]
    idx = {n: i for i, n in enumerate(order)}
    graph = {idx[n]: set() for n in order}
    for u, v in edges:
        graph[idx[u]].add(idx[v])
    res = check_consistency(
        None, onto, graph, stage_a_engine=engine,
        artefacts={idx[n]: {"in": set(by[n].inputs), "out": set(by[n].outputs)} for n in order},
        configurations=[FiniteDomain(range(by[n].n_variants)) for n in order],
        domain_terms=set(variables),
        hypothesis_vars={idx[n]: set(by[n].variables) for n in order},
        task_hypotheses={f"t{k + 1}": [idx[n] for n in t] for k, t in enumerate(tasks)},
        workflow_edges=[(f"t{a + 1}", f"t{b + 1}") for a, b in task_edges],
        param_domains={idx[n]: list(range(by[n].n_variants)) for n in order},
    )
    details = dict(res.details)
    if "cond5" in details and isinstance(details["cond5"], dict):
        v = details["cond5"]
        details["cond5"] = {**v, "hypothesis": order[v["hypothesis"]],
                            "dependency": order[v["dependency"]]}
    return {"ok": res.ok, "status": res.status, "details": details}


def ic_conditions_1_3(ve, tasks, hypotheses) -> dict:
    """Conditions 1-3 in IC semantics over the RL closure (closed world):
    every slot of the tuple has a *named* filler, every workflow task has a
    named hypothesis, every hypothesis a named implementing model."""
    from rdflib import RDF, URIRef

    graph, errors = rl_materialize(onto.world)
    typ = lambda cls: URIRef(cls.iri)  # noqa: E731

    def named(subj, prop, cls):
        return [o for o in graph.objects(URIRef(subj.iri), URIRef(prop.iri))
                if (o, RDF.type, typ(cls)) in graph]

    from hyppo.core._base import (
        Hypothesis,
        Model,
        has_for_configuration,
        has_for_hypothesis,
        has_for_model,
        has_for_workflow,
        is_implemented_by_model,
    )

    slots = {
        "O": named(ve, cr.has_for_ontology, cr.DomainOntology),
        "H": named(ve, has_for_hypothesis, Hypothesis),
        "M": named(ve, has_for_model, Model),
        "W": named(ve, has_for_workflow, Workflow),
        "C": named(ve, has_for_configuration, Configuration),
    }
    unrealised = [h.name for h in hypotheses if not named(h, is_implemented_by_model, Model)]
    slots_ok = all(slots.values())
    empty_tasks = [t.name for t in tasks if not named(t, wv.hasHypothesis, Hypothesis)]
    return {
        "rl_errors": errors,
        "condition1": {"ok": slots_ok and not unrealised,
                       "slots": {k: len(v) for k, v in slots.items()},
                       "R_unrealised": unrealised},
        "condition2": {"ok": not empty_tasks, "empty_tasks": empty_tasks},
        "condition3": {"ok": not unrealised, "unrealised": unrealised},
    }


# ---------------------------------------------------------------------------
# ABox of the experiment
# ---------------------------------------------------------------------------
def reset_abox() -> None:
    for ind in list(onto.individuals()):
        destroy_entity(ind)


@dataclass
class ABox:
    hyp: dict
    model: dict
    task: list
    version: dict
    run: dict
    ve: object
    domain: object
    variables: dict


def build_abox(prefix: str, specs, tasks, edges, *, domain_cls, quantity_cls: dict,
               vocabulary: dict, dataset: str, needs_dataset=(), experiment_id=None) -> ABox:
    """Assert what the experiment supplies: domain ontology, hypotheses (Active)
    with derived_by edges from Algorithm 1, implementing models, workflow tasks,
    the tuple slots, grounded variables, versions v0 and one run per hypothesis,
    and the dataset of the configuration."""
    by = {s.name: s for s in specs}
    with onto:
        domain = domain_cls(f"{prefix}_domain_ontology")
        H = {n: cr.Hypothesis(f"{prefix}_{n}") for n in by}
        for h in H.values():
            h.is_a.append(lc.ActiveHypothesis)
        for v in by:
            ancs = [u for u, w in edges if w == v]
            if ancs:
                H[v].derived_by = [H[a] for a in ancs]
        M = {}
        for n, s in by.items():
            M[n] = MODEL_CLASS[s.model_kind](f"{prefix}_m_{n}")
            H[n].is_implemented_by_model = M[n]
            if s.data_out == "timeseries":
                M[n].is_a.append(mc.TimeSeriesProducer)
            if s.data_in == "graph":
                M[n].is_a.append(mc.GraphConsumer)
        for u, v in edges:  # the model of a producer feeds the consumer's model
            M[u].feedsInto.append(M[v])
        T = []
        for k, t in enumerate(tasks):
            task = wv.WorkflowTask(f"{prefix}_task{k + 1}")
            task.hasHypothesis = [H[n] for n in t]
            T.append(task)
        V = {}
        for var, (qcls, _meaning) in vocabulary.items():
            ind = Variable(f"{prefix}_var_{var}")
            ind.is_a.append(quantity_cls[qcls])
            ind.has_for_varname = var
            V[var] = ind
        wf = Workflow(f"{prefix}_workflow")
        cfg = Configuration(f"{prefix}_configuration")
        ve = VirtualExperiment(f"{prefix}_ve")
        ve.has_for_ontology = [domain]
        ve.has_for_workflow = [wf]
        ve.has_for_hypothesis = list(H.values())
        ve.has_for_model = list(M.values())
        ve.has_for_configuration = [cfg]
        exp = me.Experiment(experiment_id or f"{prefix}_experiment")
        exp.usesHypothesis = list(H.values())
        ds = mc.Dataset(f"{prefix}_dataset_{dataset}")
        mcfg = mc.ModelConfig(f"{prefix}_model_config")
        mcfg.hasAvailableDataset = [ds]
        for n in needs_dataset:
            M[n].is_a.append(mc.ModelWithDatasetNeed)
            M[n].usedInConfig = [mcfg]
        ver, run = {}, {}
        for n in by:
            ver[n] = pv.HypothesisVersion(f"{prefix}_{n}_v0")
            ver[n].version_of = [H[n]]
            run[n] = pv.ExperimentRun(f"{prefix}_run_{n}")
            run[n].uses_hypothesis_version = [ver[n]]
    return ABox(H, M, T, ver, run, ve, domain, V)


# ---------------------------------------------------------------------------
# Reasoning and rule instrumentation
# ---------------------------------------------------------------------------
def _classes(ind):
    return {c.name for c in ind.is_a if hasattr(c, "name")}


def _derived_by_pairs():
    pairs = set()
    for h in cr.Hypothesis.instances():
        for a in h.derived_by:
            pairs.add((a.name, h.name))
    return pairs


def reason(*, with_markers: bool = True) -> dict:
    """Run the marker layer (rules 2, 9, 11, 13, 15) and the OWL reasoner
    (HermiT) once; return newly inferred classes, inferred derived_by facts,
    fired rule numbers and the wall time of the reasoner call."""
    before = {i.name: _classes(i) for i in onto.individuals()}
    asserted = _derived_by_pairs()
    report = apply_markers(onto, run_hermit=False) if with_markers else None
    t0 = time.perf_counter()
    sync_reasoner(infer_property_values=True, debug=0)
    dt = time.perf_counter() - t0
    inferred, fired = {}, set()
    for i in onto.individuals():
        new = _classes(i) - before.get(i.name, set())
        if new:
            inferred[i.name] = sorted(new)
            for c in new:
                if c in RULE_OF_CLASS:
                    fired.add(RULE_OF_CLASS[c])
                elif c in MARKER_RULE:
                    fired.add(MARKER_RULE[c])
    transitive = sorted(_derived_by_pairs() - asserted)
    if transitive:
        fired.add(5)
    if asserted:
        fired.add(3)
    markers = {}
    if report is not None:
        markers = {"rule2": len(report.rule2_marked), "rule9": len(report.rule9_marked),
                   "rule11": len(report.rule11_marked), "rule13": len(report.rule13_marked),
                   "rule15": len(report.rule15_marked)}
    return {"inferred": inferred, "inferred_derived_by": transitive, "markers": markers,
            "rules_fired": sorted(fired), "reasoner_seconds": round(dt, 3)}


def stale_names(abox: ABox) -> list[str]:
    return sorted(n for n, h in abox.hyp.items() if cr.StaleHypothesis in h.is_a)


def derived_stale_runs(abox: ABox) -> list[str]:
    return sorted(n for n, r in abox.run.items() if pv.DerivedStaleRun in r.is_a)


def ontology_size() -> dict:
    """Classes of the shared ontology and individuals of the current ABox."""
    return {"classes": len(list(onto.classes())), "individuals": len(list(onto.individuals())),
            "triples": len(onto.world.as_rdflib_graph())}


def time_rl_materialisation(repeats: int = 5) -> dict:
    """Median wall time of the OWL 2 RL closure (``rl_materialize``) of the
    current ABox -- the materialisation the integrity control relies on."""
    ts = []
    n_triples = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        g, _errors = rl_materialize(onto.world)
        ts.append(time.perf_counter() - t0)
        n_triples = len(g)
    ts.sort()
    return {"median_s": round(ts[len(ts) // 2], 3), "min_s": round(ts[0], 3),
            "max_s": round(ts[-1], 3), "closure_triples": n_triples, "repeats": repeats}


def time_hermit(repeats: int = 3) -> dict:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sync_reasoner(infer_property_values=True, debug=0)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_s": round(ts[len(ts) // 2], 3), "repeats": repeats}


def rule_table(fired_by_scenario: dict[str, list[int]]) -> dict[int, list[str]]:
    """rule -> scenarios in which it fired."""
    out = {r: [] for r in range(1, 20)}
    for sc, rules in fired_by_scenario.items():
        for r in rules:
            out[r].append(sc)
    return out
