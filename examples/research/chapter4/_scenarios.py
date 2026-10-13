"""Life-cycle scenarios of the chapter-4 domain experiments (shared by BGM and
HCP). Each scenario starts from a fresh ABox built by the domain's ``build``
callback, applies one step of the life cycle (or one injected defect) and
reports what the reasoner and the marker layer inferred, mapped to the rules
of tab:owl_rules."""

from __future__ import annotations

import networkx as nx
from owlready2 import OwlReadyInconsistentOntologyError, sync_reasoner

import hyppo.ontology.core_rules as cr
import hyppo.ontology.lifecycle as lc
import hyppo.ontology.model_compatibility as mc
import hyppo.ontology.multi_experiment as me
import hyppo.ontology.provenance as pv
import hyppo.ontology.quality_gates as qg
import hyppo.ontology.workflow_validation as wv
from hyppo.core._base import virtual_experiment_onto as onto

from examples.research.chapter4._domain_run import (
    ABox,
    DerivedGraph,
    cascade,
    derived_stale_runs,
    ic_conditions_1_3,
    reason,
    reset_abox,
    stale_names,
)


def _drop_active_family(h) -> None:
    """Lifecycle transition out of Active: retract Active and the classes
    inferred under it (ConflictingHypothesis, PreferredHypothesis)."""
    for c in (lc.ConflictingHypothesis, lc.PreferredHypothesis, lc.ActiveHypothesis):
        while c in h.is_a:
            h.is_a.remove(c)


def nominal_scenario(build) -> tuple[ABox, dict]:
    """Fresh ABox of the experiment, markers + reasoner once."""
    reset_abox()
    ab = build()
    return ab, reason()


def change_scenario(build, G: nx.DiGraph, src: str) -> dict:
    """A hypothesis is revised: version v0 is superseded by v1 and the old
    result is invalidated. The reasoner must mark exactly the descendants stale
    (rule 4) and the runs of the descendants derived-stale (rule 7); the
    recompute plan of Algorithm 4 must equal {src} + stale. After the plan is
    executed the recomputed hypotheses are refreshed (rule 18)."""
    reset_abox()
    ab = build()
    with onto:
        v1 = pv.HypothesisVersion(f"{ab.hyp[src].name}_v1")
        v1.version_of = [ab.hyp[src]]
        ab.version[src].superseded_by = [v1]
        ab.hyp[src].is_a.append(cr.InvalidHypothesis)
    r = reason(with_markers=False)
    stale = stale_names(ab)
    oracle = sorted(nx.descendants(G, src))
    runs = derived_stale_runs(ab)
    p_ne = cascade(G, src)
    obsolete = sorted(v.name for v in pv.ObsoleteVersion.instances())
    # execute the plan, then close the repair cycle
    with onto:
        ab.hyp[src].is_a.remove(cr.InvalidHypothesis)
        ab.run[src].uses_hypothesis_version = [v1]
    for n in stale:
        lc.refresh_hypothesis(ab.hyp[n])
    fresh = sorted(n for n, h in ab.hyp.items() if lc.FreshHypothesis in h.is_a)
    rules = set(r["rules_fired"]) | ({18} if fresh else set())
    return {
        "changed": src, "stale": stale, "oracle_descendants": oracle,
        "rule4_match": stale == oracle,
        "derived_stale_runs": runs, "rule7_match": runs == oracle,
        "obsolete_versions": obsolete,
        "plan_P_ne": p_ne, "plan_matches_marking": p_ne == sorted({src, *stale}),
        "fresh_after_recompute": fresh,
        "rules_fired": sorted(rules),
        "reasoner_seconds": r["reasoner_seconds"],
    }


def rival_scenario(build, graph: DerivedGraph, metric: dict[str, float], *,
                   lower_is_better: bool = True) -> dict:
    """Competing hypotheses (Out collision found by Algorithm 1's producer map)
    are all active in one task (rules 10, 17); the ranking by the published
    metric refutes the losers (REFUTED -> InvalidHypothesis), which are
    deprecated while active hypotheses still derive from them (rule 16); the
    survivor is confirmed (rule 19); quality gates mark the routes from the
    winner (rule 12) and the cascade marks the dependents of the losers stale
    (rule 4)."""
    reset_abox()
    ab = build()
    with onto:
        for a, b in graph.competes:
            ab.hyp[a].competes.append(ab.hyp[b])
    phase1 = reason(with_markers=True)
    conflicting = sorted(n for n, h in ab.hyp.items() if lc.ConflictingHypothesis in h.is_a)
    conflicting_tasks = sorted(t.name for t in ab.task if wv.ConflictingTask in t.is_a)
    ranked = sorted(metric, key=lambda k: metric[k], reverse=not lower_is_better)
    winner, losers = ranked[0], ranked[1:]
    # phase 2 is asserted on a fresh ABox: owlready2 keeps the phase-1
    # inferences (e.g. ConflictingHypothesis ⊑ Active) in its inference store,
    # which would clash with the lifecycle transition Active -> Deprecated.
    reset_abox()
    ab = build()
    with onto:
        for a, b in graph.competes:
            ab.hyp[a].competes.append(ab.hyp[b])
        for n in losers:
            cr.apply_epistemic_status(ab.hyp[n], "REFUTED")
            _drop_active_family(ab.hyp[n])
            ab.hyp[n].is_a.append(lc.DeprecatedHypothesis)
            ab.hyp[n].is_a.append(qg.LowQuality)
        ab.hyp[winner].is_a.append(qg.HighQuality)
        for u, v in graph.edges:
            ab.hyp[u].hasDescendant.append(ab.hyp[v])
    phase2 = reason(with_markers=True)
    return {
        "competes_pairs": [list(p) for p in graph.competes],
        "ranking": [{"hypothesis": n, "metric": metric[n]} for n in ranked],
        "phase1_active_rivals": {"conflicting_hypotheses": conflicting,
                                 "conflicting_tasks": conflicting_tasks,
                                 "rules_fired": phase1["rules_fired"]},
        "phase2_after_ranking": {
            "preferred": sorted(n for n, h in ab.hyp.items() if lc.PreferredHypothesis in h.is_a),
            "blocking_deprecation": sorted(n for n, h in ab.hyp.items()
                                           if lc.BlockingDeprecation in h.is_a),
            "promising_routes": sorted(n for n, h in ab.hyp.items() if qg.PromisingRoute in h.is_a),
            "prunable_subtrees": sorted(n for n, h in ab.hyp.items() if qg.PrunableSubtree in h.is_a),
            "stale": stale_names(ab),
            "rules_fired": phase2["rules_fired"],
        },
        "rules_fired": sorted(set(phase1["rules_fired"]) | set(phase2["rules_fired"])),
    }


def fault_orphan(build, orphan_name: str) -> dict:
    """Rule 9: a hypothesis declared in the experiment but placed in no task."""
    reset_abox()
    ab = build()
    with onto:
        h = cr.Hypothesis(orphan_name)
        h.is_a.append(lc.DraftHypothesis)
        ab.ve.has_for_hypothesis.append(h)
    r = reason(with_markers=True)
    return {"orphans": sorted(x.name for x in wv.OrphanHypothesis.instances()),
            "rules_fired": r["rules_fired"]}


def fault_dataset_missing(build) -> dict:
    """Rule 15: the configuration does not provide the dataset a model needs."""
    reset_abox()
    build()
    with onto:
        for cfg in mc.ModelConfig.instances():
            cfg.hasAvailableDataset = []
    r = reason(with_markers=True)
    return {"models_without_dataset": sorted(m.name for m in mc.DatasetNotInConfig.instances()),
            "rules_fired": r["rules_fired"]}


def fault_shared(build, shared: str, other_experiment: str) -> dict:
    """Rule 13: a second experiment reuses one of the hypotheses."""
    reset_abox()
    ab = build()
    with onto:
        e2 = me.Experiment(other_experiment)
        e2.usesHypothesis = [ab.hyp[shared]]
    r = reason(with_markers=True)
    return {"shared": sorted(x.name for x in me.SharedHypothesis.instances()),
            "rules_fired": r["rules_fired"]}


def fault_disjoint_domain(build, other_domain_cls) -> dict:
    """Rule 6: the domain ontology individual is also typed with a disjoint
    domain ontology class -- the reasoner must report an inconsistency."""
    reset_abox()
    ab = build()
    with onto:
        ab.domain.is_a.append(other_domain_cls)
    try:
        sync_reasoner(infer_property_values=True, debug=0)
        detected = False
    except OwlReadyInconsistentOntologyError:
        detected = True
    reset_abox()
    return {"inconsistency_detected": detected, "rules_fired": [6] if detected else []}


def fault_format_mismatch(build, producer: str, consumer: str) -> dict:
    """Rule 14: a time-series producer is wired directly into a graph consumer."""
    reset_abox()
    ab = build()
    with onto:
        ab.model[producer].feedsInto.append(ab.model[consumer])
    r = reason(with_markers=False)
    return {"format_mismatch": sorted(m.name for m in mc.FormatMismatch.instances()),
            "rules_fired": r["rules_fired"]}


def fault_ic(build, drop_model: str, empty_task: int) -> dict:
    """IC conditions 2 and 3 on a defective tuple: an empty workflow task and a
    hypothesis without an implementing model."""
    reset_abox()
    ab = build()
    with onto:
        ab.hyp[drop_model].is_implemented_by_model = None
        ab.task[empty_task].hasHypothesis = []
    ic = ic_conditions_1_3(ab.ve, ab.task, list(ab.hyp.values()))
    reset_abox()
    return {k: ic[k] for k in ("condition1", "condition2", "condition3")}


def fault_invariant(build, G: nx.DiGraph, hypothesis: str, check) -> dict:
    """Layer 3: a numeric domain invariant fails for one hypothesis; the
    violation becomes InvalidHypothesis and cascades (rule 4)."""
    from pydantic import ValidationError

    reset_abox()
    ab = build()
    try:
        check()
        violated, message = False, ""
    except ValidationError as exc:
        violated, message = True, exc.errors()[0]["msg"]
    if violated:
        with onto:
            ab.hyp[hypothesis].is_a.append(cr.InvalidHypothesis)
    r = reason(with_markers=False)
    stale = stale_names(ab)
    return {"hypothesis": hypothesis, "violated": violated, "message": message,
            "stale": stale, "matches_descendants": stale == sorted(nx.descendants(G, hypothesis)),
            "rules_fired": r["rules_fired"]}
