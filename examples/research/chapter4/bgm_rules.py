#!/usr/bin/env python
"""Chapter 4 -- which ontology rules fire on the BGM experiment (tab:core_coverage).

The BGM demo (bgm_cascade.py) exercises only the hypothesis graph and the
planner: the 10 hypotheses carry no equations (``g.add([])``), the 12
``derived_by`` edges are given explicitly (``g.connect``), and no reasoner is
called. This script instruments the ontology layer on the same BGM graph: it
loads the 10 hypotheses and 12 edges into the shared ontology of the complex
(all rule modules registered at import of ``hyppo.core``), asserts exactly what
the BGM experiment supplies plus one invalidation and two runs, runs the OWL
reasoner and records every class newly inferred for every individual.
Inferred classes are mapped to the rule numbers of tab:owl_rules (ch. 3).

Scenarios (each is a fresh ABox):
  * invalidate IMF;  run R_dyn uses dynamical_self_consistency@v0,
                     run R_trk uses evolutionary_tracks@v0;
  * invalidate SFR;  same runs.
Oracle: StaleHypothesis == graph descendants of the invalidated node,
DerivedStaleRun(R) <=> the run's hypothesis is a descendant.

Usage:  PYTHONPATH=. python examples/research/chapter4/bgm_rules.py
Output: examples/research/chapter4/results/bgm_rules.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import networkx as nx
from owlready2 import destroy_entity, sync_reasoner

from hyppo.core._base import virtual_experiment_onto as onto
import hyppo.ontology.core_rules as cr
import hyppo.ontology.provenance as pv

from examples.research.chapter4.bgm_cascade import EDGES, NODES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "bgm_rules.json"

# inferred class name -> rule number (tab:owl_rules; code rule 30/31 = 18/19)
RULE_OF = {
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
RUNS = {"R_dyn": "dynamical_self_consistency", "R_trk": "evolutionary_tracks"}


def build(invalid):
    for ind in list(onto.individuals()):
        destroy_entity(ind)
    with onto:
        H = {n: cr.Hypothesis(n) for n in NODES}
        for v in NODES:
            ancs = [u for u, w in EDGES if w == v]
            if ancs:
                H[v].derived_by = [H[a] for a in ancs]
        H[invalid].is_a.append(cr.InvalidHypothesis)
        R = {}
        for rname, hname in RUNS.items():
            ver = pv.HypothesisVersion(f"{hname}_v0")
            ver.version_of = [H[hname]]
            R[rname] = pv.ExperimentRun(rname)
            R[rname].uses_hypothesis_version = [ver]
    return H, R


def classes_of(ind):
    return {c.name for c in ind.is_a if hasattr(c, "name")}


def main():
    nxg = nx.DiGraph(EDGES)
    nxg.add_nodes_from(NODES)
    assert nxg.number_of_nodes() == 10 and nxg.number_of_edges() == 12
    out = {"nodes": NODES, "edges": EDGES, "scenarios": []}
    fired_all = set()
    t0 = time.time()
    for src in ("IMF", "SFR"):
        H, R = build(src)
        before = {i.name: classes_of(i) for i in onto.individuals()}
        sync_reasoner(infer_property_values=True, debug=0)
        inferred = {}
        for i in onto.individuals():
            new = classes_of(i) - before.get(i.name, set())
            if new:
                inferred[i.name] = sorted(new)
        stale = sorted(n for n in NODES if cr.StaleHypothesis in H[n].is_a)
        oracle = sorted(nx.descendants(nxg, src))
        runs = {r: pv.DerivedStaleRun in R[r].is_a for r in RUNS}
        runs_exp = {r: RUNS[r] in oracle for r in RUNS}
        fired = sorted({RULE_OF[c] for cs in inferred.values() for c in cs if c in RULE_OF})
        unmapped = sorted({c for cs in inferred.values() for c in cs if c not in RULE_OF})
        fired_all |= set(fired)
        out["scenarios"].append({
            "invalid": src, "stale": stale, "oracle_descendants": oracle,
            "rule4_match": stale == oracle,
            "derived_stale_runs": runs, "expected_runs": runs_exp,
            "rule7_match": runs == runs_exp,
            "inferred_classes": inferred, "rules_fired": fired,
            "unmapped_inferred_classes": unmapped,
        })
        print(f"invalidate {src}: stale={stale} match={stale == oracle}; "
              f"runs={runs} match={runs == runs_exp}; rules fired={fired}; "
              f"other inferred={unmapped}")
    out["rules_fired_union"] = sorted(fired_all)
    out["rules_not_fired"] = sorted(set(range(1, 20)) - fired_all)
    out["note"] = ("BGM supplies only hypotheses and derived_by edges; no models, "
                   "workflow tasks, configurations, domain ontology, lifecycle "
                   "states or competes pairs are asserted in bgm_cascade.py, so "
                   "rules depending on them have nothing to fire on.")
    out["wall_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rules fired (union): {out['rules_fired_union']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
