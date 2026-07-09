"""Series of 80 reasoner trials on the 16-node Norne HybridCRM graph
(SVD paper, paragraph "Systematic detection check").

Rule 4: 16 injections (each node invalidated once) -- the HermiT-inferred
    StaleHypothesis set must equal the graph descendants of the source.
Rule 7: 4 runs (using H5, H8, H15, H19) x 16 invalidation candidates --
    DerivedStaleRun must fire iff the candidate is a strict ancestor of
    the hypothesis used by the run.

Edges are derived by Algorithm 1 (HypothesisLattice) from the equations;
the oracle is plain graph reachability (networkx). Every trial launches
the real HermiT reasoner. Results are written to norne_battery_results.json.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx
from owlready2 import destroy_entity, sync_reasoner

from hyppo.coa._base import Equation, Structure
from hyppo.core._base import virtual_experiment_onto as onto
from hyppo.lattice_constructor._base import HypothesisLattice
import hyppo.ontology.core_rules as cr
import hyppo.ontology.provenance as pv

NORNE = [
    ("H1", "I_agg = w_ij * I_j"),
    ("H2", "q_f = a_f*q_f_prev + b_f*I_agg"),
    ("H3", "q_s = a_s*q_s_prev + b_s*I_agg"),
    ("H4", "q_c = w_f*q_f + (1-w_f)*q_s"),
    ("H5", "q_liq_phys = J*q_c + q_prim"),
    ("H6", "q_prim = q_prev*exp(-dt*taup)"),
    ("H7", "l_ml = MLP(x_hist)"),
    ("H8", "l = g*q_liq_phys + (1-g)*l_ml"),
    ("H11", "Sw = Sw_prev + (Winj - l)*dt/Vp"),
    ("H12", "krw = ((Sw-Swc)/(1-Swc-Sor))**nw"),
    ("H12b", "kro = ((1-Sw-Sor)/(1-Swc-Sor))**no"),
    ("H13", "fw = 1/(1 + kro*muw/(krw*muo))"),
    ("H14", "o_p = 1 - fw"),
    ("H15", "o = gw*o_p + (1-gw)*o_m"),
    ("GRP", "J = J0 + dJ_grp"),
    ("H19", "q_oil = l * o"),
]
RUNS = ["H5", "H8", "H15", "H19"]  # runs with 6..15 ancestors


class _Hyp:
    def __init__(self, name, formula):
        self.name = name
        self.structure = Structure([Equation(formula=formula)])

    def __repr__(self):
        return self.name


class _WF:
    def __init__(self, hyps):
        self._t = [[h] for h in hyps]

    def get_tasks(self):
        return self._t


def derive_edges():
    hyps = [_Hyp(n, f) for n, f in NORNE]
    g = HypothesisLattice(hyps, _WF(hyps)).lattice
    return sorted((u.name, v.name) for u, v in g.edges())


def build_trial(nodes, edges, invalid, run_of=None):
    for ind in list(onto.individuals()):
        destroy_entity(ind)
    with onto:
        H = {n: cr.Hypothesis(n) for n in nodes}
        for dep in nodes:
            ancs = [u for u, v in edges if v == dep]
            if ancs:
                H[dep].derived_by = [H[a] for a in ancs]
        H[invalid].is_a.append(cr.InvalidHypothesis)
        R0 = None
        if run_of is not None:
            v0 = pv.HypothesisVersion("v0")
            v0.version_of = [H[run_of]]
            R0 = pv.ExperimentRun("R0")
            R0.uses_hypothesis_version = [v0]
    return H, R0


def main():
    edges = derive_edges()
    nodes = [n for n, _ in NORNE]
    nxg = nx.DiGraph(edges)
    nxg.add_nodes_from(nodes)
    assert nxg.number_of_nodes() == 16 and nxg.number_of_edges() == 18
    assert nx.is_directed_acyclic_graph(nxg)
    assert nx.dag_longest_path_length(nxg) == 10
    print(f"Algorithm 1 graph: 16 nodes, 18 edges, depth 10 -- OK")

    t0 = time.time()
    results = {"rule4": [], "rule7": [], "edges": edges}

    # -- Rule 4: 16 injections ------------------------------------------
    total_classifications = 0
    mismatches = 0
    for i, s in enumerate(nodes, 1):
        H, _ = build_trial(nodes, edges, s)
        sync_reasoner(infer_property_values=True, debug=0)
        stale = {n for n in nodes if cr.StaleHypothesis in H[n].is_a}
        oracle = set(nx.descendants(nxg, s))
        ok = stale == oracle
        mismatches += 0 if ok else 1
        total_classifications += len(stale)
        results["rule4"].append(
            {"source": s, "stale": sorted(stale), "match": ok}
        )
        print(f"  rule4 {i:2d}/16  invalidate {s:4s} -> {len(stale):2d} stale, "
              f"{'OK' if ok else 'MISMATCH'}")
    print(f"Rule 4: {total_classifications} classifications, "
          f"{mismatches} mismatches")

    # -- Rule 7: 4 runs x 16 candidates ---------------------------------
    tp = tn = errors = 0
    for u in RUNS:
        anc = set(nx.ancestors(nxg, u))
        for x in nodes:
            H, R0 = build_trial(nodes, edges, x, run_of=u)
            sync_reasoner(infer_property_values=True, debug=0)
            got = pv.DerivedStaleRun in R0.is_a
            exp = x in anc
            if got != exp:
                errors += 1
                verdict = "ERROR"
            else:
                verdict = "TP" if got else "TN"
                tp += 1 if got else 0
                tn += 0 if got else 1
            results["rule7"].append(
                {"run_uses": u, "invalid": x, "derived_stale": got,
                 "expected": exp, "verdict": verdict}
            )
        print(f"  rule7 run uses {u:4s} ({len(anc):2d} ancestors) done")
    print(f"Rule 7: {tp} true-positive, {tn} true-negative, {errors} errors "
          f"(of 64 pairs)")

    dt = time.time() - t0
    results["summary"] = {
        "rule4_classifications": total_classifications,
        "rule4_mismatches": mismatches,
        "rule7_tp": tp, "rule7_tn": tn, "rule7_errors": errors,
        "trials": 80, "wall_seconds": round(dt, 1),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "norne_battery_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n80 trials in {dt:.0f} s -> {out}")


if __name__ == "__main__":
    main()
