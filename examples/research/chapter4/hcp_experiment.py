#!/usr/bin/env python
"""Chapter 4, §4.3 -- the HCP functional-connectivity experiment run through the core.

Domain layer: :mod:`hyppo.adapters.hcp_adapter` (3 hypothesis structures with
declared inputs, the 3-task workflow, BrainOntology, executable NumPy models of
the 4 x 4 x 3 = 48 configurations, numeric invariants). This script applies
the invariant core:

1. Algorithm 1 derives the graph atlas -> conn -> group from
   ``Out(h_a) ∩ In(h_b)`` (``GraphConstructor``, cross-checked by
   ``HypothesisGraph.build``) and compares it with the dissertation;
2. Algorithm 4 plans the recomputation on a change of every hypothesis;
3. the exhaustive enumeration of the 48 configurations is executed on a seeded
   surrogate cohort (HCP S1200 is DUA-restricted) with reuse of the results of
   the sources (definition def:12), every output passing the numeric invariants;
4. the consistency check, the reasoner and the marker layer run on the
   life-cycle scenarios (nominal, revisions with recompute, the rival GP
   connectivity model ranked by AIC) and on injected defects;
5. the published per-region decisions of §4.3 are reproduced from the reported
   statistics (sign test with Holm correction, Bayesian factor).

Usage:  PYTHONPATH=. python examples/research/chapter4/hcp_experiment.py
Output: examples/research/chapter4/results/hcp_experiment.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hyppo.adapters import hcp_adapter as N  # noqa: E402
from hyppo.ontology.core_rules import OilDomainOntology  # noqa: E402

from examples.research.chapter4 import _scenarios as S  # noqa: E402
from examples.research.chapter4._domain_run import (  # noqa: E402
    RULE_NAMES,
    build_abox,
    cascade,
    derive_graph,
    edge_diff,
    ic_conditions_1_3,
    ontology_size,
    rule_table,
    structural_check,
    time_hermit,
    time_rl_materialisation,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "hcp_experiment.json"

#: Published per-region statistics (tab:sign_test, tab:bayesian), alpha = 0.05.
REGION_STATS = {
    "Precentral Gyrus": {"p": 0.0045, "eta": 29.0, "control": False},
    "Postcentral Gyrus": {"p": 0.0003, "eta": 54.0, "control": False},
    "Planum Polare": {"p": 0.08, "eta": 7.0, "control": True},
}
ALPHA = 0.05


def derived_graph():
    return derive_graph(N.HCP_HYPOTHESES, N.HCP_TASKS, N.HCP_TASK_EDGES)


def builder(specs, tasks, edges, prefix="hcp"):
    def build():
        return build_abox(prefix, specs, tasks, edges, domain_cls=N.BrainOntology,
                          quantity_cls=N.NEURO_QUANTITY, vocabulary=N.HCP_VARIABLES,
                          dataset="HCP_S1200", needs_dataset=("h_atlas",))
    return build


def enumerate_configurations(seed: int = 0) -> dict:
    """Execute all 48 configurations in lexicographic order with reuse of the
    cached results of the sources: a result of h is reused iff the projection
    of the configuration on h and its ancestors is unchanged (def:12)."""
    from pydantic import ValidationError

    Y, genders = N.synthetic_cohort(seed=seed)
    cache: dict = {}
    runs = {"h_atlas": 0, "h_conn": 0, "h_group": 0}
    violations = []
    t0 = time.perf_counter()
    for atlas in N.ATLASES:
        for method in N.CONN_METHODS:
            for group in N.GROUPS:
                key_a, key_c, key_g = (atlas,), (atlas, method), (atlas, method, group)
                if key_a not in cache:
                    cache[key_a] = N.atlas_model(atlas, Y)
                    runs["h_atlas"] += 1
                    try:
                        N.ParcellationParams(P=N.parcellation(atlas, Y.shape[1]))
                    except ValidationError as exc:
                        violations.append((key_a, exc.errors()[0]["msg"]))
                if key_c not in cache:
                    cache[key_c] = N.conn_model(method, cache[key_a])
                    runs["h_conn"] += 1
                    for c in cache[key_c]:
                        try:
                            N.ConnectivityParams(C=c, method=method)
                        except ValidationError as exc:
                            violations.append((key_c, exc.errors()[0]["msg"]))
                            break
                if key_g not in cache:
                    cache[key_g] = N.group_model(group, cache[key_c], genders)
                    runs["h_group"] += 1
    sizes = {"men": genders.count("male"), "women": genders.count("female"),
             "mixed": len(genders)}
    try:
        N.GroupParams(sizes=sizes)
    except ValidationError as exc:
        violations.append(("groups", exc.errors()[0]["msg"]))
    return {"configurations": len(N.ATLASES) * len(N.CONN_METHODS) * len(N.GROUPS),
            "model_runs_with_reuse": runs, "total_runs_with_reuse": sum(runs.values()),
            "model_runs_without_reuse": 3 * N.CONFIG_SPACE_SIZE,
            "invariant_violations": violations, "group_sizes": sizes,
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "data_note": "seeded surrogate cohort (HCP S1200 is DUA-restricted)"}


def run(time_reasoning: bool = True, execute: bool = True) -> dict:
    g = derived_graph()
    out: dict = {"domain": "neuro (HCP)"}
    out["hypotheses"] = {
        s.name: {"label": s.label, "task": s.task + 1, "equations": list(s.equations),
                 "in": sorted(s.inputs), "out": sorted(s.outputs),
                 "variants": s.n_variants, "note": s.note}
        for s in N.HCP_HYPOTHESES
    }
    out["graph"] = {
        "nodes": g.nodes, "edges": g.edges, "n_edges": len(g.edges), "depth": g.depth,
        "competes": g.competes, "order_conflicts": g.order_conflicts,
        "hypothesis_graph_build_agrees": g.hypothesis_graph_agrees,
        "vs_dissertation": edge_diff(g.edges, N.DISSERTATION_EDGES),
    }
    out["cascade"] = {h: {"recompute": cascade(g.graph, h),
                          "fraction": len(cascade(g.graph, h)) / len(g.nodes)} for h in g.nodes}
    out["config_space"] = {"atlases": list(N.ATLASES), "conn_methods": N.CONN_METHODS,
                           "groups": N.GROUPS, "N": N.CONFIG_SPACE_SIZE,
                           "d_bar": (len(N.ATLASES) - 1) + (len(N.CONN_METHODS) - 1)
                           + (len(N.GROUPS) - 1)}
    if execute:
        out["enumeration"] = enumerate_configurations()
    out["region_decisions"] = {
        r: {**s, "significant": s["p"] < ALPHA} for r, s in REGION_STATS.items()}

    build = builder(N.HCP_HYPOTHESES, N.HCP_TASKS, g.edges)
    fired: dict[str, list[int]] = {}
    ab, r0 = S.nominal_scenario(build)
    out["ontology_size"] = ontology_size()
    out["consistency"] = {
        "ic_conditions_1_3": ic_conditions_1_3(ab.ve, ab.task, list(ab.hyp.values())),
        "structural": structural_check(N.HCP_HYPOTHESES, N.HCP_TASKS, N.HCP_TASK_EDGES,
                                       g.edges, N.HCP_VARIABLES),
    }
    out["nominal"] = {k: r0[k] for k in ("rules_fired", "markers", "reasoner_seconds")}
    fired["nominal"] = r0["rules_fired"]
    if time_reasoning:
        out["timing"] = {"rl_materialisation": time_rl_materialisation(),
                         "hermit": time_hermit()}

    out["change"] = {}
    for h in g.nodes:
        c = S.change_scenario(build, g.graph, h)
        out["change"][h] = c
        fired[f"change_{h}"] = c["rules_fired"]

    specs_r, tasks_r = N.rival_hypotheses()
    gr = derive_graph(specs_r, tasks_r, N.HCP_TASK_EDGES)
    rival = S.rival_scenario(builder(specs_r, tasks_r, gr.edges, prefix="hcpr"), gr,
                             N.rival_aic(), lower_is_better=True)
    rival["graph"] = {"n_nodes": len(gr.nodes), "n_edges": len(gr.edges), "edges": gr.edges}
    rival["consistency"] = structural_check(specs_r, tasks_r, N.HCP_TASK_EDGES, gr.edges,
                                            N.HCP_VARIABLES, engine="skip")
    out["rivals"] = rival
    fired["rivals"] = rival["rules_fired"]

    import numpy as np

    bad_P = N.parcellation("aal", 400)
    bad_P[0, :] *= 2.0  # ROI weights no longer average
    faults = {
        "orphan_hypothesis": S.fault_orphan(build, "hcp_motion_regression"),
        "dataset_missing": S.fault_dataset_missing(build),
        "shared_hypothesis": S.fault_shared(build, "h_atlas", "hcp_age_experiment"),
        "disjoint_domain": S.fault_disjoint_domain(build, OilDomainOntology),
        "format_mismatch": S.fault_format_mismatch(build, "h_atlas", "h_group"),
        "ic_conditions": S.fault_ic(build, "h_conn", 2),
        "invariant_atlas": S.fault_invariant(
            build, g.graph, "h_atlas", lambda: N.ParcellationParams(P=np.asarray(bad_P))),
    }
    out["faults"] = faults
    for k, v in faults.items():
        if "rules_fired" in v:
            fired[f"fault_{k}"] = v["rules_fired"]

    table = rule_table(fired)
    life = {"nominal", "rivals"} | {f"change_{h}" for h in g.nodes}
    out["rules"] = {
        r: {"name": RULE_NAMES[r],
            "life_cycle": sorted(s for s in table[r] if s in life),
            "fault_injection": sorted(s for s in table[r] if s not in life)}
        for r in range(1, 20)
    }
    out["rules_fired_life_cycle"] = sorted(r for r, v in out["rules"].items() if v["life_cycle"])
    out["rules_fired_fault_only"] = sorted(r for r, v in out["rules"].items()
                                           if v["fault_injection"] and not v["life_cycle"])
    out["rules_not_fired"] = sorted(r for r, v in out["rules"].items()
                                    if not v["life_cycle"] and not v["fault_injection"])
    return out


def main() -> None:
    out = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    g = out["graph"]
    print(f"HCP: {len(g['nodes'])} hypotheses, edges {g['edges']}, depth {g['depth']}; "
          f"vs dissertation {g['vs_dissertation']['only_derived']} / "
          f"{g['vs_dissertation']['only_expected']}")
    for h, c in out["cascade"].items():
        print(f"  change {h}: recompute {c['recompute']} ({len(c['recompute'])}/3)")
    e = out["enumeration"]
    print(f"  48 configurations: {e['total_runs_with_reuse']} model runs with reuse "
          f"(vs {e['model_runs_without_reuse']}), invariant violations {len(e['invariant_violations'])}")
    print(f"  consistency: {out['consistency']['structural']['status']}")
    print(f"  rules (life cycle): {out['rules_fired_life_cycle']}")
    print(f"  rules (fault injection only): {out['rules_fired_fault_only']}")
    print(f"  rules not fired: {out['rules_not_fired']}")
    if "timing" in out:
        print(f"  RL materialisation {out['timing']['rl_materialisation']['median_s']} s, "
              f"HermiT {out['timing']['hermit']['median_s']} s, size {out['ontology_size']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
