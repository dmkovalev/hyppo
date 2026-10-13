#!/usr/bin/env python
"""Chapter 4, §4.2 -- the BGM virtual experiment run through the core of the complex.

Domain layer: :mod:`hyppo.adapters.bgm_adapter` (10 hypothesis structures with
declared inputs, the 5-task workflow, GalaxyOntology, numeric invariants, the
configuration space). This script applies the invariant core to it:

1. Algorithm 1 derives the derivability graph from ``Out(h_a) ∩ In(h_b)``
   (``GraphConstructor``; cross-checked by ``HypothesisGraph.build``); the
   derived edges are compared edge by edge with the graph drawn in the
   dissertation (images/BGM_graph_standalone.tex), and the workflow of
   part4.tex is checked against condition 5;
2. Algorithm 4 plans the recomputation on an IMF / SFR change;
3. the consistency check (conditions 1-5, C3-C7, OWL 2 RL Stage A) and the
   numeric invariants run on the experiment;
4. the reasoner and the marker layer run on the life-cycle scenarios (nominal,
   IMF/SFR revision with recompute, rival IMFs ranked by chi^2) and on injected
   defects; every inferred class is mapped to the rules of tab:owl_rules;
5. the OWL 2 RL materialisation and HermiT are timed on the nominal ABox.

Usage:  PYTHONPATH=. python examples/research/chapter4/bgm_experiment.py
Output: examples/research/chapter4/results/bgm_experiment.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hyppo.adapters import bgm_adapter as B  # noqa: E402
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
OUT = HERE / "results" / "bgm_experiment.json"
NEEDS_DATA = ("dynamical_self_consistency",)  # observed kinematics / Tycho-2


def derived_graph():
    return derive_graph(B.BGM_HYPOTHESES, B.BGM_TASKS, B.BGM_TASK_EDGES)


def builder(specs, tasks, edges, prefix="bgm"):
    def build():
        return build_abox(prefix, specs, tasks, edges, domain_cls=B.GalaxyOntology,
                          quantity_cls=B.ASTRO_QUANTITY, vocabulary=B.BGM_VARIABLES,
                          dataset="Tycho2", needs_dataset=NEEDS_DATA)
    return build


def run(time_reasoning: bool = True) -> dict:
    g = derived_graph()
    out: dict = {"domain": "astro (BGM)"}
    out["hypotheses"] = {
        s.name: {"label": s.label, "task": s.task + 1, "equations": list(s.equations),
                 "in": sorted(s.inputs), "out": sorted(s.outputs),
                 "variants": s.n_variants, "note": s.note}
        for s in B.BGM_HYPOTHESES
    }
    out["workflow"] = [[n for n in t] for t in B.BGM_TASKS]
    out["graph"] = {
        "nodes": g.nodes, "edges": g.edges, "n_edges": len(g.edges), "depth": g.depth,
        "competes": g.competes, "order_conflicts": g.order_conflicts,
        "hypothesis_graph_build_agrees": g.hypothesis_graph_agrees,
        "vs_dissertation": edge_diff(g.edges, B.DISSERTATION_EDGES),
    }
    gd = derive_graph(B.BGM_HYPOTHESES, B.DISSERTATION_TASKS, B.BGM_TASK_EDGES)
    out["dissertation_workflow_check"] = {
        "workflow": B.DISSERTATION_TASKS, "n_edges": len(gd.edges),
        "consistency": structural_check(B.BGM_HYPOTHESES, B.DISSERTATION_TASKS,
                                        B.BGM_TASK_EDGES, gd.edges, B.BGM_VARIABLES,
                                        engine="skip"),
    }
    n = len(g.nodes)
    out["cascade"] = {src: {"recompute": cascade(g.graph, src),
                            "fraction": len(cascade(g.graph, src)) / n}
                      for src in ("IMF", "SFR")}
    out["config_space"] = {"axes": {k: {"hypothesis": h, "levels": lv}
                                    for k, (h, lv) in B.CONFIG_AXES.items()},
                           "N": B.CONFIG_SPACE_SIZE,
                           "d_bar": sum(len(lv) - 1 for _, lv in B.CONFIG_AXES.values())}
    out["invariants_nominal"] = B.nominal_invariant_checks()

    build = builder(B.BGM_HYPOTHESES, B.BGM_TASKS, g.edges)
    fired: dict[str, list[int]] = {}

    # -- nominal: tuple, consistency, reasoning --------------------------------
    ab, r0 = S.nominal_scenario(build)
    out["ontology_size"] = ontology_size()
    out["consistency"] = {
        "ic_conditions_1_3": ic_conditions_1_3(ab.ve, ab.task, list(ab.hyp.values())),
        "structural": structural_check(B.BGM_HYPOTHESES, B.BGM_TASKS, B.BGM_TASK_EDGES,
                                       g.edges, B.BGM_VARIABLES),
    }
    out["nominal"] = {k: r0[k] for k in ("rules_fired", "markers", "reasoner_seconds")}
    out["nominal"]["n_inferred_derived_by"] = len(r0["inferred_derived_by"])
    fired["nominal"] = r0["rules_fired"]
    if time_reasoning:
        out["timing"] = {"rl_materialisation": time_rl_materialisation(),
                         "hermit": time_hermit()}

    # -- revisions with recompute (rules 4, 7, 8, 18) --------------------------
    out["change"] = {}
    for src in ("IMF", "SFR"):
        c = S.change_scenario(build, g.graph, src)
        out["change"][src] = c
        fired[f"change_{src}"] = c["rules_fired"]

    # -- rival IMFs ranked by chi^2 (rules 10, 12, 16, 17, 19) -----------------
    specs_r, tasks_r = B.rival_hypotheses()
    gr = derive_graph(specs_r, tasks_r, B.BGM_TASK_EDGES)
    rival = S.rival_scenario(builder(specs_r, tasks_r, gr.edges, prefix="bgmr"), gr,
                             B.RIVAL_CHI2, lower_is_better=True)
    rival["graph"] = {"n_nodes": len(gr.nodes), "n_edges": len(gr.edges), "depth": gr.depth}
    rival["consistency"] = structural_check(specs_r, tasks_r, B.BGM_TASK_EDGES, gr.edges,
                                            B.BGM_VARIABLES, engine="skip")
    out["rivals"] = rival
    fired["rivals"] = rival["rules_fired"]

    # -- injected defects (detection checks) -----------------------------------
    faults = {
        "orphan_hypothesis": S.fault_orphan(build, "bgm_extinction_law"),
        "dataset_missing": S.fault_dataset_missing(build),
        "shared_hypothesis": S.fault_shared(build, "IMF", "bgm_star_counts_experiment"),
        "disjoint_domain": S.fault_disjoint_domain(build, OilDomainOntology),
        "ic_conditions": S.fault_ic(build, "local_luminosity", 2),
        "invariant_IMF": S.fault_invariant(
            build, g.graph, "IMF", lambda: B.IMFParams(alpha=2.35, m_lo=150.0, m_up=120.0)),
    }
    out["faults"] = faults
    for k in ("orphan_hypothesis", "dataset_missing", "shared_hypothesis", "disjoint_domain",
              "invariant_IMF"):
        fired[f"fault_{k}"] = faults[k]["rules_fired"]

    table = rule_table(fired)
    life = ("nominal", "change_IMF", "change_SFR", "rivals")
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
    print(f"BGM: {len(g['nodes'])} hypotheses, {g['n_edges']} derived edges, depth {g['depth']}")
    print(f"  vs dissertation: only derived {g['vs_dissertation']['only_derived']}, "
          f"only drawn {g['vs_dissertation']['only_expected']}")
    print(f"  dissertation workflow: {out['dissertation_workflow_check']['consistency']['status']}")
    for src, c in out["cascade"].items():
        print(f"  change {src}: recompute {c['recompute']} ({len(c['recompute'])}/10)")
    print(f"  consistency: {out['consistency']['structural']['status']}; "
          f"IC 1-3: {[out['consistency']['ic_conditions_1_3'][k]['ok'] for k in ('condition1', 'condition2', 'condition3')]}")
    print(f"  rules (life cycle): {out['rules_fired_life_cycle']}")
    print(f"  rules (fault injection only): {out['rules_fired_fault_only']}")
    print(f"  rules not fired: {out['rules_not_fired']}")
    if "timing" in out:
        print(f"  RL materialisation {out['timing']['rl_materialisation']['median_s']} s, "
              f"HermiT {out['timing']['hermit']['median_s']} s, size {out['ontology_size']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
