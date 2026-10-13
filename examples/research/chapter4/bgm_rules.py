#!/usr/bin/env python
"""Chapter 4 -- which ontology rules fire on the BGM experiment (tab:core_coverage).

The BGM experiment is now specified completely in the domain layer
(:mod:`hyppo.adapters.bgm_adapter`): hypothesis structures with declared inputs,
the workflow, GalaxyOntology with grounded variables, implementing models,
versions, runs and the dataset of the configuration. The graph is derived by
Algorithm 1, not entered by hand. This script runs the rule instrumentation of
:mod:`examples.research.chapter4.bgm_experiment` and reports, per rule of
tab:owl_rules, the life-cycle scenarios in which it fired (nominal run, IMF and
SFR revisions with recompute, rival IMFs ranked by chi^2) and the injected
defects that it detected (orphan hypothesis, dataset missing, shared
hypothesis, disjoint domain ontologies, numeric invariant).

Oracle of rules 4 and 7: StaleHypothesis == graph descendants of the revised
hypothesis; DerivedStaleRun(R_h) <=> h is a descendant.

Usage:  PYTHONPATH=. python examples/research/chapter4/bgm_rules.py
Output: examples/research/chapter4/results/bgm_rules.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.research.chapter4.bgm_experiment import run  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "bgm_rules.json"


def main():
    t0 = time.time()
    res = run(time_reasoning=False)
    out = {
        "nodes": res["graph"]["nodes"], "edges": res["graph"]["edges"],
        "edges_source": "Algorithm 1 (GraphConstructor) over hyppo.adapters.bgm_adapter",
        "scenarios": [
            {"invalid": src, "stale": c["stale"], "oracle_descendants": c["oracle_descendants"],
             "rule4_match": c["rule4_match"], "derived_stale_runs": c["derived_stale_runs"],
             "rule7_match": c["rule7_match"], "rules_fired": c["rules_fired"]}
            for src, c in res["change"].items()
        ],
        "rules": res["rules"],
        "rules_fired_life_cycle": res["rules_fired_life_cycle"],
        "rules_fired_fault_only": res["rules_fired_fault_only"],
        "rules_fired_union": sorted(set(res["rules_fired_life_cycle"])
                                    | set(res["rules_fired_fault_only"])),
        "rules_not_fired": res["rules_not_fired"],
        "consistency": res["consistency"]["structural"]["status"],
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for sc in out["scenarios"]:
        print(f"invalidate {sc['invalid']}: stale={sc['stale']} match={sc['rule4_match']}; "
              f"runs match={sc['rule7_match']}")
    print(f"rules fired in the life cycle: {out['rules_fired_life_cycle']}")
    print(f"rules fired only on injected defects: {out['rules_fired_fault_only']}")
    print(f"rules not fired: {out['rules_not_fired']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
