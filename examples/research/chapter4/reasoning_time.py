#!/usr/bin/env python
"""Reasoning time on the three domain ontologies of chapter 4 (sect4_compare).

For each domain the nominal ABox of the experiment is built in the shared
ontology (TBox = core + 19 rules + the three domain schemas, all loaded in one
process): domain ontology individual, hypotheses with the derived_by edges of
Algorithm 1, models, workflow tasks, the tuple slots, grounded variables,
versions and runs. Timed: the OWL 2 RL materialisation of the closure
(``rl_materialize``, owlrl -- the materialisation the integrity control relies
on, sect3_3) and the HermiT reasoner call (``sync_reasoner``), median of
repeats; sizes are reported with every measurement.

* astro -- BGM, 10 hypotheses (hyppo.adapters.bgm_adapter);
* neuro -- HCP, 3 hypotheses (hyppo.adapters.hcp_adapter);
* petro -- Norne HybridCRM, 16 hypotheses (equations of norne_battery.py,
  one task per hypothesis), OilFieldOntology.

Usage:  PYTHONPATH=. python examples/research/chapter4/reasoning_time.py
Output: examples/research/chapter4/results/reasoning_time.json
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hyppo.adapters import bgm_adapter as B  # noqa: E402
from hyppo.adapters import hcp_adapter as N  # noqa: E402
from hyppo.adapters._spec import HypothesisSpec, chain_edges  # noqa: E402
from hyppo.adapters.norne_adapter import OilFieldOntology, ReservoirParameter  # noqa: E402
from hyppo.core._base import virtual_experiment_onto as onto  # noqa: E402

from examples.research.chapter4._domain_run import (  # noqa: E402
    build_abox,
    derive_graph,
    reset_abox,
    time_hermit,
    time_rl_materialisation,
)
from examples.research.norne.norne_battery import NORNE  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "reasoning_time.json"


def petro_specs() -> tuple[list[HypothesisSpec], list[list[str]], dict]:
    from hyppo.graph_constructor._base import _rhs_variable_names

    specs, vocab = [], {}
    for k, (name, formula) in enumerate(NORNE):
        lhs, rhs = formula.split("=", 1)
        ins = frozenset(_rhs_variable_names(rhs))
        specs.append(HypothesisSpec(name, name, (formula,), ins, task=k))
        for v in ins | {lhs.strip()}:
            vocab[v] = ("ReservoirParameter", "")
    return specs, [[s.name] for s in specs], vocab


def domains():
    p_specs, p_tasks, p_vocab = petro_specs()
    return {
        "astro_BGM": dict(specs=B.BGM_HYPOTHESES, tasks=B.BGM_TASKS, domain_cls=B.GalaxyOntology,
                          quantity_cls=B.ASTRO_QUANTITY, vocabulary=B.BGM_VARIABLES,
                          dataset="Tycho2", needs=("dynamical_self_consistency",)),
        "neuro_HCP": dict(specs=N.HCP_HYPOTHESES, tasks=N.HCP_TASKS, domain_cls=N.BrainOntology,
                          quantity_cls=N.NEURO_QUANTITY, vocabulary=N.HCP_VARIABLES,
                          dataset="HCP_S1200", needs=("h_atlas",)),
        "petro_Norne": dict(specs=p_specs, tasks=p_tasks, domain_cls=OilFieldOntology,
                            quantity_cls={"ReservoirParameter": ReservoirParameter},
                            vocabulary=p_vocab, dataset="Norne", needs=("H1",)),
    }


def measure(repeats_rl: int = 5, repeats_hermit: int = 3) -> dict:
    out = {"platform": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
           "tbox_classes": len(list(onto.classes())), "domains": {}}
    for key, d in domains().items():
        g = derive_graph(d["specs"], d["tasks"], chain_edges(len(d["tasks"])))
        reset_abox()
        build_abox(key.split("_")[1].lower(), d["specs"], d["tasks"], g.edges,
                   domain_cls=d["domain_cls"], quantity_cls=d["quantity_cls"],
                   vocabulary=d["vocabulary"], dataset=d["dataset"], needs_dataset=d["needs"])
        size = {"hypotheses": len(d["specs"]), "edges": len(g.edges),
                "individuals": len(list(onto.individuals())),
                "abox_triples": len(onto.world.as_rdflib_graph())}
        out["domains"][key] = {"size": size,
                               "rl_materialisation": time_rl_materialisation(repeats_rl),
                               "hermit": time_hermit(repeats_hermit)}
        reset_abox()
    largest = max(out["domains"], key=lambda k: out["domains"][k]["size"]["individuals"])
    out["largest_by_individuals"] = largest
    return out


def main() -> None:
    out = measure()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"TBox classes: {out['tbox_classes']}  ({out['platform']})")
    for k, v in out["domains"].items():
        print(f"  {k:12s} {v['size']}  RL {v['rl_materialisation']['median_s']} s  "
              f"HermiT {v['hermit']['median_s']} s")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
