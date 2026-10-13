#!/usr/bin/env python
"""Chapter 4 -- HCP functional-connectivity cascade demo (Section 4.3, part4.tex).

Reproduces the *method* claims of the neurophysiology application from the
library, without reprocessing fMRI (the connectivity-extraction pipeline is not
part of hyppo). The brain-imaging statistics are taken from the published HCP
analysis (dissertation Tables in Section 4.3; sign test with Holm correction and
Bayesian factors); hyppo contributes the hypothesis graph, the cascade planner,
and the comparison decision.

  * the three hypotheses are structures with declared inputs
    (:mod:`hyppo.adapters.hcp_adapter`: ``X = P*Y``, ``C = Conn(X, kappa)``,
    ``C_grp = GroupMean(C, grp)``); Algorithm 1 (``GraphConstructor``,
    cross-checked by ``HypothesisGraph.build``) DERIVES the chain
    ``h_atlas -> h_conn -> h_group`` (configuration space N = 4*4*3 = 48);
  * Algorithm 4 (HypothesisGraph.plan) recomputes only affected hypotheses:
    changing the atlas re-runs all 3, the connectivity 2/3, the group method 1/3;
  * competing models per region are judged from the published sign-test p-values
    (Holm) and Bayesian factors, recovering the reported gender-difference
    conclusions.

HCP S1200 fMRI is access-restricted (DUA); only published aggregate statistics
are used here, so the demo is reproducible without the raw data.

Usage:  uv run python examples/research/chapter4/hcp_cascade.py
Output: examples/research/chapter4/results/hcp_cascade.json
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hyppo.adapters.hcp_adapter import (  # noqa: E402
    CONFIG_SPACE_SIZE,
    HCP_HYPOTHESES,
    HCP_TASK_EDGES,
    HCP_TASKS,
)
from hyppo.coa import HypothesisGraph  # noqa: E402

from examples.research.chapter4._domain_run import derive_graph  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# --- HCP hypothesis graph (Section 4.3): derived by Algorithm 1 -------------
NODES = [h.name for h in HCP_HYPOTHESES]
IDX = {n: i for i, n in enumerate(NODES)}
_GRAPH = derive_graph(HCP_HYPOTHESES, HCP_TASKS, HCP_TASK_EDGES)
EDGES = [tuple(e) for e in _GRAPH.edges]
N_VARIANTS = {h.name: h.n_variants for h in HCP_HYPOTHESES}  # config space N = 48
CONFIG_SPACE = CONFIG_SPACE_SIZE

# --- Published per-region statistics (dissertation Section 4.3 tables) ------
# sign test (Holm-corrected) p-value, Bayesian factor eta:1, alpha = 0.05.
# Precentral/Postcentral: documented gender differences; Planum Polare: control.
REGION_STATS = {
    "Precentral Gyrus": {"p": 0.0045, "eta": 29.0, "control": False},
    "Postcentral Gyrus": {"p": 0.0003, "eta": 54.0, "control": False},
    "Planum Polare": {"p": 0.08, "eta": 7.0, "control": True},
}
ALPHA = 0.05
GENERATOR_R2 = 0.8  # best generated hypothesis (GLM-seeded GP), published


def build_hcp_graph() -> HypothesisGraph:
    """Planning graph over the derived edges (Algorithm 4 uses the topology)."""
    return HypothesisGraph.from_edges(len(NODES), [(IDX[u], IDX[v]) for u, v in EDGES])


def recompute_on_change(g: HypothesisGraph, changed: set[str]) -> list[str]:
    cached = {i for i in range(len(NODES)) if NODES[i] not in changed}
    return sorted(NODES[i] for i in g.plan(cached))


def main() -> None:
    g = build_hcp_graph()
    n = len(NODES)

    cascade = {nm: recompute_on_change(g, {nm}) for nm in NODES}

    decisions = {
        region: {
            "p": s["p"],
            "eta": s["eta"],
            "significant": s["p"] < ALPHA,
            "control": s["control"],
        }
        for region, s in REGION_STATS.items()
    }

    print(f"HCP hypothesis graph: {n} nodes (chain), config space N = {CONFIG_SPACE}")
    for nm in NODES:
        print(f"  change {nm:8s} -> recompute {cascade[nm]} "
              f"= {len(cascade[nm])}/{n}")
    print("Per-region gender-difference decision (published stats):")
    for region, d in decisions.items():
        tag = "control" if d["control"] else "test"
        print(f"  {region:18s} [{tag}]: p={d['p']:.4f} eta={d['eta']:.0f}:1 "
              f"-> {'significant' if d['significant'] else 'not rejected'}")

    # method claims (Section 4.3)
    assert EDGES == [("h_atlas", "h_conn"), ("h_conn", "h_group")] and _GRAPH.depth == 2
    assert cascade["h_atlas"] == ["h_atlas", "h_conn", "h_group"]      # 3/3
    assert cascade["h_conn"] == ["h_conn", "h_group"]                   # 2/3
    assert cascade["h_group"] == ["h_group"]                           # 1/3
    assert CONFIG_SPACE == 48
    # published conclusions: test regions significant, control not rejected
    assert decisions["Precentral Gyrus"]["significant"]
    assert decisions["Postcentral Gyrus"]["significant"]
    assert not decisions["Planum Polare"]["significant"]

    out = {
        "nodes": NODES,
        "edges": EDGES,
        "edges_source": "derived by Algorithm 1 (GraphConstructor) from the "
                        "hypothesis structures of hyppo.adapters.hcp_adapter",
        "n_variants": N_VARIANTS,
        "config_space": CONFIG_SPACE,
        "cascade": {nm: {"recompute": cascade[nm],
                         "fraction": len(cascade[nm]) / n} for nm in NODES},
        "region_decisions": decisions,
        "generator_best_r2": GENERATOR_R2,
        "data_note": ("sign-test p (Holm) and Bayesian eta from the published HCP "
                      "analysis; fMRI is DUA-restricted. Graph/cascade reproduced "
                      "by Algorithm 1 / hyppo.coa.HypothesisGraph."),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hcp_cascade.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'hcp_cascade.json'}")


if __name__ == "__main__":
    main()
