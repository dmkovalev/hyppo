#!/usr/bin/env python
"""Chapter 4 -- Besancon Galaxy Model (BGM) cascade demo (Section 4.2, part4.tex).

Reproduces the *method* claims of the BGM application from the library:
  * the 10 hypotheses are structures with declared inputs
    (:mod:`hyppo.adapters.bgm_adapter`); Algorithm 1 (``GraphConstructor``,
    cross-checked by ``HypothesisGraph.build``) DERIVES the ``derived_by`` graph
    from ``Out(h_a) ∩ In(h_b)`` -- 14 edges, depth 4. Compared with the graph
    drawn in the dissertation (images/BGM_graph_standalone.tex, 12 edges) it
    has two more edges, AgeBins -> Tracks and Tracks -> Luminosity, and the
    workflow puts the tracks in their own task (see bgm_experiment.py);
  * Algorithm 4 (``HypothesisGraph.plan``) recomputes only the hypotheses
    affected by a change -- 4 of 10 (40%) when the SFR hypothesis changes
    (gamma 0.12 -> 0.25), and likewise 4 of 10 when only the IMF changes
    (cascade IMF -> star birthrate -> local luminosity -> dynamical
    self-consistency);
  * the three IMF variants are ranked by their published chi^2 on Tycho-2.

The astronomical chi^2 values are taken from the published BGM fits (Czekaj et
al. 2014, Tycho-2 thin-disc (B-V)_T); Tycho-2 (VizieR) and the BGM web model
are public but not bundled here.

Usage:  uv run python examples/research/chapter4/bgm_cascade.py
Output: examples/research/chapter4/results/bgm_cascade.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hyppo.adapters.bgm_adapter import BGM_HYPOTHESES, BGM_TASK_EDGES, BGM_TASKS, IMF_CHI2  # noqa: E402
from hyppo.coa import HypothesisGraph  # noqa: E402

from examples.research.chapter4._domain_run import derive_graph  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# --- BGM stellar-birthrate hypothesis graph (Section 4.2, fig:mass_lattice) --
# 10 hypotheses (nodes): 4 input laws + 6 derived quantities. The edges are
# NOT listed here: Algorithm 1 derives them from the hypothesis structures.
NODES = [h.name for h in BGM_HYPOTHESES]
IDX = {n: i for i, n in enumerate(NODES)}
_GRAPH = derive_graph(BGM_HYPOTHESES, BGM_TASKS, BGM_TASK_EDGES)
EDGES = [tuple(e) for e in _GRAPH.edges]   # u -> v: v is derived_by u


def build_bgm_graph() -> HypothesisGraph:
    """Planning graph over the derived edges (Algorithm 4 uses the topology)."""
    return HypothesisGraph.from_edges(len(NODES), [(IDX[u], IDX[v]) for u, v in EDGES])


def recompute_on_change(g: HypothesisGraph, changed: set[str]) -> list[str]:
    """Names of hypotheses in P_ne when ``changed`` are invalidated (cache dropped)."""
    cached = {i for i in range(len(NODES)) if NODES[i] not in changed}
    return sorted(NODES[i] for i in g.plan(cached))


def main() -> None:
    g = build_bgm_graph()

    n = len(NODES)
    change_sfr = recompute_on_change(g, {"SFR"})
    change_imf = recompute_on_change(g, {"IMF"})
    imf_ranked = sorted(IMF_CHI2.items(), key=lambda kv: kv[1])  # lower chi^2 = better

    print(f"BGM hypothesis graph (Algorithm 1): {n} nodes, {len(EDGES)} derived_by edges, "
          f"depth {_GRAPH.depth}")
    print(f"change SFR -> recompute {change_sfr}  = {len(change_sfr)}/{n} "
          f"({100 * len(change_sfr) // n}%)")
    print(f"change IMF -> recompute {change_imf}  = {len(change_imf)}/{n} "
          f"({100 * len(change_imf) // n}%)")
    print("IMF ranking by chi^2 (Tycho-2, Czekaj+2014):")
    for rank, (name, chi2) in enumerate(imf_ranked, 1):
        print(f"  {rank}. {name}: chi^2 = {chi2:.0f}")

    # method claims of Section 4.2 (10-node graph, fig:mass_lattice)
    assert n == 10 and len(EDGES) == 14 and _GRAPH.depth == 4
    assert _GRAPH.hypothesis_graph_agrees
    cascade_tail = {"star_birthrate", "local_luminosity", "dynamical_self_consistency"}
    assert set(change_sfr) == {"SFR"} | cascade_tail, change_sfr
    assert len(change_sfr) == 4 and len(change_sfr) / n == 0.4
    assert set(change_imf) == {"IMF"} | cascade_tail, change_imf
    assert len(change_imf) == 4 and len(change_imf) / n == 0.4
    assert imf_ranked[0][0] == "Kroupa (2001)"

    out = {
        "nodes": NODES,
        "edges": EDGES,
        "edges_source": "derived by Algorithm 1 (GraphConstructor) from the "
                        "hypothesis structures of hyppo.adapters.bgm_adapter",
        "depth": _GRAPH.depth,
        "n_hypotheses": n,
        "cascade": {
            "change_SFR": {"recompute": change_sfr,
                           "fraction": len(change_sfr) / n},
            "change_IMF": {"recompute": change_imf,
                           "fraction": len(change_imf) / n},
        },
        "imf_ranking_chi2": [
            {"hypothesis": k, "chi2": v, "rank": r}
            for r, (k, v) in enumerate(imf_ranked, 1)
        ],
        "data_note": ("chi^2 from Czekaj+2014 (Tycho-2 thin-disc); cascade "
                      "structure reproduced from hyppo.coa.HypothesisGraph"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bgm_cascade.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'bgm_cascade.json'}")


if __name__ == "__main__":
    main()
