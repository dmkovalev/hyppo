#!/usr/bin/env python
"""Chapter 4 -- Besancon Galaxy Model (BGM) cascade demo (Section 4.2, part4.tex).

Reproduces the *method* claims of the BGM application from the library:
  * Algorithm 1 (HypothesisGraph.build) constructs the 6-hypothesis ``derived_by``
    graph of the BGM stellar-birthrate model;
  * Algorithm 4 (HypothesisGraph.plan) recomputes only the hypotheses affected by
    a change -- 3 of 6 (50%) when the SFR hypothesis changes (gamma 0.12 -> 0.25),
    and 2 of 6 when only IMF changes;
  * the three IMF variants are ranked by their published chi^2 on Tycho-2.

The cascade structure is fully reproducible from the library. The astronomical
chi^2 values are taken from the published BGM fits (Czekaj et al. 2014, Tycho-2
thin-disc (B-V)_T); Tycho-2 (VizieR) and the BGM web model are public but not
bundled here.

Usage:  uv run python experiments/chapter4/bgm_cascade.py
Output: experiments/chapter4/results/bgm_cascade.json
"""
from __future__ import annotations

import json
from pathlib import Path

from hyppo.coa import HypothesisGraph

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# --- BGM stellar-birthrate hypothesis graph (Section 4.2) -------------------
# 6 hypotheses (nodes); a derived_by edge u -> v means "v is computed from u".
# Birthrate b(m,t) ~ f(SFR(i), rho(i, SFR(i))) * IMF(m)  (Eq. in part4.tex).
NODES = ["age_cells", "density_law", "IMF", "SFR", "rho", "birthrate"]
IDX = {n: i for i, n in enumerate(NODES)}
EDGES = [
    ("age_cells", "SFR"),    # SFR is defined per age cell
    ("SFR", "rho"),          # rho(i, SFR(i)) depends on SFR
    ("density_law", "rho"),  # rho uses the local density law
    ("SFR", "birthrate"),    # b uses SFR
    ("rho", "birthrate"),    # b uses rho
    ("IMF", "birthrate"),    # b = ... * IMF(m)
]

# Published chi^2 of IMF variants on Tycho-2 thin-disc (B-V)_T (Czekaj+2014).
IMF_CHI2 = {
    "Kroupa (2001)": 187.0,
    "Haywood (1997)": 342.0,
    "Salpeter (1955)": 518.0,
}


def build_bgm_graph() -> HypothesisGraph:
    g = HypothesisGraph()
    for _ in NODES:
        g.add([])               # planning uses only the topology
    for u, v in EDGES:
        g.connect(IDX[u], IDX[v])
    return g


def recompute_on_change(g: HypothesisGraph, changed: set[str]) -> list[str]:
    """Names of hypotheses in P_ne when ``changed`` are invalidated (cache dropped)."""
    cached = {i for i in range(len(NODES)) if NODES[i] not in changed}
    return sorted(NODES[i] for i in g.plan(cached))


def main() -> None:
    g = build_bgm_graph()
    g.build()                   # Algorithm 1

    n = len(NODES)
    change_sfr = recompute_on_change(g, {"SFR"})
    change_imf = recompute_on_change(g, {"IMF"})
    imf_ranked = sorted(IMF_CHI2.items(), key=lambda kv: kv[1])  # lower chi^2 = better

    print(f"BGM hypothesis graph: {n} nodes, {len(EDGES)} derived_by edges")
    print(f"change SFR -> recompute {change_sfr}  = {len(change_sfr)}/{n} "
          f"({100 * len(change_sfr) // n}%)")
    print(f"change IMF -> recompute {change_imf}  = {len(change_imf)}/{n}")
    print("IMF ranking by chi^2 (Tycho-2, Czekaj+2014):")
    for rank, (name, chi2) in enumerate(imf_ranked, 1):
        print(f"  {rank}. {name}: chi^2 = {chi2:.0f}")

    # method claims of Section 4.2
    assert set(change_sfr) == {"SFR", "rho", "birthrate"}, change_sfr
    assert len(change_sfr) == 3 and len(change_sfr) / n == 0.5
    assert set(change_imf) == {"IMF", "birthrate"}, change_imf
    assert imf_ranked[0][0] == "Kroupa (2001)"

    out = {
        "nodes": NODES,
        "edges": EDGES,
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
