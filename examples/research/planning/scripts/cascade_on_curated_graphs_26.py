"""Каскадный эксперимент на корпусе 26 hand-curated графов
(см. build_hand_curated_hypothesis_graphs_ext.py). Переиспользует
логику cascade_on_curated_graphs.py без дублирования."""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from cascade_on_curated_graphs import (  # noqa: E402
    R_GRID, N_REPS, build_adj, run_cascade,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GRAPHS = DATA / "hand_curated_hypothesis_graphs_26.json"
OUT_FILE = DATA / "cascade_curated_results_26.json"


def main():
    data = json.loads(GRAPHS.read_text(encoding="utf-8"))
    rng = random.Random(42)
    out = {"R_GRID": R_GRID, "N_REPS": N_REPS,
           "corpus": "hand_curated_26", "results": {}}
    keys = list(data.keys())
    for key in keys:
        g = data[key]
        n, adj = build_adj(g)
        edges = sum(len(v) for v in adj.values())
        cascade = run_cascade(n, adj, rng)
        out["results"][key] = {"n_hypotheses": n, "n_edges": edges,
                               "cascade": cascade}
        print(f"=== {key}: |H|={n}, |E|={edges} ===")
        for r in R_GRID:
            c = cascade[str(r)]
            print(f"  r={r}: ρ={c['median_rho']:.3f} "
                  f"(p05={c['p05_rho']:.3f}, p95={c['p95_rho']:.3f}), "
                  f"наивно={c['naive_1mr']:.2f}, +{c['excess_pct']:.1f}%")
    aggregate = {}
    for r in R_GRID:
        rhos = sorted(out["results"][k]["cascade"][str(r)]["median_rho"]
                      for k in keys)
        aggregate[str(r)] = {
            "median_rho": float(np.median(rhos)),
            "p05_rho": float(np.percentile(rhos, 5)),
            "p95_rho": float(np.percentile(rhos, 95)),
            "min_rho": float(min(rhos)),
            "max_rho": float(max(rhos)),
            "naive_1mr": 1 - r,
            "excess_pct": float(100 * (np.median(rhos) - (1 - r)) / (1 - r)),
        }
    out["aggregate_hand_curated_26"] = aggregate
    out["n_pipelines"] = len(keys)
    n_hyp = [out["results"][k]["n_hypotheses"] for k in keys]
    out["median_n_hypotheses"] = int(np.median(n_hyp))
    out["min_n_hypotheses"] = int(min(n_hyp))
    out["max_n_hypotheses"] = int(max(n_hyp))
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n=== Агрегат по {len(keys)} графам ===")
    for r in R_GRID:
        a = aggregate[str(r)]
        print(f"  r={r}: median ρ={a['median_rho']:.3f} "
              f"[{a['min_rho']:.3f}..{a['max_rho']:.3f}], +{a['excess_pct']:.1f}%")
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
