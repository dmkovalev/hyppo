"""Прогон cascade-эксперимента на двух hand-curated hypothesis-graphs
(rnaseq + airrflow). Сравнение с task-level и subworkflow-level."""
from __future__ import annotations
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

from hyppo.coa import HypothesisGraph

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent   # experiments/iip2026/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

GRAPHS = DATA / "hand_curated_hypothesis_graphs.json"
OUT_FILE = DATA / "cascade_curated_results.json"

R_GRID = [0.3, 0.5, 0.7, 0.9]
N_REPS = 500  # сравнительно небольшие графы → больше повторов


def algorithm_2(n: int, adj: dict, cached: set) -> set:
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    return HypothesisGraph.from_edges(n, edges).plan(cached)


def build_adj(g: dict) -> tuple[int, dict[int, set]]:
    nodes = {h["id"]: i for i, h in enumerate(g["hypotheses"])}
    n = len(nodes)
    adj = defaultdict(set)
    for e in g["subsumption_edges"]:
        adj[nodes[e["src"]]].add(nodes[e["tgt"]])
    for e in g["inter_layer_edges"]:
        adj[nodes[e["src"]]].add(nodes[e["tgt"]])
    return n, {k: set(v) for k, v in adj.items()}


def run_cascade(n: int, adj: dict, rng) -> dict:
    out = {}
    for r in R_GRID:
        n_cached = int(r * n)
        rhos = []
        for _ in range(N_REPS):
            if n_cached >= n:
                cached = set(range(n))
            else:
                cached = set(rng.sample(range(n), n_cached))
            pne = algorithm_2(n, adj, cached)
            rhos.append(len(pne) / n)
        rhos.sort()
        out[str(r)] = {
            "median_rho": float(np.median(rhos)),
            "p05_rho": float(np.percentile(rhos, 5)),
            "p95_rho": float(np.percentile(rhos, 95)),
            "naive_1mr": 1 - r,
            "excess_pct": float(100 * (np.median(rhos) - (1 - r)) / (1 - r)),
        }
    return out


def main():
    data = json.loads(GRAPHS.read_text(encoding="utf-8"))
    rng = random.Random(42)
    out = {"R_GRID": R_GRID, "N_REPS": N_REPS, "results": {}}
    pipeline_keys = list(data.keys())
    for key in pipeline_keys:
        g = data[key]
        n, adj = build_adj(g)
        edges = sum(len(v) for v in adj.values())
        cascade = run_cascade(n, adj, rng)
        out["results"][key] = {
            "n_hypotheses": n,
            "n_edges": edges,
            "cascade": cascade,
        }
        print(f"=== {key}: |H|={n}, |E|={edges} ===")
        for r in R_GRID:
            c = cascade[str(r)]
            print(f"  r={r}: ρ={c['median_rho']:.3f}  "
                  f"(p05={c['p05_rho']:.3f}, p95={c['p95_rho']:.3f}), "
                  f"наивно={c['naive_1mr']:.2f}, "
                  f"+{c['excess_pct']:.1f}% над 1-r")
        print()

    # === Агрегированная статистика по всем 20 hand-curated ===
    print("=== Aggregate medians across {n} hand-curated graphs ===".format(
        n=len(pipeline_keys)))
    aggregate = {}
    for r in R_GRID:
        rhos = [out["results"][k]["cascade"][str(r)]["median_rho"]
                for k in pipeline_keys]
        rhos.sort()
        aggregate[str(r)] = {
            "median_rho": float(np.median(rhos)),
            "p05_rho":    float(np.percentile(rhos, 5)),
            "p95_rho":    float(np.percentile(rhos, 95)),
            "min_rho":    float(min(rhos)),
            "max_rho":    float(max(rhos)),
            "naive_1mr":  1 - r,
            "excess_pct": float(100 * (np.median(rhos) - (1 - r)) / (1 - r)),
        }
        print(f"  r={r}: median ρ={aggregate[str(r)]['median_rho']:.3f}  "
              f"[min={aggregate[str(r)]['min_rho']:.3f}, "
              f"max={aggregate[str(r)]['max_rho']:.3f}], "
              f"+{aggregate[str(r)]['excess_pct']:.1f}% над 1-r")
    out["aggregate_hand_curated"] = aggregate

    n_hyp_all = [out["results"][k]["n_hypotheses"] for k in pipeline_keys]
    out["n_pipelines"] = len(pipeline_keys)
    out["median_n_hypotheses"] = int(np.median(n_hyp_all))
    out["min_n_hypotheses"] = int(min(n_hyp_all))
    out["max_n_hypotheses"] = int(max(n_hyp_all))

    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nSaved {OUT_FILE}")

    # === Сравнительная сводка с другими уровнями ===
    print("\n=== Сравнение всех уровней при r=0.7 ===")
    # task-level
    try:
        td = json.loads((DATA / "wfcommons_validation_results.json").read_text())
        task_med = float(np.median([w["rho_real"] for w in td["per_workflow"]]))
        print(f"  task-DAG (157 WfCommons):     ρ={task_med:.3f}")
    except Exception:
        pass
    # subworkflow-level (из cascade_hypothesis_results.json)
    try:
        sd = json.loads((DATA / "cascade_hypothesis_results.json").read_text())
        sub_med = sd["all_corpus"]["0.7"]["median_rho"]
        print(f"  subworkflow-агрегация:        ρ={sub_med:.3f}")
    except Exception:
        pass
    for key in pipeline_keys:
        if key not in out["results"]:
            continue
        c = out["results"][key]["cascade"]["0.7"]["median_rho"]
        print(f"  hand-curated {key:12s}:  ρ={c:.3f}")
    print(f"  hand-curated MEDIAN (N={len(pipeline_keys)}): "
          f"ρ={aggregate['0.7']['median_rho']:.3f}")
    # EDAM
    try:
        ed = json.loads((DATA / "cascade_edam_results.json").read_text())
        e_med = ed["rho_by_r"]["0.7"]["median_rho"]
        print(f"  EDAM native derived-by:       ρ={e_med:.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
