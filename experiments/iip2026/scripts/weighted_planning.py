"""
Algorithm 3 — weighted planning через min-cut (Picard 1976).

Дано: DAG (V, E), Cache ⊆ V, w: V → R_+.
Цель: min Σ_{h ∈ Pne} w(h) при A1-A3.

Picard reduction:
- s → h (cap w(h)) для h ∈ Cache
- h → t (cap ∞) для h ∉ Cache
- Каждое ребро (g, h) ∈ E (исходный DAG): h → g в reverse, cap ∞

Min s-t cut → Pe = s-side, Pne = t-side.

Сравнение с Algorithm 2 (unweighted, min |Pne|):
- При w ≡ 1 даёт тот же результат
- При log-uniform w даёт разный Pne, но меньший W(Pne)
"""
from __future__ import annotations
import io
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

from hyppo.coa import HypothesisGraph

import numpy as np
import networkx as nx

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

CACHE_DIR = CACHE / "wfcommons"
INF = 10**12  # большая capacity для "обязательных" рёбер


def parse_wfformat(path: Path) -> tuple[int, dict[int, set[int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data["workflow"]["specification"]["tasks"]
    id_to_idx = {t["id"]: i for i, t in enumerate(tasks)}
    n = len(tasks)
    adj: dict[int, set[int]] = defaultdict(set)
    for t in tasks:
        u = id_to_idx[t["id"]]
        for c in t.get("children", []):
            if c in id_to_idx:
                adj[u].add(id_to_idx[c])
    return n, dict(adj)


def algorithm_2_unweighted(n: int, adj: dict[int, set[int]],
                           cached: set[int]) -> set[int]:
    """Algorithm 2: min |Pne| через downward-closure."""
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    return HypothesisGraph.from_edges(n, edges).plan(cached)


def algorithm_3_weighted(n: int, adj: dict[int, set[int]],
                          cached: set[int],
                          weights: dict[int, float]) -> set[int]:
    """Algorithm 3: min W(Pne) через Picard min-cut.

    Конструкция:
    - source s = 'S', sink t = 'T'
    - h ∈ Cache: ребро s → h_idx с cap w(h)
    - h ∉ Cache: ребро h_idx → t с cap ∞
    - (g, h) ∈ E: ребро h_idx → g_idx с cap ∞ (reverse direction для closure)

    После min-cut: Pe = вершины достижимые из s (s-side), Pne = остальные.
    """
    G = nx.DiGraph()
    G.add_node("S")
    G.add_node("T")
    # Add all vertices
    for i in range(n):
        G.add_node(i)
    # s → h ∈ Cache with cap w(h)
    for h in range(n):
        if h in cached:
            G.add_edge("S", h, capacity=float(weights.get(h, 1.0)))
        else:
            G.add_edge(h, "T", capacity=INF)
    # Reverse edges for closure preservation: each (g, h) ∈ E → (h, g) cap ∞
    for g, vs in adj.items():
        for h in vs:
            G.add_edge(h, g, capacity=INF)
    # Min s-t cut
    cut_value, partition = nx.minimum_cut(G, "S", "T", flow_func=nx.algorithms.flow.preflow_push)
    s_side, t_side = partition
    # Pe = вершины в s-side (без S)
    p_e = (s_side - {"S"}) & set(range(n))
    p_ne = set(range(n)) - p_e
    return p_ne


def gen_weights_loguniform(n: int, low: float = 1.0, high: float = 1000.0,
                            seed: int = 42) -> dict[int, float]:
    """log-uniform weights: log w ~ U(log low, log high)."""
    rng = np.random.default_rng(seed)
    log_low, log_high = math.log(low), math.log(high)
    return {i: float(math.exp(rng.uniform(log_low, log_high))) for i in range(n)}


def compute_W(pne: set[int], weights: dict[int, float]) -> float:
    return sum(weights[h] for h in pne)


def main():
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    r_test = 0.7
    n_reps = 5  # для каждого workflow с разными случайными Cache

    # Собрать workflow JSONs из кэша
    jsons = []
    for sub in ("nextflow", "snakemake", "pegasus"):
        d = CACHE_DIR / sub
        if not d.exists():
            continue
        for p in d.rglob("*.json"):
            jsons.append((sub, p))
    print(f"Total workflows: {len(jsons)}", flush=True)

    results = []
    # Ограничимся размером — max-flow O(V²E) дорогой
    N_LIMIT = 500
    for family, path in jsons:
        try:
            n, adj = parse_wfformat(path)
        except Exception:
            continue
        if n < 10 or n > N_LIMIT:
            continue
        # log-uniform weights
        weights = gen_weights_loguniform(n, seed=hash(path.name) % 10**9)
        W_total = sum(weights.values())

        # 5 повторов с разными Cache
        ratios = []
        pne_sizes_alg2 = []
        pne_sizes_alg3 = []
        W_alg2 = []
        W_alg3 = []
        for rep in range(n_reps):
            n_cached = int(r_test * n)
            cached = set(np_rng.choice(np.arange(n), size=n_cached,
                                        replace=False).tolist())
            # Algorithm 2
            pne2 = algorithm_2_unweighted(n, adj, cached)
            W2 = compute_W(pne2, weights)
            # Algorithm 3
            try:
                pne3 = algorithm_3_weighted(n, adj, cached, weights)
            except Exception as e:
                print(f"  [{family}/{path.stem}] alg3 error: {e}", flush=True)
                continue
            W3 = compute_W(pne3, weights)

            pne_sizes_alg2.append(len(pne2))
            pne_sizes_alg3.append(len(pne3))
            W_alg2.append(W2)
            W_alg3.append(W3)
            if W3 > 0:
                ratios.append(W2 / W3)

        if not ratios:
            continue
        results.append({
            "family": family,
            "name": path.stem,
            "n": n,
            "ratio_median": float(np.median(ratios)),
            "ratio_p05": float(np.percentile(ratios, 5)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "pne_alg2_median": float(np.median(pne_sizes_alg2)),
            "pne_alg3_median": float(np.median(pne_sizes_alg3)),
            "W_alg2_median": float(np.median(W_alg2)),
            "W_alg3_median": float(np.median(W_alg3)),
        })
        print(f"  [{family:10s} | {path.stem[:30]:30s}] n={n:4d}  "
              f"|P_ne| Alg2/Alg3 = {pne_sizes_alg2[-1]:3d}/{pne_sizes_alg3[-1]:3d}  "
              f"W ratio = {ratios[-1]:.2f}×", flush=True)

    # Сводка
    if not results:
        print("No results!")
        return
    ratios_all = [r["ratio_median"] for r in results]
    print(f"\n=== Summary on {len(results)} workflows ===")
    print(f"  W(Alg2)/W(Alg3) ratio:")
    print(f"    median: {float(np.median(ratios_all)):.3f}×")
    print(f"    p05:    {float(np.percentile(ratios_all, 5)):.3f}×")
    print(f"    p95:    {float(np.percentile(ratios_all, 95)):.3f}×")
    print(f"    max:    {float(np.max(ratios_all)):.3f}×")
    pne_diffs = [r["pne_alg3_median"] / max(r["pne_alg2_median"], 1) for r in results]
    print(f"  |P_ne(Alg3)| / |P_ne(Alg2)|: median {float(np.median(pne_diffs)):.2f}×")
    print(f"    (Alg3 может выбрать БОЛЬШЕ вершин, но с меньшим суммарным весом)")

    # by family
    by_fam: dict[str, list] = defaultdict(list)
    for r in results:
        by_fam[r["family"]].append(r)
    print("\n  By family:")
    summary_by_fam = {}
    for fam, lst in by_fam.items():
        rs = [r["ratio_median"] for r in lst]
        med_r = float(np.median(rs))
        summary_by_fam[fam] = {
            "count": len(lst),
            "ratio_median": med_r,
            "ratio_p05": float(np.percentile(rs, 5)),
            "ratio_p95": float(np.percentile(rs, 95)),
        }
        print(f"    {fam:12s} (N={len(lst):3d}): W ratio median = {med_r:.2f}×")

    out = {
        "r": r_test,
        "n_reps_per_workflow": n_reps,
        "N_workflow_limit": N_LIMIT,
        "weights_distribution": "log-uniform(1, 1000)",
        "n_workflows": len(results),
        "summary_ratio_median": float(np.median(ratios_all)),
        "summary_ratio_p05": float(np.percentile(ratios_all, 5)),
        "summary_ratio_p95": float(np.percentile(ratios_all, 95)),
        "summary_ratio_max": float(np.max(ratios_all)),
        "by_family": summary_by_fam,
        "per_workflow": results,
    }
    out_path = DATA / "weighted_planning_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
