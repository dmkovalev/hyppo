"""Worker для baseline_and_overhead.py: один workflow в изолированном
процессе (обход Python 3.13 'cell' object bug в долгих циклах).

Аргументы: <wf_json> <seed>
Стандартный вывод: JSON с per_r результатами.
"""
from __future__ import annotations
import io
import json
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from hyppo.coa import HypothesisGraph


def parse_workflow(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = d.get("workflow", {}).get("specification", {})
    tasks = spec.get("tasks", [])
    if not tasks:
        return None
    id_to_idx = {t["id"]: i for i, t in enumerate(tasks)}
    n = len(tasks)
    adj = defaultdict(set)
    for t in tasks:
        u = id_to_idx[t["id"]]
        for c in t.get("children", []):
            if c in id_to_idx:
                adj[u].add(id_to_idx[c])
    runtimes = [1.0] * n
    ex_tasks = d.get("workflow", {}).get("execution", {}).get("tasks", [])
    for et in ex_tasks:
        idx = id_to_idx.get(et.get("id"))
        if idx is None:
            continue
        rt = et.get("runtimeInSeconds")
        if rt is not None and rt > 0:
            runtimes[idx] = float(rt)
    return n, dict(adj), runtimes


def algorithm_2(n, adj, cached):
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    return HypothesisGraph.from_edges(n, edges).plan(cached)


def bench_block(n, adj, cached, block_iters):
    t0 = time.perf_counter()
    pne = None
    for _ in range(block_iters):
        pne = algorithm_2(n, adj, cached)
    return time.perf_counter() - t0, pne


def adaptive_block(n, adj, cached, target_sec=0.01, cap=200):
    dt, _ = bench_block(n, adj, cached, 1)
    if dt <= 0:
        return cap
    iters = max(1, int(target_sec / dt))
    return min(iters, cap)


def main():
    wf_json = sys.argv[1]
    seed = int(sys.argv[2])
    parsed = parse_workflow(wf_json)
    if parsed is None:
        print(json.dumps({"skipped": True}))
        return
    n, adj, runtimes = parsed
    if n < 10:
        print(json.dumps({"skipped": True}))
        return
    rng = random.Random(seed)
    t_total = sum(runtimes)
    R_GRID = [0.3, 0.5, 0.7, 0.9]
    N_REPS = 10
    per_r = {}
    for r in R_GRID:
        n_cached = int(r * n)
        sample_idx = rng.sample(range(n), n_cached)
        block_iters = adaptive_block(n, adj, set(sample_idx))
        speedups = []
        pne_sizes = []
        wallclock_us = []
        for rep in range(N_REPS):
            cached = set(rng.sample(range(n), n_cached))
            dt_block, pne = bench_block(n, adj, cached, block_iters)
            wallclock_us.append((dt_block / block_iters) * 1e6)
            t_with = sum(runtimes[i] for i in pne)
            if t_with > 0:
                speedups.append(t_total / t_with)
            else:
                speedups.append(float("inf"))
            pne_sizes.append(len(pne))
        speedups.sort(); pne_sizes.sort(); wallclock_us.sort()
        def pct(arr, p):
            k = int(len(arr) * p / 100)
            return arr[min(k, len(arr) - 1)]
        per_r[r] = {
            "speedup_median": speedups[len(speedups) // 2],
            "speedup_p05": pct(speedups, 5),
            "speedup_p95": pct(speedups, 95),
            "pne_median": pne_sizes[len(pne_sizes) // 2],
            "rho_median": pne_sizes[len(pne_sizes) // 2] / n,
            "wallclock_us_median": wallclock_us[len(wallclock_us) // 2],
            "wallclock_us_p95": pct(wallclock_us, 95),
            "wallclock_us_max": max(wallclock_us),
            "block_iters": int(block_iters),
        }
    out = {
        "n": n,
        "edges": sum(len(v) for v in adj.values()),
        "t_total_sec": t_total,
        "per_r": per_r,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
