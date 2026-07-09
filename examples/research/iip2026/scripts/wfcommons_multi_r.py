"""Каскадный эффект на WfCommons при r ∈ {0.3, 0.5, 0.7, 0.9} (M2 рецензии).
Использует кэш wfcommons из предыдущего запуска."""
import io
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

ROOT = Path(__file__).resolve().parent.parent   # examples/research/iip2026/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

CACHE_WF = CACHE / "wfcommons"


def parse_wfformat(path: Path) -> tuple[int, dict[int, set[int]], int]:
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
    in_deg = defaultdict(int)
    for vs in adj.values():
        for v in vs:
            in_deg[v] += 1
    dist = [0] * n
    q = deque(i for i in range(n) if in_deg[i] == 0)
    in_d = dict(in_deg)
    while q:
        x = q.popleft()
        for v in adj.get(x, ()):
            if dist[v] < dist[x] + 1:
                dist[v] = dist[x] + 1
            in_d[v] -= 1
            if in_d[v] == 0:
                q.append(v)
    depth = max(dist) if dist else 0
    return n, dict(adj), depth


def plan(n: int, adj: dict[int, set[int]], cached: set[int]) -> int:
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    p_ne = HypothesisGraph.from_edges(n, edges).plan(cached)
    return len(p_ne)


def cascade_at_r(adj, n: int, r: float, n_reps: int, np_rng) -> float:
    arr = np.arange(n)
    n_cached = int(r * n)
    vals = []
    for _ in range(n_reps):
        cached = set(np_rng.choice(arr, size=n_cached, replace=False).tolist())
        vals.append(plan(n, adj, cached) / max(n, 1))
    return float(np.median(vals))


def main():
    np_rng = np.random.default_rng(42)
    r_values = [0.3, 0.5, 0.7, 0.9]
    n_reps = 30

    # собрать все JSON из cache
    jsons = []
    for sub in ("nextflow", "snakemake", "pegasus"):
        d = CACHE_WF / sub
        if not d.exists():
            continue
        for p in d.rglob("*.json"):
            jsons.append((sub, p))
    print(f"Total workflows in cache: {len(jsons)}", flush=True)

    # По семействам: median ρ для каждого r
    by_fam: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[float, list[float]] = defaultdict(list)
    for fam, path in jsons:
        try:
            n, adj, depth = parse_wfformat(path)
        except Exception:
            continue
        if n < 10:
            continue
        for r in r_values:
            try:
                rho = cascade_at_r(adj, n, r, n_reps, np_rng)
            except Exception:
                continue
            by_fam[fam][r].append(rho)
            overall[r].append(rho)

    summary = {}
    print("\nSummary (median ρ by family × r):")
    print(f"{'family':<12s} | " + " | ".join(f"r={r:.1f}" for r in r_values))
    print("-" * 56)
    for fam in ("nextflow", "snakemake", "pegasus"):
        row = []
        summary[fam] = {}
        for r in r_values:
            vals = by_fam[fam][r]
            if vals:
                med = float(np.median(vals))
                summary[fam][r] = {"median": med, "n": len(vals)}
                row.append(f"{med:.3f}")
            else:
                row.append("---")
        print(f"{fam:<12s} | " + " | ".join(f"{x:>6s}" for x in row))

    summary["overall"] = {}
    print(f"{'overall':<12s} | ", end="")
    for r in r_values:
        med = float(np.median(overall[r]))
        summary["overall"][r] = {"median": med, "n": len(overall[r]),
                                  "p05": float(np.percentile(overall[r], 5)),
                                  "p95": float(np.percentile(overall[r], 95))}
        print(f"{med:>6.3f}", end=" | ")
    print()

    out_path = DATA / "wfcommons_multi_r_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
