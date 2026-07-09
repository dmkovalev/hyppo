"""Measure the LIGHT parts of the asymptotic suite via the library DM core and
merge them into asymptotic_results_dm.json (which already holds the heavy build
data from dm_drive.py):
  * Algorithm 2 speedup: incremental add is O(|H|) (light); full-rebuild time is
    taken from the existing er_build medians (= Algorithm 1, already measured).
  * Algorithm 4 planning: |P_ne|/|H| vs cache rate r (graph-only, DM-independent).

Both are low-allocation, so a single clean-interpreter process is stable (unlike
the O(|H|^2) build, which trips the machine's native-heap flakiness at scale).

Run: PYTHONPATH=<hyppo-ref> C:/Python314/python.exe examples/research/planning/scripts/dm_speedup_planning.py
"""
import json
import math
import random
import time
from pathlib import Path

from hyppo.coa import HypothesisGraph


def gen_struct(rng, n_eq=5, pool=20):
    av = [f"x_{k}" for k in range(pool)]
    chosen = rng.sample(av, n_eq)
    eqs = []
    for i in range(n_eq):
        extras = rng.sample([v for v in chosen if v != chosen[i]],
                            rng.randint(1, min(3, n_eq - 1)))
        eqs.append(frozenset([chosen[i], *extras]))
    return eqs


def add_cost(n_h, rng):
    """Algorithm 2 via the library: build a graph of n_h hypotheses, then
    HypothesisGraph.add_hypothesis() -- O(|H|) incremental unions/closures."""
    g = HypothesisGraph()
    for _ in range(n_h):
        g.add(gen_struct(rng))
    g.add_hypothesis(gen_struct(rng))


def recompute_fraction(n_h, r, rng, p=0.2):
    """Algorithm 4 via the library: build an ER DAG of n_h trivial hypotheses
    (planning depends only on edges), cache a fraction r, and return the
    recompute fraction |P_ne|/|H| from HypothesisGraph.plan()."""
    g = HypothesisGraph()
    for k in range(n_h):
        g.add([frozenset({f"v{k}"})])
    for i in range(n_h):
        for j in range(i + 1, n_h):
            if rng.random() < p:
                g.connect(i, j)
    cached = set(rng.sample(range(n_h), int(r * n_h)))
    return len(g.plan(cached)) / max(n_h, 1)


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def main():
    dst = Path(__file__).resolve().parent.parent / "data" / "asymptotic_results_dm.json"
    d = json.loads(dst.read_text())
    grid = sorted(int(k) for k in d["er_build"].keys())
    n_reps = 30

    print("=== Algorithm 2 speedup (add is O(|H|); rebuild from er_build) ===", flush=True)
    speedup = {}
    print(f"{'|H|':>5} {'add ms':>12} {'rebuild ms':>12} {'k':>8}", flush=True)
    for n_h in grid:
        adds = []
        for rep in range(n_reps):
            rng = random.Random(42 + rep)
            t0 = time.perf_counter()
            add_cost(n_h, rng)
            adds.append((time.perf_counter() - t0) * 1000.0)
        am = median(adds)
        rbm = d["er_build"][str(n_h)]["median_ms"]   # full rebuild ~ build(|H|)
        speedup[str(n_h)] = {"full_median_ms": rbm, "inc_median_ms": am,
                             "speedup_x": rbm / am if am else 0.0}
        print(f"{n_h:>5} {am:>12.4f} {rbm:>12.4f} {rbm/am if am else 0:>8.1f}", flush=True)

    print("\n=== Algorithm 4 planning cascade ===", flush=True)
    rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    planning = {}
    for n_h in [10, 20, 50, 100]:
        planning[str(n_h)] = {}
        for r in rates:
            vals = [recompute_fraction(n_h, r, random.Random(42 + k)) for k in range(30)]
            planning[str(n_h)][str(r)] = median(vals)
        print(f"  |H|={n_h}: rho@r=0.5 = {planning[str(n_h)]['0.5']:.2f}", flush=True)

    d["speedup"] = speedup
    d["planning"] = planning
    d["planning_rates"] = rates
    dst.write_text(json.dumps(d, indent=2))
    print(f"\nmerged speedup + planning into {dst}", flush=True)


if __name__ == "__main__":
    main()
