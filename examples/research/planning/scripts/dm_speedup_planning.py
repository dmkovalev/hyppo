"""Measure the LIGHT parts of the asymptotic suite via the library DM core and
merge them into asymptotic_results_dm.json (which already holds the heavy build
data from dm_drive.py):
  * Algorithm 2 speedup: incremental add_hypothesis is O(|H|) (one У0 matching +
    a cheap Out&In test against each existing hypothesis); full-rebuild time is
    taken from the existing er_build medians (= Algorithm 1 build, O(|H|^2)).
  * Algorithm 4 planning: |P_ne|/|H| vs cache rate r (graph-only, DM-independent).

Faithful to the theory: hypotheses carry non-trivial inputs (so Out/In edges are
real), and only the add_hypothesis call itself is timed (setup is untimed).

Run: <hyppo-ref>/.venv/Scripts/python.exe examples/research/planning/scripts/dm_speedup_planning.py
"""
import json
import random
import time
from pathlib import Path

from hyppo.coa import HypothesisGraph

N_OUT = 5
N_IN_MAX = 3


def gen_hyps(n_h, rng, n_out=N_OUT, n_in_max=N_IN_MAX):
    """n_h structures with 5 equations and a few inputs drawn from earlier
    outputs (non-empty In(h); |E|=n_out, |V|=n_out+|In|)."""
    produced: list[str] = []
    hyps: list[list[frozenset[str]]] = []
    for idx in range(n_h):
        outs = [f"y{idx}_{k}" for k in range(n_out)]
        cap = min(n_in_max, len(produced))
        ins = rng.sample(produced, rng.randint(0, cap)) if cap else []
        eqs = []
        for k in range(n_out):
            others = [v for v in outs if v != outs[k]]
            extra = rng.sample(others, rng.randint(0, min(2, len(others))))
            eqs.append(frozenset([outs[k], *extra, *ins]))
        hyps.append(eqs)
        produced.extend(outs)
    return hyps


def gen_plan_dag(n, rng, avg_deg=2):
    """Sparse random DAG for the planning demo: each node draws 1..avg_deg edges
    from random earlier nodes. O(n) edges, partial reachability -- gives a
    graded cascade curve |P_ne|/|H| vs r (unlike a spanning chain, which would
    cascade almost everything on the first miss)."""
    edges = []
    for j in range(1, n):
        k = min(j, rng.randint(1, avg_deg))
        for i in rng.sample(range(j), k):
            edges.append((i, j))
    return edges


def add_time_ms(n_h, rng):
    """Algorithm 2: time ONLY add_hypothesis on a pre-built graph of n_h
    hypotheses (setup untimed). One У0 matching + O(|H|) Out&In tests -> O(|H|)."""
    g = HypothesisGraph()
    hyps = gen_hyps(n_h + 1, rng)
    for eqs in hyps[:-1]:
        g.add(eqs)
    t0 = time.perf_counter()
    g.add_hypothesis(hyps[-1])
    return (time.perf_counter() - t0) * 1000.0


def recompute_fraction(n_h, r, rng):
    """Algorithm 4: build a sparse DAG of n_h hypotheses (planning depends only on
    edges), cache a fraction r, return |P_ne|/|H| from HypothesisGraph.plan()."""
    g = HypothesisGraph()
    for k in range(n_h):
        g.add([frozenset({f"v{k}"})])
    for u, v in gen_plan_dag(n_h, rng):
        g.connect(u, v)
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
        adds = [add_time_ms(n_h, random.Random(42 + rep)) for rep in range(n_reps)]
        am = median(adds)
        rbm = d["er_build"][str(n_h)]["median_ms"]   # full rebuild ~ build(|H|)
        speedup[str(n_h)] = {"full_median_ms": rbm, "inc_median_ms": am,
                             "speedup_x": rbm / am if am else 0.0}
        print(f"{n_h:>5} {am:>12.5f} {rbm:>12.4f} {rbm/am if am else 0:>8.1f}", flush=True)

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
