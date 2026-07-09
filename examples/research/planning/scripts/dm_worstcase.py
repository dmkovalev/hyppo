"""Worst-case regime of Lemma 1: structures GROW with |H| (|E_i|=|V_i|=k=alpha*|H|).
With variable-disjoint complete structures and a full DAG, every pair-union is
complete, so Theta(|H|^2) transitive closures run, each O(k^2)=O(|H|^2) for sparse
equations -> total O(|H|^4). Measures the build-time exponent; expect a~4.

Pure stdlib (causal core only). Run on a clean interpreter (no numpy):
  PYTHONPATH=<hyppo-ref> python examples/research/planning/scripts/dm_worstcase.py
"""
import json
import math
import random
import sys
from pathlib import Path

from hyppo.coa import HypothesisGraph

import time


def gen_disjoint_complete(n_h, k, rng):
    """n_h variable-disjoint structures, each k complete equations over its own k
    variables, with sparse internal dependencies (each eq references its own var
    plus 1-2 others from the same structure)."""
    structs = []
    for h in range(n_h):
        names = [f"y{h}_{j}" for j in range(k)]
        eqs = []
        for i in range(k):
            others = (rng.sample([names[j] for j in range(k) if j != i],
                                 min(2, k - 1)) if k > 1 else [])
            eqs.append(frozenset([names[i], *others]))
        structs.append(eqs)
    return structs


def build_cost(n_h, alpha, rng):
    """Full DAG via the library: every i reaches every j>i, so each pair-union
    (disjoint, complete) triggers a DM transitive closure inside
    HypothesisGraph.build(). With k=alpha*|H| growing structures this realises the
    O(|H|^4) worst case of Lemma 1."""
    k = max(2, int(round(alpha * n_h)))
    hyps = gen_disjoint_complete(n_h, k, rng)
    g = HypothesisGraph()
    for h in hyps:
        g.add(h)
    for i in range(n_h):
        for j in range(i + 1, n_h):
            g.connect(i, j)
    g.build()


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    grid = [8, 12, 16, 20, 28, 40, 56]
    n_reps = 8
    medians = []
    print(f"alpha={alpha}  (k = round(alpha*|H|))")
    print(f"{'|H|':>5} {'k':>5} {'median, ms':>12}")
    for n_h in grid:
        ts = []
        for rep in range(n_reps):
            rng = random.Random(42 + rep)
            t0 = time.perf_counter()
            build_cost(n_h, alpha, rng)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        med = ts[len(ts) // 2]
        medians.append(med)
        print(f"{n_h:>5} {max(2, round(alpha*n_h)):>5} {med:>12.4f}", flush=True)

    lh = [math.log(h) for h in grid]
    lt = [math.log(m) for m in medians]
    a = slope(lh, lt)
    mx, my = sum(lh) / len(lh), sum(lt) / len(lt)
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(lh, lt))
    ss_tot = sum((y - my) ** 2 for y in lt)
    r2 = 1 - ss_res / ss_tot
    print(f"\nWorst case (growing structures, k=alpha*|H|): a = {a:.3f}  R2={r2:.4f}")

    out = {"alpha": alpha, "grid": grid, "n_reps": n_reps,
           "medians_ms": medians, "a": a, "R2": r2}
    dst = Path(__file__).resolve().parent.parent / "data" / "worstcase_dm.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
