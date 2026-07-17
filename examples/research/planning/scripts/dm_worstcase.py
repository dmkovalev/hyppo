"""Worst-case regime of Lemma 1 for the theory-faithful build: structures GROW
with |H| (k = alpha*|H| equations and ~k inputs each, so s_max, v_max = Theta(|H|))
over a FULL workflow DAG (every ordered pair comparable, Theta(|H|^2) pairs).

Algorithm 1 decides each edge by the cheap Out(h_i) & In(h_j) test -- now costing
O(v_max)=O(|H|) because In(h) is Theta(|H|) -- with NO per-pair transitive closure.
Hence build costs Theta(|H|^2 * v_max) + reachability = Theta(|H|^3): the leading
worst-case term of Lemma 1 (v_max=Theta(|H|)). Expect a~3 (NOT the a~4 of the old
per-pair-closure implementation). Only ``build`` is timed; generation and the
per-hypothesis У0 matching are untimed setup.

Pure stdlib (causal core only). Run on a clean interpreter (no numpy):
  <hyppo-ref>/.venv/Scripts/python.exe examples/research/planning/scripts/dm_worstcase.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

from hyppo.coa import HypothesisGraph


def gen_growing(n_h, k, rng):
    """n_h structures of k equations each; outputs globally unique, inputs sampled
    from earlier outputs so In(h) has Theta(k) entries overlapping earlier Out(h),
    making the Out&In test genuine O(v_max)=O(|H|) work. |E|=k, |V|=k+|In|."""
    produced = []
    hyps = []
    for h in range(n_h):
        outs = [f"y{h}_{j}" for j in range(k)]
        cap = min(k, len(produced))
        ins = rng.sample(produced, cap) if cap else []
        eqs = []
        for i in range(k):
            eq = [outs[i]]
            if k > 1:
                eq.append(outs[(i + 1) % k])          # one sibling: keeps matchable
            if ins:
                eq.append(ins[i % len(ins)])          # spread inputs -> In(h)=Theta(k)
            eqs.append(frozenset(eq))
        hyps.append(eqs)
        produced.extend(outs)
    return hyps


def make_graph(n_h, alpha, rng):
    """Setup only (untimed): growing structures + a FULL DAG (every i<j)."""
    k = max(2, int(round(alpha * n_h)))
    g = HypothesisGraph()
    for h in gen_growing(n_h, k, rng):
        g.add(h)                                      # one У0 matching each
    for i in range(n_h):
        for j in range(i + 1, n_h):
            g.connect(i, j)
    return g


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    grid = [10, 16, 24, 36, 54, 80, 120]
    n_reps = 8
    medians = []
    print(f"alpha={alpha}  (k = round(alpha*|H|))")
    print(f"{'|H|':>5} {'k':>5} {'median, ms':>12}")
    for n_h in grid:
        ts = []
        for rep in range(n_reps):
            g = make_graph(n_h, alpha, random.Random(42 + rep))   # setup: untimed
            t0 = time.perf_counter()
            g.build()                                             # timed
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
