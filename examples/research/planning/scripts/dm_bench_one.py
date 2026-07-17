"""Subprocess worker: time ONE Algorithm-1 build (HypothesisGraph.build) for a
given |H| and seed via the real library DM core. Prints a single JSON line
{n_h, seed, time_ms}. One build per process so the machine's flaky native heap
can corrupt at most a single replica; the driver retries that replica.

Faithful to the theory (Lemma 1):
  * each hypothesis is a structure with 5 equations and a few INPUT variables
    drawn from earlier hypotheses' outputs, so Out(h)/In(h) are non-trivial and
    the derivability test Out(h_i) & In(h_j) yields real edges;
  * the workflow W is a SPARSE DAG (a pipeline chain with random forward skips):
    O(|H|) edges yet Theta(|H|^2) comparable ordered pairs -- exactly the regime
    where Lemma 1's pair term O(|H|^2 * v_max) dominates and the third
    (reachability) term stays O(|H|^2) (sparse edges);
  * only ``build`` is timed (the algorithm proper); hypothesis generation and the
    per-hypothesis У0 matching are setup, done before the clock starts.
Algorithm 1 (build) decides every edge by a cheap Out&In set test per comparable
pair -- NO per-pair transitive closure (bridge, Theorem thm:build:II) -- so the
measured cost scales as O(|H|^2).

Usage: python dm_bench_one.py <n_h> <seed>
"""
import json
import random
import sys
import time

from hyppo.coa import HypothesisGraph

N_OUT = 5       # equations (outputs) per hypothesis -- bounded structure size
N_IN_MAX = 3    # at most this many input variables per hypothesis


def gen_hyps(n_h, rng, n_out=N_OUT, n_in_max=N_IN_MAX):
    """n_h hypotheses. Outputs globally unique; inputs sampled from previously
    produced outputs (non-empty In(h), real Out&In edges). Each is a valid
    structure: |E|=n_out, |V|=n_out+|In|."""
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


def gen_workflow(n, rng):
    """Sparse workflow DAG: a spanning pipeline chain i->i+1 plus a random
    forward skip per node. |E| = O(n); the transitive order is Theta(n^2)
    comparable pairs -- the O(|H|^2) worst case of Lemma 1 at sparse edges."""
    edges = [(i, i + 1) for i in range(n - 1)]
    for j in range(2, n):
        if rng.random() < 0.5:
            edges.append((rng.randrange(j - 1), j))
    return edges


def make_graph(n_h, rng):
    """Setup only (untimed): build the hypotheses (one У0 matching each) and the
    workflow edges. Returns a graph ready for :meth:`build`."""
    g = HypothesisGraph()
    for eqs in gen_hyps(n_h, rng):
        g.add(eqs)
    for u, v in gen_workflow(n_h, rng):
        g.connect(u, v)
    return g


def main():
    n_h = int(sys.argv[1])
    seed = int(sys.argv[2])
    # Warm up imports / allocator / interning so the timed build reflects steady
    # state, not cold-start cost. Cheap (|H|=20); no heap-corruption risk.
    for _ in range(3):
        make_graph(20, random.Random(7)).build()
    g = make_graph(n_h, random.Random(seed))   # setup: untimed
    t0 = time.perf_counter()
    g.build()                                   # Algorithm 1 proper: timed
    dt = (time.perf_counter() - t0) * 1000.0
    print(json.dumps({"n_h": n_h, "seed": seed, "time_ms": dt}))


if __name__ == "__main__":
    main()
