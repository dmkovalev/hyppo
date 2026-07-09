"""Subprocess worker: time ONE Algorithm-1 build (HypothesisGraph.build) for a
given |H| and seed via the real library DM core. Prints a single JSON line
{n_h, seed, time_ms}. One build per process so the machine's flaky native heap
can corrupt at most a single replica; the driver retries that replica.

Usage: python dm_bench_one.py <n_h> <seed>
"""
import json
import random
import sys
import time

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


def gen_er(n, rng, p=0.3):
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def build_lattice_cost(n_h, rng, p=0.3):
    g = HypothesisGraph()
    for _ in range(n_h):
        g.add(gen_struct(rng))
    for u, v in gen_er(n_h, rng, p):
        g.connect(u, v)
    g.build()


def main():
    n_h = int(sys.argv[1])
    seed = int(sys.argv[2])
    # Warm up imports / allocator / interning in this fresh process so the timed
    # build reflects steady state, not cold-start cost (which would inflate the
    # sub-millisecond small-|H| builds and flatten the fitted exponent). The
    # warmup is cheap (|H|=20) -- no heap-corruption risk.
    for _ in range(3):
        build_lattice_cost(20, random.Random(7))
    rng = random.Random(seed)
    t0 = time.perf_counter()
    build_lattice_cost(n_h, rng)
    dt = (time.perf_counter() - t0) * 1000.0
    print(json.dumps({"n_h": n_h, "seed": seed, "time_ms": dt}))


if __name__ == "__main__":
    main()
