"""Subprocess worker: одна точка (|H|, dag_kind, seed) → JSON в stdout."""
import json
import random
import sys
import time
from collections import defaultdict

import numpy as np


class S:
    def __init__(self, eqs):
        self.equations = eqs
        self.variables = set()
        for e in eqs:
            self.variables |= e

    def is_complete(self):
        return len(self.equations) == len(self.variables)


def union_s(a, b):
    return S(a.equations + b.equations)


def tc(struct):
    if not struct.is_complete():
        return 0
    eqs = struct.equations
    mapping = {}
    used = set()
    for idx, eq in enumerate(eqs):
        for nm in eq:
            if nm not in used:
                mapping[idx] = nm
                used.add(nm)
                break
    direct = set()
    for idx, eq in enumerate(eqs):
        if idx in mapping:
            tgt = mapping[idx]
            for nm in eq:
                if nm != tgt:
                    direct.add((nm, tgt))
    adj = defaultdict(set)
    for s_, d in direct:
        adj[s_].add(d)
    cnt = 0
    for sv in struct.variables:
        seen = set()
        st = [sv]
        while st:
            c = st.pop()
            for nb in adj[c]:
                if nb not in seen:
                    seen.add(nb)
                    cnt += 1
                    st.append(nb)
    return cnt


def build_lattice(workflow_edges, hyps):
    n = len(hyps)
    tc_calls = 0
    reach = defaultdict(set)
    adj = defaultdict(set)
    for u, v in workflow_edges:
        adj[u].add(v)
    for start in range(n):
        seen = set()
        st = [start]
        while st:
            c = st.pop()
            for nb in adj[c]:
                if nb not in seen:
                    seen.add(nb)
                    reach[start].add(nb)
                    st.append(nb)
    for i in range(n):
        for j in reach[i]:
            su = union_s(hyps[i], hyps[j])
            if su.is_complete():
                tc_calls += 1
                tc(su)
    return tc_calls


def add_hyp(_edges, hyps, h_add):
    n = len(hyps)
    tc_calls = 0
    for i in range(n):
        su = union_s(hyps[i], h_add)
        if su.is_complete():
            tc_calls += 1
            tc(su)
    return tc_calls


def gen_struct(n_eq=5, pool=20):
    av = [f"x_{k}" for k in range(pool)]
    chosen = random.sample(av, n_eq)
    eqs = []
    for i in range(n_eq):
        ev = {chosen[i]}
        nx = random.randint(1, min(3, n_eq - 1))
        ex = random.sample([v for v in chosen if v != chosen[i]], nx)
        ev.update(ex)
        eqs.append(ev)
    return S(eqs)


def gen_er(n, p=0.3):
    e = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                e.append((i, j))
    return e


def gen_ba(n, m=2):
    e = []
    deg = [0] * n
    init = min(m + 1, n)
    for i in range(init):
        for j in range(i + 1, init):
            e.append((i, j))
            deg[i] += 1
            deg[j] += 1
    for new in range(init, n):
        total = sum(deg[:new])
        if total == 0:
            tg = list(range(min(m, new)))
        else:
            probs = [deg[k] / total for k in range(new)]
            tg = list(np.random.choice(new, size=min(m, new),
                                        replace=False, p=probs))
        for t in tg:
            e.append((t, new))
            deg[t] += 1
            deg[new] += 1
    return e


def bench_build(n, kind, seed, p=0.3):
    random.seed(seed)
    np.random.seed(seed)
    hyps = [gen_struct() for _ in range(n)]
    if kind == "er":
        edges = gen_er(n, p=p)
    else:
        edges = gen_ba(n, m=2)
    t0 = time.perf_counter()
    build_lattice(edges, hyps)
    return time.perf_counter() - t0


def bench_speedup(n, seed, p=0.3):
    random.seed(seed)
    np.random.seed(seed)
    hyps = [gen_struct() for _ in range(n)]
    edges = gen_er(n, p=p)
    new_h = gen_struct()
    # full rebuild
    t0 = time.perf_counter()
    hf = hyps + [new_h]
    ef = edges + [(j, n) for j in range(n) if random.random() < p]
    build_lattice(ef, hf)
    t_full = time.perf_counter() - t0
    # incremental
    t0 = time.perf_counter()
    add_hyp(edges, hyps, new_h)
    t_inc = time.perf_counter() - t0
    return t_full, t_inc


def main():
    op = sys.argv[1]
    n = int(sys.argv[2])
    n_reps = int(sys.argv[3])
    seed_base = int(sys.argv[4])
    kind = sys.argv[5] if len(sys.argv) > 5 else "er"
    if op == "build":
        times = []
        for i in range(n_reps):
            times.append(bench_build(n, kind, seed_base + i))
        print(json.dumps({"times_s": times}))
    elif op == "speedup":
        full = []
        inc = []
        for i in range(n_reps):
            f, ic = bench_speedup(n, seed_base + i)
            full.append(f)
            inc.append(ic)
        print(json.dumps({"full_s": full, "inc_s": inc}))


if __name__ == "__main__":
    main()
