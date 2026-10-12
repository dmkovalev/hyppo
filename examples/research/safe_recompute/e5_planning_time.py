"""E5: время планирования, цепочечные ключи кэша против прямого сравнения контекстов."""
from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import networkx as nx
import numpy as np

from examples.research.safe_recompute.common import save_json
from hyppo.planner.actuality import chained_keys


def _make(n: int):
    rng = random.Random(n)
    edges = {(i, i + 1) for i in range(n - 1)}
    for i in range(n - 1):
        for _ in range(1):  # цепь даёт 1, плюс ~1 случайное ребро: средняя исходящая степень ~2
            j = rng.randint(i + 1, n - 1)
            edges.add((i, j))
    params = {i: rng.randint(0, 10**6) for i in range(n)}
    changed = rng.randrange(n)
    return sorted(edges), params, changed


def _chained(n, edges, before_keys, params2):
    cfg = {h: {"p": v} for h, v in params2.items()}
    keys = chained_keys(range(n), edges, cfg, {h: "m" for h in range(n)})
    return {h for h in range(n) if keys[h] != before_keys[h]}


def _direct(g, params1, params2):
    stale = set()
    for h in g.nodes:
        anc = nx.ancestors(g, h) | {h}
        if frozenset((a, params1[a]) for a in anc) != frozenset((a, params2[a]) for a in anc):
            stale.add(h)
    return stale


def main(sizes=(100, 300, 1000, 3000, 10000), reps=5, out=None):
    sizes = list(sizes)
    res = {k: {"sizes": [], "times": []} for k in ("chained", "direct")}
    for n in sizes:
        edges, params, changed = _make(n)
        params2 = dict(params)
        params2[changed] = params[changed] + 1
        cfg1 = {h: {"p": v} for h, v in params.items()}
        before = chained_keys(range(n), edges, cfg1, {h: "m" for h in range(n)})
        g = nx.DiGraph(edges)
        g.add_nodes_from(range(n))
        tc, td, ref = [], [], None
        for _ in range(reps):
            t = time.perf_counter()
            nc = _chained(n, edges, before, params2)
            tc.append(time.perf_counter() - t)
        res["chained"]["sizes"].append(n)
        res["chained"]["times"].append(statistics.median(tc))
        for _ in range(reps):
            t = time.perf_counter()
            nd = _direct(g, params, params2)
            td.append(time.perf_counter() - t)
        assert nc == nd, (n, len(nc), len(nd))
        res["direct"]["sizes"].append(n)
        res["direct"]["times"].append(statistics.median(td))
        print(n, res["chained"]["times"][-1], res["direct"]["times"][-1:], len(nc), flush=True)
    for k in res:
        s, t = res[k]["sizes"], res[k]["times"]
        res[k]["slope"] = float(np.polyfit(np.log(s), np.log(t), 1)[0])
    path = Path(out) if out else Path(__file__).resolve().parent / "data" / "e5_planning_time.json"
    save_json(res, path)
    return res


if __name__ == "__main__":
    main()
