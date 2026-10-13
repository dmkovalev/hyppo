"""Effect of theorem thm:safe under cache eviction: plan_mode 'cascade' vs 'safe'.

Workload: a sequence of edits (one hypothesis parameter changed per step, values
from a small set so earlier configurations recur) on real hypothesis graphs:
26 hand-curated nf-core graphs and the 16-hypothesis CRM graph (Brugge). The
result repository has a bounded capacity C (LRU eviction of records). Each step
runs Manager.orchestrate in both modes on separate repositories with the same
policy and counts model runs.

Correctness check: every model returns the digest of its chained key, i.e. a
value that depends on the parameters and models of the hypothesis and all its
sources. After every step all results in both modes must equal the current
digests: a stale reused result would fail the check.

Usage: python -m examples.research.safe_recompute.e1_eviction [steps] [seeds]
       (defaults 40 and 10; writes data/e1_eviction.json)
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import networkx as nx

from hyppo.manager import Manager

from examples.research.safe_recompute.common import DATA, crm_graph, nfcore_graphs, save_json


def graphs(limit=None):
    g = crm_graph()
    out = {"crm_brugge": (list(g.nodes), list(g.edges))}
    for name, dag in nfcore_graphs().items():
        if limit is not None and len(out) >= limit:
            break
        out[f"nfcore_{name}"] = (list(dag.nodes), list(dag.edges))
    return out


def digest(key) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_workload(nodes, edges, mode, capacity, steps, seed, policy="lru"):
    rng = random.Random(seed)
    g = nx.DiGraph(); g.add_nodes_from(nodes); g.add_edges_from(edges)
    current = {}
    calls = []

    def mk(h):
        def f(cfg):
            calls.append(h)
            return {"r2": 0.9, "v": digest(current[h])}
        return f

    models = {h: mk(h) for h in nodes}
    model_ids = {h: f"model_{h}" for h in nodes}
    lru: OrderedDict = OrderedDict()
    evict_rng = random.Random(seed + 10_000)
    cfg = {h: {"p": 0} for h in nodes}
    with tempfile.TemporaryDirectory() as tmp:
        mgr = Manager(db_path=Path(tmp) / "r.db", plan_mode=mode)
        runs = []
        for step in range(steps):
            if step:
                h = rng.choice(nodes)
                cfg = dict(cfg); cfg[h] = {"p": rng.randrange(3)}
            keys = Manager.cache_keys(g, models, cfg, model_ids)
            current.clear(); current.update(keys)
            calls.clear()
            res = mgr.orchestrate(hypotheses=nodes, workflow_edges=edges, models=models,
                                  config=cfg, model_ids=model_ids)
            runs.append(len(calls))
            for x in nodes:  # correctness: every result is actual for the current keys
                v = res[x]["metrics"].get("v")
                assert v == digest(keys[x]), (mode, step, x)
            # LRU bookkeeping: all records of this run were used. They are touched in
            # node-list order (sources before descendants within one configuration; in
            # all 26 nf-core graphs the list is topological, in CRM except H6, H15 -> H5),
            # so a source's record is older than its descendants' and is evicted first.
            # Known property of the workload; it affects the capacity C = 2|V| points.
            for x in nodes:
                k = (x, digest(keys[x]))
                lru.pop(k, None); lru[k] = keys[x]
            while len(lru) > capacity:
                if policy == "lru":
                    (x, _), key = lru.popitem(last=False)
                else:  # random eviction (e.g. size-driven), same stream in both modes
                    k = evict_rng.choice(sorted(lru, key=str))
                    key = lru.pop(k); x = k[0]
                mgr.repository.evict(x, key)
        mgr.close()
    return runs


def main(steps=40, seeds=10, out=None, graphs_limit=None):
    gs = graphs(graphs_limit)
    path = out or DATA / "e1_eviction.json"
    res_all = {}
    for policy, mult in [(p, m) for p in ("lru", "random") for m in (0.25, 0.5, 0.75, 1.0, 2.0)]:
        per_graph = {}
        for name, (nodes, edges) in gs.items():
            cap = max(1, int(mult * len(nodes)))
            tot = {"cascade": [], "safe": []}
            for seed in range(seeds):
                for mode in tot:
                    tot[mode].append(sum(run_workload(nodes, edges, mode, cap, steps, seed, policy)))
            per_graph[name] = {
                "n": len(nodes), "capacity": cap,
                "cascade_runs_median": statistics.median(tot["cascade"]),
                "safe_runs_median": statistics.median(tot["safe"]),
                "full_runs": steps * len(nodes),
            }
        ratios = [v["cascade_runs_median"] / v["safe_runs_median"] for v in per_graph.values()]
        saved = [1 - v["safe_runs_median"] / v["cascade_runs_median"] for v in per_graph.values()]
        tag = f"{policy}:{mult}"
        res_all[tag] = {
            "graphs": per_graph,
            "median_ratio_cascade_to_safe": statistics.median(ratios),
            "median_saving_of_safe": statistics.median(saved),
            "saving_iqr": ([q[0], q[2]] if len(saved) >= 2 and (q := statistics.quantiles(saved, n=4))
                           else [saved[0], saved[0]]),
            "crm_ratio": per_graph["crm_brugge"]["cascade_runs_median"] / per_graph["crm_brugge"]["safe_runs_median"],
        }
        print(f"{policy} capacity={mult}|H|: median cascade/safe = {res_all[tag]['median_ratio_cascade_to_safe']:.2f}, "
              f"median saving = {100 * res_all[tag]['median_saving_of_safe']:.0f}%, "
              f"CRM ratio = {res_all[tag]['crm_ratio']:.2f}", flush=True)
    save_json(res_all, path)
    print("written", path)
    return res_all


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    main(*a)
