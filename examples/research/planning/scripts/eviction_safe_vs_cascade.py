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

Usage: python eviction_safe_vs_cascade.py [steps] [seeds]
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "eviction_safe_vs_cascade.json"

CRM_EDGES = [
    ("H1", "H2"), ("H1", "H3"), ("H2", "H4"), ("H3", "H4"), ("H4", "H5"),
    ("H6", "H5"), ("H7", "H8"), ("H5", "H8"), ("H8", "H9"), ("H9", "H10"),
    ("H10", "H11"), ("H11", "H12"), ("H12", "H13"), ("H13", "H14"),
    ("H15", "H5"), ("H8", "H16"), ("H14", "H16"),
]


def graphs():
    out = {"crm_brugge": ([f"H{i}" for i in range(1, 17)], CRM_EDGES)}
    d = json.loads((DATA / "hand_curated_hypothesis_graphs_26.json").read_text(encoding="utf-8"))
    for name, g in d.items():
        nodes = [h["id"] for h in g["hypotheses"]]
        edges = [(e["src"], e["tgt"]) for e in g["subsumption_edges"] + g["inter_layer_edges"]]
        dag = nx.DiGraph(); dag.add_nodes_from(nodes); dag.add_edges_from(edges)
        if nx.is_directed_acyclic_graph(dag):
            out[f"nfcore_{name}"] = (nodes, edges)
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
            for x in nodes:  # LRU bookkeeping: all records of this run were used
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


def main(steps=40, seeds=10):
    gs = graphs()
    out = {}
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
        out[tag] = {
            "graphs": per_graph,
            "median_ratio_cascade_to_safe": statistics.median(ratios),
            "median_saving_of_safe": statistics.median(saved),
            "crm_ratio": per_graph["crm_brugge"]["cascade_runs_median"] / per_graph["crm_brugge"]["safe_runs_median"],
        }
        print(f"{policy} capacity={mult}|H|: median cascade/safe = {out[tag]['median_ratio_cascade_to_safe']:.2f}, "
              f"median saving = {100 * out[tag]['median_saving_of_safe']:.0f}%, "
              f"CRM ratio = {out[tag]['crm_ratio']:.2f}", flush=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    main(*a)
