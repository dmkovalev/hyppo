"""E2: безопасный план минимальной стоимости против простых базовых линий.

Планы (все допустимы по (C1)--(C3), проверяется is_feasible):
  mincost            -- plan_min_cost (теорема 3);
  N_read             -- план теоремы 1: пересчёт всех неактуальных N + чтение актуальных
                        входов и целей;
  cascade_read       -- N и все их потомки + чтение;
  N_needed_read      -- пересчёт только N ∩ (T ∪ предки T) + чтение нужных актуальных
                        входов и целей (оптимум при l = 0 для заданных целей);
  full               -- пересчёт всех вершин;
  best_simple        -- min(N_needed_read, full) (лучшая простая линия).
Основной эффект теоремы 3 -- mincost против best_simple.

Стоимость пересчёта w: для графа CRM -- время компонент из data/timing_brugge.json;
компонента вычисляется один раз и разделяется между своими гипотезами, поэтому
w[h] = время компоненты / число гипотез компоненты (CRM/6, COREY/2, BL/2, остальные /1),
сумма по гипотезам равна измеренному полному времени. Для графов nf-core -- lognormal(0,1).
Стоимость чтения l = rho * size * median(w), size ~ lognormal(0,1.5), rho in {0.01, 0.1, 1}.
Равномерное вытеснение: вершина неактуальна, если её u ~ U(0,1) < r, r in {0.5, 0.7, 0.9}
(u общие для всех r и rho; множество actual фиксируется на (граф, сид, r) и одинаково для
всех rho и обоих наборов целей). Цели: все вершины ("all") или стоки ("sinks").
"""
from __future__ import annotations

import random
import statistics
import sys
from collections import Counter

import networkx as nx

from hyppo.planner.mincost import is_feasible, plan_cost, plan_min_cost
from examples.research.safe_recompute.common import (
    DATA, crm_graph, load_json, nfcore_graphs, save_json,
)

CRM_COMPONENT = {
    "H1": "CRM", "H2": "CRM", "H3": "CRM", "H4": "CRM", "H5": "CRM", "H6": "CRM",
    "H7": "ML", "H8": "FUS", "H9": "MB", "H10": "COREY", "H11": "COREY",
    "H12": "BL", "H13": "BL", "H14": "WCTF", "H15": "GTM", "H16": "OPR",
}
RHOS = (0.01, 0.1, 1.0)
RS = (0.5, 0.7, 0.9)


def crm_costs():
    """w[h] = время компоненты / число гипотез компоненты; без запасных значений."""
    path = DATA / "timing_brugge.json"
    if not path.exists():
        raise FileNotFoundError(f"нет {path}: запустите examples/brugge_reuse_timing.py Brugge")
    pc = load_json(path).get("per_component_ms")
    if pc is None:
        raise KeyError(f"в {path} нет ключа per_component_ms")
    share = Counter(CRM_COMPONENT.values())
    missing = sorted(set(share) - set(pc))
    if missing:
        raise KeyError(f"в per_component_ms нет компонент: {missing}")
    return {h: max(pc[c] / share[c], 1e-3) for h, c in CRM_COMPONENT.items()}


def plan_with_reads(g, P, targets, actual):
    """P пересчитывается; читаются актуальные цели и входы P вне P."""
    needed = set(targets) | {u for h in P for u in g.predecessors(h)}
    return set(P), (needed - set(P)) & set(actual)


def one_case(g, w, l, targets, actual):
    nodes = set(g.nodes)
    N = nodes - set(actual)
    casc = set(N)
    for h in N:
        casc |= nx.descendants(g, h)
    anc = set(targets)
    for t in targets:
        anc |= nx.ancestors(g, t)
    plans = {
        "N_read": plan_with_reads(g, N, targets, actual),
        "cascade_read": plan_with_reads(g, casc, targets, actual),
        "N_needed_read": plan_with_reads(g, N & anc, targets, actual),
        "full": (nodes, set()),
        "mincost": plan_min_cost(g, set(actual), w, l, set(targets)),
    }
    costs = {k: plan_cost(P, R, w, l) for k, (P, R) in plans.items()}
    costs["best_simple"] = min(costs["N_needed_read"], costs["full"])
    feas = all(is_feasible(g, actual, targets, P, R) for P, R in plans.values())
    return costs, feas


def _med(xs):
    xs = list(xs)
    return statistics.median(xs) if xs else None


def summarize(rows):
    summary = {}
    for rho in RHOS:
        for t in ("all", "sinks"):
            sel = [x for x in rows if x["rho"] == rho and x["targets"] == t]
            e = {"n": len(sel),
                 "frac_N_read_gt_full": sum(x["cost_N_read"] > x["cost_full"] + 1e-9
                                            for x in sel) / len(sel)}
            for name in ("N_read", "N_needed_read", "best_simple", "cascade_read"):
                e[f"median_gain_vs_{name}"] = _med(
                    1 - x["cost_mincost"] / x[f"cost_{name}"]
                    for x in sel if x[f"cost_{name}"] > 0)
            summary[f"{rho}:{t}"] = e
    return summary


def main(seeds=20, out=None, graphs_limit=None):
    graphs = {"crm_brugge": crm_graph()}
    nf = nfcore_graphs()
    names = list(nf)
    if graphs_limit:
        names = names[:graphs_limit - 1]
    for n in names:
        graphs[f"nfcore_{n}"] = nf[n]
    crm_w = crm_costs()
    rows = []
    for gname, g in graphs.items():
        nodes = sorted(g.nodes, key=str)
        sinks = {h for h in nodes if g.out_degree(h) == 0}
        for seed in range(int(seeds)):
            rs = random.Random(f"{gname}:{seed}")
            w = dict(crm_w) if gname == "crm_brugge" else {
                h: rs.lognormvariate(0, 1) for h in nodes}
            med = statistics.median(w.values())
            size = {h: rs.lognormvariate(0, 1.5) for h in nodes}
            u = {h: rs.random() for h in nodes}
            for r in RS:
                actual = {h for h in nodes if u[h] >= r}
                for rho in RHOS:
                    l = {h: rho * size[h] * med for h in nodes}
                    for tname, targets in (("all", set(nodes)), ("sinks", sinks)):
                        c, feas = one_case(g, w, l, targets, actual)
                        rows.append({
                            "graph": gname, "seed": seed, "rho": rho, "r": r,
                            "targets": tname, "cost_mincost": c["mincost"],
                            "cost_N_read": c["N_read"],
                            "cost_N_needed_read": c["N_needed_read"],
                            "cost_best_simple": c["best_simple"],
                            "cost_cascade_read": c["cascade_read"],
                            "cost_full": c["full"], "feasible_all": feas,
                        })
    res = {"rows": rows, "summary": summarize(rows)}
    save_json(res, out if out is not None else DATA / "e2_mincost.json")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 20, a[1] if len(a) > 1 else None,
         int(a[2]) if len(a) > 2 else None)
