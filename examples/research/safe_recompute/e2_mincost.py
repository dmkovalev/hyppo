"""E2: безопасный план минимальной стоимости против N, каскада и полного пересчёта.

Планы: mincost (plan_min_cost), N+чтение остального (пересчёт только неактуальных,
остальное читается из кэша), каскад+чтение (N и все потомки), полный пересчёт.
Стоимость пересчёта w: для графа CRM -- время компонент из data/timing_brugge.json
(по гипотезам, CRM_COMPONENT); для графов nf-core -- lognormal(0,1). Стоимость
чтения l = rho * size * median(w), size ~ lognormal(0,1.5), rho in {0.01, 0.1, 1}.
Доля вытеснения r in {0.5, 0.7, 0.9}: каждая вершина независимо неактуальна с
вероятностью r (равномерное вытеснение). Цели: все вершины ("all") или стоки ("sinks").

Допустимость планов сравнения: для N и каскада P замыкается добавлением неактуальных
предшественников (close_inputs), чтобы выполнялось (C2); читаются актуальные входы P и
цели вне P. Для целей "sinks" N-план остаётся допустимым, поскольку P содержит все
неактуальные вершины, а чтение покрывает актуальные входы.
"""
from __future__ import annotations

import random
import statistics
import sys

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
    path = DATA / "timing_brugge.json"
    pc = load_json(path).get("per_component_ms", {}) if path.exists() else {}
    return {h: max(pc.get(c, 1.0), 1e-3) for h, c in CRM_COMPONENT.items()}


def close_inputs(g, P, actual):
    P = set(P)
    while True:
        add = {u for h in P for u in g.predecessors(h) if u not in actual} - P
        if not add:
            return P
        P |= add


def plan_with_reads(g, P, targets, actual):
    needed = set(targets) | {u for h in P for u in g.predecessors(h)}
    return set(P), (needed - set(P)) & set(actual)


def one_case(g, w, l, targets, actual):
    nodes = set(g.nodes)
    N = nodes - set(actual)
    casc = set(N)
    for h in N:
        casc |= set(__import__("networkx").descendants(g, h))
    plans = {}
    for name, base in (("N_read_rest", N), ("cascade_read_rest", casc)):
        P = close_inputs(g, base, actual)
        plans[name] = plan_with_reads(g, P, targets, actual)
    plans["full"] = (nodes, set())
    plans["mincost"] = plan_min_cost(g, set(actual), w, l, set(targets))
    costs = {k: plan_cost(P, R, w, l) for k, (P, R) in plans.items()}
    feas = all(is_feasible(g, actual, targets, P, R) for P, R in plans.values())
    return costs, feas


def main(seeds=20, out=None, graphs_limit=None):
    graphs = {"crm_brugge": crm_graph()}
    nf = nfcore_graphs()
    names = list(nf)
    if graphs_limit:
        names = names[:graphs_limit - 1]
    for n in names:
        graphs[f"nfcore_{n}"] = nf[n]
    crm_w = crm_costs()
    rng = random.Random(20261012)  # noqa: F841 (зафиксированная база; сиды -- r_seed)
    rows = []
    for gname, g in graphs.items():
        nodes = sorted(g.nodes, key=str)
        sinks = {h for h in nodes if g.out_degree(h) == 0}
        for seed in range(int(seeds)):
            r_seed = random.Random(seed)
            if gname == "crm_brugge":
                w = dict(crm_w)
            else:
                w = {h: r_seed.lognormvariate(0, 1) for h in nodes}
            med = statistics.median(w.values())
            size = {h: r_seed.lognormvariate(0, 1.5) for h in nodes}
            for rho in RHOS:
                l = {h: rho * size[h] * med for h in nodes}
                for r in RS:
                    actual = {h for h in nodes if r_seed.random() >= r}
                    for tname, targets in (("all", set(nodes)), ("sinks", sinks)):
                        c, feas = one_case(g, w, l, targets, actual)
                        rows.append({
                            "graph": gname, "seed": seed, "rho": rho, "r": r,
                            "targets": tname, "cost_mincost": c["mincost"],
                            "cost_N_read_rest": c["N_read_rest"],
                            "cost_cascade_read_rest": c["cascade_read_rest"],
                            "cost_full": c["full"], "feasible_all": feas,
                        })
    summary = {}
    for rho in RHOS:
        for t in ("all", "sinks"):
            sel = [x for x in rows if x["rho"] == rho and x["targets"] == t
                   and x["cost_N_read_rest"] > 0]
            summary[f"{rho}:{t}"] = {
                "n": len(sel),
                "median_gain_vs_N": statistics.median(
                    1 - x["cost_mincost"] / x["cost_N_read_rest"] for x in sel),
                "median_gain_vs_cascade": statistics.median(
                    1 - x["cost_mincost"] / x["cost_cascade_read_rest"]
                    for x in sel if x["cost_cascade_read_rest"] > 0),
            }
    res = {"rows": rows, "summary": summary}
    save_json(res, out if out is not None else DATA / "e2_mincost.json")
    return res


if __name__ == "__main__":
    main(*[int(x) for x in sys.argv[1:2]])
