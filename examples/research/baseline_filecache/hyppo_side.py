# examples/research/baseline_filecache/hyppo_side.py
"""Сторона комплекса: гипотезы со структурами (формулы «выход = f(входы)»),
граф алгоритмом 1 (HypothesisLattice), план алгоритмом 4 (plan_cascade).

Сущности хранилища — отдельные гипотезы-хранилища (store_*), выход которых
(gains_uto, corey, wstatus, ...) входит в уравнения потребителей. Ревизия сущности
= изменение гипотезы-хранилища; план — её потомки по derived_by.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from hyppo.coa._base import Equation, Structure
from hyppo.coa.graph import plan_cascade
from hyppo.lattice_constructor._base import HypothesisLattice

from .config import tok


class _Hyp:
    def __init__(self, name: str, formulas: list[str]):
        self.name = name
        self.structure = Structure([Equation(formula=f) for f in formulas])

    def __repr__(self) -> str:
        return self.name


class _Workflow:
    def __init__(self, hyps):
        self._tasks = [[h] for h in hyps]

    def get_tasks(self):
        return self._tasks


@dataclass
class Model:
    nodes: list[str]
    edges: list[tuple[str, str]]
    node_job: dict[str, str]
    build_seconds: float
    extra_declarations: int = 0
    store_nodes: dict[str, set[str]] = field(default_factory=dict)


def _hypotheses(grid: dict) -> tuple[list[_Hyp], dict[str, str], dict[str, set[str]]]:
    hyps: list[_Hyp] = []
    node_job: dict[str, str] = {}
    store: dict[str, set[str]] = {e: set() for e in ("gains", "tau", "well_status", "corey_ref", "ridge")}
    for c in grid["cons"]:
        hyps.append(_Hyp(f"crm_fit_{c}", [f"p1_{c} = crm(prodw, inj)",
                                          f"graw_{c} = crmg(prodw, inj)",
                                          f"traw_{c} = crmt(prodw, inj)"]))
        node_job[f"crm_fit_{c}"] = f"p1/{c}.npy"
        hyps.append(_Hyp(f"store_gains_{c}", [f"gains_{c} = keep(graw_{c})"]))
        hyps.append(_Hyp(f"store_tau_{c}", [f"tau_{c} = keep(traw_{c})"]))
        store["gains"].add(f"store_gains_{c}")
        store["tau"].add(f"store_tau_{c}")
    hyps.append(_Hyp("store_corey", ["corey = keep(lab)"]))
    hyps.append(_Hyp("store_status", ["wstatus = keep(ops)"]))
    hyps.append(_Hyp("par_ridge", ["ridge = keep(cfg)"]))
    store["corey_ref"].add("store_corey")
    store["well_status"].add("store_status")
    store["ridge"].add("par_ridge")
    for c in grid["cons"]:
        for a in grid["alphas"]:
            A = tok(a)
            n = f"fit_liq_{c}_{A}"
            hyps.append(_Hyp(n, [f"hyb_{c}_{A} = liq(p1_{c}, gains_{c}, tau_{c}, wstatus, ridge, alpha_{A})"]))
            node_job[n] = f"hyb/{c}_{a}.npy"
    for no in grid["corey"]:
        for nw in grid["corey"]:
            n = f"fit_wct_{tok(no)}_{tok(nw)}"
            hyps.append(_Hyp(n, [f"wct_{tok(no)}_{tok(nw)} = bl(corey, no_{tok(no)}, nw_{tok(nw)})"]))
            node_job[n] = f"wct/{no}_{nw}.npy"
    for c in grid["cons"]:
        for a in grid["alphas"]:
            for no in grid["corey"]:
                for nw in grid["corey"]:
                    A, NO, NW = tok(a), tok(no), tok(nw)
                    n = f"fit_opr_{c}_{A}_{NO}_{NW}"
                    hyps.append(_Hyp(n, [f"opr_{c}_{A}_{NO}_{NW} = oprf(hyb_{c}_{A}, wct_{NO}_{NW}, corey, wstatus)"]))
                    node_job[n] = f"opr/{c}_{a}_{no}_{nw}.json"
    return hyps, node_job, store


def build(grid: dict) -> Model:
    hyps, node_job, store = _hypotheses(grid)
    t0 = time.perf_counter()
    lattice = HypothesisLattice(hyps, _Workflow(hyps)).lattice
    dt = time.perf_counter() - t0
    return Model(
        nodes=[h.name for h in hyps],
        edges=[(u.name, v.name) for u, v in lattice.edges()],
        node_job=node_job,
        build_seconds=dt,
        store_nodes=store,
    )


def plan_jobs(model: Model, revision: str) -> set[str]:
    changed = model.store_nodes[revision]
    cached = set(model.nodes) - changed
    p_ne = plan_cascade(model.nodes, model.edges, cached)
    return {model.node_job[n] for n in p_ne if n in model.node_job}


def timed_plan(model: Model, revision: str) -> tuple[set[str], float]:
    t0 = time.perf_counter()
    res = plan_jobs(model, revision)
    return res, time.perf_counter() - t0
