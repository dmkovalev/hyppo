"""Сквозной пример гл. 2 диссертации: фрагменты графа CRM гл. 4 и его модификации.

Закрепляет утверждения текста разд. 2.1--2.5: алгоритм 1 даёт 19 рёбер при глубине
10; карта производителей находит соперника H4' (модификация 1); расщеплённый блок
S_w <-> f_w даёт конфликт порядка со свидетелем леммы о балансе (избыток 1), который
отбраковывает условие 5, а составная гипотеза H9-12 его устраняет (модификация 2);
каскады правки H10 и нарушения инварианта H3; вытеснение записи H4: N != Cl(N).
"""

import networkx as nx
import pytest

from examples.research.chapter2 import crm_running_example as ex
from examples.research.safe_recompute.common import CRM_EDGES
from hyppo.manager import Manager
from hyppo.ontology.consistency import Status, check_consistency


def _key(s):
    return int("".join(c for c in s if c.isdigit()) or 0)


def test_base_graph_is_the_chapter4_graph():
    _, g = ex.build("base")
    assert g.number_of_nodes() == 16
    assert set(g.edges()) == ex.CRM_EDGES_19 == set(CRM_EDGES)
    assert nx.dag_longest_path_length(g) == 10


def test_rival_found_by_producer_map_not_by_pairs():
    gc, g = ex.build("rival")
    assert ex.competes_pairs(gc) == {("H4", "H4'")}
    # одна задача: парный детектор пару не перечисляет, рёбер между ними нет
    assert not g.has_edge("H4", "H4'") and not g.has_edge("H4'", "H4")
    assert {("H1", "H4'"), ("H4'", "H5"), ("H4'", "H7")} <= set(g.edges())
    assert g.number_of_edges() == 22
    assert ex.signature(gc, "H4'")[1] == ex.signature(gc, "H4")[1] == {"q_c"}


def _cond5(variant):
    gc, g = ex.build(variant)
    hyps, wf = ex.spec(variant)
    ids = {n: i for i, n in enumerate(hyps)}
    tasks = {f"t{k + 1}": [ids[h.name] for h in t] for k, t in enumerate(wf.get_tasks())}
    edges = [(f"t{a + 1}", f"t{b + 1}") for a, b in wf.get_task_edges()]
    sigs = {ids[n]: dict(zip(("in", "out"), ex.signature(gc, n))) for n in hyps}
    adj = {ids[n]: {ids[v] for v in g.successors(n)} for n in hyps}
    res = check_consistency(None, None, adj, run_hermit=False, task_hypotheses=tasks,
                            workflow_edges=edges, artefacts=sigs)
    return res, ids


def test_split_block_is_an_order_conflict_with_balance_witness():
    gc, g = ex.build("split")
    assert ex.order_conflicts(gc) == {("H9*", "H12")}
    assert ex.balance_excess(gc, "H9*", "H12") == 1
    assert not g.has_edge("H12", "H9*")
    res, ids = _cond5("split")
    assert not res.ok and res.status == Status.COND5_VIOLATED
    assert res.details["cond5"]["hypothesis"] == ids["H9*"]
    assert res.details["cond5"]["dependency"] == ids["H12"]


def test_merged_composite_hypothesis_is_a_nontrivial_dm_block():
    gc, g = ex.build("merged")
    assert ex.order_conflicts(gc) == set() and ex.competes_pairs(gc) == set()
    assert ex.dm_blocks(gc, "H9-12") == [{"S_w", "k_rw", "k_ro", "f_w"}]
    assert (g.number_of_nodes(), g.number_of_edges(), nx.dag_longest_path_length(g)) == (13, 15, 8)
    res, _ = _cond5("merged")
    assert res.details["cond5"].startswith("passed")


def test_cascades_of_running_example():
    _, g = ex.build("base")
    assert ex.cascade(g, "H10") == {"H10", "H12", "H13", "H14", "H16"}
    stale = nx.descendants(g, "H3")
    assert sorted(stale, key=_key) == ["H4", "H5", "H7", "H8", "H9", "H10", "H11",
                                        "H12", "H13", "H14", "H16"]


NODES = [f"H{i}" for i in range(1, 17)]


def _models(calls):
    def mk(h):
        def f(cfg):
            calls.append(h)
            return {"r2": 0.9, "v": cfg.get("p", 0)}
        return f
    return {h: mk(h) for h in NODES}


def _run(mgr, cfg, calls):
    calls.clear()
    res = mgr.orchestrate(hypotheses=NODES, workflow_edges=sorted(ex.CRM_EDGES_19),
                          models=_models(calls), config=cfg)
    return sorted(calls, key=_key), res


@pytest.mark.parametrize("mode, expected", [
    ("cascade", ["H4", "H5", "H7", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H16"]),
    ("safe", ["H4"]),
])
def test_eviction_of_crm_record(tmp_path, mode, expected):
    """Запись H4 вытеснена: N = {H4}, Cl(N) --- H4 и десять потомков (thm:safe (iv))."""
    mgr = Manager(db_path=tmp_path / "r.db", plan_mode=mode)
    calls = []
    cfg = {h: {"p": 1} for h in NODES}
    first, res1 = _run(mgr, cfg, calls)
    assert first == NODES
    keys = mgr.cache_keys(nx.DiGraph(sorted(ex.CRM_EDGES_19)), _models([]), cfg)
    assert mgr.repository.evict("H4", keys["H4"])
    again, res2 = _run(mgr, cfg, calls)
    assert again == expected
    assert {h: r["metrics"] for h, r in res2.items()} == {h: r["metrics"] for h, r in res1.items()}
    mgr.close()
