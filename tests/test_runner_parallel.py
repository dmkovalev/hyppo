"""Level-parallel execution of P_ne (Runner(max_workers > 1)).

The plan semantics (thm:1) only require that a hypothesis runs after its
sources; hypotheses of one topological level are independent and may run
concurrently. These tests pin: same results as sequential execution, the
dependency order, real concurrency, the SKIPPED cascade, repository writes
from the calling thread, and the Manager pass-through.
"""

import random
import threading
import time

import networkx as nx
import pytest

from hyppo.manager import Manager
from hyppo.metadata_repository import MetadataRepository
from hyppo.runner import Runner


def _random_dag(n, p, seed):
    rng = random.Random(seed)
    nodes = [f"h{i}" for i in range(n)]
    edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    return nodes, edges


def _models(nodes, log=None, delay=0.0):
    def mk(h):
        def f(cfg):
            t0 = time.perf_counter()
            if delay:
                time.sleep(delay)
            if log is not None:
                log.append((h, t0, time.perf_counter()))
            return {"r2": 0.9, "value": cfg.get("x", 0) + len(h)}
        return f
    return {h: mk(h) for h in nodes}


@pytest.mark.parametrize("seed", range(5))
def test_parallel_results_equal_sequential(seed):
    nodes, edges = _random_dag(25, 0.15, seed)
    order = list(nx.topological_sort(nx.DiGraph(edges) if edges else nx.DiGraph()))
    order += [h for h in nodes if h not in order]
    plan = {"p_ne": order, "p_e": set()}
    cfg = {h: {"x": i} for i, h in enumerate(nodes)}
    seq = Runner(max_workers=1).execute(plan, _models(nodes), cfg, graph_edges=edges)
    par = Runner(max_workers=4).execute(plan, _models(nodes), cfg, graph_edges=edges)
    assert seq == par


def test_every_hypothesis_starts_after_its_sources_finish():
    nodes, edges = _random_dag(20, 0.2, 7)
    log = []
    Runner(max_workers=6).execute(
        {"p_ne": nodes, "p_e": set()}, _models(nodes, log, delay=0.005), graph_edges=edges
    )
    start = {h: a for h, a, _ in log}
    end = {h: b for h, _, b in log}
    for u, v in edges:
        assert end[u] <= start[v], (u, v)


def test_independent_hypotheses_really_run_concurrently():
    barrier = threading.Barrier(2, timeout=5)

    def meet(cfg):
        barrier.wait()  # deadlocks (BrokenBarrierError) unless both run at once
        return {"r2": 0.9}

    res = Runner(max_workers=2, max_retries=1).execute(
        {"p_ne": ["a", "b"], "p_e": set()}, {"a": meet, "b": meet}
    )
    assert res["a"]["status"] == res["b"]["status"] == "SUCCESS"


def test_failure_skips_descendants_but_not_independent_branch():
    def boom(cfg):
        raise RuntimeError("model crashed")

    models = {"a": boom, "b": lambda c: {"r2": 0.9}, "c": lambda c: {"r2": 0.9},
              "d": lambda c: {"r2": 0.9}}
    res = Runner(max_workers=3, max_retries=2).execute(
        {"p_ne": ["a", "c", "b", "d"], "p_e": set()}, models,
        graph_edges=[("a", "b"), ("b", "d")],
    )
    assert res["a"]["status"] == "FAILED"
    assert res["b"]["status"] == res["d"]["status"] == "SKIPPED"
    assert res["c"]["status"] == "SUCCESS"


def test_parallel_run_saves_to_sqlite_repository(tmp_path):
    repo = MetadataRepository(db_path=tmp_path / "r.db")
    nodes, edges = _random_dag(12, 0.2, 3)
    res = Runner(repository=repo, max_workers=4).execute(
        {"p_ne": nodes, "p_e": set()}, _models(nodes, delay=0.002), graph_edges=edges
    )
    assert all(r["status"] == "SUCCESS" for r in res.values())
    assert all(repo.has_result(h, {}) for h in nodes)
    repo.close()


def test_levels_and_cycle_detection():
    assert Runner._levels(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")]) == [
        ["a", "b"], ["c"], ["d"]]
    with pytest.raises(ValueError):
        Runner._levels(["a", "b"], [("a", "b"), ("b", "a")])


def test_manager_passes_max_workers_and_matches_sequential(tmp_path):
    nodes, edges = _random_dag(15, 0.2, 11)
    kw = dict(hypotheses=nodes, workflow_edges=edges, models=_models(nodes))
    seq = Manager(db_path=tmp_path / "s.db").orchestrate(**kw)
    mgr = Manager(db_path=tmp_path / "p.db", max_workers=4)
    assert mgr.runner.max_workers == 4
    assert mgr.orchestrate(**kw) == seq
    mgr.close()
