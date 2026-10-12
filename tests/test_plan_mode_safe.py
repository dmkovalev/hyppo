"""Planner mode 'safe' (theorem thm:safe): recompute exactly N.

After a source's record is evicted, 'cascade' recomputes the source and all its
descendants (Cl(N)), 'safe' recomputes the source only and reuses the still
actual descendants. After a change of a source parameter both modes coincide:
the chained keys already make every descendant miss the cache.
"""

import pytest

from hyppo.manager import Manager

EDGES = [("a", "b"), ("b", "c"), ("a", "d")]
NODES = ["a", "b", "c", "d"]


def _models(calls):
    def mk(h):
        def f(cfg):
            calls.append(h)
            return {"r2": 0.9, "v": cfg.get("p", 0)}
        return f
    return {h: mk(h) for h in NODES}


def _run(mgr, cfg, calls):
    calls.clear()
    res = mgr.orchestrate(hypotheses=NODES, workflow_edges=EDGES, models=_models(calls), config=cfg)
    return sorted(calls), res


@pytest.mark.parametrize("mode, expected", [("cascade", ["a", "b", "c", "d"]), ("safe", ["a"])])
def test_eviction_of_a_source(tmp_path, mode, expected):
    mgr = Manager(db_path=tmp_path / "r.db", plan_mode=mode)
    calls = []
    cfg = {h: {"p": 1} for h in NODES}
    first, res1 = _run(mgr, cfg, calls)
    assert first == NODES
    keys = mgr.cache_keys(__import__("networkx").DiGraph(EDGES), _models([]), cfg)
    assert mgr.repository.evict("a", keys["a"])
    again, res2 = _run(mgr, cfg, calls)
    assert again == expected
    assert {h: r["metrics"] for h, r in res2.items()} == {h: r["metrics"] for h, r in res1.items()}
    mgr.close()


@pytest.mark.parametrize("mode", ["cascade", "safe"])
def test_parameter_change_coincides(tmp_path, mode):
    mgr = Manager(db_path=tmp_path / "r.db", plan_mode=mode)
    calls = []
    cfg = {h: {"p": 1} for h in NODES}
    _run(mgr, cfg, calls)
    cfg2 = dict(cfg, b={"p": 2})
    again, _ = _run(mgr, cfg2, calls)
    assert again == ["b", "c"]  # b and its descendant; a, d reused
    mgr.close()


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        Manager(db_path=tmp_path / "r.db", plan_mode="lazy")
