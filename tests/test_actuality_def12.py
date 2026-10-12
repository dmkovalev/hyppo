"""Actuality of a stored result (definition def:12) in the planner and manager.

(a) A change of a SOURCE's model with unchanged configurations must miss the
    cache for the source and for every descendant (chained keys, conditions
    (1)-(3)); before, the key (hypothesis, own config) still hit.
(b) A source marked StaleHypothesis by the ontology layer (e.g. SUPERSEDED by a
    rival, nothing else changed) must be recomputed with its descendants
    (condition (4), witness delta_1' of prop:hidden_stale).
"""

import networkx as nx

from hyppo.manager import Manager


def _run(mgr, models, model_ids=None, stale=None):
    calls = []

    def wrap(h, fn):
        def f(cfg):
            calls.append(h)
            return fn(cfg)
        return f

    res = mgr.orchestrate(
        hypotheses=["a", "b", "c"],
        workflow_edges=[("a", "b"), ("b", "c")],
        models={h: wrap(h, fn) for h, fn in models.items()},
        model_ids=model_ids,
        stale=stale,
    )
    return calls, res


def test_source_model_change_misses_source_and_descendants(tmp_path):
    mgr = Manager(db_path=tmp_path / "c.db")
    models = {h: (lambda cfg: {"r2": 0.9}) for h in "abc"}
    ids_v1 = {"a": "crm@v1", "b": "ml@v1", "c": "opr@v1"}
    calls, _ = _run(mgr, models, ids_v1)
    assert sorted(calls) == ["a", "b", "c"]          # cold run
    calls, _ = _run(mgr, models, ids_v1)
    assert calls == []                              # everything actual
    ids_v2 = dict(ids_v1, a="crm@v2")               # source model revised
    calls, _ = _run(mgr, models, ids_v2)
    assert sorted(calls) == ["a", "b", "c"]          # source + descendants
    mgr.close()


def test_stale_marker_forces_recompute_without_any_other_change(tmp_path):
    mgr = Manager(db_path=tmp_path / "s.db")
    models = {h: (lambda cfg: {"r2": 0.9}) for h in "abc"}
    ids = {"a": "m_a", "b": "m_b", "c": "m_c"}
    _run(mgr, models, ids)
    calls, _ = _run(mgr, models, ids)
    assert calls == []
    calls, _ = _run(mgr, models, ids, stale={"b"})   # b SUPERSEDED, files/configs same
    assert sorted(calls) == ["b", "c"]
    mgr.close()


def test_planner_chained_keys_and_stale_marker():
    from hyppo.metadata_repository import SharedCache
    from hyppo.planner._base import build_optimal_plan, cache_keys_for

    class _M:
        def __init__(self, name):
            self.name = name

    class _H:
        def __init__(self, name, model):
            self.name = name
            self.is_implemented_by_model = [_M(model)]
            self.is_a = []

    class _Lat:
        def __init__(self, g):
            self.hyp_graph = g
            self.hypotheses = list(g.nodes())

    a, b, c = _H("a", "crm@v1"), _H("b", "ml"), _H("c", "opr")
    lat = _Lat(nx.DiGraph([(a, b), (b, c)]))
    cache = SharedCache(":memory:")
    for h, k in cache_keys_for(lat, {}).items():
        cache.save_result(h.name, k, {"r2": 0.9})
    assert build_optimal_plan({}, lat, cache).cached == {a, b, c}

    a.is_implemented_by_model = [_M("crm@v2")]       # source model revised
    plan = build_optimal_plan({}, lat, cache)
    assert plan.needs_execution == {a, b, c}

    a.is_implemented_by_model = [_M("crm@v1")]       # back to the cached model
    from hyppo.ontology.core_rules import StaleHypothesis

    b.is_a.append(StaleHypothesis)                    # ontology marks b stale
    plan = build_optimal_plan({}, lat, cache)
    assert plan.cached == {a} and plan.needs_execution == {b, c}
    cache.close()
