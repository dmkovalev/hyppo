"""End-to-end witness delta_1' of prop:hidden_stale (C8).

b and b2 compete (alternative models of one quantity); a is derived from b, h
from a. After b2 is recomputed (new model version) and beats b by Delta AIC > 10,
b becomes SUPERSEDED; the cascade marks a and h stale although their configs,
models and inputs are unchanged, and the next plan recomputes exactly a and h.
"""

import itertools

from hyppo.manager import Manager

_k = itertools.count()


def _models(aic_b2, calls):
    def mk(h, metrics):
        def f(cfg):
            calls.append(h)
            return dict(metrics)
        return f

    return {
        "b": mk("b", {"r2": 0.9, "aic": 10.0}),
        "b2": mk("b2", {"r2": 0.9, "aic": aic_b2}),
        "a": mk("a", {"r2": 0.9}),
        "h": mk("h", {"r2": 0.9}),
    }


def _run(mgr, aic_b2, b2_version, **kw):
    calls = []
    res = mgr.orchestrate(
        hypotheses=["b", "b2", "a", "h"],
        workflow_edges=[("b", "a"), ("a", "h")],
        models=_models(aic_b2, calls),
        model_ids={"b": "b@1", "b2": f"b2@{b2_version}", "a": "a@1", "h": "h@1"},
        competes={"b": {"b2"}, "b2": {"b"}},
        **kw,
    )
    return calls, res


def test_superseded_source_makes_descendants_stale_graph_path(tmp_path):
    mgr = Manager(db_path=tmp_path / "w.db")
    calls, res = _run(mgr, aic_b2=50.0, b2_version=1)          # b wins
    assert sorted(calls) == ["a", "b", "b2", "h"]
    assert res["b"]["epistemic_status"] == "SUPPORTED"
    calls, res = _run(mgr, aic_b2=-5.0, b2_version=2)          # b2 revised, now better
    assert calls == ["b2"]                                      # only the rival recomputed
    assert res["b"]["epistemic_status"] == "SUPERSEDED"
    assert mgr.pending_stale == {"a", "h"}                      # hidden staleness found
    calls, _ = _run(mgr, aic_b2=-5.0, b2_version=2)            # nothing else changed
    assert sorted(calls) == ["a", "h"]                          # ... yet a, h recomputed
    assert mgr.pending_stale == set()
    mgr.close()


def test_superseded_source_cascade_derived_by_ontology(tmp_path):
    from hyppo.core._base import Hypothesis, virtual_experiment_onto as onto
    import hyppo.ontology.core_rules as cr  # noqa: F401 (registers rule 4)

    k = next(_k)
    with onto:
        ind = {n: Hypothesis(f"w_{n}_{k}") for n in ("b", "b2", "a", "h")}
        ind["a"].derived_by = [ind["b"]]
        ind["h"].derived_by = [ind["a"]]
    mgr = Manager(db_path=tmp_path / "o.db")
    _run(mgr, aic_b2=50.0, b2_version=1, individuals=ind, ontology=onto)
    _, res = _run(mgr, aic_b2=-5.0, b2_version=2, individuals=ind, ontology=onto)
    assert res["b"]["epistemic_status"] == "SUPERSEDED"
    assert cr.InvalidHypothesis in ind["b"].is_a                # status -> marker
    assert mgr.pending_stale == {"a", "h"}                      # derived by RL
    mgr.close()
