"""Rivalry does not suppress dependency (definition def:derived, variant A).

h_a (task t1) computes x and y; h_b (task t2, after t1) computes y from x.
They compete on y (output collision) AND h_b reads h_a's output x. The
derivability edge h_a -> h_b must exist, the pair must be recorded as competes,
and a change of h_a must recompute h_b (theorem thm:1 over a complete L).
"""

import os
import tempfile

from hyppo.coa import HypothesisGraph
from hyppo.manager import Manager

A = [frozenset({"x", "u"}), frozenset({"y", "x"})]  # x = f(u), y = g(x)
B = [frozenset({"y", "x"})]                         # y = k(x)


def test_edge_and_competes_coexist():
    g = HypothesisGraph()
    a = g.add(A, inputs={"u"})
    b = g.add(B, inputs={"x"})
    g.connect(a, b)
    assert g.build() == [(a, b)]
    assert (a, b) in g.competes_pairs
    assert g.order_conflicts == []


def test_counter_supply_still_suppresses_the_edge():
    g = HypothesisGraph()
    a = g.add([frozenset({"x", "z"})], inputs={"z"})   # In(a) = {z}
    b = g.add([frozenset({"z", "x"})], inputs={"x"})   # Out(b) = {z}: counter-supply
    g.connect(a, b)
    assert g.build() == []
    assert (a, b) in g.order_conflicts


def test_incremental_add_agrees_with_build():
    g = HypothesisGraph()
    a = g.add(A, inputs={"u"})
    g.build()
    b = g.add_hypothesis(B, inputs={"x"}, preds=[a])
    assert (a, b) in g.hyp_graph


def test_change_of_rival_source_recomputes_the_dependent_rival():
    calls = []

    def mk(h):
        def f(cfg):
            calls.append(h)
            return {"r2": 0.9}
        return f

    d = tempfile.mkdtemp()
    m = Manager(db_path=os.path.join(d, "r.db"))
    kw = dict(hypotheses=["a", "b"], workflow_edges=[("a", "b")],
              models={"a": mk("a"), "b": mk("b")},
              structures={"a": {"equations": A, "inputs": {"u"}}, "b": {"equations": B, "inputs": {"x"}}})
    m.orchestrate(config={"a": {"p": 1}, "b": {}}, **kw)
    calls.clear()
    m.orchestrate(config={"a": {"p": 2}, "b": {}}, **kw)
    assert sorted(calls) == ["a", "b"]
    m.close()
