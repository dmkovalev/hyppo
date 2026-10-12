"""Algorithm 1 conforms to the dissertation's definition of derivability.

Pins the four points where the implementation must follow the theory:

* C1 -- an edge h_i -> h_j requires tau(h_i) < tau(h_j) (the workflow routes h_i
  before h_j): no edges inside a task, between parallel tasks, or against the
  workflow;
* C2 -- condition У3 has priority: an output collision (competes) or a
  counter-supply (order conflict) yields NO derivability edge;
* C3 -- with declared inputs (У2) the signature is Out = V \\ In, independent of
  the matching, and У0 is checked;
* C4 -- order conflicts are decided by reachability, not by list position.
"""

import pytest

from hyppo.coa import HypothesisGraph
from hyppo.lattice_constructor._base import HypothesisLattice


class _Eq:
    def __init__(self, formula):
        self.formula = formula


class _Structure:
    def __init__(self, formula):
        self.equations = [_Eq(formula)]


class _Hyp:
    def __init__(self, name, formula):
        self.name = name
        self.structure = _Structure(formula)

    def __repr__(self):  # pragma: no cover
        return self.name


class _Chain:
    """Ordered task list (a chain workflow)."""

    def __init__(self, tasks):
        self._tasks = tasks

    def get_tasks(self):
        return self._tasks


class _Dag(_Chain):
    """Task list plus explicit task edges (a DAG workflow)."""

    def __init__(self, tasks, edges):
        super().__init__(tasks)
        self._edges = edges

    def get_task_edges(self):
        return self._edges


def _edges(hl):
    return {(u.name, v.name) for u, v in hl.lattice.edges()}


# ---- C1: tau-order ---------------------------------------------------------
def test_no_edge_against_the_workflow():
    a, b = _Hyp("a", "x = c0"), _Hyp("b", "y = x + 1")
    hl = HypothesisLattice([a, b], _Chain([[b], [a]]))  # consumer scheduled first
    assert _edges(hl) == set()


def test_no_edge_inside_one_task():
    a, b = _Hyp("a", "x = c0"), _Hyp("b", "y = x + 1")
    hl = HypothesisLattice([a, b], _Chain([[a, b]]))
    assert _edges(hl) == set()


def test_no_edge_between_parallel_tasks():
    a, b = _Hyp("a", "x = c0"), _Hyp("b", "y = x + 1")
    # tasks 0 and 1 are incomparable in the DAG (no path between them)
    hl = HypothesisLattice([a, b], _Dag([[a], [b]], edges=[]))
    assert _edges(hl) == set()


def test_edge_along_the_workflow():
    a, b = _Hyp("a", "x = c0"), _Hyp("b", "y = x + 1")
    hl = HypothesisLattice([a, b], _Dag([[a], [b]], edges=[(0, 1)]))
    assert _edges(hl) == {("a", "b")}


# ---- C2: У3 priority -------------------------------------------------------
def test_output_collision_gives_no_edge_but_competes():
    a = _Hyp("a", "x = c0")
    b = _Hyp("b", "x = y + 1")  # also produces x, and uses nothing of a
    c = _Hyp("c", "y = x + 2")  # consumes x
    hl = HypothesisLattice([a, c, b], _Chain([[a], [c], [b]]))
    assert ("a", "b") not in _edges(hl)
    assert b in hl.competes(a)


def test_counter_supply_gives_no_edge_but_order_conflict():
    a = _Hyp("a", "x = z + 1")   # needs z ...
    b = _Hyp("b", "z = x * 2")   # ... which b produces later, and b uses x
    hl = HypothesisLattice([a, b], _Chain([[a], [b]]))
    assert _edges(hl) == set()
    assert (a, b) in hl.order_conflicts()


# ---- C4: order conflicts by reachability -----------------------------------
def test_parallel_tasks_are_not_an_order_conflict():
    a = _Hyp("a", "x = z + 1")
    b = _Hyp("b", "z = c0")
    hl = HypothesisLattice([a, b], _Dag([[a], [b]], edges=[]))
    assert hl.order_conflicts() == set()


# ---- HypothesisGraph: C2, C3 -----------------------------------------------
def test_graph_declared_inputs_fix_the_signature():
    g = HypothesisGraph()
    # one equation over {u, v}: which one is the output is NOT decided by the
    # matching when the model declares In = {u}
    i = g.add([{"u", "v"}], inputs={"u"})
    assert g._out[i] == {"v"} and g._in[i] == {"u"}
    j = g.add([{"u", "v"}], inputs={"v"})
    assert g._out[j] == {"u"} and g._in[j] == {"v"}


def test_graph_checks_condition_U0():
    g = HypothesisGraph()
    with pytest.raises(ValueError):
        g.add([{"u", "v", "w"}], inputs={"u"})  # |E| + |In| != |V|
    with pytest.raises(ValueError):
        g.add([{"u"}], inputs={"q"})  # declared input not in the structure


def test_graph_U3_priority():
    g = HypothesisGraph()
    p = g.add([{"x"}], inputs=set())          # produces x
    q = g.add([{"x", "y"}], inputs={"y"})     # also produces x -> competes
    r = g.add([{"x", "z"}], inputs={"x"})     # consumes x
    g.connect(p, q)
    g.connect(q, r)
    edges = set(g.build())
    assert (p, q) not in edges and (p, q) in g.competes_pairs
    assert (p, r) in edges


def test_graph_counter_supply_is_order_conflict():
    g = HypothesisGraph()
    a = g.add([{"x", "z"}], inputs={"z"})   # needs z
    b = g.add([{"z", "x"}], inputs={"x"})   # produces z later, consumes x
    g.connect(a, b)
    assert g.build() == []
    assert (a, b) in g.order_conflicts
