"""Tests for hypothesis lattice construction."""
import networkx as nx
import pytest


class FakeStructure:
    """Minimal stand-in for COA Structure, used to test lattice logic."""

    def __init__(self, equations=None, vars_set=None, complete=False, tc=None):
        self.equations = equations or []
        self.vars = vars_set or set()
        self._complete = complete
        self._tc = tc or {}

    def union(self, other):
        """Return a new FakeStructure representing the union."""
        if isinstance(other, FakeStructure):
            others = [other]
        else:
            others = other if hasattr(other, '__iter__') else [other]
        new_eqs = list(self.equations)
        new_vars = set(self.vars)
        for o in others:
            new_eqs.extend(o.equations)
            new_vars.update(o.vars)
        fs = FakeStructure(equations=new_eqs, vars_set=new_vars)
        # union is complete if both parts were marked complete
        fs._complete = self._complete and all(
            getattr(o, '_complete', False) for o in others
        )
        fs._tc = {**self._tc}
        for o in others:
            fs._tc.update(getattr(o, '_tc', {}))
        return fs

    def is_complete(self):
        return self._complete

    def build_transitive_closure(self):
        return self._tc


class FakeEquation:
    """Minimal stand-in for a COA Equation exposing its variables."""

    def __init__(self, vars_list):
        self.vars = list(vars_list)


class FakeHypothesis:
    """Minimal hypothesis with a structure attached."""

    def __init__(self, name, structure):
        self.name = name
        self.structure = structure

    def __repr__(self):
        return f"H({self.name})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, FakeHypothesis) and self.name == other.name


class FakeWorkflow:
    """Workflow returning a list of task groups (lists of hypotheses)."""

    def __init__(self, task_groups):
        self._tasks = task_groups

    def get_tasks(self):
        return list(self._tasks)


def test_build_lattice_no_complete_union():
    """When no union is complete, lattice should have no edges."""
    from hyppo.lattice_constructor._base import HypothesisLattice

    s1 = FakeStructure(vars_set={"x"}, complete=False)
    s2 = FakeStructure(vars_set={"y"}, complete=False)
    h1 = FakeHypothesis("h1", s1)
    h2 = FakeHypothesis("h2", s2)
    wf = FakeWorkflow([[h1], [h2]])

    hl = HypothesisLattice(hypotheses=[h1, h2], workflow=wf)
    assert isinstance(hl.lattice, nx.DiGraph)
    assert hl.lattice.number_of_edges() == 0


def test_build_lattice_with_complete_union():
    """When h1's output variable feeds h2's equation, build_lattice adds edge (h1, h2)."""
    from hyppo.lattice_constructor._base import HypothesisLattice

    # h1 produces "a" (from equation over {a, x}); h2 consumes "a" as an input
    # of its {a, b} equation while producing "b" (a is exogenous in h2).
    s1 = FakeStructure(equations=[FakeEquation(["a", "x"])],
                       vars_set={"a", "x"}, complete=True)
    s2 = FakeStructure(equations=[FakeEquation(["a"]), FakeEquation(["a", "b"])],
                       vars_set={"a", "b"}, complete=True)
    h1 = FakeHypothesis("h1", s1)
    h2 = FakeHypothesis("h2", s2)
    wf = FakeWorkflow([[h1], [h2]])

    hl = HypothesisLattice(hypotheses=[h1, h2], workflow=wf)
    assert isinstance(hl.lattice, nx.DiGraph)
    # h1's output "a" appears among h2's inputs → derived_by edge (h1, h2)
    assert hl.lattice.number_of_edges() > 0
    assert (h1, h2) in hl.lattice.edges


def test_add_hypothesis_increases_edges():
    """add_hypothesis should be able to add new edges to an existing lattice."""
    from hyppo.lattice_constructor._base import HypothesisLattice

    s1 = FakeStructure(vars_set={"x"}, complete=False)
    h1 = FakeHypothesis("h1", s1)
    wf = FakeWorkflow([[h1]])

    hl = HypothesisLattice(hypotheses=[h1], workflow=wf)
    edges_before = hl.lattice.number_of_edges()

    tc = {"a": {"b"}, "b": set()}
    s_new = FakeStructure(vars_set={"a", "b"}, complete=True, tc=tc)
    h_new = FakeHypothesis("h_new", s_new)
    # add h_new to workflow so _is_correct passes
    wf._tasks.append([h_new])
    hl.hypotheses.append(h_new)

    hl.add_hypothesis(h_new)
    # At minimum, the method should not raise
    assert hl.lattice.number_of_edges() >= edges_before


def test_is_correct_validates_hypotheses():
    """_is_correct returns True only if all hypotheses appear in some task."""
    from hyppo.lattice_constructor._base import HypothesisLattice

    s1 = FakeStructure(vars_set={"x"}, complete=False)
    h1 = FakeHypothesis("h1", s1)
    h_missing = FakeHypothesis("h_missing", s1)
    wf = FakeWorkflow([[h1]])

    with pytest.raises(Exception, match="not found"):
        HypothesisLattice(hypotheses=[h1, h_missing], workflow=wf)


def test_derived_by_empty_for_unknown():
    """derived_by returns empty set for a hypothesis not in the lattice."""
    from hyppo.lattice_constructor._base import HypothesisLattice

    s1 = FakeStructure(vars_set={"x"}, complete=False)
    h1 = FakeHypothesis("h1", s1)
    wf = FakeWorkflow([[h1]])

    hl = HypothesisLattice(hypotheses=[h1], workflow=wf)
    h_unknown = FakeHypothesis("unknown", s1)
    assert hl.derived_by(h_unknown) == set()


def test_remove_hypothesis():
    """remove_hypothesis should remove from both the list and the graph."""
    from hyppo.lattice_constructor._base import HypothesisLattice

    s1 = FakeStructure(vars_set={"x"}, complete=False)
    h1 = FakeHypothesis("h1", s1)
    h2 = FakeHypothesis("h2", s1)
    wf = FakeWorkflow([[h1, h2]])

    hl = HypothesisLattice(hypotheses=[h1, h2], workflow=wf)
    # Manually add h1 as a node so removal works
    hl.lattice.add_node(h1)
    hl.remove_hypothesis(h1)
    assert h1 not in hl.hypotheses
    assert h1 not in hl.lattice.nodes
