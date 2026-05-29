from hyppo.coa import causal


def test_variables_and_completeness():
    eqs = [frozenset({"x_1"}), frozenset({"x_1", "x_2"})]
    assert causal.variables(eqs) == {"x_1", "x_2"}
    assert causal.is_complete(eqs)
    assert not causal.is_complete([frozenset({"x_1", "x_2", "x_3"})])


def test_perfect_matching_assigns_contained_vars():
    eqs = [frozenset({"x_0", "x_1"}), frozenset({"x_1", "x_2"}),
           frozenset({"x_0", "x_2"})]
    m = causal.perfect_matching(eqs)
    assert m is not None
    assert len(m) == 3
    for i, v in m.items():
        assert v in eqs[i]
    assert len(set(m.values())) == 3


def test_perfect_matching_none_when_singular():
    eqs = [frozenset({"x_0"}), frozenset({"x_0"})]
    assert causal.perfect_matching(eqs) is None
