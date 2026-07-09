import random
from itertools import combinations

from hyppo.coa import causal


def test_variables_and_completeness():
    eqs = [frozenset({"x_1"}), frozenset({"x_1", "x_2"})]
    assert causal.variables(eqs) == {"x_1", "x_2"}
    assert causal.is_complete(eqs)
    assert not causal.is_complete([frozenset({"x_1", "x_2", "x_3"})])


def test_perfect_matching_assigns_contained_vars():
    eqs = [
        frozenset({"x_0", "x_1"}),
        frozenset({"x_1", "x_2"}),
        frozenset({"x_0", "x_2"}),
    ]
    m = causal.perfect_matching(eqs)
    assert m is not None
    assert len(m) == 3
    for i, v in m.items():
        assert v in eqs[i]
    assert len(set(m.values())) == 3


def test_perfect_matching_none_when_singular():
    eqs = [frozenset({"x_0"}), frozenset({"x_0"})]
    assert causal.perfect_matching(eqs) is None


def _brute_minimal(eqs):
    """Reference: inclusion-minimal complete subsets of equations (exponential)."""
    n = len(eqs)
    complete = []
    for r in range(1, n + 1):
        for c in combinations(range(n), r):
            sub = [eqs[i] for i in c]
            if causal.is_complete(sub):
                complete.append(frozenset(c))
    return {cs for cs in complete if not any(o < cs for o in complete)}


def _random_complete(n_eq, rng):
    """n_eq equations over exactly n_eq vars with a built-in perfect matching."""
    eqs = []
    for i in range(n_eq):
        own = f"x_{i}"
        k = rng.randint(0, min(2, n_eq - 1))
        extras = rng.sample([j for j in range(n_eq) if j != i], k) if k else []
        eqs.append(frozenset({own, *(f"x_{j}" for j in extras)}))
    return eqs


def test_minimal_blocks_named_example():
    eqs = [
        frozenset({"x_1"}),
        frozenset({"x_2"}),
        frozenset({"x_3"}),
        frozenset({"x_1", "x_2", "x_3", "x_4", "x_5"}),
        frozenset({"x_1", "x_3", "x_4", "x_5"}),
        frozenset({"x_4", "x_6"}),
        frozenset({"x_5", "x_7"}),
    ]
    got = causal.minimal_blocks(eqs)
    assert set(got) == _brute_minimal(eqs)
    assert set(got) == {frozenset({0}), frozenset({1}), frozenset({2})}


def test_minimal_blocks_equiv_bruteforce_exhaustive():
    rng = random.Random(42)
    for _ in range(3000):
        n = rng.randint(1, 8)
        eqs = _random_complete(n, rng)
        if not causal.is_complete(eqs):
            continue
        assert set(causal.minimal_blocks(eqs)) == _brute_minimal(eqs)


def test_block_decomposition_partitions_all_equations():
    eqs = _random_complete(7, random.Random(1))
    blocks = causal.block_decomposition(eqs)
    covered = set()
    for b in blocks:
        assert not (covered & b)
        covered |= b
    assert covered == set(range(len(eqs)))


def test_causal_mapping_valid():
    eqs = [
        frozenset({"x_0", "x_1"}),
        frozenset({"x_1", "x_2"}),
        frozenset({"x_0", "x_2"}),
    ]
    m = causal.causal_mapping(eqs)
    for i, v in m.items():
        assert v in eqs[i]
    assert len(set(m.values())) == 3


def test_transitive_closure_triangular():
    eqs = [
        frozenset({"x_0"}),
        frozenset({"x_0", "x_1"}),
        frozenset({"x_0", "x_1", "x_2"}),
    ]
    tc = causal.transitive_closure(eqs)
    assert tc["x_0"] == {"x_1", "x_2"}
    assert tc["x_2"] == set()
    assert "x_0" not in tc["x_0"]


def test_transitive_closure_excludes_self_in_cycle():
    eqs = [
        frozenset({"x_0", "x_1"}),
        frozenset({"x_1", "x_2"}),
        frozenset({"x_0", "x_2"}),
    ]
    tc = causal.transitive_closure(eqs)
    for v in ("x_0", "x_1", "x_2"):
        assert v not in tc[v]


def test_stress_10k_no_crash():
    """10000 random complete structures through the full core in one process.
    The pure core has no native deps, so this must complete without segfault."""
    rng = random.Random(2026)
    runs = 0
    for _ in range(10000):
        n = rng.randint(2, 12)
        eqs = _random_complete(n, rng)
        if not causal.is_complete(eqs):
            continue
        m = causal.causal_mapping(eqs)
        assert m is not None
        for i, v in m.items():  # bug-fix invariant
            assert v in eqs[i]
        causal.minimal_blocks(eqs)
        causal.transitive_closure(eqs)
        runs += 1
    assert runs > 9000
