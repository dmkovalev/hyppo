"""Tests for hyppo.coa.graph.HypothesisGraph (Algorithms 1, 2, 4 of the paper).

Covers construction, the build lattice (Alg 1), incremental add (Alg 2), the
cascade-aware plan (Alg 4), and a polynomial-scaling smoke test for build.
"""

import math
import random

from hyppo.coa import HypothesisGraph

# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_add_and_len():
    g = HypothesisGraph()
    assert g.add([frozenset({"a", "b"}), frozenset({"b"})]) == 0
    assert g.add([frozenset({"c"})]) == 1
    assert len(g) == 2


def test_from_edges_builds_graph():
    g = HypothesisGraph.from_edges(3, [(0, 1), (1, 2)])
    assert len(g) == 3
    # linear chain 0->1->2: caching only root still recomputes 1,2 (cascade)
    assert g.plan({0}) == {1, 2}
    assert g.plan({0, 1, 2}) == set()


def test_connect_out_of_range_raises():
    g = HypothesisGraph()
    g.add([frozenset({"a"})])
    try:
        g.connect(0, 5)
    except IndexError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected IndexError")


# --------------------------------------------------------------------------
# Algorithm 1: build
# --------------------------------------------------------------------------


def test_build_links_complete_union():
    """Two complete, variable-disjoint structures -> their union is complete,
    so the reachable pair appears as a lattice edge."""
    g = HypothesisGraph()
    g.add([frozenset({"a", "b"}), frozenset({"b"})])  # complete: 2 eq, vars {a,b}
    g.add([frozenset({"c", "d"}), frozenset({"d"})])  # complete: 2 eq, vars {c,d}
    g.connect(0, 1)
    assert g.build() == [(0, 1)]


def test_build_skips_incomplete_union():
    """Variable-overlapping structures whose union is not complete -> no edge."""
    g = HypothesisGraph()
    g.add([frozenset({"a", "b"}), frozenset({"b"})])
    g.add([frozenset({"a", "c"}), frozenset({"c"})])  # union has 4 eq, 3 vars
    g.connect(0, 1)
    assert g.build() == []


def test_build_only_reachable_pairs():
    """No edges -> nothing reachable -> empty lattice even if unions complete."""
    g = HypothesisGraph()
    g.add([frozenset({"a", "b"}), frozenset({"b"})])
    g.add([frozenset({"c", "d"}), frozenset({"d"})])
    assert g.build() == []


# --------------------------------------------------------------------------
# Algorithm 2: add_hypothesis
# --------------------------------------------------------------------------


def test_add_hypothesis_appends_node():
    g = HypothesisGraph()
    g.add([frozenset({"a", "b"}), frozenset({"b"})])
    idx = g.add_hypothesis([frozenset({"c", "d"}), frozenset({"d"})])
    assert idx == 1
    assert len(g) == 2


# --------------------------------------------------------------------------
# Algorithm 4: plan (cascade)
# --------------------------------------------------------------------------


def _chain(n):
    """A linear derived_by chain 0 -> 1 -> ... -> n-1 of trivial hypotheses."""
    g = HypothesisGraph()
    for k in range(n):
        g.add([frozenset({f"v{k}"})])
    for k in range(n - 1):
        g.connect(k, k + 1)
    return g


def test_plan_nothing_cached_recomputes_all():
    g = _chain(4)
    assert g.plan(set()) == {0, 1, 2, 3}


def test_plan_all_cached_recomputes_none():
    g = _chain(4)
    assert g.plan({0, 1, 2, 3}) == set()


def test_plan_cascade_invalidates_downstream():
    """Caching only the root still forces every downstream node to recompute,
    because each is itself non-cached (the cascade)."""
    g = _chain(4)
    assert g.plan({0}) == {1, 2, 3}


def test_plan_mid_chain_cache_does_not_protect_downstream():
    """Caching a middle node does not save its descendants if they are uncached."""
    g = _chain(4)
    # node 2 cached, 0/1/3 not -> all uncached recomputed; 2 also pulled in as a
    # dependent of 1's cascade.
    assert g.plan({2}) == {0, 1, 2, 3}


def test_plan_protected_subtree():
    """Caching a node AND all its ancestors protects it from recompute."""
    g = _chain(4)
    # cache the whole prefix 0,1,2 -> only leaf 3 recomputes
    assert g.plan({0, 1, 2}) == {3}


# --------------------------------------------------------------------------
# scaling: build exponent is polynomial (sanity, small grid)
# --------------------------------------------------------------------------


def _gen_struct(rng, n_eq=5, pool=20):
    av = [f"x_{k}" for k in range(pool)]
    chosen = rng.sample(av, n_eq)
    eqs = []
    for i in range(n_eq):
        extras = rng.sample(
            [v for v in chosen if v != chosen[i]], rng.randint(1, min(3, n_eq - 1))
        )
        eqs.append(frozenset([chosen[i], *extras]))
    return eqs


def _build_graph(n_h, rng, p=0.3):
    g = HypothesisGraph()
    for _ in range(n_h):
        g.add(_gen_struct(rng))
    for i in range(n_h):
        for j in range(i + 1, n_h):
            if rng.random() < p:
                g.connect(i, j)
    return g


def test_build_exponent_is_polynomial():
    import time

    h_values = [10, 20, 30, 50, 70, 100]
    means = []
    for n_h in h_values:
        ts = []
        for rep in range(8):
            rng = random.Random(42 + rep)
            g = _build_graph(n_h, rng)
            t0 = time.perf_counter()
            g.build()
            ts.append(time.perf_counter() - t0)
        means.append(sum(ts) / len(ts))
    lh = [math.log(h) for h in h_values]
    lt = [math.log(m) for m in means]
    n = len(lh)
    mx, my = sum(lh) / n, sum(lt) / n
    a = sum((x - mx) * (y - my) for x, y in zip(lh, lt)) / sum(
        (x - mx) ** 2 for x in lh
    )
    assert a < 2.8, f"exponent {a:.3f} too high -- expected near-quadratic"
