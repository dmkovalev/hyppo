"""Theorem thm:build (I), the bridge to causal ordering: for a consistent pair
(h_a, h_b) the cheap data-flow criterion Out(h_a) ∩ In(h_b) ≠ ∅ holds iff the
transitive closure C^+ over the exogenised union S̄_ab contains a pair
(x_a, x_b) with x_a ∈ Out(h_a) and x_b ∈ Out(h_b) (preimages in non-trivial
equations of different hypotheses). Checked against the library's own closure
on random consistent pairs with in-hypothesis cycles."""

import random

import pytest

from hyppo.coa import causal

from tests.test_block_purity import _hypothesis


@pytest.mark.parametrize("seed", range(300))
def test_bridge_equivalence(seed):
    rng = random.Random(seed)
    pool = {f"p{i}" for i in range(6)}
    e_a, out_a, in_a = _hypothesis(rng, "a", rng.randint(1, 5), pool)
    # with probability ~1/2 h_b may read outputs of h_a (an edge), else only the pool
    src = pool | out_a if rng.random() < 0.5 else pool
    e_b, out_b, in_b = _hypothesis(rng, "b", rng.randint(1, 5), src)

    x_ab = (in_a | in_b) - out_a
    union = e_a + e_b + [frozenset({x}) for x in sorted(x_ab)]
    tc = causal.transitive_closure(union)
    assert tc is not None

    cheap = bool(out_a & in_b)
    closure = any(tc[x] & out_b for x in out_a)
    assert cheap == closure
