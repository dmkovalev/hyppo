"""Corollary thm:build:I (purity of blocks): for a consistent pair (h_a, h_b)
(conditions У0 for both, disjoint equations, condition У3) no irreducible
Dulmage-Mendelsohn block of the exogenised union S̄_ab = E_a ∪ E_b ∪ T_X contains
equations of both hypotheses.

Random consistent pairs are generated with in-hypothesis cycles (so that
non-trivial blocks really occur); h_b may consume outputs of h_a (an edge), but
never produces a variable h_a uses or produces (У3).
"""

import random

import pytest

from hyppo.coa import causal


def _hypothesis(rng, tag, k, inputs):
    """k equations over outputs tag0..tag{k-1}; equation i contains its own output,
    its cyclic successor (an algebraic loop -> a non-trivial block) and a few
    declared inputs. Complete after exogenisation: |E| + |In| = |V| (У0)."""
    outs = [f"{tag}{i}" for i in range(k)]
    eqs = []
    for i in range(k):
        eq = {outs[i]}
        if k > 1 and rng.random() < 0.7:
            eq.add(outs[(i + 1) % k])
        if inputs:
            eq |= set(rng.sample(sorted(inputs), rng.randint(0, min(2, len(inputs)))))
        eqs.append(frozenset(eq))
    used_in = set().union(*eqs) - set(outs)
    return eqs, set(outs), used_in


@pytest.mark.parametrize("seed", range(200))
def test_no_block_mixes_two_consistent_hypotheses(seed):
    rng = random.Random(seed)
    pool = {f"p{i}" for i in range(6)}
    e_a, out_a, in_a = _hypothesis(rng, "a", rng.randint(1, 5), pool)
    # h_b reads from the pool and from Out(h_a) -> possible derivability edge
    e_b, out_b, in_b = _hypothesis(rng, "b", rng.randint(1, 5), pool | out_a)
    assert not (out_a & out_b) and not (in_a & out_b)  # condition У3

    x_ab = (in_a | in_b) - out_a
    trivial = [frozenset({x}) for x in sorted(x_ab)]
    union = e_a + e_b + trivial
    blocks = causal.block_decomposition(union)
    assert blocks is not None, "the consistent union must be complete (lemma lem:balance)"

    idx_a = set(range(len(e_a)))
    idx_b = set(range(len(e_a), len(e_a) + len(e_b)))
    for blk in blocks:
        assert not (blk & idx_a and blk & idx_b), f"block {sorted(blk)} mixes h_a and h_b"


def test_generator_produces_non_trivial_blocks():
    """Guard: the random pairs do exercise non-trivial (size > 1) blocks."""
    sizes = []
    for seed in range(200):
        rng = random.Random(seed)
        e_a, _, in_a = _hypothesis(rng, "a", rng.randint(1, 5), {"p0", "p1"})
        triv = [frozenset({x}) for x in sorted(in_a)]
        sizes += [len(b) for b in causal.block_decomposition(e_a + triv)]
    assert max(sizes) > 1
