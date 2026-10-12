"""Theorem thm:build (III): incremental Algorithm 2 equals the full rebuild of
Algorithm 1 when reachability between existing tasks is preserved; otherwise the
graph falls back to a full rebuild (still equal)."""

import random

import pytest

from hyppo.coa import HypothesisGraph


def _gen(n, rng):
    hyps, produced = [], []
    for idx in range(n):
        outs = [f"y{idx}_{k}" for k in range(3)]
        cap = min(3, len(produced))
        ins = rng.sample(produced, rng.randint(0, cap)) if cap else []
        eqs = [frozenset([o, *ins]) for o in outs]
        hyps.append((eqs, set(ins)))
        produced.extend(outs)
    return hyps


def _workflow(n, rng):
    edges = [(i, i + 1) for i in range(n - 1)]
    for j in range(2, n):
        if rng.random() < 0.5:
            edges.append((rng.randrange(j - 1), j))
    return edges


@pytest.mark.parametrize("seed", range(100))
def test_incremental_equals_full_rebuild(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 25)
    hyps = _gen(n + 1, rng)
    edges = _workflow(n + 1, rng)
    preds = [u for u, v in edges if v == n]

    full = HypothesisGraph()
    for eqs, ins in hyps:
        full.add(eqs, inputs=ins)
    for u, v in edges:
        full.connect(u, v)
    expected = set(full.build())

    inc = HypothesisGraph()
    for eqs, ins in hyps[:-1]:
        inc.add(eqs, inputs=ins)
    for u, v in edges:
        if v != n:
            inc.connect(u, v)
    inc.build()
    inc.add_hypothesis(hyps[-1][0], inputs=hyps[-1][1], preds=preds)
    assert not inc.last_add_was_rebuild
    assert inc.lattice == expected
    assert set(inc.competes_pairs) == set(full.competes_pairs)
    assert set(inc.order_conflicts) == set(full.order_conflicts)


def test_new_paths_trigger_full_rebuild():
    """A new task t1 -> new -> t2 between unrelated tasks breaks the precondition
    of thm:build (III); the graph rebuilds and still equals Algorithm 1."""
    g = HypothesisGraph()
    a = g.add([{"x"}], inputs=set())
    b = g.add([{"x2", "x"}], inputs={"x"})  # consumes x, but a and b unrelated
    g.build()
    assert g.lattice == set()
    g.add_hypothesis([{"m"}], inputs=set(), preds=[a], succs=[b])
    assert g.last_add_was_rebuild
    assert (a, b) in g.lattice
