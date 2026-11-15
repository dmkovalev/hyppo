"""Regression checks for the dissertation/implementation discrepancies."""
import json

import pytest

from hyppo.coa import HypothesisGraph
from hyppo.planner import actuality
from hyppo.runner import Runner


def run_pair(a, b, weights=None):
    return Runner().execute(
        {'p_ne': ['a', 'b'], 'p_e': set()},
        {'a': lambda _: a, 'b': lambda _: b},
        competes={'a': {'b'}, 'b': {'a'}}, ranking_weights=weights,
    )


def test_combined_winner_is_not_superseded_by_aic_alone():
    result = run_pair({'r2': .99, 'aic': 120}, {'r2': .8, 'aic': 100}, {'r2': 1.})
    assert result['a']['epistemic_status'] == 'SUPPORTED'


def test_aicc_can_reverse_aic_winner():
    result = run_pair({'r2': .9, 'aic': 90, 'n': 30, 'k': 15},
                      {'r2': .9, 'aic': 100, 'n': 30, 'k': 2})
    assert result['a']['epistemic_status'] == 'SUPERSEDED'
    assert result['b']['epistemic_status'] == 'SUPPORTED'


def test_correct_both_rivals_when_only_one_requires_aicc():
    result = run_pair({'r2': .9, 'aic': 90, 'n': 100, 'k': 30},
                      {'r2': .9, 'aic': 100, 'n': 100, 'k': 1})
    assert result['a']['epistemic_status'] == 'SUPERSEDED'


@pytest.mark.parametrize('weights', [{'r2': -1, 'aic': 2}, {'r2': .5}, {}, {'unknown': 1}])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        run_pair({'r2': .9, 'aic': 100}, {'r2': .8, 'aic': 120}, weights)


def test_incomplete_aicc_metadata_rejected():
    with pytest.raises(ValueError):
        run_pair({'r2': .9, 'aic': 100, 'n': 30, 'k': 15}, {'r2': .8, 'aic': 120})


def test_invalid_aicc_denominator_rejected():
    with pytest.raises(ValueError):
        run_pair({'r2': .9, 'aic': 100, 'n': 30, 'k': 29},
                 {'r2': .8, 'aic': 120, 'n': 30, 'k': 1})


def test_hashing_shared_ancestor_once_and_preserving_keys(monkeypatch):
    n = 30
    roots = [f'r{i}' for i in range(n)]
    leaves = [f'l{i}' for i in range(n)]
    nodes = roots + ['hub'] + leaves
    edges = [(r, 'hub') for r in roots] + [('hub', leaf) for leaf in leaves]
    digest = actuality._digest
    counts = []

    def counted(key):
        counts.append(len(key['up']))
        return digest(key)

    monkeypatch.setattr(actuality, '_digest', counted)
    keys = actuality.chained_keys(nodes, edges, epochs={'hub': 2})
    assert len(counts) == len(nodes)
    assert sum(counts) == len(edges)
    old_keys = {}
    for h in nodes:
        old_keys[h] = {'cfg': {}, 'model': None, 'up': {
            p: digest(old_keys[p]) + ('@2' if p == 'hub' else '')
            for p in sorted(u for u, v in edges if v == h)
        }}
    assert json.dumps(keys, sort_keys=True) == json.dumps(old_keys, sort_keys=True)
    assert keys == actuality.chained_keys(nodes, reversed(edges), epochs={'hub': 2})


def test_bad_topological_order_rejected():
    with pytest.raises(ValueError):
        actuality.chained_keys(['b', 'a'], [('a', 'b')])


def test_insertion_checks_smaller_boundary(monkeypatch):
    g = HypothesisGraph()
    for i in range(60):
        g.add([frozenset({f'x{i}'})])
    for i in range(59):
        g.connect(i, i+1)
    g.build()
    reach, calls = g._reach_from, []

    def counted(node, adj):
        calls.append(node)
        return reach(node, adj)

    monkeypatch.setattr(g, '_reach_from', counted)
    g.add_hypothesis([frozenset({'new'})], preds=range(30), succs=[59])
    assert not g.last_add_was_rebuild
    assert len(calls) == 3  # one check, two incremental traversals
    incremental = set(g.hyp_graph)
    assert incremental == set(g.build())


def test_cycle_insertion_does_not_mutate_graph():
    g = HypothesisGraph()
    for x in ('x', 'y'):
        g.add([frozenset({x})])
    g.connect(0, 1)
    g.build()
    with pytest.raises(ValueError):
        g.add_hypothesis([frozenset({'z'})], preds=[1], succs=[0])
    assert len(g) == 2


def test_manager_forwards_ranking_weights():
    from hyppo.manager import Manager
    result = Manager().orchestrate(
        ['a', 'b'], [],
        {'a': lambda _: {'r2': .99, 'aic': 120}, 'b': lambda _: {'r2': .8, 'aic': 100}},
        competes={'a': {'b'}, 'b': {'a'}}, ranking_weights={'r2': 1.},
    )
    assert result['a']['epistemic_status'] == 'SUPPORTED'


@pytest.mark.parametrize('n', [100, 200])
@pytest.mark.parametrize('shared_descendant', [False, True])
def test_cascade_work_is_linear(monkeypatch, n, shared_descendant):
    import hyppo.coa.graph as graph_module

    edges = [(i, n - 1) for i in range(n - 1)] if shared_descendant else []
    graph = HypothesisGraph.from_edges(n, edges)
    work = [0]

    class CountedSet(set):
        def __contains__(self, value):
            work[0] += 1
            return super().__contains__(value)

        def add(self, value):
            work[0] += 1
            return super().add(value)

        def __ior__(self, other):
            # Count elements visited by bulk updates, not just Python calls.
            work[0] += len(other)
            return super().__ior__(other)

    monkeypatch.setattr(graph_module, 'set', CountedSet, raising=False)
    cached = {n - 1} if shared_descendant else set()
    assert graph.plan(cached) == set(range(n))
    assert work[0] <= 5 * (n + len(edges))


def test_cascade_preserves_cached_independent_branch():
    graph = HypothesisGraph.from_edges(6, [(0, 2), (1, 2), (2, 3), (4, 5)])
    assert graph.plan({1, 2, 3, 4, 5}) == {0, 2, 3}
    assert graph.plan(set(range(6))) == set()
