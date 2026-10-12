"""Проверка минимального по стоимости безопасного плана (обобщение thm:safe, сведение к min-cut).

1. На случайных ОАГ до 7 вершин оптимум plan_min_cost совпадает по стоимости с полным
   перебором всех 3^n назначений (не получена / прочитана / пересчитана), а сам план допустим.
2. Частный случай l == 0, T == V: recompute == N, read == V \\ N (thm:safe(ii)).
3. Безопасность (п. (a)): на всех назначениях для n <= 5 план, удовлетворяющий (C1), (C2),
   даёт истинные значения во всех выборочных мирах тогда и только тогда, когда Rd ⊆ A;
   план plan_min_cost безопасен.
"""
import itertools
import random

import networkx as nx
import pytest

from hyppo.planner.mincost import is_feasible, plan_cost, plan_min_cost


def random_dag(n, p, rng):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j)
    return g


def random_instance(rng, n_max=7, zero_l=False, all_targets=False):
    n = rng.randint(1, n_max)
    g = random_dag(n, rng.choice([0.2, 0.4, 0.6]), rng)
    actual = {h for h in g if rng.random() < 0.6}
    w = {h: rng.choice([0, 1, 2, 3, 5, 8]) for h in g}
    l = {h: 0 if zero_l else rng.choice([0, 1, 2, 4, 9]) for h in g}
    if all_targets:
        targets = set(g)
    else:
        targets = {h for h in g if rng.random() < 0.4}
    return g, actual, w, l, targets


def all_assignments(g):
    nodes = list(g)
    for states in itertools.product((0, 1, 2), repeat=len(nodes)):  # 0 нет, 1 чтение, 2 пересчёт
        P = {h for h, s in zip(nodes, states) if s == 2}
        Rd = {h for h, s in zip(nodes, states) if s == 1}
        yield P, Rd


def brute_force_opt(g, actual, w, l, targets):
    best = None
    for P, Rd in all_assignments(g):
        if is_feasible(g, actual, targets, P, Rd):
            c = plan_cost(P, Rd, w, l)
            if best is None or c < best:
                best = c
    return best


@pytest.mark.parametrize("seed", range(400))
def test_matches_brute_force(seed):
    rng = random.Random(seed)
    g, actual, w, l, targets = random_instance(rng)
    P, Rd = plan_min_cost(g, actual, w, l, targets)
    assert is_feasible(g, actual, targets, P, Rd)
    assert plan_cost(P, Rd, w, l) == pytest.approx(brute_force_opt(g, actual, w, l, targets))


@pytest.mark.parametrize("seed", range(200))
def test_special_case_equals_N(seed):
    rng = random.Random(10_000 + seed)
    g, actual, w, l, targets = random_instance(rng, zero_l=True, all_targets=True)
    N = set(g) - actual
    P, Rd = plan_min_cost(g, actual, w, l, targets)
    assert P == N
    assert Rd == set(g) - N


def test_recompute_actual_when_reading_is_expensive():
    # a -> b; b актуальна, но её чтение дороже пересчёта; a актуальна и дешёва в чтении.
    g = nx.DiGraph([("a", "b")])
    P, Rd = plan_min_cost(g, {"a", "b"}, {"a": 10, "b": 1}, {"a": 1, "b": 5}, {"b"})
    assert (P, Rd) == ({"b"}, {"a"})
    # если и вход дорог, выгоднее читать b
    P, Rd = plan_min_cost(g, {"a", "b"}, {"a": 10, "b": 1}, {"a": 7, "b": 5}, {"b"})
    assert (P, Rd) == (set(), {"b"})


# ---------- безопасность в модели миров ----------

def random_world(g, actual, rng):
    """Истинные значения и записи: актуальные записи честны, неактуальные отсутствуют или устарели."""
    order = list(nx.topological_sort(g))
    tables = {h: {} for h in g}

    def f(h, ins):
        key = tuple(ins)
        if key not in tables[h]:
            tables[h][key] = rng.randint(0, 3)
        return tables[h][key]

    y = {}
    for h in order:
        y[h] = f(h, [y[p] for p in sorted(g.predecessors(h))])
    store = {}
    for h in g:
        if h in actual:
            store[h] = y[h]
        elif rng.random() < 0.5:
            store[h] = (y[h] + 1) % 4  # запись есть, но получена до ревизии и расходится
    return order, f, y, store


def execute(g, order, f, store, P, Rd):
    z = {}
    for h in order:
        if h in P:
            pa = sorted(g.predecessors(h))
            if any(p not in z for p in pa):
                return None
            z[h] = f(h, [z[p] for p in pa])
        elif h in Rd:
            if h not in store:
                return None
            z[h] = store[h]
    return z


def safe_in_worlds(g, actual, P, Rd, rng, worlds=8):
    for _ in range(worlds):
        order, f, y, store = random_world(g, actual, rng)
        z = execute(g, order, f, store, P, Rd)
        if z is None or any(z[h] != y[h] for h in z):
            return False
    return True


@pytest.mark.parametrize("seed", range(60))
def test_safety_characterization(seed):
    rng = random.Random(20_000 + seed)
    g, actual, w, l, targets = random_instance(rng, n_max=5)
    for P, Rd in all_assignments(g):
        got = P | Rd
        closed = targets <= got and all(set(g.predecessors(h)) <= got for h in P)
        if not closed:
            continue
        # (a): при выполненных (C1), (C2) безопасность равносильна Rd ⊆ A
        expect = Rd <= actual
        # мир с устаревшей записью строится детерминированно: принудительно портим запись
        if expect:
            assert safe_in_worlds(g, actual, P, Rd, rng)
        else:
            bad = next(iter(Rd - actual))
            order, f, y, store = random_world(g, actual, rng)
            store[bad] = (y[bad] + 1) % 4
            z = execute(g, order, f, store, P, Rd)
            assert z is None or any(z[h] != y[h] for h in z)
    P, Rd = plan_min_cost(g, actual, w, l, targets)
    assert safe_in_worlds(g, actual, P, Rd, rng)
