"""Minimax optimality of the safe plan N among ADAPTIVE plans (strengthening of thm:safe).

Model (as in test_safe_plan_theorem.py, copied, that file is not changed):
  world = (f_old, f_new); f_new differs from f_old only at direct consumers of U;
  records (h, ctx) are honest: their value is computed with f_old in ctx;
  N = {h : no record with the current ctx, or h is a strict descendant of U}.
An adaptive strategy processes hypotheses in topological order; at h it sees the metadata,
the values of all records and the values it has already produced, and either recomputes h
(evaluates f_new[h] at its own values of the sources) or assigns a value from what it has seen.

Checks:
  (a) strategy-independent core of the proof: in the free (Herbrand) world W_H, for every h in N
      there is a consistent world W' with the same record values and the same values of all
      g != h, but with a different value of h; hence any safe strategy recomputes all of N in W_H.
      For the greedy early-cutoff strategy G we check directly: G is safe and recomputes N in W_H.
  (b) there are worlds in which G is safe and recomputes strictly less than N.
  (c) the richness assumption is needed: with a one-element value domain nothing must be recomputed.
"""
import itertools
import random

import pytest


# ---------------------------------------------------------------- copied helpers
def dags(n):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for mask in range(1 << len(pairs)):
        yield [p for k, p in enumerate(pairs) if mask >> k & 1]


def anc_sets(n, edges):
    pa = {j: [i for i, k in edges if k == j] for j in range(n)}
    anc = {}
    for j in range(n):
        s, st = set(), list(pa[j])
        while st:
            x = st.pop()
            if x not in s:
                s.add(x)
                st.extend(pa[x])
        anc[j] = s
    return pa, anc


def desc(n, edges, U):
    out = set(U)
    ch = {i: [k for j, k in edges if j == i] for i in range(n)}
    st = list(U)
    while st:
        x = st.pop()
        for y in ch[x]:
            if y not in out:
                out.add(y)
                st.append(y)
    return out


def ctx(h, theta, anc):
    return tuple(sorted((g, theta[g]) for g in anc[h] | {h}))


# ---------------------------------------------------------------- worlds
def evaluate(n, pa, theta, f):
    y = {}
    for h in range(n):  # 0..n-1 is a topological order
        y[h] = f[h](theta[h], tuple(y[p] for p in pa[h]))
    return y


def build_store(n, pa, recs, f_old):
    store = {}
    for (h, c) in recs:
        th = dict(c)
        full = {g: th.get(g, 0) for g in range(n)}  # params outside ctx do not affect h
        store[(h, c)] = evaluate(n, pa, full, f_old)[h]
    return store


def herbrand(n, consumers):
    """Free world: a value is the term of its computation; revised consumers add a tag."""
    f_old = [(lambda h: lambda t, ins: ("f", h, t, ins))(h) for h in range(n)]
    f_new = [(lambda h: lambda t, ins: ("rev" if h in consumers else "f", h, t, ins))(h)
             for h in range(n)]
    return f_old, f_new


def table_world(n, pa, consumers, rng, dom):
    def tbl():
        return {}

    def mk(table, h):
        def f(t, ins):
            key = (t, ins)
            if key not in table:
                table[key] = rng.choice(dom)
            return table[key]
        return f

    old_t = [tbl() for _ in range(n)]
    new_t = [old_t[h] if h not in consumers else tbl() for h in range(n)]
    return [mk(old_t[h], h) for h in range(n)], [mk(new_t[h], h) for h in range(n)]


# ---------------------------------------------------------------- the greedy early-cutoff strategy
def greedy(n, pa, theta, f_new, store, cur, consumers, DU):
    """Adaptive strategy G. Uses only metadata, record values and its own values;
    touches the world only through recomputation f_new[h](theta_h, own source values)."""
    z, recomputed = {}, set()
    for h in range(n):
        if (h, cur[h]) in store and h not in DU:
            z[h] = store[(h, cur[h])]          # actual record (h not in N)
            continue
        hit = None
        if h not in consumers:                 # f_h has not changed since the records were made
            for (g, c), val in store.items():
                if g != h or dict(c)[h] != theta[h]:
                    continue
                th = dict(c)
                ok = True
                for p in pa[h]:                # the inputs of that record are known iff the
                    kp = (p, tuple(sorted((x, th[x]) for x in dict(cur[p]))))
                    if kp not in store or store[kp] != z[p]:   # sources' records exist
                        ok = False
                        break
                if ok:
                    hit = val
                    break
        if hit is not None:
            z[h] = hit                          # early cutoff: same f_h, same inputs
        else:
            z[h] = f_new[h](theta[h], tuple(z[p] for p in pa[h]))
            recomputed.add(h)
    return z, recomputed


# ---------------------------------------------------------------- adversary of the proof
def masked_world(n, pa, theta, f_old, f_new, consumers, h, y, nu):
    """W': change f_h at its current point to a fresh value nu and let every child of h
    treat nu as the old value y_h. Changes are applied to f_old too unless h is a consumer."""
    pt = (theta[h], tuple(y[p] for p in pa[h]))

    def patch_h(f):
        return lambda t, ins: nu if (t, ins) == pt else f(t, ins)

    def patch_child(f, k):
        i = pa[k].index(h)
        return lambda t, ins: f(t, ins[:i] + ((y[h],) if ins[i] == nu else (ins[i],)) + ins[i + 1:])

    g_old, g_new = list(f_old), list(f_new)
    g_new[h] = patch_h(f_new[h])
    if h not in consumers:
        g_old[h] = patch_h(f_old[h])
    for k in range(n):
        if h in pa[k]:
            g_old[k] = patch_child(g_old[k], k)
            g_new[k] = patch_child(g_new[k], k)
    return g_old, g_new


def scenarios(n_max, seed, per_graph=6):
    rng = random.Random(seed)
    for n in range(1, n_max + 1):
        for edges in dags(n):
            pa, anc = anc_sets(n, edges)
            for _ in range(per_graph):
                theta = {h: rng.randint(0, 2) for h in range(n)}
                cur = {h: ctx(h, theta, anc) for h in range(n)}
                recs = set()
                for h in range(n):
                    if rng.random() < 0.6:
                        recs.add((h, cur[h]))
                    for _k in range(2):
                        if rng.random() < 0.5:
                            old = {g: rng.randint(0, 2) for g in range(n)}
                            recs.add((h, ctx(h, old, anc)))
                U = {h for h in range(n) if rng.random() < 0.15}
                consumers = {k for (j, k) in edges if j in U}
                DU = desc(n, edges, consumers)
                N = {h for h in range(n) if (h, cur[h]) not in recs or h in DU}
                yield rng, n, edges, pa, theta, cur, recs, consumers, DU, N


# ---------------------------------------------------------------- tests
@pytest.mark.parametrize("seed", [1, 2])
def test_minimax_adversary_construction(seed):
    """(a) core: in W_H every h in N is indistinguishable from a world with another y_h."""
    cases = witnessed = 0
    for rng, n, edges, pa, theta, cur, recs, consumers, DU, N in scenarios(4, seed):
        f_old, f_new = herbrand(n, consumers)
        store = build_store(n, pa, recs, f_old)
        y = evaluate(n, pa, theta, f_new)
        cases += 1
        for h in N:
            g_old, g_new = masked_world(n, pa, theta, f_old, f_new, consumers, h, y, ("nu", h))
            # consistency: only consumers of U may differ between old and new functions
            for g in range(n):
                if g not in consumers:
                    for t in range(3):
                        for ins in itertools.product([("probe",), y.get(0)], repeat=len(pa[g])):
                            assert g_old[g](t, ins) == g_new[g](t, ins)
            store2 = build_store(n, pa, recs, g_old)
            y2 = evaluate(n, pa, theta, g_new)
            assert store2 == store, (edges, recs, h)            # same record values
            assert all(y2[g] == y[g] for g in range(n) if g != h)  # same everything else
            assert y2[h] != y[h]
            witnessed += 1
    assert cases > 0 and witnessed > 0


@pytest.mark.parametrize("seed", [1, 2])
def test_greedy_is_safe_and_hits_N_in_free_world(seed):
    """(a) for G: safe in random worlds (small and rich domains); recomputes exactly N in W_H.
    (b) strict gain exists: in some worlds G recomputes a proper subset of N."""
    strict = 0
    for rng, n, edges, pa, theta, cur, recs, consumers, DU, N in scenarios(4, seed):
        f_old, f_new = herbrand(n, consumers)
        store = build_store(n, pa, recs, f_old)
        z, R = greedy(n, pa, theta, f_new, store, cur, consumers, DU)
        assert z == evaluate(n, pa, theta, f_new)
        assert R == N, (edges, recs, consumers, N, R)
        for dom in ((0, 1), (0, 1, 2, 3)):
            for _w in range(15):
                f_old, f_new = table_world(n, pa, consumers, rng, dom)
                store = build_store(n, pa, recs, f_old)
                z, R = greedy(n, pa, theta, f_new, store, cur, consumers, DU)
                assert z == evaluate(n, pa, theta, f_new)   # safety
                assert R <= N
                strict += R < N
    assert strict > 0


def test_strict_gain_example():
    """(b) a -> b, parameter of a changed; old records a@0 and b@(0, 0) exist. N = {a, b}.
    In a world with f_a(1) = f_a(0), G recomputes only a and reuses b (early cutoff)."""
    n, edges = 2, [(0, 1)]
    pa, anc = anc_sets(n, edges)
    theta = {0: 1, 1: 0}
    cur = {h: ctx(h, theta, anc) for h in range(n)}
    recs = {(0, ctx(0, {0: 0, 1: 0}, anc)), (1, ctx(1, {0: 0, 1: 0}, anc))}
    N = {h for h in range(n) if (h, cur[h]) not in recs}
    assert N == {0, 1}
    f = [lambda t, ins: 7, lambda t, ins: ins[0] + t + 1]   # f_a does not depend on its parameter
    store = build_store(n, pa, recs, f)
    z, R = greedy(n, pa, theta, f, store, cur, set(), set())
    assert z == evaluate(n, pa, theta, f) and R == {0}
    f_h, _ = herbrand(n, set())                              # worst case: free world
    store = build_store(n, pa, recs, f_h)
    _, R = greedy(n, pa, theta, f_h, store, cur, set(), set())
    assert R == N


def test_richness_assumption_is_needed():
    """(c) with a one-element value domain the strategy 'never recompute' is safe."""
    n, edges = 2, [(0, 1)]
    pa, anc = anc_sets(n, edges)
    theta = {0: 1, 1: 0}
    f = [lambda t, ins: 0, lambda t, ins: 0]
    assert evaluate(n, pa, theta, f) == {0: 0, 1: 0}   # every world gives the only value
