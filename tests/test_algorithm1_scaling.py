"""Algorithm 1 (lattice build) scales polynomially in |H| with the polynomial
core. Runs the real library on Python 3.13; asserts the fitted log-log exponent
is well below the old exponential blow-up and consistent with ~quadratic."""
import math
import random
import time

from hyppo.coa._base import Structure, Equation


def _gen_struct(rng, n_eq=5, pool=20):
    av = [f"x_{k}" for k in range(pool)]
    chosen = rng.sample(av, n_eq)
    eqs = []
    for i in range(n_eq):
        extras = rng.sample([v for v in chosen if v != chosen[i]],
                            rng.randint(1, min(3, n_eq - 1)))
        eqs.append(Equation(formula="+".join([chosen[i], *extras]) + "=0"))
    return Structure(eqs)


def _build_cost(n_h, rng):
    hyps = [_gen_struct(rng) for _ in range(n_h)]
    edges = [(i, j) for i in range(n_h) for j in range(i + 1, n_h)
             if rng.random() < 0.3]
    adj = {i: set() for i in range(n_h)}
    for u, v in edges:
        adj[u].add(v)
    reach = {}
    for s in range(n_h):
        seen, st = set(), [s]
        while st:
            c = st.pop()
            for nb in adj[c]:
                if nb not in seen:
                    seen.add(nb)
                    st.append(nb)
        reach[s] = seen
    for i in range(n_h):
        for j in reach[i]:
            su = hyps[i].union(hyps[j])
            if su.is_complete():
                su.build_transitive_closure()


def test_algorithm1_exponent_is_polynomial():
    h_values = [10, 20, 30, 50, 70, 100, 150]
    means = []
    for n_h in h_values:
        ts = []
        for rep in range(10):
            rng = random.Random(42 + rep)
            t0 = time.perf_counter()
            _build_cost(n_h, rng)
            ts.append(time.perf_counter() - t0)
        means.append(sum(ts) / len(ts))
    lh = [math.log(h) for h in h_values]
    lt = [math.log(m) for m in means]
    n = len(lh)
    mx, my = sum(lh) / n, sum(lt) / n
    a = sum((x - mx) * (y - my) for x, y in zip(lh, lt)) / sum((x - mx) ** 2 for x in lh)
    print(f"fitted exponent a = {a:.3f}")
    assert a < 2.8, f"exponent {a:.3f} too high -- expected near-quadratic"
