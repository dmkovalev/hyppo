"""Brute-force check of the safe-plan theorem on small DAGs.

World: deterministic f_h(theta_h, inputs) -> {0,1}. A revision of u in U (status/axioms, not
reflected in models or parameters) changes the functions of the direct consumers of u after the
records were made; u's own result stays valid (as in def:12 and the cascade of thm:integrity IV).
Repository: records (h, ctx) with ctx = params of h and all its sources; value computed with OLD f.
cached(h) <=> record with current ctx exists and h is not a strict descendant of U.  N = not cached.
Plan P executes: h in P recomputed from inputs (parents in P recomputed, else stored);
h not in P read from the record with the current ctx (missing -> failure).
Safe(P) <=> for EVERY world consistent with the metadata, the result equals full recomputation.

Checks (i) N ⊆ P => safe in all sampled worlds; (ii) for every h in N a world exists in which
N minus {h} fails; (iii) if the repository is closed under sources, N is closed downwards.
"""
import itertools, random, sys

import pytest

def dags(n):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for mask in range(1 << len(pairs)):
        yield [p for k, p in enumerate(pairs) if mask >> k & 1]

def anc_sets(n, edges):
    pa = {j: [i for i, k in edges if k == j] for j in range(n)}
    anc = {}
    for j in range(n):
        s = set()
        st = list(pa[j])
        while st:
            x = st.pop()
            if x not in s:
                s.add(x); st.extend(pa[x])
        anc[j] = s
    return pa, anc

def desc(n, edges, U):
    out = set(U); ch = {i: [k for j, k in edges if j == i] for i in range(n)}
    st = list(U)
    while st:
        x = st.pop()
        for y in ch[x]:
            if y not in out:
                out.add(y); st.append(y)
    return out

def ctx(h, theta, anc):
    return tuple(sorted((g, theta[g]) for g in anc[h] | {h}))

def evaluate(n, pa, theta, f):
    y = {}
    for h in range(n):  # 0..n-1 is a topological order
        y[h] = f[h][(theta[h], tuple(y[p] for p in pa[h]))]
    return y

def random_f(n, pa, rng):
    f = []
    for h in range(n):
        tbl = {}
        for t in (0, 1):
            for ins in itertools.product((0, 1), repeat=len(pa[h])):
                tbl[(t, ins)] = rng.randint(0, 1)
        f.append(tbl)
    return f

def flip(tbl):
    """A revised consumer: some input combination now gives a different value."""
    t = dict(tbl)
    k = next(iter(t))
    t[k] = 1 - t[k]
    return t


def run_plan(n, pa, theta, f_new, store, P, cur_ctx):
    z = {}
    for h in range(n):
        if h in P:
            ins = []
            for p in pa[h]:
                if p in P:
                    ins.append(z[p])
                else:
                    key = (p, cur_ctx[p])
                    if key not in store:
                        return None
                    ins.append(store[key])
            z[h] = f_new[h][(theta[h], tuple(ins))]
        else:
            key = (h, cur_ctx[h])
            if key not in store:
                return None
            z[h] = store[key]
    return z

def check(n_max=4, worlds=60, seed=1):
    rng = random.Random(seed)
    stats = dict(cases=0, safe_ok=0, nec_ok=0, closed_ok=0, closed_cases=0, strict=0)
    for n in range(1, n_max + 1):
        for edges in dags(n):
            pa, anc = anc_sets(n, edges)
            for _ in range(6):  # random scenario: current params, record contexts, U
                theta = {h: rng.randint(0, 1) for h in range(n)}
                cur = {h: ctx(h, theta, anc) for h in range(n)}
                # candidate record contexts: current or a random old parameter vector
                recs = set()
                for h in range(n):
                    if rng.random() < 0.7:
                        recs.add((h, cur[h]))
                    if rng.random() < 0.5:
                        old = {g: rng.randint(0, 1) for g in range(n)}
                        recs.add((h, ctx(h, old, anc)))
                U = {h for h in range(n) if rng.random() < 0.15}
                children_U = {k for (j, k) in edges if j in U}
                DU = desc(n, edges, children_U)  # strict descendants of U
                N = {h for h in range(n) if (h, cur[h]) not in recs or h in DU}
                stats['cases'] += 1

                def build_store(f_old):
                    store = {}
                    for (h, c) in recs:
                        th = dict(c)
                        # values of the sub-context: evaluate h and its sources with OLD f
                        full = {g: th.get(g, 0) for g in range(n)}
                        y = evaluate(n, pa, full, f_old)
                        store[(h, c)] = y[h]
                    return store

                # (i) N ⊆ P is safe in every sampled world
                ok = True
                for _w in range(worlds):
                    f_old = random_f(n, pa, rng)
                    f_new = [dict(f_old[h]) if h not in children_U else flip(f_old[h]) for h in range(n)]
                    store = build_store(f_old)
                    truth = evaluate(n, pa, theta, f_new)
                    for extra in (set(), set(range(n)), {g for g in range(n) if rng.random() < 0.5}):
                        z = run_plan(n, pa, theta, f_new, store, N | extra, cur)
                        if z != truth:
                            ok = False
                stats['safe_ok'] += ok
                if not ok:
                    print('SAFETY VIOLATION', n, edges, theta, recs, U, N); sys.exit(1)

                # (ii) every h in N is necessary: some world breaks N \ {h}
                nec = True
                for h in N:
                    found = False
                    for _w in range(400):
                        f_old = random_f(n, pa, rng)
                        f_new = [dict(f_old[g]) if g not in children_U else random_f(n, pa, rng)[g] for g in range(n)]
                        store = build_store(f_old)
                        truth = evaluate(n, pa, theta, f_new)
                        if run_plan(n, pa, theta, f_new, store, N - {h}, cur) != truth:
                            found = True; break
                    nec &= found
                    if not found:
                        print('NECESSITY NOT WITNESSED', n, edges, theta, recs, U, N, h); sys.exit(1)
                stats['nec_ok'] += nec

                # (iii) repository closed under sources => N closed downwards
                closed = all((g, cur[g]) in recs for (h, c) in recs if c == cur[h] for g in anc[h])
                if closed:
                    stats['closed_cases'] += 1
                    clN = desc(n, edges, N)
                    stats['closed_ok'] += (clN == N)
                    if clN != N:
                        print('CLOSURE VIOLATION', n, edges, recs, U, N, clN); sys.exit(1)
                else:
                    stats['strict'] += (desc(n, edges, N) != N)
    return stats



@pytest.mark.parametrize("seed", [1, 2])
def test_safe_plan_theorem_brute_force(seed):
    """Theorem thm:safe: N is the unique minimal safe plan; closed repository => Cl(N) = N."""
    s = check(n_max=4, worlds=30, seed=seed)
    assert s["safe_ok"] == s["cases"] and s["nec_ok"] == s["cases"]
    assert s["closed_ok"] == s["closed_cases"] > 0
    assert s["strict"] > 0  # without closure the cascade Cl(N) is strictly larger than N


def test_runner_repository_is_closed_under_sources(tmp_path):
    """Condition of thm:safe (iii) holds for the implementation: a stored result of a
    hypothesis at its chained key implies stored results of all its sources at
    their keys, across runs with parameter changes and model failures."""
    import networkx as nx
    from hyppo.manager import Manager

    rng = random.Random(7)
    checked = 0
    for trial in range(15):
        n = rng.randint(2, 6)
        nodes = [f"h{i}" for i in range(n)]
        edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
        mgr = Manager(db_path=tmp_path / f"r{trial}.db")
        for run in range(4):
            cfg = {h: {"p": rng.randint(0, 2)} for h in nodes}
            fail = {h for h in nodes if rng.random() < 0.15}

            def mk(h, bad):
                def f(c):
                    if bad:
                        raise RuntimeError("model failure")
                    return {"r2": 0.9}
                return f

            models = {h: mk(h, h in fail) for h in nodes}
            mgr.orchestrate(hypotheses=nodes, workflow_edges=edges, config=cfg, models=models)
            g = nx.DiGraph(); g.add_nodes_from(nodes); g.add_edges_from(edges)
            keys = Manager.cache_keys(g, models, cfg)
            for h in nodes:
                if mgr.repository.has_result(h, keys[h]):
                    checked += 1
                    for parent in g.predecessors(h):
                        assert mgr.repository.has_result(parent, keys[parent]), (trial, run, h, parent)
        mgr.close()
    assert checked > 50
