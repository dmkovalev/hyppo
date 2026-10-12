# tests/test_causal_refine.py
from hyppo.planner import causal_refine as cr


def _two_output_pair():
    s = {"a": {"ex": frozenset({"x", "theta"}), "ey": frozenset({"y", "phi"})},
         "b": {"ez": frozenset({"z", "x"})}}
    inputs = {"a": {"theta", "phi"}, "b": {"x"}}
    return s, inputs, [("a", "b")]


def test_change_of_unread_parameter_keeps_consumer_actual():
    s, inputs, edges = _two_output_pair()
    assert cr.n_caus(s, inputs, edges, changed_params={("a", "phi")}, stale=set()) == {"a"}
    assert cr.n_caus(s, inputs, edges, changed_params={("a", "theta")}, stale=set()) == {"a", "b"}


def test_single_output_hypotheses_coincide_with_N():
    s = {"a": {"e": frozenset({"x", "t"})}, "b": {"e": frozenset({"y", "x"})}, "c": {"e": frozenset({"z", "y"})}}
    inputs = {"a": {"t"}, "b": {"x"}, "c": {"y"}}
    edges = [("a", "b"), ("b", "c")]
    assert cr.n_caus(s, inputs, edges, changed_params={("a", "t")}, stale=set()) == {"a", "b", "c"}


def test_stale_marks_propagate_at_hypothesis_level():
    s, inputs, edges = _two_output_pair()
    assert cr.n_caus(s, inputs, edges, changed_params=set(), stale={"a"}) == {"a", "b"}


def test_affected_variables_of_edit_is_forward_closure():
    st = {"e1": frozenset({"x", "u"}), "e2": frozenset({"y", "x"}), "e3": frozenset({"w", "v"})}
    assert cr.affected_variables(st, {"e1"}, inputs={"u", "v"}) == {"x", "y"}
    assert cr.affected_variables(st, {"e3"}, inputs={"u", "v"}) == {"w"}


# ───────────── property test: N_caus vs an independent reference (varlevel_check.py) ─────────────
# The generators and the reference trace below are copied (not imported) from the research
# script temp/align/varlevel_check.py. The reference builds D block-wise from the
# Dulmage--Mendelsohn decomposition (every variable of a block depends on all of Pre(B) and
# Theta(B)), with parameters carried by equation subsets and global exogenous parameters X:x;
# the library builds D edge-wise from a perfect matching over per-hypothesis parameters. The
# reference trace is additionally validated semantically by brute-force world evaluation.
import hashlib  # noqa: E402
import random  # noqa: E402

from hyppo.coa import causal  # noqa: E402

_P_DOM, _Q = 3, 3


def _hyp(name, eqs, inputs, params, task):
    eqs = [frozenset(e) for e in eqs]
    V = set().union(*eqs)
    out = V - set(inputs)
    assert len(eqs) == len(out)
    params = {p: set(c) for p, c in params.items()}
    params["M:" + name] = set(range(len(eqs)))  # model identifier: carrier = all equations
    return {"name": name, "eqs": eqs, "In": set(inputs), "Out": out, "params": params, "task": task}


def _gen_instance(rng, p_coll=0.15):
    nh = rng.randint(2, 4)
    exo = [f"x{i}" for i in range(rng.randint(1, 3))]
    hyps, produced = [], []
    for i in range(nh):
        k = rng.randint(1, 3)
        outs = []
        for j in range(k):
            cands = [y for y in produced if y not in outs and not any(y in hh["In"] for hh in hyps)]
            if cands and rng.random() < p_coll:
                outs.append(rng.choice(cands))
            else:
                outs.append(f"y{i}{j}")
        avail = list(exo) + [y for y in dict.fromkeys(produced) if y not in outs]
        ins = rng.sample(avail, rng.randint(0, min(3, len(avail))))
        eqs = []
        for j in range(k):
            s = {outs[j]}
            for o in outs:
                if o != outs[j] and rng.random() < 0.3:
                    s.add(o)
            for x in ins:
                if rng.random() < 0.5:
                    s.add(x)
            eqs.append(s)
        for x in ins:
            if not any(x in e for e in eqs):
                rng.choice(eqs).add(x)
        params = {}
        for t in range(rng.randint(1, 2)):
            car = {j for j in range(k) if rng.random() < 0.5} or {rng.randrange(k)}
            params[f"t{i}{t}"] = car
        hyps.append(_hyp(f"h{i}", eqs, ins, params, i))
        produced += [o for o in outs if o not in produced]
    return _RefInst(hyps)


def _gen_repo(rng, inst, q_change=0.25, p_keep=0.85, p_u=0.1):
    c = {p: rng.randrange(_Q) for p in inst.params}
    runs = []
    for _ in range(rng.randint(1, 3)):
        r = dict(c)
        for p in r:
            if rng.random() < q_change:
                r[p] = (r[p] + rng.randint(1, _Q - 1)) % _Q
        runs.append(r)
    kept = {(h["name"], j) for h in inst.H for j in range(len(runs)) if rng.random() < p_keep}
    U = {h["name"] for h in inst.H if rng.random() < p_u}
    return c, runs, kept, U


class _RefInst:
    def __init__(self, hyps):
        self.H = hyps
        self.by = {h["name"]: h for h in hyps}
        self.parents = {h["name"]: set() for h in hyps}
        self.children = {h["name"]: set() for h in hyps}
        for a in hyps:
            for b in hyps:
                if a["task"] < b["task"] and a["Out"] & b["In"]:
                    self.parents[b["name"]].add(a["name"])
                    self.children[a["name"]].add(b["name"])
        self.prod, self.exo = {}, set()
        for b in hyps:
            for x in b["In"]:
                ps = sorted(a for a in self.parents[b["name"]] if x in self.by[a]["Out"])
                self.prod[(x, b["name"])] = ps
                if not ps:
                    self.exo.add(x)
        self.params = {p for h in hyps for p in h["params"]} | {"X:" + x for x in self.exo}
        self.nodes, self.preds = [("p", p) for p in sorted(self.params)], {}
        for n in self.nodes:
            self.preds[n] = []
        for h in hyps:
            for x in sorted(h["In"]):
                n = ("in", x, h["name"])
                ps = self.prod[(x, h["name"])]
                self.preds[n] = [("v", x, a) for a in ps] if ps else [("p", "X:" + x)]
                self.nodes.append(n)
            for phi, pre, th in self._blocks(h):
                for v in sorted(phi):
                    n = ("v", v, h["name"])
                    self.preds[n] = [("in", u, h["name"]) if u in h["In"] else ("v", u, h["name"])
                                     for u in sorted(pre)] + [("p", t) for t in sorted(th)]
                    self.nodes.append(n)
        fp = {}
        for n in self.nodes:  # nodes are in topological order
            fp[n] = frozenset([n[1]]) if n[0] == "p" else frozenset().union(*(fp[p] for p in self.preds[n]))
        self.Pi = {h["name"]: frozenset().union(*(fp[("v", v, h["name"])] for v in h["Out"])) for h in hyps}

    @staticmethod
    def _blocks(h):
        k = len(h["eqs"])
        blocks, m = causal._block_decomposition_with_matching(
            list(h["eqs"]) + [frozenset([x]) for x in sorted(h["In"])])
        out = []
        for B in blocks:
            Bi = sorted(i for i in B if i < k)
            if not Bi:
                continue
            phi = {m[i] for i in Bi}
            vs = set().union(*(h["eqs"][i] for i in Bi))
            th = {p for p, car in h["params"].items() if car & set(Bi)}
            out.append((frozenset(phi), frozenset(vs - phi), frozenset(th)))
        res, done = [], set()
        while len(res) < len(out):  # topological order of blocks
            blk = next(b for b in out if b not in res and {v for v in b[1] if v in h["Out"]} <= done)
            res.append(blk)
            done |= blk[0]
        return res

    def descendants(self, U):
        seen, st = set(), list(U)
        while st:
            for ch in self.children[st.pop()]:
                if ch not in seen:
                    seen.add(ch)
                    st.append(ch)
        return seen

    def evaluate(self, cfg, wf):
        val = {}
        for n in self.nodes:
            val[n] = cfg[n[1]] if n[0] == "p" else wf(n, tuple(val[p] for p in self.preds[n]))
        return val

    def path_to(self, src, targets):
        succ = {}
        for n, ps in self.preds.items():
            for p in ps:
                succ.setdefault(p, []).append(n)
        prev, queue = {src: None}, [src]
        while queue:
            n = queue.pop(0)
            if n in targets:
                path = []
                while n is not None:
                    path.append(n)
                    n = prev[n]
                return path[::-1]
            for s in succ.get(n, ()):
                if s not in prev:
                    prev[s] = n
                    queue.append(s)
        return None


def _to_library(inst):
    """Research instance -> (structures, inputs, edges, param map research-name -> pairs)."""
    structures, inputs, pmap = {}, {}, {}
    for h in inst.H:
        n = h["name"]
        st = {}
        for i, e in enumerate(h["eqs"]):
            st[f"e{i}"] = frozenset(e | {p for p, car in h["params"].items() if i in car})
        structures[n] = st
        inputs[n] = set(h["In"]) | set(h["params"])
        for p in h["params"]:
            pmap.setdefault(p, set()).add((n, p))
        for x in h["In"]:
            if not inst.prod[(x, n)]:
                pmap.setdefault("X:" + x, set()).add((n, x))
    edges = [(a, b) for b, ps in inst.parents.items() for a in sorted(ps)]
    return structures, inputs, edges, pmap


def _rand_world(seed):
    def wf(n, args):
        d = hashlib.blake2b(repr((seed, n, args)).encode(), digest_size=4).digest()
        return int.from_bytes(d, "little") % _P_DOM
    return wf


def _path_world(inst, path, c):
    """Witness world: along the path the value is passed on, everything else is constant 0."""
    pos = {n: i for i, n in enumerate(path)}
    theta = path[0][1]

    def wf(n, args):
        i = pos.get(n)
        if not i:
            return 0
        k = inst.preds[n].index(path[i - 1])
        return (1 if args[k] == c[theta] else 0) if i == 1 else args[k]
    return wf


def test_n_caus_matches_reference_on_random_instances():
    rng = random.Random(20261012)
    checks = strict = 0
    for it in range(300):
        inst = _gen_instance(rng)
        _, _, _, U = _gen_repo(rng, inst)
        structures, inputs, edges, pmap = _to_library(inst)
        marked = set(U) | inst.descendants(U)
        c0 = {p: 0 for p in inst.params}
        worlds = [_rand_world((it, w)) for w in range(6)]
        base = [inst.evaluate(c0, wf) for wf in worlds]
        for p in sorted(inst.params):
            ref = {h for h in inst.by if p in inst.Pi[h]}
            for stale in (set(), U):
                got = cr.n_caus(structures, inputs, edges, changed_params=pmap[p], stale=stale)
                assert got == ref | (marked if stale else set()), (it, p, stale)
                checks += 1
            owners = {h for h, _ in pmap[p]}
            if ref < owners | inst.descendants(owners):  # strictly below the hypothesis-level N
                strict += 1
            # semantic validation of the reference: sufficiency (random worlds) ...
            r = dict(c0)
            r[p] = 1
            for wf, v0 in zip(worlds, base):
                v1 = inst.evaluate(r, wf)
                for h in set(inst.by) - ref:
                    assert all(v0[("v", v, h)] == v1[("v", v, h)] for v in inst.by[h]["Out"]), (it, p, h)
            # ... and necessity (constructive witness world along a path in D)
            for h in ref:
                path = inst.path_to(("p", p), {("v", v, h) for v in inst.by[h]["Out"]})
                assert path is not None
                wf = _path_world(inst, path, r)
                v0, v1 = inst.evaluate(c0, wf), inst.evaluate(r, wf)
                assert any(v0[("v", v, h)] != v1[("v", v, h)] for v in inst.by[h]["Out"]), (it, p, h)
    assert checks > 0
    assert strict > 0  # the refinement is non-trivial on this sample
