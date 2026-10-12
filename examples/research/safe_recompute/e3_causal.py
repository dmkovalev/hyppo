"""E3: причинное уточнение безопасного плана (N_caus против N уровня гипотез).

Для каждого одиночного изменения параметра сравниваются
  N      -- план уровня гипотез: гипотезы-владельцы параметра и все их потомки в графе гипотез;
  N_caus -- hyppo.planner.causal_refine.n_caus (гипотезы, у которых изменённый параметр
            причинно влияет хотя бы на один выход), stale = пусто.
Теорема: N_caus ⊆ N; равенство, если каждый параметр влияет на все выходы своей гипотезы;
укрупнение гипотез сохраняет N_caus (на уровне переменных), но может увеличить N.

Экземпляры и модель параметров скопированы из исследовательского скрипта
temp/align/varlevel_check.py (не импортируются): гипотеза -- система уравнений, In(h),
параметры с носителями (подмножествами уравнений), идентификатор модели h.M с носителем
"все уравнения", экзогенные входы -- глобальные параметры X:x (один параметр на все
гипотезы-читатели). Рёбра графа гипотез: a -> b, если a предшествует b по порядку задач и
Out(a) ∩ In(b) ≠ ∅ (вариант А при коллизиях выходов). Перевод в формат causal_refine -- как в
tests/test_causal_refine.py: параметр становится входной переменной уравнений своего носителя,
X:x передаётся парами (h, x) для всех читателей.

Блоки результата (data/e3_causal.json):
  projectile_exact   -- сквозной пример гл. 2 с точными сигнатурами h2^0, h2^1 (24 параметра);
  projectile_aligned -- контроль: сигнатуры выровнены, уточнение пусто;
  crm_fine           -- граф CRM/Norne из 16 гипотез по одному уравнению (формулы
                        examples/norne_alg1_graph.py); одновыходные гипотезы, поэтому N_caus = N;
  crm_coarse         -- тот же граф, укрупнённый до 9 компонент исполнения
                        (CRM = H1..H6, ML, FUS, MB, COREY = 2 гипотезы, BL = 2 гипотезы, WCTF,
                        GTM, OPR; varlevel_check.NORNE_COMP). N считается на графе компонент и
                        измеряется числом покрытых исходных гипотез; N_caus -- на мелком графе,
                        и проверяется, что множество затронутых выходных переменных в
                        укрупнённой системе даёт те же исходные гипотезы (инвариантность (iv));
  synthetic          -- случайные системы gen_instance из varlevel_check с одним изменением:
                        число уравнений (= выходов) гипотезы k = 1 с вероятностью 1 - q и
                        k ~ U{2, 3} с вероятностью q (в оригинале k ~ U{1, 2, 3}); остальное без
                        изменений (2-4 гипотезы, 1-3 экзогенных входа, p_coll = 0.15, 1-2
                        параметра + модель). Генератор с фиксированным зерном для каждого q.

Имена гипотез CRM/Norne соответствуют графу CRM гл. 4 (common.CRM_EDGES) так:
GRP = H15 (GTM), H11 = H9 (MB), H12/H12b = H10/H11 (COREY), H13/H14 = H12/H13 (BL),
H15 = H14 (WCTF), H19 = H16 (OPR), остальные совпадают. Граф по формулам отличается от
CRM_EDGES одним местом: krw и kro (COREY) вычисляются параллельно из Sw, а не цепочкой,
поэтому рёбер 18, а не 17.
"""
from __future__ import annotations

import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

from hyppo.planner import causal_refine as cr
from examples.research.safe_recompute.common import DATA, save_json

SEED = 20261012
QS = (0.0, 0.25, 0.5, 0.75)


# ───────────────────────────── модель (по varlevel_check.py) ─────────────────────────────
def hyp(name, eqs, inputs, params, task, model_param=True):
    eqs = [frozenset(e) for e in eqs]
    V = set().union(*eqs)
    if not set(inputs) <= V:
        raise ValueError(f"{name}: inputs not in equations")
    out = V - set(inputs)
    if len(eqs) != len(out):
        raise ValueError(f"{name}: |E| != |Out|")
    params = {p: set(c) for p, c in params.items()}
    if model_param:
        params[f"{name}.M"] = set(range(len(eqs)))
    return {"name": name, "eqs": eqs, "In": set(inputs), "Out": out, "params": params, "task": task}


class Inst:
    def __init__(self, hyps, prec):
        self.H = hyps
        self.by = {h["name"]: h for h in hyps}
        self.parents = {h["name"]: set() for h in hyps}
        for a in hyps:
            for b in hyps:
                if a is b or not prec(a["task"], b["task"]):
                    continue
                if a["In"] & b["Out"]:
                    raise ValueError(f"order conflict {a['name']}->{b['name']}")
                if a["Out"] & b["In"]:
                    self.parents[b["name"]].add(a["name"])
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.by)
        self.G.add_edges_from((a, b) for b, ps in self.parents.items() for a in ps)
        self.exo_readers = defaultdict(set)
        for b in hyps:
            for x in b["In"]:
                if not any(x in self.by[a]["Out"] for a in self.parents[b["name"]]):
                    self.exo_readers[x].add(b["name"])

    def to_library(self):
        """-> (structures, inputs, edges, pmap: research parameter -> set of (h, name) pairs)."""
        structures, inputs, pmap = {}, {}, {}
        for h in self.H:
            n = h["name"]
            structures[n] = {f"e{i}": frozenset(e | {p for p, car in h["params"].items() if i in car})
                             for i, e in enumerate(h["eqs"])}
            inputs[n] = set(h["In"]) | set(h["params"])
            for p in h["params"]:
                pmap.setdefault(p, set()).add((n, p))
        for x, readers in self.exo_readers.items():
            pmap["X:" + x] = {(n, x) for n in readers}
        edges = sorted(self.G.edges)
        return structures, inputs, edges, pmap

    def n_hyp(self, owners):
        """Hypothesis-level plan N: owners and all their descendants."""
        res = set(owners)
        for h in owners:
            res |= nx.descendants(self.G, h)
        return res


def single_changes(inst):
    """[(parameter, N, N_caus)] for every single parameter change, stale = ∅."""
    s, i, e, pmap = inst.to_library()
    rows = []
    for p in sorted(pmap):
        nc = cr.n_caus(s, i, e, changed_params=pmap[p], stale=set())
        n = inst.n_hyp({h for h, _ in pmap[p]})
        assert nc <= n, (p, nc, n)
        rows.append((p, n, nc))
    return rows


def summarize(rows):
    return {"cases": len(rows),
            "strict_cases": sum(nc < n for _, n, nc in rows),
            "n_mean": statistics.fmean(len(n) for _, n, _ in rows),
            "n_caus_mean": statistics.fmean(len(nc) for _, _, nc in rows)}


# ───────────────────────────── снаряд (гл. 2) ─────────────────────────────
def projectile(exact):
    """Сквозной пример гл. 2 (f1--f9). exact=True: точные сигнатуры h2^0, h2^1."""
    W = nx.DiGraph([("t1", "t2"), ("t2", "t3"), ("t3", "t4"), ("t3", "t5"), ("t3", "t6"), ("t4", "t7")])
    prec = lambda a, b: b in nx.descendants(W, a)  # noqa: E731
    H = [
        hyp("h1a", [{"Cd"}], [], {"cd0": {0}}, "t1"),
        hyp("h1b", [{"Re", "rho", "vt", "d", "mu"}, {"Cd", "Re"}, {"vt", "m", "g", "rho", "d", "Cd"}],
            ["rho", "d", "mu", "m", "g"], {"psi": {1}}, "t1"),
        hyp("h1", [{"b", "mu", "d"}, {"c", "Cd", "rho", "d"}], ["Cd", "d", "mu", "rho"],
            {"kS": {0}, "kN": {1}}, "t2"),
    ]
    if exact:
        H += [hyp("h20", [{"traj", "v0", "theta", "g"}], ["v0", "theta", "g"], {}, "t3"),
              hyp("h21", [{"traj", "b", "m", "v0", "theta"}], ["b", "m", "v0", "theta"], {}, "t3"),
              hyp("h22", [{"traj", "b", "c", "m", "v0", "theta"}], ["b", "c", "m", "v0", "theta"],
                  {"rk_step": {0}}, "t3")]
    else:
        for k in range(3):
            H.append(hyp(f"h2{k}", [{"traj", "b", "c", "m", "v0", "theta"}],
                         ["b", "c", "m", "v0", "theta"], {"rk_step": {0}} if k == 2 else {}, "t3"))
    H += [hyp("h3", [{"R", "traj", "theta"}], ["traj", "theta"], {"grid": {0}}, "t4"),
          hyp("h4", [{"Hmax", "traj"}], ["traj"], {}, "t5"),
          hyp("h5", [{"E", "traj", "m"}], ["traj", "m"], {}, "t6"),
          hyp("h6", [{"thstar", "R", "theta"}], ["R", "theta"], {"tgrid": {0}}, "t7")]
    return Inst(H, prec)


def projectile_block(exact):
    rows = single_changes(projectile(exact))
    out = summarize(rows)
    out["strict"] = [{"param": p, "N": len(n), "N_caus": len(nc), "saved": sorted(n - nc)}
                     for p, n, nc in rows if nc < n]
    return out


# ───────────────────────────── CRM/Norne ─────────────────────────────
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
NORNE = [  # examples/norne_alg1_graph.py (HYPS), формулы дословно
    ("H1", "I_agg = w_ij * I_j", "I_agg"), ("H2", "q_f = a_f*q_f_prev + b_f*I_agg", "q_f"),
    ("H3", "q_s = a_s*q_s_prev + b_s*I_agg", "q_s"), ("H4", "q_c = w_f*q_f + (1-w_f)*q_s", "q_c"),
    ("H5", "q_liq_phys = J*q_c + q_prim", "q_liq_phys"), ("H6", "q_prim = q_prev*exp(-dt*taup)", "q_prim"),
    ("H7", "l_ml = MLP(x_hist)", "l_ml"), ("H8", "l = g*q_liq_phys + (1-g)*l_ml", "l"),
    ("H11", "Sw = Sw_prev + (Winj - l)*dt/Vp", "Sw"),
    ("H12", "krw = ((Sw-Swc)/(1-Swc-Sor))**nw", "krw"), ("H12b", "kro = ((1-Sw-Sor)/(1-Swc-Sor))**no", "kro"),
    ("H13", "fw = 1/(1 + kro*muw/(krw*muo))", "fw"), ("H14", "o_p = 1 - fw", "o_p"),
    ("H15", "o = gw*o_p + (1-gw)*o_m", "o"), ("GRP", "J = J0 + dJ_grp", "J"), ("H19", "q_oil = l * o", "q_oil"),
]
NORNE_TASKS = [["H1"], ["H2", "H3"], ["H4"], ["H6"], ["GRP"], ["H5"], ["H7"], ["H8"], ["H11"],
               ["H12", "H12b"], ["H13"], ["H14"], ["H15"], ["H19"]]
NORNE_DATA = {"I_j", "q_f_prev", "q_s_prev", "q_prev", "dt", "x_hist", "Winj", "Sw_prev", "o_m", "Swc", "Sor"}
NORNE_FUN = {"exp", "MLP"}
NORNE_COMP = {"GTM": ["GRP"], "CRM": ["H1", "H2", "H3", "H4", "H6", "H5"], "ML": ["H7"], "FUS": ["H8"],
              "MB": ["H11"], "COREY": ["H12", "H12b"], "BL": ["H13", "H14"], "WCTF": ["H15"], "OPR": ["H19"]}


def _norne_spec():
    outs = {n: o for n, _, o in NORNE}
    spec = {}
    for n, f, o in NORNE:
        ids = set(IDENT.findall(f)) - NORNE_FUN
        vars_ = {v for v in ids if v in outs.values() or v in NORNE_DATA}
        spec[n] = (vars_, vars_ - {o}, sorted(ids - vars_), o)
    return spec


def norne(components=False):
    spec = _norne_spec()
    task = {h: i for i, t in enumerate(NORNE_TASKS) for h in t}
    if not components:
        H = [hyp(n, [spec[n][0]], spec[n][1], {f"{n}.{p}": {0} for p in spec[n][2]}, task[n])
             for n, _, _ in sorted(NORNE, key=lambda r: task[r[0]])]
        return Inst(H, lambda a, b: a < b)
    H = []
    for ci, (cname, comp) in enumerate(NORNE_COMP.items()):
        eqs, params, allout, allin = [], {}, set(), set()
        for n in comp:
            params[f"{n}.M"] = {len(eqs)}  # same parameter names as in the fine instance
            for p in spec[n][2]:
                params[f"{n}.{p}"] = {len(eqs)}
            eqs.append(spec[n][0])
            allout.add(spec[n][3])
            allin |= spec[n][1]
        H.append(hyp(cname, eqs, allin - allout, params, ci, model_param=False))
    return Inst(H, lambda a, b: a < b)


def crm_blocks():
    fine, coarse = norne(False), norne(True)
    producer = {o: n for n, (_, _, _, o) in _norne_spec().items()}
    size = {c: len(m) for c, m in NORNE_COMP.items()}
    fs, fi, fe, fpm = fine.to_library()
    cs, ci, ce, cpm = coarse.to_library()
    assert set(fpm) == set(cpm)
    D = cr.influence_graph(cs, ci, ce)
    rows = []
    for p in sorted(fpm):
        n_fine = fine.n_hyp({h for h, _ in fpm[p]})
        n_caus = cr.n_caus(fs, fi, fe, changed_params=fpm[p], stale=set())
        n_coarse_c = coarse.n_hyp({h for h, _ in cpm[p]})
        n_caus_c = cr.n_caus(cs, ci, ce, changed_params=cpm[p], stale=set())
        # (iv): affected output variables of the coarse system -> the same fine hypotheses
        seeds = {("p", h, x) for h, x in cpm[p]}
        reached = set().union(*(nx.descendants(D, s) for s in seeds))
        var_level = {producer[x] for k, _, x in reached if k == "v"}
        assert var_level == n_caus, (p, var_level, n_caus)
        assert n_caus <= n_fine, p
        rows.append({"param": p, "N_fine": len(n_fine), "N_coarse": sum(size[c] for c in n_coarse_c),
                     "N_caus": len(n_caus), "N_caus_coarse_components": sum(size[c] for c in n_caus_c)})
    m = len(rows)
    crm_fine = {"hypotheses": len(fine.H), "edges": fine.G.number_of_edges(), "cases": m,
                "strict_cases": sum(r["N_caus"] < r["N_fine"] for r in rows),
                "n_mean": statistics.fmean(r["N_fine"] for r in rows),
                "n_caus_mean": statistics.fmean(r["N_caus"] for r in rows)}
    crm_coarse = {"components": len(coarse.H), "edges": coarse.G.number_of_edges(), "cases": m,
                  "n_mean": statistics.fmean(r["N_coarse"] for r in rows),
                  "n_caus_mean": statistics.fmean(r["N_caus"] for r in rows),
                  "n_caus_components_mean": statistics.fmean(r["N_caus_coarse_components"] for r in rows),
                  "frac_coarse_gt_caus": sum(r["N_coarse"] > r["N_caus"] for r in rows) / m,
                  "frac_coarse_gt_fine": sum(r["N_coarse"] > r["N_fine"] for r in rows) / m,
                  "variable_level_invariance_ok": True,
                  "rows": rows}
    return crm_fine, crm_coarse


# ───────────────────────────── случайные системы ─────────────────────────────
def gen_instance(rng, q, p_coll=0.15):
    """varlevel_check.gen_instance; k = 1 w.p. 1 - q, else k ~ U{2, 3}."""
    nh = rng.randint(2, 4)
    exo = [f"x{i}" for i in range(rng.randint(1, 3))]
    hyps, produced = [], []
    for i in range(nh):
        k = rng.randint(2, 3) if rng.random() < q else 1
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
        hyps.append(hyp(f"h{i}", eqs, ins, params, i))
        produced += [o for o in outs if o not in produced]
    return Inst(hyps, lambda a, b: a < b)


def synthetic_block(n_inst):
    res = {}
    for qi, q in enumerate(QS):
        rng = random.Random(SEED + qi)
        ratios, strict_inst, multi, nh = [], 0, 0, 0
        for _ in range(n_inst):
            inst = gen_instance(rng, q)
            rows = single_changes(inst)
            ratios += [len(nc) / len(n) for _, n, nc in rows]
            strict_inst += any(nc < n for _, n, nc in rows)
            multi += sum(len(h["Out"]) >= 2 for h in inst.H)
            nh += len(inst.H)
            if all(len(h["Out"]) == 1 for h in inst.H):
                assert all(nc == n for _, n, nc in rows)  # equality criterion of the theorem
        res[str(q)] = {"instances": n_inst, "cases": len(ratios),
                       "multi_output_share": multi / nh,
                       "median_ratio": statistics.median(ratios),
                       "mean_ratio": statistics.fmean(ratios),
                       "frac_strict": sum(r < 1 for r in ratios) / len(ratios),
                       "frac_instances_strict": strict_inst / n_inst}
    return res


def main(synthetic=500, out=None):
    crm_fine, crm_coarse = crm_blocks()
    res = {"seed": SEED,
           "projectile_exact": projectile_block(True),
           "projectile_aligned": projectile_block(False),
           "crm_fine": crm_fine,
           "crm_coarse": crm_coarse,
           "synthetic": synthetic_block(synthetic)}
    save_json(res, Path(out) if out is not None else DATA / "e3_causal.json")
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = main()
    for k in ("projectile_exact", "projectile_aligned", "crm_fine"):
        print(k, {a: b for a, b in r[k].items() if a != "strict"})
    print("crm_coarse", {a: b for a, b in r["crm_coarse"].items() if a != "rows"})
    for q, v in r["synthetic"].items():
        print("synthetic q =", q, v)
