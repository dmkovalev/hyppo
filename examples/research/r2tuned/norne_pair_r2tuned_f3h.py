#!/usr/bin/env python3
"""Кандидатная пара Norne F-3H -> E-4AH на конвейере _r2tuned (оценка замены F-4H).

Тот же конвейер, что norne_pair_r2tuned.py: маска остановок (дебит > 1), спад
первичной добычи от даты OPEN, 9 водяных + 4 газовых (WAG) нагнетательных в
пластовых м3, r2c по точкам маски, без лагов, in-sample.

Дополнительно: (1) многостартовая подгонка (сетка начальных f, tau, tau_p) ---
устойчивость R^2/AIC/коэффициента; (2) отложенный хвост: подгонка по первым 70 %
работающих точек E-4AH, R^2 на оставшихся 30 %; (3) полный CRM месторождения ---
коэффициент F-3H -> E-4AH и крупнейшие получатели закачки F-3H; (4) расстояние
по сетке (центроиды COMPDAT) от E-4AH до нагнетательных.

Запуск: .venv/Scripts/python.exe examples/research/r2tuned/norne_pair_r2tuned_f3h.py
Результат: examples/research/r2tuned/norne_pair_r2tuned_f3h.json
"""
from __future__ import annotations
import os, sys, json, math, itertools, collections
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
os.environ.setdefault("HYPPO_BRUGGE_RUN", r"F:\git-repos\diss\thesis\papers\brugge_run")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.optimize import least_squares

from examples.brugge_reuse_timing import _run_dir
from examples.research.r2tuned.models_r2tuned import (r2c, fit_crm_masked, _model,
                                                      norne_gas_injection)
from examples.research.r2tuned.brugge_reuse_timing_r2tuned import load_field_r2tuned

HERE = os.path.dirname(os.path.abspath(__file__))
SCH = _run_dir() + "/norne_schedule.sch"

LIQ, WIN, t, oil, wat, lo, hi, t_start = load_field_r2tuned("Norne", with_gas=True)
d = np.load(_run_dir() + "/norne_perwell.npz", allow_pickle=True)
producers = [str(x) for x in d["producers"]]
_, gnames = norne_gas_injection(SCH, LIQ.shape[0])
inj_names = [str(x) for x in d["injectors"]] + [g + "(gas)" for g in gnames]
T = LIQ.shape[0]
a = producers.index("E-4AH")
b3, b4 = inj_names.index("F-3H"), inj_names.index("F-4H")
q = LIQ[:, a]
mask = q > 1
idx = np.nonzero(mask)[0]
n = int(mask.sum())
out = {"n_points": n, "t_start_E4AH_day": float(t_start[a]),
       "E4AH_first_last_step": [int(idx.min()), int(idx.max())]}

F0 = (0.05, 0.1, 0.3, 0.7)
TAU0 = (10.0, 100.0, 1000.0)
TAUP0 = (100.0, 300.0, 1000.0)


def fit(cols, m, x0f=0.1, x0tau=100.0, x0taup=300.0):
    Ni = len(cols)
    I = WIN[:, cols] if cols else np.zeros((T, 0))
    lo_ = np.r_[np.zeros(Ni), np.full(Ni, 1.0), 0.0, 1.0]
    up_ = np.r_[np.ones(Ni), np.full(Ni, 1e4), np.inf, 1e5]
    x0 = np.r_[np.full(Ni, x0f), np.full(Ni, x0tau), q[m][:5].mean(), x0taup]
    x0 = np.clip(x0, lo_ + 1e-6, up_ - 1e-6)
    r = least_squares(lambda x: (_model(x, I, t, t_start[a], Ni) - q)[m], x0,
                      bounds=(lo_, up_), x_scale="jac", max_nfev=300)
    return r.x, _model(r.x, I, t, t_start[a], Ni)


def score(cols, x, pr, m):
    rss = float(((q - pr)[m] ** 2).sum()); nn = int(m.sum()); k = 2 + 2 * len(cols)
    return {"R2": r2c(pr, q, m), "rss": rss, "k": k,
            "AIC": nn * math.log(rss / nn) + 2 * k,
            "gains": [round(float(g), 3) for g in x[:len(cols)]]}


def default(cols):
    x, pr = fit(cols, mask)
    return score(cols, x, pr, mask)


def multistart(cols):
    runs = []
    for f0, ta, tp in itertools.product(F0 if cols else (0.1,), TAU0 if cols else (100.0,), TAUP0):
        x, pr = fit(cols, mask, f0, ta, tp)
        s = score(cols, x, pr, mask); s["x0"] = [f0, ta, tp]; runs.append(s)
    best = min(runs, key=lambda s: s["AIC"])
    return best, runs


# 1) пары, стартовая точка как в norne_pair_r2tuned.py
sets = {"primary": [], "primary+F-3H": [b3], "primary+F-4H": [b4],
        "primary+F-3H+F-4H": [b3, b4], "primary+all": list(range(WIN.shape[1]))}
res = {k: default(v) for k, v in sets.items()}
for k, v in res.items():
    print(f"E-4AH {k:18s}: R2={v['R2']:.3f} AIC={v['AIC']:.1f} gains={v['gains']}")
out["models_default_start"] = res
out["dAIC_F3H"] = res["primary"]["AIC"] - res["primary+F-3H"]["AIC"]
out["dAIC_F4H"] = res["primary"]["AIC"] - res["primary+F-4H"]["AIC"]
out["dAIC_F4H_given_F3H"] = res["primary+F-3H"]["AIC"] - res["primary+F-3H+F-4H"]["AIC"]
print(f"n={n}; dAIC F-3H={out['dAIC_F3H']:.1f}, F-4H={out['dAIC_F4H']:.1f}, "
      f"F-4H|F-3H={out['dAIC_F4H_given_F3H']:.1f}")

# 2) многостартовая устойчивость
ms = {}
for k in ("primary", "primary+F-3H", "primary+F-4H"):
    best, runs = multistart(sets[k])
    R2s = [r["R2"] for r in runs]; Gs = [r["gains"][0] for r in runs if r["gains"]]
    ms[k] = {"best": best, "n_starts": len(runs),
             "R2_min": min(R2s), "R2_max": max(R2s),
             "gain_min": min(Gs) if Gs else None, "gain_max": max(Gs) if Gs else None,
             "gain_at_best": best["gains"][0] if best["gains"] else None}
    print(f"multistart {k:14s}: {len(runs)} стартов, R2 [{min(R2s):.3f}, {max(R2s):.3f}], "
          f"best R2={best['R2']:.3f} AIC={best['AIC']:.1f} gains={best['gains']}"
          + (f", f in [{min(Gs):.3f}, {max(Gs):.3f}]" if Gs else ""))
out["multistart"] = ms
out["dAIC_F3H_multistart"] = ms["primary"]["best"]["AIC"] - ms["primary+F-3H"]["best"]["AIC"]
out["dAIC_F4H_multistart"] = ms["primary"]["best"]["AIC"] - ms["primary+F-4H"]["best"]["AIC"]
print(f"dAIC (лучший старт): F-3H={out['dAIC_F3H_multistart']:.1f}, "
      f"F-4H={out['dAIC_F4H_multistart']:.1f}")

# 3) отложенный хвост: первые 70 % работающих точек -> оставшиеся 30 %
cut = idx[int(0.7 * len(idx))]
mtr = mask & (np.arange(T) < cut); mte = mask & (np.arange(T) >= cut)
hold = {}
for k in ("primary", "primary+F-3H", "primary+F-4H"):
    x, pr = fit(sets[k], mtr)
    hold[k] = {"R2_train": r2c(pr, q, mtr), "R2_test": r2c(pr, q, mte),
               "gains": [round(float(g), 3) for g in x[:len(sets[k])]]}
    print(f"holdout {k:14s}: R2 train={hold[k]['R2_train']:.3f} test={hold[k]['R2_test']:.3f} "
          f"gains={hold[k]['gains']}")
out["holdout"] = {"n_train": int(mtr.sum()), "n_test": int(mte.sum()), **hold}

# 4) когда закачивает F-3H относительно работы E-4AH
inj3 = np.nonzero(WIN[:, b3] > 0)[0]; inj4 = np.nonzero(WIN[:, b4] > 0)[0]
out["F3H_inj_steps"] = [int(inj3.min()), int(inj3.max())]
out["F4H_inj_steps"] = [int(inj4.min()), int(inj4.max())]
out["corr_q_E4AH_vs_inj_on_mask"] = {
    "F-3H": float(np.corrcoef(q[mask], WIN[mask, b3])[0, 1]),
    "F-4H": float(np.corrcoef(q[mask], WIN[mask, b4])[0, 1])}
print(f"F-3H закачка шаги {out['F3H_inj_steps']}, F-4H {out['F4H_inj_steps']}, "
      f"E-4AH работа {out['E4AH_first_last_step']}; corr={out['corr_q_E4AH_vs_inj_on_mask']}")

# 5) полный CRM месторождения
params, p1 = fit_crm_masked(LIQ, WIN, t, 0, T, t_start)
Ni = WIN.shape[1]
G = np.array([p[:Ni] if p is not None else np.zeros(Ni) for p in params])
out["fullfield_R2"] = r2c(p1, LIQ, LIQ > 1)
out["fullfield_gain_F3H_E4AH"] = float(G[a, b3])
out["fullfield_gain_F4H_E4AH"] = float(G[a, b4])
out["fullfield_F3H_top"] = sorted(((float(G[i, b3]), p) for i, p in enumerate(producers)),
                                  reverse=True)[:6]
out["fullfield_E4AH_gains"] = sorted(((float(G[a, j]), nm) for j, nm in enumerate(inj_names)),
                                     reverse=True)
out["fullfield_F3H_rank_for_E4AH"] = [nm for _, nm in out["fullfield_E4AH_gains"]].index("F-3H") + 1
print(f"полный CRM: R2={out['fullfield_R2']:.3f}; F-3H->E-4AH={G[a, b3]:.3f} "
      f"(место {out['fullfield_F3H_rank_for_E4AH']}), F-4H->E-4AH={G[a, b4]:.3f}; "
      "крупнейшие F-3H: " + ", ".join(f"{p}={g:.3f}" for g, p in out["fullfield_F3H_top"]))

# 6) геометрия: центроиды перфораций (COMPDAT, I-J) и расстояния до E-4AH
c = collections.defaultdict(list); inC = False
for l in open(SCH, encoding="latin-1"):
    s = l.strip()
    if s.startswith("COMPDAT"): inC = True; continue
    if inC and s.startswith("/"): inC = False; continue
    if inC and s.startswith("'"):
        p = s.split(); c[p[0].strip("'")].append((int(p[1]), int(p[2])))
cen = {w: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for w, v in c.items()}
dist = lambda u, v: math.hypot(cen[u][0] - cen[v][0], cen[u][1] - cen[v][1])
out["centroids_IJ"] = {w: cen[w] for w in ("E-4AH", "E-4H", "F-3H", "F-4H")}
out["dist_cells_E4AH_to_inj"] = sorted((round(dist("E-4AH", w), 1), w)
                                       for w in cen if w[0] in "CF")
out["F3H_nearest_producers"] = sorted((round(dist("F-3H", w), 1), w)
                                      for w in cen if w[0] in "BDEK")[:6]
print("расстояния E-4AH -> нагнетательные (ячейки):", out["dist_cells_E4AH_to_inj"])
print("ближайшие к F-3H добывающие:", out["F3H_nearest_producers"])

json.dump(out, open(os.path.join(HERE, "norne_pair_r2tuned_f3h.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
