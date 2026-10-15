#!/usr/bin/env python3
"""Пара сегмента G Norne (F-4H -> E-4AH) на конвейере _r2tuned (разд. sect4_wf_bo).

Исходные числа (R^2 0.41 -> 0.57, коэффициент 0.27, dAIC 38, n=135) дал
F:/git-repos/diss/temp/norne_pair3.py: pywaterflood CRM на окне работы E-4AH
[первый..последний ненулевой шаг], 57 из 135 точек --- остановки (нулевой дебит),
которые входили в подгонку и в R^2; закачка газа не учитывалась; первичный член
отсчитывался от первого шага окна.

Здесь: fit_crm_masked (подгонка только по работающим точкам, дебит > 1),
спад первичной добычи от даты OPEN в WCONHIST, вход --- 9 водяных + 4 газовых
(WAG) нагнетательных в пластовых м3, R^2 --- r2c по тем же точкам маски.
Без лагов добычи, подгонка in-sample (как в исходном расчёте).

Запуск: .venv/Scripts/python.exe examples/research/r2tuned/norne_pair_r2tuned.py
Результат: examples/research/r2tuned/norne_pair_r2tuned.json
"""
from __future__ import annotations
import os, sys, json, math
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
os.environ.setdefault("HYPPO_BRUGGE_RUN", r"F:\git-repos\diss\thesis\papers\brugge_run")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

from examples.brugge_reuse_timing import _run_dir
from examples.research.r2tuned.models_r2tuned import (r2c, fit_crm_masked,
                                                      norne_gas_injection)
from examples.research.r2tuned.brugge_reuse_timing_r2tuned import load_field_r2tuned

HERE = os.path.dirname(os.path.abspath(__file__))

LIQ, WIN, t, oil, wat, lo, hi, t_start = load_field_r2tuned("Norne", with_gas=True)
d = np.load(_run_dir() + "/norne_perwell.npz", allow_pickle=True)
producers = [str(x) for x in d["producers"]]
_, gnames = norne_gas_injection(_run_dir() + "/norne_schedule.sch", LIQ.shape[0])
inj_names = [str(x) for x in d["injectors"]] + [g + "(gas)" for g in gnames]
T = LIQ.shape[0]
a = producers.index("E-4AH"); b = inj_names.index("F-4H")
q = LIQ[:, [a]]
mask = np.zeros((T, 1), bool); mask[:, 0] = q[:, 0] > 1
n = int(mask.sum())
out = {"n_points": n, "t_start_E4AH_day": float(t_start[a]),
       "window_steps_old": None}
nz = np.nonzero(q[:, 0] > 0)[0]
out["window_steps_old"] = int(nz.max() - nz.min() + 1)
out["shutin_in_old_window"] = int((q[nz.min():nz.max() + 1, 0] <= 1).sum())


def fit_one(cols):
    I = WIN[:, cols] if cols else np.zeros((T, 0))
    params, p1 = fit_crm_masked(q, I, t, 0, T, t_start[[a]])
    pr = p1[:, [0]]
    rss = float(((q - pr)[mask] ** 2).sum())
    k = 2 + 2 * len(cols)
    return {"R2": r2c(pr, q, mask), "rss": rss, "k": k,
            "AIC": n * math.log(rss / n) + 2 * k,
            "gains": [round(float(g), 3) for g in params[0][:len(cols)]]}


res = {"primary": fit_one([]), "primary+F-4H": fit_one([b]),
       "primary+all": fit_one(list(range(WIN.shape[1])))}
for k, v in res.items():
    print(f"E-4AH {k:14s}: R2={v['R2']:.3f}  AIC={v['AIC']:.1f}  gains={v['gains']}")
dA = res["primary"]["AIC"] - res["primary+F-4H"]["AIC"]
print(f"n={n}, dAIC(F-4H)={dA:.1f}")
out["models"] = res; out["dAIC_F4H"] = dA

# Скрининг: какая одиночная нагнетательная лучше всего объясняет E-4AH
scr = []
for j, nm in enumerate(inj_names):
    if not (WIN[:, j] > 0).any():
        continue
    r = fit_one([j])
    scr.append((res["primary"]["AIC"] - r["AIC"], r["R2"], r["gains"][0], nm))
scr.sort(reverse=True)
print("скрининг одиночных нагнетательных (dAIC, R2, коэф.):")
for s in scr:
    print(f"  {s[3]:12s} dAIC={s[0]:6.1f} R2={s[1]:.3f} f={s[2]:.3f}")
out["screen"] = [{"inj": s[3], "dAIC": s[0], "R2": s[1], "gain": s[2]} for s in scr]

# Полный CRM месторождения (все добывающие, все нагнетательные, in-sample)
params, p1 = fit_crm_masked(LIQ, WIN, t, 0, T, t_start)
Ni = WIN.shape[1]
G = np.array([p[:Ni] if p is not None else np.zeros(Ni) for p in params])
mall = LIQ > 1
out["fullfield_R2"] = r2c(p1, LIQ, mall)
out["fullfield_gain_F4H_E4AH"] = float(G[a, b])
top = sorted(((float(G[i, b]), p) for i, p in enumerate(producers)), reverse=True)[:6]
out["fullfield_F4H_top"] = top
print(f"полный CRM: R2={out['fullfield_R2']:.3f}; F-4H->E-4AH={G[a, b]:.3f}; "
      "крупнейшие F-4H: " + ", ".join(f"{p}={g:.3f}" for g, p in top))

json.dump(out, open(os.path.join(HERE, "norne_pair_r2tuned.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
