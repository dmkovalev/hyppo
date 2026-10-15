"""Абляция уточнений CRM на Norne (in-sample, корректный R^2, с закачкой газа)."""
import os, sys, json
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
import numpy as np
import examples.research.r2tuned.models_r2tuned as M
from examples.research.r2tuned.brugge_reuse_timing_r2tuned import load_field_r2tuned
LIQ, WIN, t, oil, wat, lo, hi, ts = load_field_r2tuned("Norne")
m = LIQ > 1; out = {}
def run(name, LIQfit, tstart):
    _, p1 = M.fit_crm_masked(LIQfit, WIN, t, lo, hi, tstart)
    out[name] = round(M.r2c(p1, LIQ, m), 3); print(name, out[name], flush=True)
# (1) без маски: нули остановок как наблюдения (подменяем LIQ<=1 на 1.5, чтобы попали в маску подгонки, цель 0)
class Z(np.ndarray): pass
_orig = M.fit_crm_masked
def fit_nomask(LIQf, W, tt, a, b, tst):
    T, Np = LIQf.shape; p = np.zeros((T, Np))
    for j in range(Np):
        if (LIQf[a:b, j] > 1).sum() < 3: continue
        q = LIQf[:, j]; Ni = W.shape[1]
        lo_ = np.r_[np.zeros(Ni), np.full(Ni, 1.0), 0.0, 1.0]; up = np.r_[np.ones(Ni), np.full(Ni, 1e4), np.inf, 1e5]
        x0 = np.clip(np.r_[np.full(Ni, .1), np.full(Ni, 100.), q[q > 1][:5].mean(), 300.], lo_ + 1e-6, up - 1e-6)
        r = M.least_squares(lambda x: M._model(x, W, tt, tst[j], Ni) - q, x0, bounds=(lo_, up), x_scale="jac", max_nfev=300)
        p[:, j] = M._model(r.x, W, tt, tst[j], Ni)
    return None, p
run("mask+start (итог)", LIQ, ts)
run("mask, первичный от t=0", LIQ, np.zeros_like(ts))
_, p = fit_nomask(LIQ, WIN, t, lo, hi, ts); out["без маски, start"] = round(M.r2c(p, LIQ, m), 3); print(out)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "ablation_r2tuned.json"), "w"), indent=1, ensure_ascii=False)
