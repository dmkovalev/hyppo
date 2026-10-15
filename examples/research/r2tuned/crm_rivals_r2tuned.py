"""Переоценка соперников H4 || H4' (fig:crm_rivals, разд. sect4_wf_rivals) на конвейере _r2tuned.

Отличия от examples/research/chapter2/crm_rivals_brugge.py (исходный скрипт не меняется):
  * данные/окно/дата ввода --- load_field_r2tuned("Brugge"): подгонка на [0, 0.7T) = [0, 93),
    t_start[j] --- первый шаг с дебитом > 1;
  * подгонка --- fit_crm_masked (потери только по работающим точкам LIQ > 1 окна,
    первичная добыча от даты ввода, 0 <= f_ij <= 1);
    H2--H4: tau на каждую пару (Ni tau на добывающую), H4': одна tau на добывающую;
  * R^2 --- r2c (SSE и SST по одним и тем же точкам маски);
  * AIC/AICc/BIC --- по остаткам на работающих точках окна подгонки [0, 93); n = их число;
    k = фактическое число оценённых параметров (по подогнанным скважинам).
Отложенная часть [93, 133) в подгонке не участвует.

Запуск: .venv/Scripts/python.exe examples/research/r2tuned/crm_rivals_r2tuned.py
Результат: examples/research/r2tuned/crm_rivals_r2tuned.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
os.environ.setdefault("HYPPO_BRUGGE_RUN", r"F:\git-repos\diss\thesis\papers\brugge_run")
from examples.research.r2tuned.brugge_reuse_timing_r2tuned import load_field_r2tuned  # noqa: E402
from examples.research.r2tuned.models_r2tuned import (fit_crm_masked, q_CRM_perpair,  # noqa: E402
                                                      r2c)

THETA_AIC, THETA_SUP = 10.0, 0.7
HERE = os.path.dirname(os.path.abspath(__file__))


def _model_pp(x, I, t, t0, Ni):
    """Одна tau на добывающую: CRM-P с одинаковой tau для всех нагнетательных."""
    gains, tau, q0, taup = x[:Ni], x[Ni], x[Ni + 1], x[Ni + 2]
    q = q_CRM_perpair(I, t, gains, np.full(Ni, tau))
    on = t >= t0
    prim = np.zeros_like(t)
    prim[on] = q0 * np.exp(-(t[on] - t0) / taup)
    return q + prim


def fit_crm_masked_perproducer(LIQ, WIN, t, s0, hi, t_start):
    T, Np = LIQ.shape
    Ni = WIN.shape[1]
    I = np.ascontiguousarray(WIN, float)
    tt = np.ascontiguousarray(t, float)
    p1 = np.zeros((T, Np))
    params = []
    lo = np.r_[np.zeros(Ni), 1.0, 0.0, 1.0]
    up = np.r_[np.ones(Ni), 1e4, np.inf, 1e5]
    for j in range(Np):
        m = np.zeros(T, bool)
        m[s0:hi] = LIQ[s0:hi, j] > 1
        if m.sum() < 3:
            params.append(None)
            continue
        q = LIQ[:, j]
        x0 = np.r_[np.full(Ni, 0.1), 100.0, q[m][:5].mean(), 300.0]
        x0 = np.clip(x0, lo + 1e-6, up - 1e-6)
        r = least_squares(lambda x: (_model_pp(x, I, tt, t_start[j], Ni) - q)[m], x0,
                          bounds=(lo, up), x_scale="jac", max_nfev=300)
        params.append(r.x)
        p1[:, j] = _model_pp(r.x, I, tt, t_start[j], Ni)
    return params, p1


def main():
    LIQ, WIN, t, _oil, _wat, lo, hi, t_start = load_field_r2tuned("Brugge")
    T, Np = LIQ.shape
    Ni = WIN.shape[1]
    work = LIQ > 1
    m_all = work.copy()                                   # весь ряд [0, T), как в tab:r2_fields
    m_fit = np.zeros_like(work); m_fit[lo:hi] = work[lo:hi]
    m_hold = np.zeros_like(work); m_hold[hi:] = work[hi:]

    fits = {"H2-H4 (per-pair)": fit_crm_masked, "H4' (per-producer)": fit_crm_masked_perproducer}
    out = {"field": "Brugge", "pipeline": "r2tuned", "T": int(T), "Np": int(Np), "Ni": int(Ni),
           "fit_window": [int(lo), int(hi)], "t_start_idx": [int(np.searchsorted(t, x)) for x in t_start],
           "rivals": {}, "series": {"t": t.tolist(),
                                    "observed_total": np.where(work, LIQ, 0).sum(1).tolist()},
           "perwell": {"observed": np.round(LIQ, 3).tolist()}}
    for label, fn in fits.items():
        params, pred = fn(LIQ, WIN, t, lo, hi, t_start)
        res = (LIQ - pred)[m_fit]
        n = int(m_fit.sum()); rss = float((res ** 2).sum())
        k = int(sum(len(p) for p in params if p is not None))
        ll = n * np.log(rss / n)
        aic = ll + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
        bic = ll + k * np.log(n)
        taus = np.concatenate([p[Ni:2 * Ni] if label.startswith("H2") else p[Ni:Ni + 1]
                               for p in params if p is not None])
        out["rivals"][label] = {
            "k": k, "n_fit": n, "n_over_k": n / k, "rss_fit": rss, "n_wells_fitted":
            int(sum(p is not None for p in params)),
            "AIC": float(aic), "AICc": float(aicc), "BIC": float(bic),
            "R2_table": r2c(pred, LIQ, m_all), "R2_fit": r2c(pred, LIQ, m_fit),
            "R2_holdout": r2c(pred, LIQ, m_hold),
            "tau_range": [float(taus.min()), float(taus.max())],
        }
        out["series"][label] = np.where(work, pred, 0).sum(1).tolist()
        out["perwell"][label] = np.round(pred, 3).tolist()

    R = out["rivals"]
    for crit in ("AIC", "AICc", "BIC"):
        vals = {a: v[crit] for a, v in R.items()}
        best = min(vals.values())
        z = sum(np.exp(-(x - best) / 2) for x in vals.values())
        for a, v in R.items():
            v["d" + crit] = float(vals[a] - best)
            v["w_" + crit] = float(np.exp(-(vals[a] - best) / 2) / z)
    for a, v in R.items():
        v["akaike_weight"] = v["w_AIC"]
        v["status"] = ("SUPERSEDED" if v["dAIC"] > THETA_AIC else
                       "SUPPORTED" if v["R2_table"] >= THETA_SUP else "REFUTED")
    A, B = "H2-H4 (per-pair)", "H4' (per-producer)"
    out["delta_H4_minus_H4p"] = {c: R[A][c] - R[B][c] for c in ("AIC", "AICc", "BIC")}

    dest = os.path.join(HERE, "crm_rivals_r2tuned.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for a, v in R.items():
        print(f"{a:20s} k={v['k']} n={v['n_fit']} n/k={v['n_over_k']:.2f} "
              f"R2 all={v['R2_table']:.4f} fit={v['R2_fit']:.4f} hold={v['R2_holdout']:.4f} "
              f"AIC={v['AIC']:.1f} AICc={v['AICc']:.1f} BIC={v['BIC']:.1f} "
              f"w={v['akaike_weight']:.3g} tau={v['tau_range']} -> {v['status']}")
    print("H4 - H4':", {c: round(x, 1) for c, x in out["delta_H4_minus_H4p"].items()})
    print("saved", dest)


if __name__ == "__main__":
    main()
