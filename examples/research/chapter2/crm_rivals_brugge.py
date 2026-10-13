#!/usr/bin/env python3
"""Переоценка соперников сквозного примера гл. 2 на открытых данных Brugge.

Соперничающая пара (модификация 1 графа CRM гл. 4): величину q_c вычисляют
  * многоканальная цепочка H2--H4 (стандартный конвейер гл. 4: CRM с постоянной
    времени для каждой пары <<нагнетательная--добывающая>>, tau_selection="per-pair");
  * одноканальный соперник H4' (одна постоянная времени на добывающую скважину,
    tau_selection="per-producer").
Обе модели подгоняются тем же конвейером, что и в табл. R^2 гл. 4
(examples/brugge_reuse_timing.py: окно подгонки [24 : 0,7 T], первичная добыча,
ограничения "up-to one"). AIC/BIC считаются на окне подгонки по гауссовым остаткам
(AIC = n ln(RSS/n) + 2k), R^2 --- как в табл. гл. 4 (все месяцы t >= 24, LIQ > 1),
а также отдельно на окне подгонки и на отложенной части.

Запуск (из корня репозитория):
    PYTHONPATH=. python examples/research/chapter2/crm_rivals_brugge.py
Результат: examples/research/chapter2/data/crm_rivals_brugge.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.brugge_reuse_timing import load_field, r2a  # noqa: E402

from pywaterflood import CRM  # noqa: E402

warnings.filterwarnings("ignore")

THETA_AIC = 10.0


def fit(sel: str, LIQ, WIN, t, s0, hi):
    c = CRM(primary=True, tau_selection=sel, constraints="up-to one")
    c.fit(production=LIQ[s0:hi], injection=WIN[s0:hi], time=t[s0:hi])
    pred = np.asarray(c.predict(injection=WIN, time=t)).reshape(LIQ.shape)
    return c, pred


def n_params(sel: str, Np: int, Ni: int) -> int:
    # коэффициенты связности + постоянные времени + первичная добыча (gain, tau)
    taus = Np * Ni if sel == "per-pair" else Np
    return Np * Ni + taus + 2 * Np


def main() -> None:
    LIQ, WIN, t, _oil, _wat, s0, hi = load_field("Brugge")
    T, Np = LIQ.shape
    Ni = WIN.shape[1]
    m_all = (LIQ > 1).copy(); m_all[:s0] = False
    m_fit = np.zeros_like(m_all); m_fit[s0:hi] = LIQ[s0:hi] > 1
    m_hold = np.zeros_like(m_all); m_hold[hi:] = LIQ[hi:] > 1

    rivals = {"H2-H4 (per-pair)": "per-pair", "H4' (per-producer)": "per-producer"}
    out = {"field": "Brugge", "T": int(T), "Np": int(Np), "Ni": int(Ni),
           "fit_window": [int(s0), int(hi)], "rivals": {}, "series": {}}
    obs_total = np.where(LIQ > 1, LIQ, 0).sum(1)
    out["series"]["t"] = [float(x) for x in t]
    out["series"]["observed_total"] = [float(x) for x in obs_total]
    out["perwell"] = {"observed": np.round(LIQ, 3).tolist()}
    for label, sel in rivals.items():
        c, pred = fit(sel, LIQ, WIN, t, s0, hi)
        res = (LIQ - pred)[m_fit]
        n = int(m_fit.sum()); rss = float((res ** 2).sum())
        k = n_params(sel, Np, Ni)
        aic = n * np.log(rss / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
        bic = n * np.log(rss / n) + k * np.log(n)
        out["rivals"][label] = {
            "tau_selection": sel, "k": k, "n_fit": n, "rss_fit": rss,
            "AIC": float(aic), "AICc": float(aicc), "BIC": float(bic),
            "R2_table": r2a(pred, LIQ, m_all),
            "R2_fit": r2a(pred, LIQ, m_fit),
            "R2_holdout": r2a(pred, LIQ, m_hold),
            "tau_range": [float(np.min(c.tau)), float(np.max(c.tau))],
        }
        out["series"][label] = [float(x) for x in np.where(LIQ > 1, pred, 0).sum(1)]
        out["perwell"][label] = np.round(pred, 3).tolist()

    aics = {k: v["AIC"] for k, v in out["rivals"].items()}
    best = min(aics, key=aics.get)
    a_min = aics[best]
    z = sum(np.exp(-(a - a_min) / 2) for a in aics.values())
    for k, v in out["rivals"].items():
        v["dAIC"] = float(aics[k] - a_min)
        v["akaike_weight"] = float(np.exp(-(aics[k] - a_min) / 2) / z)
        v["status"] = "SUPERSEDED" if v["dAIC"] > THETA_AIC else (
            "SUPPORTED" if v["R2_table"] >= 0.7 else "REFUTED")
    out["winner_AIC"] = best
    bics = {k: v["BIC"] for k, v in out["rivals"].items()}
    out["winner_BIC"] = min(bics, key=bics.get)

    dest = Path(__file__).resolve().parent / "data" / "crm_rivals_brugge.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in out["rivals"].items():
        print(f"{k:20s} k={v['k']:4d} n={v['n_fit']} R2(tab)={v['R2_table']:.4f} "
              f"R2(fit)={v['R2_fit']:.4f} R2(hold)={v['R2_holdout']:.4f} "
              f"AIC={v['AIC']:.1f} BIC={v['BIC']:.1f} dAIC={v['dAIC']:.1f} "
              f"w={v['akaike_weight']:.3g} tau={v['tau_range']} -> {v['status']}")
    print("winner AIC:", out["winner_AIC"], " winner BIC:", out["winner_BIC"])
    print("saved", dest)


if __name__ == "__main__":
    main()
