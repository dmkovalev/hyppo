#!/usr/bin/env python3
"""Посткважинный R^2 компонент модели на Brugge (к табл. tab:r2_fields гл. 4).

Табличные R^2 (CRM, Hybrid, WCT) в brugge_reuse_timing.py --- объединённые
(pooled) по всем скважинам: сумма квадратов остатков по всем (t, j) делится на
сумму квадратов отклонений от общего среднего. На Brugge большая часть
добывающих скважин в окне подгонки работает с постоянным дебитом жидкости,
поэтому объединённый R^2 в основном отражает различия между скважинами, а не
качество воспроизведения динамики каждой скважины.

Здесь тот же конвейер (те же функции fit_crm / fit_hybrid / fit_wct из
examples/brugge_reuse_timing.py, те же окна) пересчитывается по скважинам.

Окна (как в конвейере гл. 4, load_field("Brugge")):
* окно подгонки   [s0, hi) = [24, int(0.7 T)) = [24, 93) --- на нём оцениваются CRM и Hybrid;
* отложенное окно [hi, T) = [93, 133)                   --- на нём CRM и Hybrid не подгонялись.
WCT (Бакли--Леверетт, fit_wct) подгоняется по скважинам на всём ряде (окна нет),
поэтому его посткважинный R^2 считается на всём ряде и он in-sample.

Для скважины j: R^2_j = 1 - SSE / SST по месяцам с дебитом жидкости > 1,
SST --- от среднего этой скважины в этом окне.

Исключения:
* CRM, Hybrid (дебит жидкости): скважина исключается из окна, если коэффициент
  вариации дебита жидкости CV = std / mean (по месяцам с дебитом > 1 в окне)
  меньше CV_MIN = 0.01 --- постоянный дебит, SST ~ 0, R^2 не определён;
* WCT (обводнённость): скважина исключается, если максимальная обводнённость
  за ряд меньше WCT_BT = 0.05 (прорыва воды нет, обводнённость ~ 0, SST ~ 0).

Запуск (из корня репозитория):
    HYPPO_BRUGGE_RUN=.../thesis/papers/brugge_run PYTHONPATH=. \
        python examples/research/chapter4/brugge_r2_perwell.py
Результат: examples/research/chapter4/results/brugge_r2_perwell.json
"""
from __future__ import annotations

import json
import os

import numpy as np

from examples.brugge_reuse_timing import (fit_crm, fit_hybrid, fit_wct,
                                          load_field, r2a)

CV_MIN = 0.01
WCT_BT = 0.05
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "brugge_r2_perwell.json")


def r2_well(pred, y, m):
    y = y[m]; p = pred[m]
    sst = ((y - y.mean()) ** 2).sum()
    if sst <= 0:
        return None
    return float(1 - ((y - p) ** 2).sum() / sst)


def cv(x):
    return float(x.std() / x.mean()) if len(x) and x.mean() > 0 else 0.0


def summary(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "median": round(float(med), 3),
            "q1": round(float(q1), 3), "q3": round(float(q3), 3),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
            "n_negative": int((v < 0).sum())}


def main():
    LIQ, WIN, t, oil, wat, s0, hi = load_field("Brugge")
    T, Np = LIQ.shape
    _, p1 = fit_crm(LIQ, WIN, t, s0, hi)
    hyb, r2h_pooled = fit_hybrid(p1, LIQ, WIN, s0, hi, Np)
    wct_pred, r2w_pooled = fit_wct(LIQ, wat, oil)
    wct = np.where(LIQ > 1, wat / np.maximum(LIQ, 1), 0.0)

    mval = (LIQ > 1).copy(); mval[:s0] = False
    pooled = {
        # ровно как в brugge_reuse_timing.run_components
        "CRM": round(r2a(p1, LIQ, mval), 3),            # месяцы s0..T (окно подгонки + отложенное)
        "Hybrid": round(r2h_pooled, 3),                  # окно подгонки
        "WCT": round(r2w_pooled, 3),                     # весь ряд
        "CRM_fit_window": round(r2a(p1[s0:hi], LIQ[s0:hi], LIQ[s0:hi] > 1), 3),
        "CRM_heldout": round(r2a(p1[hi:], LIQ[hi:], LIQ[hi:] > 1), 3),
        "Hybrid_heldout": round(r2a(hyb[hi:], LIQ[hi:], LIQ[hi:] > 1), 3),
    }

    windows = {"fit": (s0, hi), "heldout": (hi, T)}
    wells = []
    res = {}
    for wname, (a, b) in windows.items():
        per = {"CRM": [], "Hybrid": []}
        const = []
        for j in range(Np):
            m = LIQ[a:b, j] > 1
            c = cv(LIQ[a:b, j][m])
            rec = {"well": j, "window": wname, "n_months": int(m.sum()),
                   "cv_liq": round(c, 4)}
            if m.sum() < 5 or c < CV_MIN:
                rec["excluded_constant_rate"] = True
                const.append(j)
            else:
                rec["excluded_constant_rate"] = False
                rc = r2_well(p1[a:b, j], LIQ[a:b, j], m)
                rh = r2_well(hyb[a:b, j], LIQ[a:b, j], m)
                rec["CRM"] = round(rc, 4); rec["Hybrid"] = round(rh, 4)
                per["CRM"].append(rc); per["Hybrid"].append(rh)
            wells.append(rec)
        res[wname] = {"window_months": [a, b], "constant_rate_wells": const,
                      "n_constant": len(const), "n_included": Np - len(const),
                      **{k: summary(v) for k, v in per.items()}}

    wct_rows, wct_vals, no_bt = [], [], []
    for j in range(Np):
        m = LIQ[:, j] > 1
        mx = float(wct[m, j].max()) if m.any() else 0.0
        r = r2_well(wct_pred[:, j], wct[:, j], m) if m.sum() >= 5 else None
        row = {"well": j, "max_wct": round(mx, 5),
               "R2": None if r is None else round(r, 4),
               "excluded_no_breakthrough": mx < WCT_BT}
        wct_rows.append(row)
        if mx < WCT_BT:
            no_bt.append(j)
        elif r is not None:
            wct_vals.append(r)
    res["WCT_full_series"] = {"no_breakthrough_wells": no_bt,
                              "n_included": len(wct_vals),
                              "WCT": summary(wct_vals), "wells": wct_rows}

    out = {"field": "Brugge", "n_producers": Np, "T": T, "s0": s0, "hi": hi,
           "cv_min": CV_MIN, "wct_breakthrough": WCT_BT,
           "constant_rate_definition": "CV = std/mean дебита жидкости по месяцам с дебитом > 1 в окне < cv_min",
           "no_breakthrough_definition": "max обводнённости по ряду < wct_breakthrough",
           "note_wct": "WCT подгоняется по скважинам на всём ряде: посткважинный R^2 WCT in-sample",
           "pooled": pooled, "per_well_summary": res, "wells": wells}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    short = {k: {kk: vv for kk, vv in v.items() if kk != "wells"} for k, v in res.items()}
    print(json.dumps({"pooled": pooled, "per_well_summary": short}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
