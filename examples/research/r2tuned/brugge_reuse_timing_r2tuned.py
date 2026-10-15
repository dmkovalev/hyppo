#!/usr/bin/env python3
"""Вариант _r2tuned бенчмарка examples/brugge_reuse_timing.py (табл. tab:r2_fields,
tab:reuse_timing гл. 4). Исходный скрипт и его JSON не изменяются.

Отличия (подробно --- в models_r2tuned.py):
* R^2 --- корректный объединённый (SST от среднего по точкам маски), r2c;
* CRM --- CRM-P per-pair, 'up-to one', подгонка только по работающим точкам,
  спад первичной добычи от даты ввода скважины; для Norne во вход CRM добавлена
  закачка газа (WAG-нагнетательные C-1H, C-3H, C-4H, C-4AH) в пластовых м3 (B_g);
* Hybrid --- та же гребневая регрессия без лага добычи, alpha по отложенному хвосту;
* окно: Brugge --- подгонка CRM/Hybrid на [0, 0.7T), R^2 на всём ряде [0, T);
  Norne --- подгонка и R^2 на всём ряде (in-sample, как в исходном протоколе);
  все четыре компонента оцениваются на одном и том же окне.
* WCT (Бакли--Леверетт) --- без изменений (fit_wct исходного конвейера).

Запуск:  .venv/Scripts/python.exe examples/research/r2tuned/brugge_reuse_timing_r2tuned.py Brugge|Norne
Результат: examples/research/r2tuned/timing_<поле>_r2tuned.json
"""
from __future__ import annotations
import os, sys, time, json
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
os.environ.setdefault("HYPPO_BRUGGE_RUN", r"F:\git-repos\diss\thesis\papers\brugge_run")
import numpy as np

from examples.brugge_reuse_timing import (_run_dir, fit_wct, build_graph, _Lat, NODES,
                                          COMPONENT)
from examples.research.r2tuned.models_r2tuned import (r2c, fit_crm_masked, fit_hybrid_masked,
                                                      norne_open_dates, norne_gas_injection,
                                                      BG_NORNE)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_field_r2tuned(name, with_gas=True):
    """-> LIQ, WIN, t, oil, wat, fit_lo, fit_hi, t_start (сут ввода скважин)."""
    if name == "Brugge":
        d = np.load(_run_dir() + "/brugge_perwell.npz", allow_pickle=True)
        dw = np.load(_run_dir() + "/brugge_oilwater.npz", allow_pickle=True)
        LIQ = d["production"].astype(float); WIN = d["injection"].astype(float)
        t = d["time"].astype(float)
        oil = dw["oil"].astype(float); wat = dw["water"].astype(float)
        T, Np = LIQ.shape
        first = [np.argmax(LIQ[:, j] > 1) if (LIQ[:, j] > 1).any() else T - 1 for j in range(Np)]
        t_start = t[np.array(first)]
        return LIQ, WIN, t, oil, wat, 0, int(0.7 * T), t_start
    d = np.load(_run_dir() + "/norne_perwell.npz", allow_pickle=True)
    LIQ = d["liq"].astype(float); WIN = d["inj"].astype(float); t = d["time"].astype(float)
    oil = d["oil"].astype(float); wat = d["water"].astype(float)
    T = LIQ.shape[0]
    sch = _run_dir() + "/norne_schedule.sch"
    od = norne_open_dates(sch, T)
    t_start = np.array([t[od.get(w, T - 1)] for w in d["producers"]])
    if with_gas:
        G, _ = norne_gas_injection(sch, T)
        WIN = np.hstack([WIN, G * BG_NORNE])
    return LIQ, WIN, t, oil, wat, 0, T, t_start


def run_components(comps, data, cache, per_component_ms=None, ev=None):
    LIQ, WIN, t, oil, wat, lo, hi, t_start = data
    T, Np = LIQ.shape
    ev_lo, ev_hi = ev if ev else (0, T)
    m = np.zeros_like(LIQ, bool); m[ev_lo:ev_hi] = LIQ[ev_lo:ev_hi] > 1
    mo = np.zeros_like(LIQ, bool); mo[ev_lo:ev_hi] = oil[ev_lo:ev_hi] > 1
    wct = np.where(LIQ > 1, wat / np.maximum(LIQ, 1), 0.0)
    pc = per_component_ms if per_component_ms is not None else {}
    pc.update({k: 0.0 for k in ("CRM", "ML", "FUS", "MB", "COREY", "BL", "WCTF", "GTM", "OPR")})
    pc["GTM"] = 0.001
    t_total, r2 = 0.0, {}
    if "CRM" in comps:
        ts = time.perf_counter()
        _, cache["p1"] = fit_crm_masked(LIQ, WIN, t, lo, hi, t_start)
        dt = time.perf_counter() - ts; t_total += dt; pc["CRM"] += dt * 1000
        r2["CRM"] = r2c(cache["p1"], LIQ, m)
    if "ML" in comps or "FUS" in comps:
        ts = time.perf_counter()
        cache["hyb"], cache["alpha"] = fit_hybrid_masked(cache["p1"], LIQ, WIN, lo, hi, Np)
        dt = time.perf_counter() - ts; t_total += dt; pc["ML"] += dt * 500; pc["FUS"] += dt * 500
        r2["Hybrid"] = r2c(cache["hyb"], LIQ, m)
    if {"MB", "COREY", "BL", "WCTF"} & comps:
        ts = time.perf_counter()
        cache["wct"], _ = fit_wct(LIQ, wat, oil)
        dt = time.perf_counter() - ts; t_total += dt
        for k in ("MB", "COREY", "BL", "WCTF"):
            pc[k] += dt * 250
        r2["WCT"] = r2c(cache["wct"], wct, m)
    if "OPR" in comps and cache.get("hyb") is not None and cache.get("wct") is not None:
        ts = time.perf_counter()
        opr = cache["hyb"] * (1 - cache["wct"])
        dt = time.perf_counter() - ts; t_total += dt; pc["OPR"] += dt * 1000
        r2["OPR"] = r2c(opr, oil, mo)
    return t_total, r2


def main():
    field = sys.argv[1] if len(sys.argv) > 1 else "Brugge"
    from hyppo.planner._base import build_optimal_plan, cache_keys_for
    from hyppo.metadata_repository import SharedCache
    data = load_field_r2tuned(field)
    LIQ = data[0]; T = LIQ.shape[0]
    all_comps = {"CRM", "ML", "FUS", "MB", "COREY", "BL", "WCTF", "OPR"}
    cache, pcm = {}, {}
    t_full, r2_full = run_components(all_comps, data, cache, pcm)
    print(f"{field}: {LIQ.shape[1]} доб., {data[1].shape[1]} нагн. входов, {T} шагов; "
          f"подгонка [{data[5]},{data[6]}), R^2 на [0,{T})")
    for k, v in r2_full.items():
        print(f"    {k:8s} R2 = {v:.3f}")
    print(f"    alpha = {cache['alpha']}; полный пересчёт {t_full*1000:.0f} мс")

    # отложенная выборка (доп. проверка, другой протокол): подгонка CRM/Hybrid на [lo, 0.7T)
    hold = {}
    hi_h = int(0.7 * T)
    dh = data[:6] + (hi_h,) + data[7:]
    ch = {}
    run_components({"CRM", "ML", "FUS"}, dh, ch)
    ch["wct"] = cache["wct"]
    _, rh = run_components({"OPR"}, dh, ch, ev=(hi_h, T))
    oil = data[3]
    m = np.zeros_like(LIQ, bool); m[hi_h:] = LIQ[hi_h:] > 1
    # скважины, у которых в окне подгонки есть >= 12 работающих шагов (для новых скважин
    # CRM без истории дебита неприменима); WCT в отложенной оценке in-sample
    seen = (LIQ[:hi_h] > 1).sum(0) >= 12
    ms = m & seen[None, :]
    mos = np.zeros_like(LIQ, bool); mos[hi_h:] = oil[hi_h:] > 1; mos &= seen[None, :]
    hold = {"fit_window": [data[5], hi_h], "eval_window": [hi_h, T],
            "all_wells": {"CRM": r2c(ch["p1"], LIQ, m), "Hybrid": r2c(ch["hyb"], LIQ, m),
                          "OPR": rh["OPR"]},
            "wells_with_history": {"n_wells": int((ms.any(0)).sum()),
                                   "n_points": int(ms.sum()), "n_points_all": int(m.sum()),
                                   "CRM": r2c(ch["p1"], LIQ, ms), "Hybrid": r2c(ch["hyb"], LIQ, ms),
                                   "OPR": r2c(ch["hyb"] * (1 - cache["wct"]), oil, mos)}}
    print("  отложенная выборка:", json.dumps(hold, ensure_ascii=False))

    scenarios = [("H10"), ("H7"), ("H8"), ("H2"), ("H1")]
    g = build_graph()
    results = []
    for h_edit in scenarios:
        sc = SharedCache(":memory:")
        keys = cache_keys_for(_Lat(g), {})
        for h in NODES:
            if h != h_edit:
                sc.save_result(h, keys[h], {"r2": 0.9})
        plan = build_optimal_plan({}, _Lat(g), sc)
        p_ne = set(plan.needs_execution)
        comps = {COMPONENT[h] for h in p_ne}
        cache2 = {k: cache[k] for k in ("p1", "hyb", "wct")}
        t_incr, _ = run_components(comps, data, cache2)
        results.append({"edit": h_edit, "pne_size": len(p_ne), "components": sorted(comps),
                        "incr_ms": t_incr * 1000, "speedup": round(t_full / t_incr, 1)})
        print(f"  правка {h_edit}: |P_ne|={len(p_ne)} t_incr={t_incr*1000:.0f} мс "
              f"ускорение {t_full/t_incr:.1f}x")
    out = {"field": field, "full_ms": t_full * 1000,
           "r2": {k: round(v, 3) for k, v in r2_full.items()}, "alpha": cache["alpha"],
           "protocol": {"fit_window": [data[5], data[6]], "eval_window": [0, T],
                        "r2": "pooled, SST from mean over masked points"},
           "holdout": hold, "per_component_ms": pcm, "scenarios": results}
    dest = os.path.join(HERE, f"timing_{field.lower()}_r2tuned.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("JSON:", dest)


if __name__ == "__main__":
    main()
