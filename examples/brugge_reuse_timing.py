#!/usr/bin/env python3
"""Полностью работающий пример: реальный ускорение полного пересчёта vs
инкрементального повторного использования на реальных данных Brugge.

Исполняет РЕАЛЬНЫЕ компоненты модели (pywaterflood CRM, гребневая Hybrid,
Бакли--Леверетт WCT, OPR) на РЕАЛЬНЫх данных Brugge, через РЕАЛЬНЫЙ планировщик
Hyppo (build_optimal_plan) и кэш SharedCache. Для серии правок одной гипотезы
измеряет wall-clock: полный пересчёт (все 4 компонента) vs инкрементальный
(только P_ne --- множество, вычисленное планировщиком по теореме 1).

Запуск:
    cd hyppo-ref
    uv run --extra data python examples/brugge_reuse_timing.py
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.getcwd())

from pywaterflood import CRM
import networkx as nx
# hyppo импортируется лениво внутри main(), чтобы модуль (и его golden-тест)
# не подтягивал owlready2/JVM на этапе импорта.

RUN = "/mnt/f/git-repos/diss/thesis/papers/brugge_run"


def _run_dir():
    """Переносимый поиск каталога с данными Brugge/Norne (локально и в CI)."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.environ.get("HYPPO_BRUGGE_RUN"),
        RUN,
        os.path.join(os.getcwd(), "thesis/papers/brugge_run"),
        os.path.join(os.getcwd(), "../thesis/papers/brugge_run"),
        os.path.join(here, "..", "..", "thesis/papers/brugge_run"),  # из examples/
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "Каталог brugge_run не найден. Задайте HYPPO_BRUGGE_RUN или разместите данные "
        "в thesis/papers/brugge_run."
    )

# ───────────────────────── данные месторождения ─────────────────────────
def load_field(name):
    """Brugge: окно [24:0.7T]; Norne: подгонка на всех данных (s0=0)."""
    if name == "Brugge":
        d = np.load(_run_dir() + "/brugge_perwell.npz", allow_pickle=True)
        LIQ = d["production"].astype(float)
        WIN = d["injection"].astype(float)
        t = d["time"].astype(float)
        dw = np.load(_run_dir() + "/brugge_oilwater.npz", allow_pickle=True)
        oil = dw["oil"].astype(float); wat = dw["water"].astype(float)
        s0, fit_all = 24, False
    else:  # Norne
        d = np.load(_run_dir() + "/norne_perwell.npz", allow_pickle=True)
        LIQ = d["liq"].astype(float)
        WIN = d["inj"].astype(float)
        t = d["time"].astype(float)
        oil = d["oil"].astype(float); wat = d["water"].astype(float)
        s0, fit_all = 0, True
    T = LIQ.shape[0]
    fit_hi = T if fit_all else int(T * 0.7)
    return LIQ, WIN, t, oil, wat, s0, fit_hi

def r2a(Yh, Y, mask):
    Yh = np.asarray(Yh, float); Y = np.asarray(Y, float)
    return float(1 - ((Y - Yh)[mask] ** 2).sum() / ((Y - Y.mean())[mask] ** 2).sum())

# ───────────────────── модельные компоненты (РЕАЛЬНЫЕ) ─────────────────────
def fw_curve(no, nw, Swc=0.25, Sor=0.25, muo=1.0, muw=0.3):
    Sw = np.linspace(Swc, 1 - Sor, 400)
    krw = ((Sw - Swc) / (1 - Swc - Sor)) ** nw
    kro = ((1 - Sw - Sor) / (1 - Swc - Sor)) ** no
    fw = 1 / (1 + kro * muw / np.maximum(krw, 1e-12) / muo)
    return Sw, fw

def fit_crm(LIQ, WIN, t, s0, fit_hi):
    c = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
    c.fit(production=LIQ[s0:fit_hi], injection=WIN[s0:fit_hi], time=t[s0:fit_hi])
    p1 = np.asarray(c.predict(injection=WIN, time=t)).reshape(LIQ.shape)
    return c, p1

def fit_hybrid(p1, LIQ, WIN, s0, hi, Np):
    # Признаки: [1, CRM-прогноз p1, средняя закачка]. Без лага фактической
    # добычи q_{t-1}: он даёт R²≈1.0 (тривиальный авторегрессионный прогноз),
    # не характеризующий качество модели.
    def feat(a, b):
        n = (b - a) * Np
        return np.stack([np.ones(n), p1[a:b].reshape(-1, order="F"),
                         np.repeat(WIN[a:b].mean(1)[:, None], Np, axis=1).reshape(-1, order="F")], axis=1)
    Xtr = feat(s0, hi); ytr = LIQ[s0:hi].reshape(-1, order="F"); bw = None
    for al in [0.1, 1, 10, 100]:
        w = np.linalg.solve(Xtr.T @ Xtr + al * np.eye(3), Xtr.T @ ytr)
        r2 = r2a((Xtr @ w).reshape(hi - s0, Np, order="F"), LIQ[s0:hi], (LIQ[s0:hi] > 1))
        if bw is None or r2 > bw[0]:
            bw = (r2, al, w)
    T = LIQ.shape[0]
    hyb = (feat(0, T) @ bw[2]).reshape(T, Np, order="F")
    return hyb, bw[0]

def fit_wct(LIQ, wat, oil):
    Np = LIQ.shape[1]
    wct = np.where(LIQ > 1, wat / np.maximum(LIQ, 1), 0.0)
    cum = np.cumsum(LIQ, axis=0); best = None
    for no in [1.5, 2, 2.5, 3]:
        for nw in [1.5, 2, 2.5, 3]:
            Sw, f = fw_curve(no, nw); pred = np.zeros_like(wct)
            for j in range(Np):
                drv = cum[:, j]; mx = drv.max()
                if mx <= 0: continue
                m = LIQ[:, j] > 1; y = wct[m, j]
                if m.sum() < 5 or y.std() < 1e-6: continue
                bp = None
                for a in np.linspace(0.3, 3, 25):
                    for sh in [.5, .55, .6, .65, .7]:
                        Swe = np.clip(.25 + a * (drv / mx) * (sh - .25), .25, sh)
                        pr = np.interp(Swe, Sw, f)
                        r2 = 1 - ((y - pr[m]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                        if bp is None or r2 > bp[0]: bp = (r2, a, sh)
                if bp:
                    pred[:, j] = np.interp(np.clip(.25 + bp[1] * (drv / mx) * (bp[2] - .25), .25, bp[2]), Sw, f)
            r2 = r2a(pred, wct, LIQ > 1)
            if best is None or r2 > best[0]: best = (r2, no, nw, pred)
    return best[3], best[0]

def fit_opr(hyb, wct_pred, oil):
    opr = hyb * (1 - wct_pred)
    return opr, r2a(opr, oil, oil > 1)

# ───────────────── граф 16 гипотез (рис. lattice_crm) ─────────────────
NODES = [f"H{i}" for i in range(1, 17)]
# ребро (a, b): b выведена из a (a -> b); H8->H9 --- межветвевое
EDGES = [
    ("H1", "H2"), ("H1", "H3"), ("H2", "H4"), ("H3", "H4"), ("H4", "H5"),
    ("H6", "H5"), ("H7", "H8"), ("H5", "H8"),          # ветвь жидкости
    ("H8", "H9"),                                       # межветвевое: материальный баланс использует l
    ("H9", "H10"), ("H10", "H11"), ("H11", "H12"), ("H12", "H13"), ("H13", "H14"),
    ("H15", "H5"),                                      # ГТМ модулирует продуктивность
    ("H8", "H16"), ("H14", "H16"),                      # OPR = l * (1 - wct)
]
# гипотеза -> компонента модели (что реально исполняется)
COMPONENT = {
    "H1": "CRM", "H2": "CRM", "H3": "CRM", "H4": "CRM", "H5": "CRM", "H6": "CRM",
    "H7": "ML", "H8": "FUS",
    "H9": "MB", "H10": "COREY", "H11": "COREY", "H12": "BL", "H13": "BL", "H14": "WCTF",
    "H15": "GTM", "H16": "OPR",
}
COMPONENT_DESC = {"CRM": "подгонка CRM (pywaterflood)", "ML": "ML-коррекция (гребневая)",
                  "FUS": "слияние LPR", "MB": "материальный баланс", "COREY": "Кори-проницаемости",
                  "BL": "Бакли--Леверетт WCT", "WCTF": "слияние WCT", "GTM": "модуляция ГТМ",
                  "OPR": "прогноз нефти"}

class _Lat:
    def __init__(self, g): self.lattice = g; self.hypotheses = list(g.nodes())

def build_graph():
    g = nx.DiGraph(); g.add_nodes_from(NODES); g.add_edges_from(EDGES); return g

# ───────────────────── timed execution of a component set ─────────────────────
def run_components(comps, data, cache):
    """Реально исполняет компоненты из множества comps, переиспользуя кэш."""
    LIQ, WIN, t, oil, wat, s0, hi = data
    T, Np = LIQ.shape
    mval = (LIQ > 1).copy(); mval[:s0] = False
    t_total = 0.0
    r2 = {}
    if "CRM" in comps:
        ts = time.perf_counter()
        c, p1 = fit_crm(LIQ, WIN, t, s0, hi)
        cache["p1"] = p1
        t_total += time.perf_counter() - ts
        r2["CRM"] = r2a(p1, LIQ, mval)
    p1 = cache["p1"]
    if "ML" in comps or "FUS" in comps:
        ts = time.perf_counter()
        hyb, r2h = fit_hybrid(p1, LIQ, WIN, s0, hi, Np)
        cache["hyb"] = hyb
        t_total += time.perf_counter() - ts
        r2["Hybrid"] = r2h
    hyb = cache.get("hyb")
    if {"MB", "COREY", "BL", "WCTF"} & comps:
        ts = time.perf_counter()
        wct_pred, r2w = fit_wct(LIQ, wat, oil)
        cache["wct"] = wct_pred
        t_total += time.perf_counter() - ts
        r2["WCT"] = r2w
    wct_pred = cache.get("wct")
    if "OPR" in comps and hyb is not None and wct_pred is not None:
        ts = time.perf_counter()
        _, r2o = fit_opr(hyb, wct_pred, oil)
        t_total += time.perf_counter() - ts
        r2["OPR"] = r2o
    return t_total, r2

def main():
    field = sys.argv[1] if len(sys.argv) > 1 else "Brugge"
    from hyppo.planner._base import build_optimal_plan
    from hyppo.metadata_repository import SharedCache
    print("=" * 72)
    print(f"РЕАЛЬНЫЙ БЕНЧМАРК: полный пересчёт vs инкрементальное переиспользование ({field})")
    print("=" * 72)
    data = load_field(field)
    LIQ = data[0]
    print(f"{field}: {LIQ.shape[1]} продюсеров, {LIQ.shape[0]} месяцев\n")

    # 1) базовый прогон: исполнить ВСЕ компоненты, заполнить кэш
    all_comps = {"CRM", "ML", "FUS", "MB", "COREY", "BL", "WCTF", "OPR"}
    cache = {}
    t_full, r2_full = run_components(all_comps, data, cache)
    print("Базовый прогон (все компоненты) --- R²:")
    for k, v in r2_full.items():
        print(f"    {k:8s} R² = {v:.3f}")
    print(f"    ВРЕМЯ полного пересчёта: {t_full*1000:.0f} мс\n")

    # 2) типология правок и соответствующее множество компонентов
    #    P_ne вычисляется РЕАЛЬНЫМ планировщиком; отображение гипотеза->компонента
    #    даёт множество компонентов к пересчёту.
    scenarios = [
        ("правка WCT (показатели Кори, H10)", "H10"),    # дешёвая: поздняя правка
        ("правка ML-коррекции (H7)",          "H7"),
        ("правка слияния жидкости (H8)",      "H8"),     # межветвевая: тянет WCT
        ("правка CRM-канала (H2)",            "H2"),     # дорогая: ранняя правка
        ("правка агрегации закачки (H1)",     "H1"),     # самая дорогая: корень
    ]

    g = build_graph()
    print(f"граф гипотез: {g.number_of_nodes()} узлов, {g.number_of_edges()} рёбер\n")
    print(f"{'сценарий':42s} {'|P_ne|':>6s} {'компоненты':>10s} {'t_инкр, мс':>10s} {'ускорение':>9s}")
    print("-" * 82)

    results = []
    for label, h_edit in scenarios:
        # РЕАЛЬНЫЙ планировщик: кэшируем все 16 гипотез, КРОМЕ h_edit
        # (его результат инвалидирован правкой). build_optimal_plan строит P_ne
        # каскадом по достижимости (алгоритм 4, теорема 1).
        sc = SharedCache(":memory:")
        for h in NODES:
            if h == h_edit:
                continue  # результат этой гипотезы устарел --- нет в кэше
            sc.save_result(h, {}, {"r2": 0.9})
        plan = build_optimal_plan({}, _Lat(g), sc)
        p_ne = set(plan.needs_execution)
        # отображение гипотез P_ne в исполняемые компоненты модели
        comps = {COMPONENT[h] for h in p_ne}
        cache2 = {"p1": cache["p1"], "hyb": cache.get("hyb"), "wct": cache.get("wct")}
        t_incr, _ = run_components(comps, data, cache2)
        speedup = t_full / t_incr if t_incr > 0 else float("inf")
        results.append((label, h_edit, len(p_ne), sorted(comps), t_incr, speedup))
        print(f"{label:42s} {len(p_ne):>6d} {len(comps):>10d} {t_incr*1000:>10.1f} {speedup:>8.1f}×")

    print("-" * 82)
    print(f"\nВЫВОД: полный пересчёт = {t_full*1000:.0f} мс; инкрементальный --- от "
          f"{min(r[4] for r in results)*1000:.1f} до {max(r[4] for r in results)*1000:.1f} мс "
          f"(ускорение {min(r[5] for r in results):.1f}×--{max(r[5] for r in results):.1f}×).")
    print("Измерение использовало реальные данные Brugge, реальную подгонку CRM (pywaterflood)")
    print("и реальное множество P_ne (достижимость в графе гипотез --- алгоритм 4, теорема 1).")

    out = {"field": field, "full_ms": t_full * 1000,
           "r2": {k: round(v, 3) for k, v in r2_full.items()},
           "scenarios": [{"edit": r[1], "pne_size": r[2], "components": r[3],
                          "incr_ms": r[4] * 1000, "speedup": round(r[5], 1)} for r in results]}
    with open("/tmp/brugge_reuse_timing.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nJSON сохранён: /tmp/brugge_reuse_timing.json")

if __name__ == "__main__":
    main()
