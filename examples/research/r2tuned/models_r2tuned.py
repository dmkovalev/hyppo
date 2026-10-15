"""Уточнённые компоненты модели заводнения (вариант _r2tuned) к табл. tab:r2_fields.

Отличия от examples/brugge_reuse_timing.py (исходный конвейер не изменяется):

1. Метрика R^2 (r2c). В исходной r2a среднее в SST берётся по ВСЕМУ массиву,
   включая точки вне маски (остановленные / ещё не введённые скважины, нули),
   а суммы --- только по маске. Это не R^2: SST завышается на n*(mean_mask -
   mean_all)^2 и R^2 искусственно растёт (на Norne CRM: 0.671 по r2a против
   0.032 по корректной формуле). r2c: SST от среднего по тем же точкам маски.

2. CRM (fit_crm_masked). Те же уравнения CRM-P с постоянной времени для каждой
   пары (pywaterflood.crm.q_CRM_perpair, ограничения 'up-to one': 0<=f_ij<=1)
   и экспоненциальный спад первичной добычи, но:
   (а) функция потерь считается только по точкам, где скважина работает
       (дебит > 1): периоды остановки --- управляющее воздействие, а не отклик
       пласта, и трактуются как пропуски, а не как наблюдения нулевого дебита;
   (б) спад первичной добычи отсчитывается от даты ввода скважины t_j^0
       (у pywaterflood q_i = production[0] * gain: у скважин, введённых позже
       первого шага, первичный член тождественно равен нулю). Для Norne t_j^0
       --- первая дата со статусом OPEN в WCONHIST расписания (не зависит от
       замеренных дебитов); для Brugge расписания нет, t_j^0 --- первый шаг с
       ненулевым дебитом жидкости (дата ввода скважины).

3. Hybrid (fit_hybrid_masked). Та же гребневая регрессия на [1, q_CRM, средняя
   закачка], без лага фактической добычи; подгонка только по работающим точкам;
   alpha выбирается по отложенному хвосту окна подгонки (последние 30 %), а не
   по R^2 на тех же точках, затем модель переобучается на всём окне.

4. OPR оценивается на том же окне, что CRM/Hybrid ([s0, T)), а не на всём ряде
   с периодом запуска [0, s0), исключённым из подгонки CRM.

WCT (Бакли--Леверетт) не меняется (fit_wct исходного конвейера), меняется
только метрика.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import least_squares


def q_CRM_perpair(injection, time, gains, taus):
    """CRM-P для одной добывающей: та же рекурсия, что pywaterflood.crm.q_CRM_perpair
    (совпадение до 1e-11 проверяет check_equiv_r2tuned.py), векторизовано по
    нагнетательным; на этих размерах быстрее вызова Rust-ядра pywaterflood."""
    T = injection.shape[0]
    dt = np.diff(time, prepend=time[0])
    E = np.exp(-dt[:, None] / np.asarray(taus)[None, :])
    A = 1 - E
    out = np.zeros_like(injection)
    for k in range(1, T):
        out[k] = out[k - 1] * E[k] + A[k] * injection[k]
    return out @ gains


# Объёмный коэффициент газа для пересчёта закачки газа Norne в пластовые м3:
# B_g = (p_sc/T_sc) Z T / p = 0.00351 * 0.9 * 371 K / 270 бар ~ 0.0043 м3/ст.м3
# (пластовое давление ~270 бар, T ~98 C, Z ~0.9 --- оценка; PVT-таблиц в данных нет).
BG_NORNE = 0.0043


def norne_gas_injection(sch_path, n_steps):
    """Закачка газа Norne (WCONINJE 'GAS', ст.м3/сут) -> массив (T, Ng) и имена скважин."""
    raw = open(sch_path, encoding="latin-1").read()
    txt = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
    segs = re.split(r"\b(DATES|WCONINJE)\b", txt)
    t, G = -1, {}
    for i in range(1, len(segs) - 1, 2):
        kw, body = segs[i], segs[i + 1]
        if kw == "DATES":
            t += 1
            continue
        for rec in body.split("/"):
            q = re.findall(r"'([^']+)'", rec)
            if len(q) >= 2 and q[1] == "GAS" and "'RATE'" in rec:
                G.setdefault(q[0], {})[max(t, 0)] = float(rec.split("'RATE'")[1].split()[0])
    names = sorted(G)
    return np.array([[G[w].get(k, 0.0) for w in names] for k in range(n_steps)]), names


def r2c(Yh, Y, mask):
    """Корректный объединённый R^2: SSE и SST по одним и тем же точкам маски."""
    y = np.asarray(Y, float)[mask]
    yh = np.asarray(Yh, float)[mask]
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def norne_open_dates(sch_path, n_steps):
    """Индекс первого шага DATES, на котором скважина объявлена OPEN в WCONHIST."""
    raw = open(sch_path, encoding="latin-1").read()
    txt = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
    segs = re.split(r"\b(DATES|WCONHIST)\b", txt)
    t, first = -1, {}
    for i in range(1, len(segs) - 1, 2):
        kw, body = segs[i], segs[i + 1]
        if kw == "DATES":
            t += 1
        else:
            for rec in body.split("/"):
                q = re.findall(r"'([^']+)'", rec)
                if len(q) >= 2 and q[1] == "OPEN" and q[0] not in first:
                    first[q[0]] = max(t, 0)
    assert t + 1 == n_steps, (t + 1, n_steps)
    return first


def _model(x, I, t, t0, Ni):
    gains, taus, q0, taup = x[:Ni], x[Ni:2 * Ni], x[2 * Ni], x[2 * Ni + 1]
    q = q_CRM_perpair(I, t, gains, taus)
    on = t >= t0
    prim = np.zeros_like(t)
    prim[on] = q0 * np.exp(-(t[on] - t0) / taup)
    return q + prim


def fit_crm_masked(LIQ, WIN, t, s0, hi, t_start):
    """CRM-P (per-pair tau), подгонка по работающим точкам окна [s0, hi).

    t_start[j] --- время ввода скважины j (сут). Возвращает (params, p1)."""
    T, Np = LIQ.shape
    Ni = WIN.shape[1]
    I = np.ascontiguousarray(WIN, float)
    tt = np.ascontiguousarray(t, float)
    p1 = np.zeros((T, Np))
    params = []
    lo = np.r_[np.zeros(Ni), np.full(Ni, 1.0), 0.0, 1.0]
    up = np.r_[np.ones(Ni), np.full(Ni, 1e4), np.inf, 1e5]
    for j in range(Np):
        m = np.zeros(T, bool)
        m[s0:hi] = LIQ[s0:hi, j] > 1
        if m.sum() < 3:
            params.append(None)
            continue
        q = LIQ[:, j]
        x0 = np.r_[np.full(Ni, 0.1), np.full(Ni, 100.0), q[m][:5].mean(), 300.0]
        x0 = np.clip(x0, lo + 1e-6, up - 1e-6)
        r = least_squares(lambda x: (_model(x, I, tt, t_start[j], Ni) - q)[m], x0,
                          bounds=(lo, up), x_scale="jac", max_nfev=300)
        params.append(r.x)
        p1[:, j] = _model(r.x, I, tt, t_start[j], Ni)
    return params, p1


def _feat(p1, WIN, a, b, Np):
    n = (b - a) * Np
    return np.stack([np.ones(n), p1[a:b].reshape(-1, order="F"),
                     np.repeat(WIN[a:b].mean(1)[:, None], Np, axis=1).reshape(-1, order="F")], axis=1)


def _ridge(X, y, al):
    return np.linalg.solve(X.T @ X + al * np.eye(X.shape[1]), X.T @ y)


def fit_hybrid_masked(p1, LIQ, WIN, s0, hi, Np, alphas=(0.1, 1, 10, 100)):
    """Гребневая коррекция CRM; alpha --- по отложенному хвосту окна подгонки."""
    T = LIQ.shape[0]
    cut = s0 + int(0.7 * (hi - s0))
    Xa = _feat(p1, WIN, s0, cut, Np); ya = LIQ[s0:cut].reshape(-1, order="F")
    ma = (LIQ[s0:cut] > 1).reshape(-1, order="F")
    Xb = _feat(p1, WIN, cut, hi, Np); yb = LIQ[cut:hi].reshape(-1, order="F")
    mb = (LIQ[cut:hi] > 1).reshape(-1, order="F")
    best = None
    for al in alphas:
        w = _ridge(Xa[ma], ya[ma], al)
        sse = ((Xb[mb] @ w - yb[mb]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, al)
    X = _feat(p1, WIN, s0, hi, Np); y = LIQ[s0:hi].reshape(-1, order="F")
    mm = (LIQ[s0:hi] > 1).reshape(-1, order="F")
    w = _ridge(X[mm], y[mm], best[1])
    return (_feat(p1, WIN, 0, T, Np) @ w).reshape(T, Np, order="F"), best[1]
