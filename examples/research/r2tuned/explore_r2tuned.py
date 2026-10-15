"""Исследование: R^2 компонент с уточнёнными CRM/Hybrid при разных протоколах."""
import os, sys, time, json
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
os.environ.setdefault("HYPPO_BRUGGE_RUN", r"F:\git-repos\diss\thesis\papers\brugge_run")
import numpy as np
from examples.brugge_reuse_timing import load_field, fit_wct, _run_dir
from examples.research.r2tuned.models_r2tuned import (r2c, fit_crm_masked, fit_hybrid_masked,
                                                      norne_open_dates)


def starts(field, LIQ, t):
    T, Np = LIQ.shape
    if field == "Norne":
        d = np.load(_run_dir() + "/norne_perwell.npz", allow_pickle=True)
        od = norne_open_dates(_run_dir() + "/norne_schedule.sch", T)
        return np.array([t[od.get(w, T - 1)] for w in d["producers"]])
    first = np.array([np.argmax(LIQ[:, j] > 1) if (LIQ[:, j] > 1).any() else T - 1 for j in range(Np)])
    return t[first]


def evaluate(field, s0, hi, ev_lo, ev_hi, wct_pred=None, tag=""):
    LIQ, WIN, t, oil, wat, _, _ = load_field(field)
    T, Np = LIQ.shape
    ts = starts(field, LIQ, t)
    t0 = time.perf_counter()
    _, p1 = fit_crm_masked(LIQ, WIN, t, s0, hi, ts)
    tc = time.perf_counter() - t0
    hyb, al = fit_hybrid_masked(p1, LIQ, WIN, s0, hi, Np)
    if wct_pred is None:
        wct_pred, _ = fit_wct(LIQ, wat, oil)
    wct = np.where(LIQ > 1, wat / np.maximum(LIQ, 1), 0.0)
    m = np.zeros_like(LIQ, bool); m[ev_lo:ev_hi] = LIQ[ev_lo:ev_hi] > 1
    mo = np.zeros_like(LIQ, bool); mo[ev_lo:ev_hi] = oil[ev_lo:ev_hi] > 1
    opr = hyb * (1 - wct_pred)
    res = {"CRM": r2c(p1, LIQ, m), "Hybrid": r2c(hyb, LIQ, m), "WCT": r2c(wct_pred, wct, m),
           "OPR": r2c(opr, oil, mo), "alpha": al, "crm_s": round(tc, 1)}
    print(f"{field} {tag} fit[{s0},{hi}) eval[{ev_lo},{ev_hi}):",
          {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
    return res, wct_pred


if __name__ == "__main__":
    out = {}
    # Brugge
    LIQ = load_field("Brugge")[0]; T = LIQ.shape[0]; hi = int(0.7 * T)
    r, w = evaluate("Brugge", 24, hi, 24, T, tag="table-protocol")
    out["Brugge_table"] = r
    out["Brugge_holdout"], _ = evaluate("Brugge", 24, hi, hi, T, w, tag="holdout")
    out["Brugge_from0"], _ = evaluate("Brugge", 0, hi, 0, T, w, tag="from0")
    # Norne
    LIQ = load_field("Norne")[0]; T = LIQ.shape[0]; hi = int(0.7 * T)
    r, w = evaluate("Norne", 0, T, 0, T, tag="table-protocol(in-sample)")
    out["Norne_table"] = r
    out["Norne_holdout"], _ = evaluate("Norne", 0, hi, hi, T, w, tag="holdout")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "explore_r2tuned.json"), "w"), indent=1)
