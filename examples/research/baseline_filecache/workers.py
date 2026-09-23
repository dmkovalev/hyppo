# examples/research/baseline_filecache/workers.py
"""Воркеры конвейера Brugge. Сущности хранилища читаются в рантайме:
fit_liq — gains, tau, well_status; fit_wct — corey_ref; fit_opr — corey_ref, well_status."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from pywaterflood import CRM

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.research.baseline_filecache import metastore  # noqa: E402

CONSTRAINTS = {"uto": "up-to one", "pos": "positive"}


def _run_dir() -> Path:
    env = os.environ.get("HYPPO_BRUGGE_RUN")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "thesis" / "papers" / "brugge_run"


def load_data():
    d = np.load(_run_dir() / "brugge_perwell.npz", allow_pickle=True)
    LIQ = d["production"].astype(float)
    WIN = d["injection"].astype(float)
    t = d["time"].astype(float)
    dw = np.load(_run_dir() / "brugge_oilwater.npz", allow_pickle=True)
    return LIQ, WIN, t, dw["oil"].astype(float), dw["water"].astype(float), 24, int(LIQ.shape[0] * 0.7)


def digest(arr) -> str:
    return hashlib.sha256(np.round(np.asarray(arr, float), 9).tobytes()).hexdigest()


def fw_curve(no, nw, Swc, Sor, muo=1.0, muw=0.3):
    Sw = np.linspace(Swc, 1 - Sor, 400)
    krw = ((Sw - Swc) / (1 - Swc - Sor)) ** nw
    kro = ((1 - Sw - Sor) / (1 - Swc - Sor)) ** no
    return Sw, 1 / (1 + kro * muw / np.maximum(krw, 1e-12) / muo)


def cmd_seed(a):
    LIQ, *_ = load_data()
    metastore.write(a.meta, "corey_ref", "*", np.array([0.25, 0.25]))
    metastore.write(a.meta, "well_status", "*", np.ones(LIQ.shape[1]))


def cmd_crm_fit(a):
    LIQ, WIN, t, _, _, s0, hi = load_data()
    c = CRM(primary=True, tau_selection="per-pair", constraints=CONSTRAINTS[a.constraints])
    c.fit(production=LIQ[s0:hi], injection=WIN[s0:hi], time=t[s0:hi])
    p1 = np.asarray(c.predict(injection=WIN, time=t)).reshape(LIQ.shape)
    Path(a.p1).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.p1, p1)
    metastore.write(a.meta, "gains", a.constraints, np.array(c.gains, float))
    metastore.write(a.meta, "tau", a.constraints, np.array(c.tau, float))


def cmd_fit_liq(a):
    LIQ, WIN, t, _, _, s0, hi = load_data()
    p1 = np.load(a.p1)
    _, gains = metastore.read(a.meta, "gains", a.constraints)
    _, tau = metastore.read(a.meta, "tau", a.constraints)
    _, status = metastore.read(a.meta, "well_status", "*")
    T, Np = LIQ.shape
    gw = WIN @ gains.T
    tau_p = np.maximum(tau.reshape(Np, -1).mean(axis=1), 1e-6)
    dt = float(np.median(np.diff(t)))
    lam = 1.0 - np.exp(-dt / tau_p)
    filt = np.zeros_like(gw)
    filt[0] = gw[0]
    for k in range(1, T):
        filt[k] = filt[k - 1] + lam * (gw[k] - filt[k - 1])

    def feat(lo, hi_):
        n = (hi_ - lo) * Np
        return np.stack([np.ones(n), p1[lo:hi_].reshape(-1, order="F"),
                         np.repeat(WIN[lo:hi_].mean(1)[:, None], Np, axis=1).reshape(-1, order="F"),
                         filt[lo:hi_].reshape(-1, order="F")], axis=1)

    Xtr, ytr = feat(s0, hi), LIQ[s0:hi].reshape(-1, order="F")
    pen = float(a.alpha) * float(a.ridge)
    w = np.linalg.solve(Xtr.T @ Xtr + pen * np.eye(4), Xtr.T @ ytr)
    hyb = (feat(0, T) @ w).reshape(T, Np, order="F") * status[None, :]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, hyb)


def cmd_fit_wct(a):
    LIQ, _, _, _, wat, _, _ = load_data()
    _, cr = metastore.read(a.meta, "corey_ref", "*")
    Swc, Sor = float(cr[0]), float(cr[1])
    Sw, f = fw_curve(float(a.no), float(a.nw), Swc, Sor)
    wct = np.where(LIQ > 1, wat / np.maximum(LIQ, 1), 0.0)
    cum = np.cumsum(LIQ, axis=0)
    pred = np.zeros_like(wct)
    for j in range(LIQ.shape[1]):
        drv, mx = cum[:, j], cum[:, j].max()
        m = LIQ[:, j] > 1
        if mx <= 0 or m.sum() < 5 or wct[m, j].std() < 1e-6:
            continue
        y, best = wct[m, j], None
        for aa in np.linspace(0.3, 3, 25):
            for sh in np.linspace(Swc + 0.25, 1 - Sor, 5):
                pr = np.interp(np.clip(Swc + aa * (drv / mx) * (sh - Swc), Swc, sh), Sw, f)
                r2 = 1 - ((y - pr[m]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                if best is None or r2 > best[0]:
                    best = (r2, aa, sh)
        pred[:, j] = np.interp(np.clip(Swc + best[1] * (drv / mx) * (best[2] - Swc), Swc, best[2]), Sw, f)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, pred)


def cmd_fit_opr(a):
    _, _, _, oil, _, _, _ = load_data()
    _, cr = metastore.read(a.meta, "corey_ref", "*")
    _, status = metastore.read(a.meta, "well_status", "*")
    mob = (1 - cr[0] - cr[1]) / 0.5           # поправка на подвижную нефть к эталону Swc=Sor=0.25
    opr = np.load(a.hyb) * (1 - np.load(a.wct)) * mob * status[None, :]
    m = oil > 1
    r2 = float(1 - ((oil - opr)[m] ** 2).sum() / ((oil - oil.mean())[m] ** 2).sum())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"r2": r2, "digest": digest(opr)}, fh)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seed"); p.add_argument("--meta", required=True); p.set_defaults(fn=cmd_seed)
    p = sub.add_parser("crm_fit")
    for k in ("--constraints", "--p1", "--meta"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_crm_fit)
    p = sub.add_parser("fit_liq")
    for k in ("--constraints", "--alpha", "--ridge", "--p1", "--meta", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_fit_liq)
    p = sub.add_parser("fit_wct")
    for k in ("--no", "--nw", "--meta", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_fit_wct)
    p = sub.add_parser("fit_opr")
    for k in ("--hyb", "--wct", "--meta", "--out"):
        p.add_argument(k, required=True)
    p.set_defaults(fn=cmd_fit_opr)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
