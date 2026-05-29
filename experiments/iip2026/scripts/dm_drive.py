"""Canonical driver for Algorithm 1's |H|-exponent, measured through the library
(HypothesisGraph.build -> DM core) with full process isolation: each replica is
one fresh `dm_bench_one.py` subprocess, so the machine's flaky native heap can
corrupt at most a single build, which is simply retried. Pools 30 reps per |H|
on the dissertation grid [10..500], fits a log-log power law with a bootstrap CI,
and writes data/asymptotic_results_dm.json (er_build + er_powerlaw).

Run from hyppo-ref:
  .venv/Scripts/python.exe experiments/iip2026/scripts/dm_drive.py
(any interpreter works; the driver allocates nothing heavy -- the workers do.)
"""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

PY = sys.executable
WORKER = str(Path(__file__).resolve().parent / "dm_bench_one.py")
GRID = [10, 25, 50, 100, 200, 300, 400, 500]
N_REPS = 30
MAX_RETRY = 12


def one_time(n_h, seed):
    """Run one isolated build; return time_ms or None if the worker crashed."""
    p = subprocess.run([PY, WORKER, str(n_h), str(seed)],
                       capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        return None
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])["time_ms"]
    except Exception:
        return None


def measure(n_h):
    times = []
    for rep in range(N_REPS):
        seed = 42 + rep
        for attempt in range(MAX_RETRY):
            t = one_time(n_h, seed)
            if t is not None:
                times.append(t)
                break
            print(f"  |H|={n_h} rep {rep} attempt {attempt} crashed; retry",
                  flush=True)
        else:
            raise SystemExit(f"|H|={n_h} rep {rep} failed after {MAX_RETRY} retries")
    return times


def pct(sv, q):
    pos = q * (len(sv) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sv[lo] if lo == hi else sv[lo] + (sv[hi] - sv[lo]) * (pos - lo)


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def main():
    er_build, medians = {}, []
    print(f"{'|H|':>5} {'median, ms':>14} {'p05':>12} {'p95':>12}", flush=True)
    for n_h in GRID:
        ts = sorted(measure(n_h))
        med, p05, p95 = pct(ts, 0.5), pct(ts, 0.05), pct(ts, 0.95)
        mean = sum(ts) / len(ts)
        std = math.sqrt(sum((t - mean) ** 2 for t in ts) / len(ts))
        er_build[str(n_h)] = {"median_ms": med, "mean_ms": mean, "std_ms": std,
                              "p05_ms": p05, "p95_ms": p95, "n_reps": len(ts)}
        medians.append(med)
        print(f"{n_h:>5} {med:>14.4f} {p05:>12.4f} {p95:>12.4f}", flush=True)

    lh = [math.log(h) for h in GRID]
    lt = [math.log(m) for m in medians]
    a = slope(lh, lt)
    mx, my = sum(lh) / len(lh), sum(lt) / len(lt)
    b = my - a * mx
    a0 = math.exp(b)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(lh, lt))
    ss_tot = sum((y - my) ** 2 for y in lt)
    r2 = 1 - ss_res / ss_tot

    rng = random.Random(42)
    n = len(lh)
    samp = sorted(slope([lh[k] for k in idx], [lt[k] for k in idx])
                  for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(2000)))
    lo, hi = samp[50], samp[1949]
    print(f"\nDM core [10..500]: a = {a:.3f}  95%-CI [{lo:.3f}, {hi:.3f}]  "
          f"a0={a0:.6f}  R2={r2:.4f}", flush=True)

    out = {"platform": "Windows 11", "python": "3.13", "algorithm": "DM (Kuhn+Tarjan)",
           "h_grid": GRID, "n_reps": N_REPS, "er_build": er_build,
           "er_powerlaw": {"a": a, "a0": a0, "R2": r2, "ci95": [lo, hi]}}
    dst = Path(__file__).resolve().parent.parent / "data" / "asymptotic_results_dm.json"
    # preserve speedup/planning if already present (added by dm_speedup_planning.py)
    if dst.exists():
        prev = json.loads(dst.read_text())
        for k in ("speedup", "planning", "planning_rates"):
            if k in prev:
                out[k] = prev[k]
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
