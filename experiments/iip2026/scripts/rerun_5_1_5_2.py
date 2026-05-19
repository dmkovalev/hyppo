"""
Честный ре-run экспериментов 5.1 (асимптотика) и 5.2 (speedup) с
subprocess-изоляцией каждой точки |H| (обход регрессии Python 3.13
с замыканиями state-leakage между долгими повторами в одном процессе).
"""
from __future__ import annotations
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WORKER = Path(__file__).parent / "_bench_one.py"


def power_law_fit(xs, ys):
    lx = np.log(np.array(xs, dtype=float))
    ly = np.log(np.array(ys, dtype=float))
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(np.exp(b)), r2


def bootstrap_ci(xs, ys, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    n = len(lx)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, _ = np.polyfit(lx[idx], ly[idx], 1)
        samples.append(a)
    samples.sort()
    return float(samples[int(0.025 * n_boot)]), float(samples[int(0.975 * n_boot)])


def run_worker(op, n, n_reps, seed_base, kind="er"):
    """Один subprocess на одну реплику — полная изоляция от Python 3.13 state leakage."""
    if op == "build":
        all_times = []
        for i in range(n_reps):
            r = subprocess.run(
                [sys.executable, str(WORKER), op, str(n), "1", str(seed_base + i), kind],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                print(f"    WARN: replica {i} failed: {r.stderr[:200]}", flush=True)
                continue
            all_times.extend(json.loads(r.stdout.strip())["times_s"])
        return {"times_s": all_times}
    else:  # speedup
        full_all = []
        inc_all = []
        for i in range(n_reps):
            r = subprocess.run(
                [sys.executable, str(WORKER), op, str(n), "1", str(seed_base + i), kind],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                print(f"    WARN: replica {i} failed", flush=True)
                continue
            d = json.loads(r.stdout.strip())
            full_all.extend(d["full_s"])
            inc_all.extend(d["inc_s"])
        return {"full_s": full_all, "inc_s": inc_all}


def stat(times_s):
    arr = np.array(times_s) * 1000  # ms
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p05_ms": float(np.percentile(arr, 5)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_reps": len(arr),
    }


def main():
    h_values = [10, 25, 50, 100, 200, 300, 400, 500]
    n_reps = 30
    print(f"Platform: {platform.system()} {platform.release()}", flush=True)
    print(f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}", flush=True)

    out = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h_grid": h_values,
        "n_reps": n_reps,
    }

    # --- 5.1 ER build ---
    print("\n=== 5.1 ER build_lattice ===", flush=True)
    er_b = {}
    for h in h_values:
        t0 = time.time()
        data = run_worker("build", h, n_reps, 42, "er")
        er_b[h] = stat(data["times_s"])
        print(f"  |H|={h:4d}: median={er_b[h]['median_ms']:.2f} ms, "
              f"CI=[{er_b[h]['p05_ms']:.2f}, {er_b[h]['p95_ms']:.2f}] "
              f"({time.time()-t0:.1f}s)", flush=True)
    out["er_build"] = er_b

    a_er, a0_er, r2_er = power_law_fit(h_values, [er_b[h]["median_ms"] for h in h_values])
    lo, hi = bootstrap_ci(h_values, [er_b[h]["median_ms"] for h in h_values])
    out["er_powerlaw"] = {"a": a_er, "a0": a0_er, "R2": r2_er, "ci_lo": lo, "ci_hi": hi}
    print(f"\n  Power-law: T = {a0_er:.4f} * |H|^{a_er:.4f}, R²={r2_er:.4f}, CI=[{lo:.3f}, {hi:.3f}]")

    # --- 5.1b BA build ---
    print("\n=== 5.1b BA build_lattice ===", flush=True)
    ba_b = {}
    for h in h_values:
        t0 = time.time()
        data = run_worker("build", h, n_reps, 42, "ba")
        ba_b[h] = stat(data["times_s"])
        print(f"  |H|={h:4d}: median={ba_b[h]['median_ms']:.2f} ms ({time.time()-t0:.1f}s)",
              flush=True)
    out["ba_build"] = ba_b

    a_ba, a0_ba, r2_ba = power_law_fit(h_values, [ba_b[h]["median_ms"] for h in h_values])
    lo, hi = bootstrap_ci(h_values, [ba_b[h]["median_ms"] for h in h_values])
    out["ba_powerlaw"] = {"a": a_ba, "a0": a0_ba, "R2": r2_ba, "ci_lo": lo, "ci_hi": hi}
    print(f"\n  Power-law BA: T = {a0_ba:.4f} * |H|^{a_ba:.4f}, R²={r2_ba:.4f}, CI=[{lo:.3f}, {hi:.3f}]")

    # --- 5.2 Speedup ---
    print("\n=== 5.2 Speedup ===", flush=True)
    su = {}
    for h in h_values:
        t0 = time.time()
        data = run_worker("speedup", h, n_reps, 42)
        full = np.array(data["full_s"]) * 1000
        inc = np.array(data["inc_s"]) * 1000
        su[h] = {
            "full_median_ms": float(np.median(full)),
            "full_p05_ms": float(np.percentile(full, 5)),
            "full_p95_ms": float(np.percentile(full, 95)),
            "inc_median_ms": float(np.median(inc)),
            "inc_p05_ms": float(np.percentile(inc, 5)),
            "inc_p95_ms": float(np.percentile(inc, 95)),
            "speedup_x": float(np.median(full)) / max(float(np.median(inc)), 1e-9),
        }
        print(f"  |H|={h:4d}: full={su[h]['full_median_ms']:8.2f} ms, "
              f"inc={su[h]['inc_median_ms']:.3f} ms, "
              f"speedup={su[h]['speedup_x']:7.1f}x ({time.time()-t0:.1f}s)",
              flush=True)
    out["speedup"] = su

    out_path = Path(__file__).resolve().parent.parent / "data" / "asymptotic_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
