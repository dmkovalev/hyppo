"""M3 + M2a одним прогоном на 157 WfCommons-экземплярах.
Subprocess isolation per workflow (Python 3.13 'cell' bug в долгих циклах).

Для каждого WfFormat JSON:
  - V = задачи (вершины графа гипотез)
  - E = parent/children рёбра (= derived_by)
  - t_eval(h) = runtimeInSeconds из execution.tasks
  - Cache ⊆ V, мощность r·|V|, случайная выборка

M3 (баseline speedup) — за r ∈ {0.3, 0.5, 0.7, 0.9}:
  T_no_cache = Σ_{h ∈ V} t_eval(h)
  T_with_cache = Σ_{h ∈ Pne} t_eval(h)
  speedup = T_no_cache / T_with_cache

M2a (overhead Algorithm 2) — wall-clock самого планирования
(time.perf_counter; адаптивный блок из ~10мс прогонов
для устойчивости к шуму планировщика ОС).
"""
from __future__ import annotations
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent   # examples/research/planning/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

CACHE_DIR = CACHE / "wfcommons"
OUT_FILE = DATA / "baseline_and_overhead.json"
WORKER = Path(__file__).parent / "_bench_one_baseline.py"


def main():
    R_GRID = [0.3, 0.5, 0.7, 0.9]
    N_REPS = 10

    jsons: list[tuple[str, Path]] = []
    for fam in ("nextflow", "snakemake", "pegasus"):
        d = CACHE_DIR / fam
        if not d.exists():
            continue
        for p in d.rglob("*.json"):
            jsons.append((fam, p))
    print(f"Total candidate JSONs: {len(jsons)}", flush=True)

    results: list[dict] = []
    skipped = 0
    for i, (family, path) in enumerate(jsons):
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(path), str(42 + i)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            print(f"  [{path.stem}] FAIL: {proc.stderr[:200]}", flush=True)
            skipped += 1
            continue
        try:
            d = json.loads(proc.stdout.strip())
        except Exception as e:
            print(f"  [{path.stem}] parse error: {e}", flush=True)
            skipped += 1
            continue
        if d.get("skipped"):
            skipped += 1
            continue
        d["family"] = family
        d["name"] = path.stem
        results.append(d)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(jsons)} (skipped {skipped})",
                  flush=True)

    print(f"\nProcessed {len(results)}, skipped {skipped}", flush=True)

    # === Сводка по r ===
    summary_by_r: dict[float, dict] = {}
    for r in R_GRID:
        sp = [w["per_r"][str(r)]["speedup_median"] for w in results
              if np.isfinite(w["per_r"][str(r)]["speedup_median"])]
        rho = [w["per_r"][str(r)]["rho_median"] for w in results]
        wc = [w["per_r"][str(r)]["wallclock_us_median"] for w in results]
        wc_max_per = [w["per_r"][str(r)]["wallclock_us_max"] for w in results]
        ns = [w["n"] for w in results]
        bi = [w["per_r"][str(r)]["block_iters"] for w in results]
        summary_by_r[r] = {
            "n_workflows": len(sp),
            "speedup_median": float(np.median(sp)),
            "speedup_p05": float(np.percentile(sp, 5)),
            "speedup_p95": float(np.percentile(sp, 95)),
            "speedup_max": float(np.max(sp)),
            "rho_median": float(np.median(rho)),
            "rho_p05": float(np.percentile(rho, 5)),
            "rho_p95": float(np.percentile(rho, 95)),
            "wallclock_us_median": float(np.median(wc)),
            "wallclock_us_p95": float(np.percentile(wc, 95)),
            "wallclock_us_max": float(np.max(wc_max_per)),
            "block_iters_median": float(np.median(bi)),
            "block_iters_min": int(np.min(bi)),
            "block_iters_max": int(np.max(bi)),
            "n_median": float(np.median(ns)),
            "n_max": int(np.max(ns)),
        }

    print("\n=== M3: speedup (no-cache vs Algorithm 2) ===")
    print(f"{'r':>6s} | {'med':>10s} | {'p95':>10s} | {'max':>10s}")
    for r in R_GRID:
        s = summary_by_r[r]
        print(f"{r:6.2f} | {s['speedup_median']:>9.2f}× | "
              f"{s['speedup_p95']:>9.2f}× | {s['speedup_max']:>9.2f}×")
    print("\n=== M2a: wall-clock Алгоритма 2 (мс, адаптивный блок) ===")
    print(f"{'r':>6s} | {'med мкс':>10s} | {'p95 мкс':>10s} | "
          f"{'max мкс':>10s} | {'block med':>10s} | {'|V| max':>8s}")
    for r in R_GRID:
        s = summary_by_r[r]
        print(f"{r:6.2f} | {s['wallclock_us_median']:>9.1f} | "
              f"{s['wallclock_us_p95']:>9.1f} | {s['wallclock_us_max']:>9.1f} | "
              f"{s['block_iters_median']:>9.0f} | {s['n_max']:>8d}")

    out = {
        "R_GRID": R_GRID,
        "N_REPS": N_REPS,
        "n_workflows": len(results),
        "skipped": skipped,
        "measurement": "adaptive block (~10ms target), wallclock = block_time / block_iters",
        "by_r": {str(k): v for k, v in summary_by_r.items()},
        "per_workflow": results,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
