"""Worst case of Lemma 1 (s_max, v_max = Theta(|H|), full DAG) on the same grid
as the typical-case experiment: |H| in {10,25,50,100,200,300,400,500}, 30 runs
per point (seed 42+i), bootstrap CI (B=2000) for the log-log slope of medians.
Only ``build`` is timed. Output: data/worstcase_dm_full.json

  <hyppo-ref>/.venv/Scripts/python.exe examples/research/planning/scripts/dm_worstcase_full.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dm_worstcase import make_graph, slope  # noqa: E402

GRID = [10, 25, 50, 100, 200, 300, 400, 500]
N_REPS = 30
B = 2000


def fit(grid, meds):
    lh = [math.log(h) for h in grid]
    lt = [math.log(m) for m in meds]
    a = slope(lh, lt)
    mx, my = sum(lh) / len(lh), sum(lt) / len(lt)
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(lh, lt))
    ss_tot = sum((y - my) ** 2 for y in lt)
    return a, 1 - ss_res / ss_tot


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def main():
    times = {}
    for n_h in GRID:
        ts = []
        for rep in range(N_REPS):
            g = make_graph(n_h, 1.0, random.Random(42 + rep))   # setup, untimed
            t0 = time.perf_counter()
            g.build()
            ts.append((time.perf_counter() - t0) * 1000.0)
        times[n_h] = ts
        print(f"|H|={n_h:>4}  median={median(ts):.3f} ms", flush=True)

    meds = [median(times[h]) for h in GRID]
    a, r2 = fit(GRID, meds)
    local = [math.log(meds[i + 1] / meds[i]) / math.log(GRID[i + 1] / GRID[i])
             for i in range(len(GRID) - 1)]

    rng = random.Random(0)
    boots = []
    for _ in range(B):
        bm = [median([rng.choice(times[h]) for _ in range(N_REPS)]) for h in GRID]
        boots.append(fit(GRID, bm)[0])
    boots.sort()
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]
    # slope on the three largest sizes (closest to the asymptotic regime)
    a_tail = slope([math.log(h) for h in GRID[-3:]], [math.log(m) for m in meds[-3:]])

    print(f"\nworst case: a={a:.3f}  R2={r2:.4f}  95% bootstrap CI [{lo:.3f}; {hi:.3f}]")
    print("local slopes:", [round(x, 2) for x in local])
    print(f"tail slope (300,400,500): {a_tail:.3f}")

    out = {"grid": GRID, "n_reps": N_REPS, "B": B, "alpha": 1.0,
           "medians_ms": meds, "a": a, "R2": r2, "ci95": [lo, hi],
           "local_slopes": local, "tail_slope": a_tail,
           "all_ms": {str(h): times[h] for h in GRID}}
    dst = Path(__file__).resolve().parent.parent / "data" / "worstcase_dm_full.json"
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
