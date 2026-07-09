"""M7a. Bootstrap 95% CI для медиан ρ_real по семействам WfCommons."""
import io
import json
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

DATA_IN = DATA / "wfcommons_validation_results.json"
OUT_FILE = DATA / "bootstrap_ci_cascade.json"


def bootstrap_median_ci(xs, B=2000, level=0.95, seed=42):
    import random
    random.seed(seed)
    arr = list(xs)
    n = len(arr)
    meds = []
    for _ in range(B):
        sample = [arr[random.randint(0, n - 1)] for _ in range(n)]
        sample.sort()
        meds.append(sample[n // 2] if n % 2 else 0.5 * (sample[n // 2 - 1] + sample[n // 2]))
    meds.sort()
    alpha = (1 - level) / 2
    lo_idx = int(B * alpha)
    hi_idx = int(B * (1 - alpha))
    return float(meds[lo_idx]), float(meds[hi_idx])


def main():
    d = json.loads(DATA_IN.read_text(encoding="utf-8"))
    rows = d["per_workflow"]

    # Группировка под Table 2: nextflow, snakemake, pegasus (агрегат), all
    def group(fam: str) -> str:
        if fam.startswith("pegasus"):
            return "pegasus"
        return fam

    groups = sorted({group(r["family"]) for r in rows})
    out = {}
    for fam in groups + ["all"]:
        if fam == "all":
            xs = [r["rho_real"] for r in rows]
        else:
            xs = [r["rho_real"] for r in rows if group(r["family"]) == fam]
        lo, hi = bootstrap_median_ci(xs)
        out[fam] = {
            "n": len(xs),
            "median": float(np.median(xs)),
            "ci95_lo": lo,
            "ci95_hi": hi,
        }
        print(f"{fam:20s} N={len(xs):3d}: median={np.median(xs):.3f}, "
              f"CI95=[{lo:.3f}; {hi:.3f}]")
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
