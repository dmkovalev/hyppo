#!/usr/bin/env python
"""Chapter 4: Parametric sensitivity sweep over WCT spike magnitude.

Shows that the hypothesis-management mechanism works across a range of
regime-change severities, not just at one cherry-picked point.

Usage:
    uv run python experiments/chapter4/synthetic_sweep.py

Output:
    experiments/chapter4/results/synthetic_sweep.json
    experiments/chapter4/figures/sweep_mape_vs_spike.png
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.chapter4.synthetic_ab import (
    RANDOM_SEED,
    N_PRODUCERS,
    N_INJECTORS,
    N_MONTHS_HISTORY,
    N_MONTHS_VALIDATION,
    WCT_SPIKE_WELL,
    generate_synthetic_field,
    phase1_initial_training,
    phase3a_baseline,
    phase3b_treatment,
    compute_mape,
    compute_r2,
    forecast_with_model,
)

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SPIKE_MAGNITUDES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
N_SEEDS = 10


def run_single(spike_mag: float, seed: int) -> dict:
    """Run one A/B pair with given spike magnitude and seed."""
    rng = np.random.default_rng(seed)

    # Regenerate field with custom spike magnitude
    field = generate_synthetic_field(rng)

    # Override the spike magnitude
    spike_idx = field["producers"].index(WCT_SPIKE_WELL)
    regime_month = N_MONTHS_HISTORY + 3
    # Undo the default spike, apply custom
    field["wct"][spike_idx, regime_month:] -= 0.35  # remove default
    field["wct"][spike_idx, regime_month:] += spike_mag
    field["wct"] = np.clip(field["wct"], 0.05, 0.95)

    # Recompute OPR with updated WCT
    n_months = field["n_months"]
    opr = np.zeros((N_PRODUCERS, n_months))
    for p in range(N_PRODUCERS):
        for j in range(N_INJECTORS):
            tau_months = max(1, int(field["tau_matrix"][j, p] / 30))
            shifted = np.roll(field["injection"][j], tau_months)
            shifted[:tau_months] = field["injection"][j, 0]
            opr[p] += field["f_matrix"][j, p] * shifted
        opr[p] *= (1.0 - field["wct"][p])
    opr += rng.normal(0, 0.5, opr.shape)
    opr = np.clip(opr, 0.1, None)
    field["opr"] = opr

    model_v1 = phase1_initial_training(field)
    baseline = phase3a_baseline(field, model_v1)
    treatment = phase3b_treatment(field, model_v1)

    return {
        "spike_mag": spike_mag,
        "seed": seed,
        "mape_baseline": baseline["mape_opr"],
        "mape_treatment": treatment["mape_opr"],
        "r2_baseline": baseline["r2_opr"],
        "r2_treatment": treatment["r2_opr"],
        "delta_mape": round(baseline["mape_opr"] - treatment["mape_opr"], 2),
    }


def main():
    print("=" * 70)
    print("Chapter 4: Parametric sweep over WCT spike magnitude")
    print(f"  Magnitudes: {SPIKE_MAGNITUDES}")
    print(f"  Seeds per point: {N_SEEDS}")
    print("=" * 70)

    all_results = []
    summary_rows = []

    for mag in SPIKE_MAGNITUDES:
        runs = []
        for s in range(N_SEEDS):
            seed = RANDOM_SEED + s
            r = run_single(mag, seed)
            runs.append(r)
            all_results.append(r)

        mape_bl = [r["mape_baseline"] for r in runs]
        mape_tr = [r["mape_treatment"] for r in runs]
        delta = [r["delta_mape"] for r in runs]

        row = {
            "spike_magnitude": mag,
            "mape_baseline_mean": round(np.mean(mape_bl), 2),
            "mape_baseline_std": round(np.std(mape_bl), 2),
            "mape_treatment_mean": round(np.mean(mape_tr), 2),
            "mape_treatment_std": round(np.std(mape_tr), 2),
            "delta_mape_mean": round(np.mean(delta), 2),
            "delta_mape_std": round(np.std(delta), 2),
        }
        summary_rows.append(row)
        print(f"  spike={mag:.2f}: "
              f"MAPE_bl={row['mape_baseline_mean']:.1f}+-{row['mape_baseline_std']:.1f}, "
              f"MAPE_tr={row['mape_treatment_mean']:.1f}+-{row['mape_treatment_std']:.1f}, "
              f"delta={row['delta_mape_mean']:+.1f}+-{row['delta_mape_std']:.1f}")

    # Save
    output = {
        "experiment": "chapter4_synthetic_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_seeds": N_SEEDS,
        "spike_magnitudes": SPIKE_MAGNITUDES,
        "summary": summary_rows,
        "raw": all_results,
    }
    out_path = RESULTS_DIR / "synthetic_sweep.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n  Results: {out_path}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mags = [r["spike_magnitude"] for r in summary_rows]
        bl_mean = [r["mape_baseline_mean"] for r in summary_rows]
        bl_std = [r["mape_baseline_std"] for r in summary_rows]
        tr_mean = [r["mape_treatment_mean"] for r in summary_rows]
        tr_std = [r["mape_treatment_std"] for r in summary_rows]
        delta_mean = [r["delta_mape_mean"] for r in summary_rows]
        delta_std = [r["delta_mape_std"] for r in summary_rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.errorbar(mags, bl_mean, yerr=bl_std, fmt="r--s", label="Baseline (no hyppo)", capsize=4)
        ax1.errorbar(mags, tr_mean, yerr=tr_std, fmt="g-^", label="Treatment (with hyppo)", capsize=4)
        ax1.set_xlabel("WCT spike magnitude (pp)")
        ax1.set_ylabel("MAPE OPR, %")
        ax1.set_title("MAPE vs regime-change severity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(mags, delta_mean, width=0.035, yerr=delta_std, capsize=4, color="steelblue", alpha=0.8)
        ax2.set_xlabel("WCT spike magnitude (pp)")
        ax2.set_ylabel("Delta MAPE (pp), baseline - treatment")
        ax2.set_title("Improvement from hypothesis management")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig_path = FIGURES_DIR / "sweep_mape_vs_spike.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure: {fig_path}")
    except ImportError:
        print("  [skip] matplotlib not installed")

    print(f"\n{'=' * 70}")
    print("Sweep complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
