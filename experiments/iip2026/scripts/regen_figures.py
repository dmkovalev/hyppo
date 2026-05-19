"""Перерисовать asymp_build_lattice.pdf и asymp_add_hypothesis.pdf
из честных данных в thesis/papers/asymptotic_results.json."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # experiments/iip2026/
DATA = ROOT / "data" / "asymptotic_results.json"
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = json.loads(DATA.read_text())
    hs = sorted(int(k) for k in d["er_build"].keys())
    er_med = [d["er_build"][str(h)]["median_ms"] for h in hs]
    er_lo = [d["er_build"][str(h)]["p05_ms"] for h in hs]
    er_hi = [d["er_build"][str(h)]["p95_ms"] for h in hs]
    a = d["er_powerlaw"]["a"]
    a0 = d["er_powerlaw"]["a0"]
    r2 = d["er_powerlaw"]["R2"]

    # --- Figure 1: build_lattice asymptotic ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(hs, er_med, "o-", color="C0", lw=1.5, ms=5, label="median build time")
    ax.fill_between(hs, er_lo, er_hi, color="C0", alpha=0.15, label="5-95% range")
    h_fit = np.linspace(min(hs), max(hs), 100)
    t_fit = a0 * h_fit ** a
    ax.plot(h_fit, t_fit, "--", color="C3", lw=1.2,
            label=rf"fit $T = {a0:.4f} \cdot |H|^{{{a:.2f}}}$, $R^2={r2:.4f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|H|$ (number of hypotheses)")
    ax.set_ylabel(r"Build time $T$, ms")
    ax.set_title("Asymptotic build_lattice scaling (ER, p=0.3)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "asymp_build_lattice.pdf")
    print(f"Saved {OUT/'asymp_build_lattice.pdf'}")

    # --- Figure 2: speedup ---
    su = d["speedup"]
    full = [su[str(h)]["full_median_ms"] for h in hs]
    inc = [su[str(h)]["inc_median_ms"] for h in hs]
    sx = [su[str(h)]["speedup_x"] for h in hs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))
    ax1.plot(hs, full, "o-", color="C0", lw=1.5, ms=5, label="full rebuild")
    ax1.plot(hs, inc, "s-", color="C1", lw=1.5, ms=5, label="incremental add")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"$|H|$"); ax1.set_ylabel("time, ms")
    ax1.set_title("Full rebuild vs incremental add")
    ax1.legend(fontsize=9); ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(hs, sx, "^-", color="C2", lw=1.5, ms=6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$|H|$"); ax2.set_ylabel("Speedup, ×")
    ax2.set_title("Speedup of incremental add over rebuild")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "asymp_add_hypothesis.pdf")
    print(f"Saved {OUT/'asymp_add_hypothesis.pdf'}")


if __name__ == "__main__":
    main()
