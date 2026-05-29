"""Rebuild the three asymptotic figures from the DM-core measurement
(asymptotic_results_dm.json) into thesis/images/:
  asymp_build_lattice.pdf   (Algorithm 1 exponent)
  asymp_add_hypothesis.pdf  (Algorithm 2 speedup)
  asymp_planning_cache.pdf  (Algorithm 4 cascade)
Short matplotlib script (no heavy loop / no owlready). Run on the .venv that has
matplotlib; retry if the flaky heap hits.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data" / "asymptotic_results_dm.json"
WORST = HERE.parent.parent / "data" / "worstcase_dm.json"
IMG = HERE.parents[4] / "thesis" / "images"


def main():
    d = json.loads(DATA.read_text())
    hs = sorted(int(k) for k in d["er_build"].keys())
    IMG.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: build_lattice asymptotic ---
    med = [d["er_build"][str(h)]["median_ms"] for h in hs]
    lo = [d["er_build"][str(h)]["p05_ms"] for h in hs]
    hi = [d["er_build"][str(h)]["p95_ms"] for h in hs]
    a = d["er_powerlaw"]["a"]
    a0 = d["er_powerlaw"]["a0"]
    ci = d["er_powerlaw"].get("ci95", [a, a])
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(hs, med, "o-", color="C0", lw=1.5, ms=5, label="median build time")
    ax.fill_between(hs, lo, hi, color="C0", alpha=0.15, label="5-95% range")
    hf = np.linspace(min(hs), max(hs), 100)
    # exponent reported with its bootstrap CI (R^2 on log-log does not discriminate
    # the exponent -- it stays ~1 for any integer power -- so it is not shown)
    ax.plot(hf, a0 * hf ** a, "--", color="C3", lw=1.2,
            label=rf"fit $T\propto|H|^{{{a:.2f}}}$ (95% CI [{ci[0]:.2f}, {ci[1]:.2f}])")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|H|$"); ax.set_ylabel(r"Build time $T$, ms")
    ax.set_title(r"Построение графа гипотез (DM-ядро, ER $p=0{,}3$)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(IMG / "asymp_build_lattice.pdf"); plt.close(fig)
    print(f"saved asymp_build_lattice.pdf (a={a:.3f})")

    # --- Figure 2: speedup ---
    if "speedup" in d:
        su = d["speedup"]
        full = [su[str(h)]["full_median_ms"] for h in hs]
        inc = [su[str(h)]["inc_median_ms"] for h in hs]
        sx = [su[str(h)]["speedup_x"] for h in hs]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))
        ax1.plot(hs, full, "o-", color="C0", lw=1.5, ms=5, label="полная перестройка (Алг.1)")
        ax1.plot(hs, inc, "s-", color="C1", lw=1.5, ms=5, label="инкрем. добавление (Алг.2)")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel(r"$|H|$"); ax1.set_ylabel("время, мс")
        ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)
        ax2.plot(hs, sx, "^-", color="C2", lw=1.5, ms=6)
        ax2.set_xscale("log"); ax2.set_xlabel(r"$|H|$"); ax2.set_ylabel(r"ускорение $k$")
        ax2.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(IMG / "asymp_add_hypothesis.pdf"); plt.close(fig)
        print(f"saved asymp_add_hypothesis.pdf (k@{hs[-1]}={sx[-1]:.0f}x)")

    # --- Figure 3: planning cascade ---
    if "planning" in d:
        rates = d["planning_rates"]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for idx, nh in enumerate(sorted(int(k) for k in d["planning"].keys())):
            ys = [d["planning"][str(nh)][str(r)] for r in rates]
            ax.plot(rates, ys, "o-", color=f"C{idx}", lw=1.3, ms=4, label=rf"$|H|={nh}$")
        ax.plot(rates, [1 - r for r in rates], "k--", alpha=0.4, label=r"идеал $1-r$")
        ax.set_xlabel(r"доля кэша $r$"); ax.set_ylabel(r"доля пересчёта $|P_{ne}|/|H|$")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(IMG / "asymp_planning_cache.pdf"); plt.close(fig)
        print("saved asymp_planning_cache.pdf")

    # --- Figure 4: complexity landscape O(|H|), O(|H|^2), O(|H|^4) ---
    if WORST.exists() and "speedup" in d:
        wc = json.loads(WORST.read_text())
        add_t = [d["speedup"][str(h)]["inc_median_ms"] for h in hs]   # Alg 2: O(|H|)
        wh = [int(x) for x in wc["grid"]]
        wt = list(wc["medians_ms"])                                   # Alg 1 worst: O(|H|^4)
        fig, ax = plt.subplots(figsize=(6.0, 4.2))

        def _guide(h0, t0, slope, lo, hgh, **kw):
            xx = np.linspace(lo, hgh, 50)
            ax.plot(xx, t0 * (xx / h0) ** slope, **kw)

        # measured series
        ax.plot(hs, add_t, "s-", color="C1", lw=1.4, ms=5,
                label=r"добавление гипотезы (Алг.2)")
        ax.plot(hs, med, "o-", color="C0", lw=1.4, ms=5,
                label=r"построение графа, средний случай (Алг.1)")
        ax.plot(wh, wt, "^-", color="C3", lw=1.4, ms=6,
                label=r"построение графа, худший случай (Алг.1)")
        # reference slopes anchored at each series' first point
        _guide(hs[0], add_t[0], 1, hs[0], hs[-1], ls=":", color="C1", alpha=0.6,
               label=r"$\propto|H|$")
        _guide(hs[0], med[0], 2, hs[0], hs[-1], ls=":", color="C0", alpha=0.6,
               label=r"$\propto|H|^2$")
        _guide(wh[0], wt[0], 4, wh[0], wh[-1], ls=":", color="C3", alpha=0.6,
               label=r"$\propto|H|^4$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$|H|$"); ax.set_ylabel("время, мс")
        ax.set_title(r"Ландшафт сложности: $O(|H|)$, $O(|H|^2)$, $O(|H|^4)$")
        ax.legend(fontsize=7.5, loc="upper left", ncol=2); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(IMG / "asymp_complexity.pdf"); plt.close(fig)
        print(f"saved asymp_complexity.pdf (worst a={wc['a']:.2f})")


if __name__ == "__main__":
    main()
