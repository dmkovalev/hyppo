# Experiments — Информатика и её применения (ИиП) 2026

Reproducibility package for **"Планирование виртуальных экспериментов
с повторным использованием вычисленных фрагментов"** by D. Yu. Kovalev,
submitted to the journal *Informatika i Ee Primeneniya* (FRC IU RAS).

## Contents

| Path | Purpose |
|---|---|
| `scripts/asymptotic_validation.py` | Original Section 5.1/5.2 implementation (build_lattice, add_hypothesis, plan). Reference for the algorithms in the paper. |
| `scripts/_bench_one.py` | Worker process for benchmarking a single replica (subprocess-isolated to avoid Python 3.13 state-leakage on long loops). |
| `scripts/rerun_5_1_5_2.py` | Honest re-run driver for Sections 5.1 (asymptotic `build_lattice`) and 5.2 (incremental-add speedup). Produces `data/asymptotic_results.json`. |
| `scripts/regen_figures.py` | Regenerates `asymp_build_lattice.pdf` and `asymp_add_hypothesis.pdf` from `data/asymptotic_results.json`. |
| `scripts/wfcommons_validation.py` | Section 5.3 cascade-effect experiment on 157 real workflow execution traces from [WfCommons](https://wfcommons.org) (Nextflow nf-core, Snakemake RASflow, Pegasus). Produces `data/wfcommons_validation_results.json` and `out/wfcommons_vs_synthetic_cascade.pdf`. |
| `data/asymptotic_results.json` | Honest measurements for Table 1 (speedup) and the power-law fit `a = 2.19, R² = 0.9999` for ER and `a_BA = 1.32, R² = 0.9998` for BA. |
| `data/wfcommons_validation_results.json` | Per-workflow cascade ρ at r=0.7 for all 157 instances + per-family medians. |

## Reproduction

Requirements: Python 3.11+ (3.13 tested), `numpy`, `matplotlib`. No
external dependencies beyond those. Internet access required for
`wfcommons_validation.py` (downloads ~7 MB of workflow traces from GitHub).

```bash
cd experiments/iip2026

# Sections 5.1 + 5.2 — asymptotic build + speedup (~3-5 min)
python scripts/rerun_5_1_5_2.py

# Section 5.3 — cascade on real workflows (~5-10 min, downloads traces)
python scripts/wfcommons_validation.py

# Regenerate figures from JSON (instant)
python scripts/regen_figures.py
```

All output goes into `data/` (JSON) and `out/` (PDF figures), both
relative to `experiments/iip2026/`. The repository ships with the JSON
already populated (the values used in the paper); rerunning will
overwrite with measurements from the current machine.

## Hardware notes

The honest measurements in `data/asymptotic_results.json` were taken on
a Windows 11 / x86-64 workstation, Python 3.13.3, NumPy 2.4.4. Absolute
millisecond timings in Table 1 of the paper depend on the host machine;
the **power-law exponent** `a` and **speedup ratios** are far more
stable across hardware. Expect:

- ER `build_lattice`: T ≈ a₀ · |H|^2.19 (R² ≈ 0.9999)
- BA `build_lattice`: T ≈ a₀ · |H|^1.32 (R² ≈ 0.9998)
- Speedup at |H|=500: ~300× (range 250–400× across machines)

## Citing

```bibtex
@article{Kovalev2026Planning,
  author  = {Kovalev, D. Yu.},
  title   = {Planning virtual experiments with reuse of computed fragments},
  journal = {Informatics and Applications},
  year    = {2026},
  note    = {submitted}
}
```

Backing reference for the cascade-effect experiment:

```bibtex
@article{Coleman2022WfCommons,
  author  = {Coleman, T. and Casanova, H. and Pottier, L. and others},
  title   = {WfCommons: A framework for enabling scientific workflow
             research and development},
  journal = {Future Generation Computer Systems},
  volume  = {128},
  pages   = {16--27},
  year    = {2022},
  doi     = {10.1016/j.future.2021.09.043}
}
```
