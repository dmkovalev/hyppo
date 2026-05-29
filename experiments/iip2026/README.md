# Experiments — Информатика и её применения (ИиП) 2026

Reproducibility package for **"Планирование виртуальных экспериментов
с повторным использованием вычисленных фрагментов"** by D. Yu. Kovalev,
submitted to the journal *Informatika i Ee Primeneniya* (FRC IU RAS).

## Single source of truth

The three algorithms of the paper live in the **library**, not in these
scripts — every measurement calls them, so the reported numbers are
reproducible from `hyppo` itself:

| Paper | Library entry point |
|---|---|
| Algorithm 1 — `build_lattice` | `hyppo.coa.HypothesisGraph.build()` |
| Algorithm 2 — `add_hypothesis` | `hyppo.coa.HypothesisGraph.add_hypothesis()` |
| Algorithm 4 — `plan` (cascade) | `hyppo.coa.HypothesisGraph.plan()` |

`HypothesisGraph` delegates all causal work to `hyppo.coa.causal`, the pure
Dulmage–Mendelsohn core (Kuhn matching + Tarjan SCC + BFS closure), giving the
polynomial complexity of Lemma 1 — `O(|H|² · s_max · v_max)`. The earlier
standalone *greedy* reimplementation (which produced the stale `a ≈ 2.19`) has
been removed; the scripts below are thin measurement harnesses over the library.

## Contents

| Path | Purpose |
|---|---|
| `scripts/dm_drive.py` | **Section 5.1** — Algorithm 1 build-time exponent on the grid `[10..500]`, driven by `HypothesisGraph.build`. Spawns one isolated `dm_bench_one.py` subprocess **per replica** (process isolation against this machine's flaky native heap), pools 30 reps/size, fits a log-log power law with a bootstrap CI. Writes `data/asymptotic_results_dm.json` (`er_build` + `er_powerlaw`). |
| `scripts/dm_bench_one.py` | Subprocess worker: warms up, then times **one** `HypothesisGraph.build` for a given `|H|`/seed. Prints one JSON line. |
| `scripts/dm_speedup_planning.py` | **Section 5.2 + 5.4** — Algorithm 2 speedup (incremental `add_hypothesis` is `O(|H|)`; full-rebuild baseline taken from `er_build`) and Algorithm 4 planning cascade (`|P_ne|/|H|` vs cache rate `r`, via `HypothesisGraph.plan`). Merges `speedup` + `planning` into `asymptotic_results_dm.json`. Light allocation → stable in one process. |
| `scripts/dm_worstcase.py` | Worst-case regime of Lemma 1: structures **grow** with `|H|` (`k = α·|H|`), variable-disjoint, full DAG → every pair-union complete → `O(|H|⁴)`. Measures the build exponent (expect `a ≈ 4`). Writes `data/worstcase_dm.json`. |
| `scripts/dm_figure.py` | Regenerates `asymp_build_lattice.pdf`, `asymp_add_hypothesis.pdf`, `asymp_planning_cache.pdf` (into `../../thesis/images/`) from `asymptotic_results_dm.json`. Needs matplotlib. |
| `scripts/wfcommons_validation.py` | **Section 5.3** — cascade-effect experiment on 157 real workflow execution traces from [WfCommons](https://wfcommons.org) (Nextflow nf-core, Snakemake RASflow, Pegasus). Writes `data/wfcommons_validation_results.json`. |
| `scripts/algorithm1_bench.py` | OWL 2 DL consistency-check timing (HermiT Stage A + Stage B C3/C4/C5) — separate "Algorithm 1" of the ontology section; needs Java 11+. |
| `data/asymptotic_results_dm.json` | DM measurements: build power law `a ≈ 2.11` (`R² ≈ 0.9999`), Algorithm 2 speedup, Algorithm 4 planning. |
| `data/worstcase_dm.json` | Worst-case build exponent (growing structures). |
| `data/wfcommons_validation_results.json` | Per-workflow cascade ρ at `r=0.7` for all 157 instances + per-family medians. |

## Reproduction

Requirements: Python 3.11+ (stdlib only for the DM scripts — `numpy`/`matplotlib`
are needed **only** by `dm_figure.py` and `wfcommons_validation.py`). The DM
measurement scripts deliberately avoid numpy: on the reference Windows host the
native heap is flaky under sustained heavy allocation, so `dm_drive.py` isolates
each replica in its own process and retries any that crash.

```bash
cd experiments/iip2026

# Section 5.1 — Algorithm 1 build exponent (process-isolated, ~2-4 min)
python scripts/dm_drive.py

# Section 5.2 + 5.4 — Algorithm 2 speedup + Algorithm 4 planning (light, fast)
python scripts/dm_speedup_planning.py

# Worst case O(|H|^4) (growing structures)
python scripts/dm_worstcase.py

# Section 5.3 — cascade on real workflows (~5-10 min, downloads traces)
python scripts/wfcommons_validation.py

# Regenerate figures from JSON (needs matplotlib)
python scripts/dm_figure.py
```

## Hardware notes

Absolute millisecond timings depend on the host machine; the **power-law
exponent** `a` and **speedup ratios** are far more stable. On the reference
Windows 11 / x86-64 host expect:

- ER `build_lattice` (Algorithm 1): `T ≈ a₀ · |H|^2.11` (`R² ≈ 0.9999`),
  consistent with Lemma 1's `O(|H|²)` at fixed structure size.
- Worst case (structures growing as `k = |H|`): `a ≈ 3.7–4.0` (finite-size
  approach to the `O(|H|⁴)` bound of Lemma 1).
- Algorithm 2 speedup at `|H|=500`: `≈ 100–120×` (grows ~linearly with `|H|`,
  since rebuild is `O(|H|²)` and incremental add is `O(|H|)`).

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
