# Experiments

Research experiments backing the dissertation chapters and papers. Not
part of the published `gedanken`/`hyppo` package (not installed, not
covered by the packaging tests), but always driven through the real
`hyppo` library — no standalone reimplementations of the algorithms.

## Contents

- `chapter4/`: dissertation Chapter 4 experiments (BGM and HCP cascade
  studies, synthetic A/B and honest-baseline comparisons, sensitivity
  sweep); `figures/` and `results/` hold generated PNG/JSON artifacts.
- `iip2026/`: reproducibility package for the "Информатика и её
  применения" 2026 submission — see `iip2026/README.md` for the full
  script index (Algorithm 1/2/4 benchmarks, WfCommons/Gene Ontology/
  nf-core cascade validation). `iip2026/scripts/` also holds ad hoc
  table/figure generators (e.g. an untracked `aggregate_table_171.py`)
  not yet wired into the reproducibility index.
- `norne/`: standalone Norne field battery experiments
  (`norne_battery.py`, `full_experiment.py`) with cached JSON results.

## Data

Some scripts download or expect local datasets (WfCommons traces, Gene
Ontology `go-basic.obo`, nf-core DAGs, Brugge/Norne/Volve field data)
that are gitignored and not bundled with the repository.
