# Research

Research experiments backing the dissertation chapters and papers. Not
part of the published `gedanken`/`hyppo` package (not installed, not
covered by the packaging tests), but always driven through the real
`hyppo` library — no standalone reimplementations of the algorithms.

## Contents

- `chapter4/`: dissertation Chapter 4 experiments (BGM and HCP cascade
  studies, synthetic A/B and honest-baseline comparisons, sensitivity
  sweep); `figures/` and `results/` hold generated PNG/JSON artifacts.
  `bgm_experiment.py` / `hcp_experiment.py` run the two domain layers
  (`hyppo/adapters/bgm_adapter.py`, `hcp_adapter.py`: hypothesis structures,
  workflow, domain OWL ontology, numeric invariants) through the core:
  graph derived by Algorithm 1, consistency check, reasoner and the 19 rules
  on the life-cycle scenarios and injected defects; `reasoning_time.py`
  times the RL materialisation and HermiT on the three domain ontologies.
- `planning/`: reproducibility package for the dissertation's planning
  chapter — see `planning/README.md` for the full script index
  (Algorithm 1/2/4 benchmarks, WfCommons/Gene Ontology/nf-core cascade
  validation). `planning/scripts/` also holds ad hoc table/figure
  generators (e.g. an untracked `aggregate_table_171.py`) not yet wired
  into the reproducibility index.
- `norne/`: standalone Norne field battery experiments
  (`norne_battery.py`, `full_experiment.py`) with cached JSON results.

## Data

Some scripts download or expect local datasets (WfCommons traces, Gene
Ontology `go-basic.obo`, nf-core DAGs, Brugge/Norne/Volve field data)
that are gitignored and not bundled with the repository.
