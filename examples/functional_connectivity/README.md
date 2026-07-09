# Functional Connectivity Example

Legacy example applying hypothesis-driven virtual experiments to brain
functional-connectivity analysis on Human Connectome Project (HCP) data.
It predates the current `hyppo.coa`/`hyppo.lattice_constructor` platform
API and is kept for reference, not exercised by CI.

## Contents

- `_hcp_brain.py`: OWL ontology (owlready2) for the HCP brain domain
  (subjects, ROIs, atlases, functional connectivity).
- `extract_ROI.py`: extracts region-of-interest time series from HCP
  NIfTI images (`nilearn`) and writes per-subject connectivity matrices.
- `get_functions.py`: PySpark UDF for distributed symbolic-regression
  feature extraction.
- `compare_models_legacy.py`: compares symbolic-regression / statistical
  models against a legacy DCI (causal discovery) baseline.
- `main.py`: legacy pipeline entry point wiring `ve_manager`/`ve_runner`
  modules (not part of the current `hyppo` package) to a config file.
- `job_template.yml`: cluster job template for running the extraction
  pipeline.

## Requirements

Not covered by the project's standard `uv sync` extras: `nilearn`,
`nibabel`, `pyspark`, `gplearn`, `statsmodels`, `dci`, `graphviz`, and
access to HCP imaging data (not distributed with this repo).
