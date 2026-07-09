"""Layer 2 (markers) — PROCEDURAL run on a real Norne ABox."""
import os,sys; sys.path.insert(0,r"F:\git-repos\diss\hyppo-ref")
from hyppo.core._base import (virtual_experiment_onto as onto, Hypothesis, Model,
    Workflow, Configuration, VirtualExperiment, is_implemented_by_model)
import hyppo.ontology.core_rules as cr
import hyppo.ontology.provenance as pv
import hyppo.ontology.workflow_validation as wv
import hyppo.ontology.quality_gates as qg
import hyppo.ontology.multi_experiment as me
import hyppo.ontology.model_compatibility as mc
import hyppo.ontology.lifecycle as lc
from hyppo.ontology.markers import apply_markers

for ind in list(onto.individuals()): ind.destroy()

# === Build a Norne-like ABox with scenarios for each marker rule ===
PROD = [f"P{j}" for j in range(1,11)]  # 10 producers
INJ  = ["C1H","C2H","C3H"]             # 3 injectors

with onto:
    # Hypotheses: producers + injectors
    h_prod = {p: Hypothesis(f"norne_{p}") for p in PROD}
    h_inj  = {i: Hypothesis(f"norne_{i}") for i in INJ}
    # derived_by: each producer depends on injectors (simplified: top-3)
    for j,p in enumerate(PROD):
        h_prod[p].derived_by = [h_inj[INJ[j%3]]]
    h_forecast = Hypothesis("norne_forecast")
    h_forecast.derived_by = [h_prod[PROD[0]], h_prod[PROD[1]]]

    # Models
    m_phys = cr.PhysicsModel("norne_m_phys")
    m_ml   = cr.DataDrivenModel("norne_m_ml")
    m_hyb  = cr.HybridModel("norne_m_hyb")
    h_prod[PROD[0]].is_implemented_by_model = m_phys
    h_prod[PROD[1]].is_implemented_by_model = m_ml

    # --- Rule 2 scenario (CompleteExperiment): VE with all 5 slots ---
    ve_full = VirtualExperiment("norne_VE_full")
    oildom  = cr.OilDomainOntology("NorneOntology")
    wf      = Workflow("norne_WF")
    cfg     = Configuration("norne_CFG")
    ve_full.has_for_ontology      = [oildom]
    ve_full.has_for_workflow      = [wf]
    ve_full.has_for_hypothesis    = [h_prod[PROD[0]]]
    ve_full.has_for_model         = [m_phys]
    ve_full.has_for_configuration = [cfg]
    # incomplete VE (missing config) — should NOT be marked
    ve_incomplete = VirtualExperiment("norne_VE_incomplete")
    ve_incomplete.has_for_ontology = [oildom]
    ve_incomplete.has_for_workflow = [wf]
    ve_incomplete.has_for_hypothesis = [h_prod[PROD[1]]]
    ve_incomplete.has_for_model = [m_ml]
    # (no has_for_configuration)

    # --- Rule 9 scenario (OrphanHypothesis) ---
    # Tasks reference SOME producers; others are orphans
    task1 = wv.WorkflowTask("norne_task1")
    task1.hasHypothesis = [h_prod[PROD[0]], h_prod[PROD[1]], h_prod[PROD[2]]]
    task2 = wv.WorkflowTask("norne_task2")
    task2.hasHypothesis = [h_prod[PROD[3]], h_prod[PROD[4]]]
    # P6..P10 NOT in any task → orphans (procedurally detected)

    # --- Rule 11 scenario (PrunableSubtree) ---
    # LowQuality root with all-LowQuality descendants
    hq = qg.HighQuality("norne_hq")
    lq_root = qg.LowQuality("norne_lq_root")
    lq_child1 = qg.LowQuality("norne_lq_child1")
    lq_child2 = qg.LowQuality("norne_lq_child2")
    lq_root.hasDescendant = [lq_child1, lq_child2]
    # lq_root: all descendants LowQuality → PrunableSubtree

    # --- Rule 13 scenario (SharedHypothesis) ---
    h_shared = Hypothesis("norne_shared")
    exp1 = me.Experiment("norne_exp1")
    exp2 = me.Experiment("norne_exp2")
    exp1.usesHypothesis = [h_shared]
    exp2.usesHypothesis = [h_shared]

    # --- Rule 15 scenario (DatasetNotInConfig) ---
    m_need = mc.ModelWithDatasetNeed("norne_m_need")
    cfg2 = mc.ModelConfig("norne_cfg2")
    m_need.usedInConfig = [cfg2]
    # cfg2 has NO hasAvailableDataset → DatasetNotInConfig
    # Also: a model WITH a dataset (should NOT be marked)
    m_ok = mc.ModelWithDatasetNeed("norne_m_ok")
    cfg3 = mc.ModelConfig("norne_cfg3")
    ds = mc.Dataset("norne_ds1")
    cfg3.hasAvailableDataset = [ds]
    m_ok.usedInConfig = [cfg3]

# === Run Layer 2 procedurally ===
print("Running apply_markers() — Layer 2 (CWA checks, procedural)...")
report = apply_markers(onto, run_hermit=True)

print(f"\n=== MarkerReport ===")
print(f"Rule 2 (CompleteExperiment):     {len(report.rule2_marked)} marked")
for iri in report.rule2_marked:
    print(f"  {iri}")
print(f"Rule 9 (OrphanHypothesis):       {len(report.rule9_marked)} marked")
for iri in report.rule9_marked:
    print(f"  {iri.split('#')[-1]}")
print(f"Rule 11 (PrunableSubtree):       {len(report.rule11_marked)} marked")
for iri in report.rule11_marked:
    print(f"  {iri.split('#')[-1]}")
print(f"Rule 13 (SharedHypothesis):      {len(report.rule13_marked)} marked")
for iri in report.rule13_marked:
    print(f"  {iri.split('#')[-1]}")
print(f"Rule 15 (DatasetNotInConfig):    {len(report.rule15_marked)} marked")
for iri in report.rule15_marked:
    print(f"  {iri.split('#')[-1]}")
print(f"Rolled back (inconsistency):     {len(report.rolled_back)}")
print(f"HermiT skipped:                  {report.hermit_skipped}")

# Verify: incomplete VE NOT marked, model-with-dataset NOT marked
ve_inc_ok = cr.CompleteExperiment not in ve_incomplete.is_a
m_ok_clean = mc.DatasetNotInConfig not in m_ok.is_a
print(f"\nNegative checks:")
print(f"  Incomplete VE NOT CompleteExperiment: {ve_inc_ok}")
print(f"  Model WITH dataset NOT DatasetNotInConfig: {m_ok_clean}")
print(f"\n>>> Layer 2 (markers) run PROCEDURALLY on Norne ABox — all 5 rules executed.")
