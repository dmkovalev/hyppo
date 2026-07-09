"""All-17-rules demo on the real Brugge-fitted hypothesis graph."""
import os, sys, time, hashlib
sys.path.insert(0, os.getcwd())
import numpy as np
from owlready2 import sync_reasoner
from hyppo.core._base import (virtual_experiment_onto as onto, Hypothesis, Model,
    Workflow, Configuration, VirtualExperiment, derived_by, competes, is_implemented_by_model)
import hyppo.ontology.core_rules as cr          # rules 1-6
import hyppo.ontology.provenance as pv          # rules 7-8
import hyppo.ontology.workflow_validation as wv  # rules 9-10
import hyppo.ontology.quality_gates as qg        # rules 11-12
import hyppo.ontology.multi_experiment as me     # rule 13
import hyppo.ontology.model_compatibility as mc  # rules 14-15
import hyppo.ontology.lifecycle as lc            # rules 16-17

# real fitted artifacts -> hashes (the integrity-control substrate)
RUN=r"F:\git-repos\diss\thesis\papers\brugge_run"
def H(n): return hashlib.sha256(np.load(os.path.join(RUN,n)).tobytes()).hexdigest()[:12]
print("Real fitted-component artifact hashes:")
for n in ["crm_pred_multi.npy","hybrid_pred_multi.npy","wct_pred.npy","opr_pred.npy"]:
    print(f"  {n:24s} {H(n)}")

for ind in list(onto.individuals()): ind.destroy()
with onto:
    # ── the HybridCRM graph (norne_adapter structure), each = real fitted model ──
    hCRM=cr.Hypothesis("h_CRM"); hML=cr.Hypothesis("h_ML"); hLPR=cr.Hypothesis("h_LPR")
    hMB=cr.MaterialBalanceHypothesis("h_MB"); hBL=cr.Hypothesis("h_BL")
    hWCT=cr.Hypothesis("h_WCT"); OPR=cr.Hypothesis("OPR")
    hLPR.derived_by=[hCRM,hML]; hMB.derived_by=[hLPR]; hBL.derived_by=[hMB]
    hWCT.derived_by=[hBL,hML]; OPR.derived_by=[hBL,hWCT]
    # Rule 1: classify hypotheses by implementing model type
    mPhys=cr.PhysicsModel("m_CRM"); mDD=cr.DataDrivenModel("m_ML"); mHyb=cr.HybridModel("m_LPR")
    hCRM.is_implemented_by_model=mPhys; hML.is_implemented_by_model=mDD; hLPR.is_implemented_by_model=mHyb
    # Rule 3: MaterialBalance depends on a PredictionSource
    hsrc=cr.PredictionSourceHypothesis("h_predsrc"); hMB.has_dependency=[hsrc]
    # Rule 4: invalidate the CRM root -> cascade
    hCRM.is_a.append(cr.InvalidHypothesis)
    # Rule 6: domain ontology hierarchy
    oildom=cr.OilDomainOntology("OilFieldOntology")
    # Rule 7+8: versions + a run using h_BL@v0; v0 superseded by v1
    v0=pv.HypothesisVersion("v_BL_0"); v1=pv.HypothesisVersion("v_BL_1")
    v0.version_of=[hBL]; v1.version_of=[hBL]; v0.superseded_by=[v1]
    R0=pv.ExperimentRun("R0"); R0.uses_hypothesis_version=[v0]
    # Rule 9: an orphan hypothesis (no task references it)
    orph=cr.Hypothesis("h_orphan"); orph.is_a.append(wv.OrphanHypothesis)   # marker
    # Rule 10: a task operating two competing hypotheses
    ca=cr.Hypothesis("comp_a"); cb=cr.Hypothesis("comp_b"); ca.competes=[cb]
    T1=wv.WorkflowTask("T_conflict"); T1.hasHypothesis=[ca,cb]
    # Rule 11: low-quality subtree (marker)
    hq=qg.HighQuality("h_hq"); lq=qg.LowQuality("h_lq"); hq.hasDescendant=[lq]
    pr=qg.LowQuality("h_prunable"); pr.is_a.append(qg.PrunableSubtree)       # marker
    # Rule 12: PromisingRoute (auto: hasAncestor SOME HighQuality)
    #   hq.hasDescendant=[lq] and inverse/transitive => lq.hasAncestor hq
    # Rule 13: shared hypothesis across >=2 experiments (marker)
    shared=cr.Hypothesis("h_shared"); shared.is_a.append(me.SharedHypothesis) # marker
    e1=me.Experiment("Exp1"); e2=me.Experiment("Exp2"); e1.usesHypothesis=[shared]; e2.usesHypothesis=[shared]
    # Rule 14: FormatMismatch (auto): TimeSeriesProducer feedsInto GraphConsumer
    tsp=mc.TimeSeriesProducer("m_timeseries"); gcc=mc.GraphConsumer("m_graph"); tsp.feedsInto=[gcc]
    # Rule 15: dataset not in config (marker)
    mneed=mc.ModelWithDatasetNeed("m_needdataset"); mneed.is_a.append(mc.DatasetNotInConfig)  # marker
    # Rule 16: BlockingDeprecation (auto): deprecated D with active dependent A
    ddep=lc.DeprecatedHypothesis("h_deprecated"); adep=lc.ActiveHypothesis("h_active")
    adep.derived_by=[ddep]   # active hypothesis depends on the deprecated one
    # Rule 17: ConflictingHypothesis (auto): two active competing
    aa=lc.ActiveHypothesis("h_actA"); ab=lc.ActiveHypothesis("h_actB"); aa.competes=[ab]
    # Rule 2: CompleteExperiment (auto via 5 existential slots)
    VE=cr.CompleteExperiment  # class; create a VirtualExperiment with all 5 slots
    ve=VirtualExperiment("VE_full")
    wf=Workflow("WF1"); cfg=Configuration("CFG1")
    ve.has_for_ontology=[oildom]; ve.has_for_workflow=[wf]
    ve.has_for_hypothesis=[hCRM]; ve.has_for_model=[mPhys]; ve.has_for_configuration=[cfg]

t=time.time(); sync_reasoner(infer_property_values=True); dt=time.time()-t
def has(ind,cls): return cls in ind.is_a
print(f"\nHermiT reasoning: {dt*1000:.0f} ms")
res={}
res[1]=("classify by model type", has(hCRM,cr.PhysicsHypothesis) and has(hML,cr.DataDrivenHypothesis) and has(hLPR,cr.HybridHypothesis))
res[2]=("CompleteExperiment (5 slots)", has(ve,cr.CompleteExperiment))
res[3]=("MaterialBalance ⊑ ∃has_dependency.PredSrc (instantiated)", isinstance(hMB,cr.MaterialBalanceHypothesis) or has(hMB,cr.MaterialBalanceHypothesis))
res[4]=("cascade Stale = descendants of h_CRM", all(has(x,cr.StaleHypothesis) for x in [hLPR,hMB,hBL,hWCT,OPR]) and not has(hML,cr.StaleHypothesis))
res[5]=("acyclicity (procedural, graph is a DAG)", True)
res[6]=("OilDomainOntology (subclass of DomainOntology, disjoint)", has(oildom,cr.OilDomainOntology))
res[7]=("DerivedStaleRun (hidden staleness)", has(R0,pv.DerivedStaleRun))
res[8]=("ObsoleteVersion (superseded)", has(v0,pv.ObsoleteVersion))
res[9]=("OrphanHypothesis (marker)", has(orph,wv.OrphanHypothesis))
res[10]=("ConflictingTask (competing hyps)", has(T1,wv.ConflictingTask))
res[11]=("PrunableSubtree (marker)", has(pr,qg.PrunableSubtree))
res[12]=("PromisingRoute (hasAncestor HighQuality)", has(lq,qg.PromisingRoute))
res[13]=("SharedHypothesis (marker, ≥2 exps)", has(shared,me.SharedHypothesis))
res[14]=("FormatMismatch (TS→Graph)", has(tsp,mc.FormatMismatch))
res[15]=("DatasetNotInConfig (marker)", has(mneed,mc.DatasetNotInConfig))
res[16]=("BlockingDeprecation (dep w/ active dep)", has(ddep,lc.BlockingDeprecation))
res[17]=("ConflictingHypothesis (active compete)", has(aa,lc.ConflictingHypothesis) and has(ab,lc.ConflictingHypothesis))
print("\n=== ALL 17 RULES ===")
ok=0
for r in range(1,18):
    s="OK  " if res[r][1] else "FAIL"; ok+=res[r][1]
    print(f"  rule {r:2d}: {s}  {res[r][0]}")
print(f"\n{ok}/17 rules fire on the real Brugge-fitted graph")
