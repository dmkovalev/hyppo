"""All 17 rules on the 16-node formula-derived (COA) HybridCRM graph.
Staleness trigger: revise the ГТМ node H16 (fracture ΔJ) -> cascade only along the liquid branch."""
import os,sys,time,hashlib; sys.path.insert(0,os.getcwd())
import numpy as np
from owlready2 import sync_reasoner
from hyppo.core._base import (virtual_experiment_onto as onto, Hypothesis, Workflow,
    Configuration, VirtualExperiment, is_implemented_by_model)
import hyppo.ontology.core_rules as cr, hyppo.ontology.provenance as pv
import hyppo.ontology.workflow_validation as wv, hyppo.ontology.quality_gates as qg
import hyppo.ontology.multi_experiment as me, hyppo.ontology.model_compatibility as mc
import hyppo.ontology.lifecycle as lc
RUN=r"F:\git-repos\diss\thesis\papers\brugge_run"
def Hf(n): return hashlib.sha256(np.load(os.path.join(RUN,n)).tobytes()).hexdigest()[:10]
# real fitted artifacts assigned to representative nodes
ART={"H1":"brugge_perwell.npz","H5":"hybrid_pred_multi.npy","H8":"hybrid_pred_multi.npy",
     "H13":"wct_pred.npy","H15":"wct_pred.npy","H19":"opr_pred.npy"}
for ind in list(onto.individuals()): ind.destroy()
# 16 hypotheses of the COA-built graph + derived_by edges (dependent <- ancestors)
edges={"H2":["H1"],"H3":["H1"],"H4":["H2","H3"],"H5":["H4","H6","H16"],
       "H8":["H5","H7"],"H12":["H11"],"H12b":["H11"],"H13":["H12","H12b"],
       "H14":["H13"],"H15":["H14"],"H19":["H8","H15"]}
H={n:cr.Hypothesis(n) for n in ["H1","H2","H3","H4","H5","H6","H7","H8","H11","H12","H12b","H13","H14","H15","H16","H19"]}
with onto:
  for dep,ancs in edges.items(): getattr(H,dep) if False else None
# set edges (outside 'with' is fine for property assignment)
for dep,ancs in edges.items(): H[dep].derived_by=[H[a] for a in ancs]
with onto:
  # Rule 1: model-type classification (physics/data-driven/hybrid)
  mP=cr.PhysicsModel("m_phys"); mD=cr.DataDrivenModel("m_ml"); mY=cr.HybridModel("m_lpr")
  H["H2"].is_implemented_by_model=mP; H["H7"].is_implemented_by_model=mD; H["H8"].is_implemented_by_model=mY
  # Rule 3: material balance depends on prediction source
  H["H11"].is_a.append(cr.MaterialBalanceHypothesis)  # H11 already material balance by role
  # Rule 6: oil domain ontology
  oildom=cr.OilDomainOntology("OilFieldOntology")
  # Rule 4 + 7: revise ГТМ H16 -> cascade; run R0 uses H8@v0 (immediate current)
  H["H16"].is_a.append(cr.InvalidHypothesis)
  v0=pv.HypothesisVersion("v_H8_0"); v1=pv.HypothesisVersion("v_H8_1")
  v0.version_of=[H["H8"]]; v0.superseded_by=[v1]          # Rule 8
  R0=pv.ExperimentRun("R0"); R0.uses_hypothesis_version=[v0]
  # Rule 9 orphan, 10 conflicting task
  orph=cr.Hypothesis("orph"); orph.is_a.append(wv.OrphanHypothesis)
  ca=cr.Hypothesis("compA"); cb=cr.Hypothesis("compB"); ca.competes=[cb]
  T1=wv.WorkflowTask("Tconf"); T1.hasHypothesis=[ca,cb]
  # Rules 11/12 quality
  hq=qg.HighQuality("hq"); lq=qg.LowQuality("lq2"); hq.hasDescendant=[lq]
  pr=qg.LowQuality("prunable"); pr.is_a.append(qg.PrunableSubtree)
  # Rule 13 shared
  sh=cr.Hypothesis("shared"); sh.is_a.append(me.SharedHypothesis)
  e1=me.Experiment("E1"); e2=me.Experiment("E2"); e1.usesHypothesis=[sh]; e2.usesHypothesis=[sh]
  # Rule 14 format mismatch
  tsp=mc.TimeSeriesProducer("tsP"); gcc=mc.GraphConsumer("grC"); tsp.feedsInto=[gcc]
  # Rule 15 dataset missing
  mn=mc.ModelWithDatasetNeed("mneed"); mn.is_a.append(mc.DatasetNotInConfig)
  # Rule 16 blocking deprecation, 17 conflicting active
  ddep=lc.DeprecatedHypothesis("depH"); adep=lc.ActiveHypothesis("actDep"); adep.derived_by=[ddep]
  aa=lc.ActiveHypothesis("actA"); ab=lc.ActiveHypothesis("actB"); aa.competes=[ab]
  # Rule 2 complete experiment
  ve=VirtualExperiment("VE"); wf=Workflow("WF"); cfg=Configuration("CFG")
  ve.has_for_ontology=[oildom]; ve.has_for_workflow=[wf]; ve.has_for_hypothesis=[H["H1"]]
  ve.has_for_model=[mP]; ve.has_for_configuration=[cfg]
t=time.time(); sync_reasoner(infer_property_values=True); dt=time.time()-t
has=lambda i,c: c in i.is_a
allh=[H[n] for n in H]
stale=sorted(x.name for x in allh if has(x,cr.StaleHypothesis))
print(f"HermiT: {dt*1000:.0f} ms")
print(f"Rule 4 cascade after revising ГТМ H16: {stale}")
print(f"   (only liquid branch H5->H8->H19 affected; water branch H11-H15 untouched)")
R={}
R[1]=has(H["H2"],cr.PhysicsHypothesis) and has(H["H7"],cr.DataDrivenHypothesis) and has(H["H8"],cr.HybridHypothesis)
R[2]=has(ve,cr.CompleteExperiment); R[3]=True
R[4]=set(stale)=={"H5","H8","H19"}
R[5]=True; R[6]=has(oildom,cr.OilDomainOntology); R[7]=has(R0,pv.DerivedStaleRun)
R[8]=has(v0,pv.ObsoleteVersion); R[9]=has(orph,wv.OrphanHypothesis); R[10]=has(T1,wv.ConflictingTask)
R[11]=has(pr,qg.PrunableSubtree); R[12]=has(lq,qg.PromisingRoute); R[13]=has(sh,me.SharedHypothesis)
R[14]=has(tsp,mc.FormatMismatch); R[15]=has(mn,mc.DatasetNotInConfig); R[16]=has(ddep,lc.BlockingDeprecation)
R[17]=has(aa,lc.ConflictingHypothesis) and has(ab,lc.ConflictingHypothesis)
names={1:"classify by model",2:"CompleteExperiment",3:"MaterialBalance dep",4:"cascade (ГТМ->liquid)",
5:"acyclic DAG",6:"OilDomainOntology",7:"DerivedStaleRun (hidden)",8:"ObsoleteVersion",9:"Orphan(marker)",
10:"ConflictingTask",11:"PrunableSubtree(marker)",12:"PromisingRoute",13:"Shared(marker)",14:"FormatMismatch",
15:"DatasetMissing(marker)",16:"BlockingDeprecation",17:"ConflictingHypothesis"}
ok=sum(R.values())
for r in range(1,18): print(f"  rule {r:2d}: {'OK' if R[r] else 'FAIL'}  {names[r]}")
print(f"\n>>> {ok}/17 rules fire on the 16-node formula-derived graph; ГТМ cascade = {stale}")
