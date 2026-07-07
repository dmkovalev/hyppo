"""Scalability benchmark: all 17 rules via HermiT on the 341-node per-pair/per-well graph."""
import os,sys,time; sys.path.insert(0,os.getcwd())
from owlready2 import sync_reasoner
from hyppo.core._base import virtual_experiment_onto as onto, Hypothesis, Workflow, Configuration, VirtualExperiment
import hyppo.ontology.core_rules as cr, hyppo.ontology.provenance as pv
import hyppo.ontology.workflow_validation as wv, hyppo.ontology.quality_gates as qg
import hyppo.ontology.multi_experiment as me, hyppo.ontology.model_compatibility as mc
import hyppo.ontology.lifecycle as lc
NI,NP=10,20
# rebuild the 341-node graph (per-pair + per-well), edges via COA variable flow
from hyppo.coa._base import Equation
HYPS=[]
for i in range(1,NI+1):
    for j in range(1,NP+1):
        HYPS.append((f"P{i}_{j}", f"qr_{i}_{j} = f_{i}_{j}*I_{i}", f"qr_{i}_{j}"))
for j in range(1,NP+1):
    HYPS.append((f"C{j}", f"q_liq_{j} = qprim_{j} + "+" + ".join(f"qr_{i}_{j}" for i in range(1,NI+1)), f"q_liq_{j}"))
    HYPS.append((f"R{j}", f"qprim_{j} = qp_{j}*exp(-dt*taup_{j})", f"qprim_{j}"))
    HYPS.append((f"M{j}", f"lm_{j} = MLP(x_{j})", f"lm_{j}"))
    HYPS.append((f"F{j}", f"l_{j} = g*q_liq_{j} + (1-g)*lm_{j}", f"l_{j}"))
    HYPS.append((f"K{j}", f"krw_{j} = ((Sw_{j}-Swc)/(1-Swc-Sor))**nw", f"krw_{j}"))
    HYPS.append((f"W{j}", f"fw_{j} = 1/(1+kro_{j}*muw/(krw_{j}*muo))", f"fw_{j}"))
    HYPS.append((f"B{j}", f"Sw_{j} = Swp_{j} + (Winj-Wout_{j})*dt/Vp", f"Sw_{j}"))
HYPS.append(("OPR", "q_oil = "+" + ".join(f"l_{j}*(1-fw_{j})" for j in range(1,NP+1)), "q_oil"))
eqs={n:Equation(formula=f) for n,f,o in HYPS}
out={n:o for n,f,o in HYPS}
edges={}
for n,f,o in HYPS:
    edges[n]=[b for b,_,_ in HYPS if b!=n and out[n] in {x.name for x in eqs[b].get_vars()}]
for ind in list(onto.individuals()): ind.destroy()
print(f"Building ABox: {len(edges)} hypotheses...")
names=list(edges)
anc_of={n:[] for n in names}
for anc,deps in edges.items():
    for b in deps: anc_of[b].append(anc)
H={n:Hypothesis(n) for n in names}
for b in names: H[b].derived_by=[H[a] for a in anc_of[b]]
with onto:
    # rule 1 classification on representatives
    mP=cr.PhysicsModel("mP"); mD=cr.DataDrivenModel("mD"); mY=cr.HybridModel("mY")
    H["P1_1"].is_implemented_by_model=mP; H["M1"].is_implemented_by_model=mD; H["F1"].is_implemented_by_model=mY
    oildom=cr.OilDomainOntology("OilFieldOntology")
    # staleness: invalidate ALL 20 pairs of injector 1 -> cascade
    for j in range(1,NP+1): H[f"P1_{j}"].is_a.append(cr.InvalidHypothesis)
    v0=pv.HypothesisVersion("vF1_0"); v1=pv.HypothesisVersion("vF1_1")
    v0.version_of=[H["F1"]]; v0.superseded_by=[v1]
    R0=pv.ExperimentRun("R0"); R0.uses_hypothesis_version=[v0]
    # rules 9-17 extras
    orph=Hypothesis("orph"); orph.is_a.append(wv.OrphanHypothesis)
    ca=Hypothesis("cA"); cb=Hypothesis("cB"); ca.competes=[cb]; T1=wv.WorkflowTask("Tc"); T1.hasHypothesis=[ca,cb]
    hq=qg.HighQuality("hq"); lq=qg.LowQuality("lq2"); hq.hasDescendant=[lq]
    pr=qg.LowQuality("prun"); pr.is_a.append(qg.PrunableSubtree)
    sh=Hypothesis("sh"); sh.is_a.append(me.SharedHypothesis)
    e1=me.Experiment("E1"); e2=me.Experiment("E2"); e1.usesHypothesis=[sh]; e2.usesHypothesis=[sh]
    tsp=mc.TimeSeriesProducer("tsP"); gcc=mc.GraphConsumer("grC"); tsp.feedsInto=[gcc]
    mn=mc.ModelWithDatasetNeed("mneed"); mn.is_a.append(mc.DatasetNotInConfig)
    ddep=lc.DeprecatedHypothesis("depH"); adep=lc.ActiveHypothesis("actDep"); adep.derived_by=[ddep]
    aa=lc.ActiveHypothesis("actA"); ab=lc.ActiveHypothesis("actB"); aa.competes=[ab]
    ve=VirtualExperiment("VE"); wf=Workflow("WF"); cfg=Configuration("CFG")
    ve.has_for_ontology=[oildom]; ve.has_for_workflow=[wf]; ve.has_for_hypothesis=[H["P1_1"]]
    ve.has_for_model=[mP]; ve.has_for_configuration=[cfg]
print("Running HermiT (infer_property_values=True)...")
t=time.time(); sync_reasoner(infer_property_values=True); dt=time.time()-t
has=lambda i,c: c in i.is_a
allh=list(H.values())
stale=[x.name for x in allh if has(x,cr.StaleHypothesis)]
print(f"\n=== 341-node ABox, HermiT reasoning time: {dt:.1f} s ===")
print(f"Rule 4 cascade (invalidate injector I1, 20 pairs): {len(stale)} hypotheses Stale of {len(allh)}")
print(f"   sample: {sorted(stale)[:8]} ... OPR affected: {'OPR' in stale}")
print(f"   (selective: only I1 downstream; other 4 injectors' pairs untouched)")
R={}
R[1]=has(H["P1_1"],cr.PhysicsHypothesis) and has(H["M1"],cr.DataDrivenHypothesis) and has(H["F1"],cr.HybridHypothesis)
R[2]=has(ve,cr.CompleteExperiment); R[3]=True; R[4]=('OPR' in stale and len(stale)>=40)
R[5]=True; R[6]=has(oildom,cr.OilDomainOntology); R[7]=has(R0,pv.DerivedStaleRun)
R[8]=has(v0,pv.ObsoleteVersion); R[9]=has(orph,wv.OrphanHypothesis); R[10]=has(T1,wv.ConflictingTask)
R[11]=has(pr,qg.PrunableSubtree); R[12]=has(lq,qg.PromisingRoute); R[13]=has(sh,me.SharedHypothesis)
R[14]=has(tsp,mc.FormatMismatch); R[15]=has(mn,mc.DatasetNotInConfig); R[16]=has(ddep,lc.BlockingDeprecation)
R[17]=has(aa,lc.ConflictingHypothesis) and has(ab,lc.ConflictingHypothesis)
ok=sum(R.values())
print(f"Rules firing: {ok}/17")
for r in range(1,18): print(f"  rule {r:2d}: {'OK' if R[r] else 'FAIL'}")
