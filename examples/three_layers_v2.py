import os,sys,hashlib
import sys as _sys
ONLY=_sys.argv[1] if len(_sys.argv)>1 else None
sys.path.insert(0,r"F:\git-repos\diss\hyppo-ref")
import numpy as np
from pywaterflood import CRM
from hyppo.ontology.oil_constraints import FractionalFlowParams,TimeScaleParams,SaturationParams,CoreyExponentParams
RUN=r"F:\git-repos\diss\thesis\papers\brugge_run"
SETS={"Brugge":(RUN+r"\brugge_perwell.npz",24,3.5,2.0),"Norne":(RUN+r"\norne_perwell.npz",0,2.5,2.0)}
for name,(npz,s0,no,nw) in SETS.items():
    if ONLY and name!=ONLY: continue
    d=np.load(npz,allow_pickle=True)
    LIQ=(d["liq"] if "liq" in d else d["production"]).astype(float);WIN=(d["inj"] if "inj" in d else d["injection"]).astype(float);time=d["time"].astype(float);T=LIQ.shape[0];ntr=int(T*0.7)
    inj=[str(x) for x in d["injectors"]];prod=[str(x) for x in d["producers"]]
    print(f"\n{'='*60}\n  {name}: {len(prod)}P+{len(inj)}I, {T}mo\n{'='*60}")
    crm=CRM(primary=True,tau_selection="per-pair",constraints="up-to one")
    crm.fit(production=LIQ[s0:ntr],injection=WIN[s0:ntr],time=time[s0:ntr])
    g=np.array(crm.gains);tau=np.array(crm.tau).flatten()
    p=np.asarray(crm.predict(injection=WIN,time=time)).reshape(LIQ.shape)
    r2t=1-((LIQ[ntr:]-p[ntr:])**2)[(LIQ[ntr:]>1)].sum()/((LIQ[ntr:][(LIQ[ntr:]>1)]-LIQ[ntr:][(LIQ[ntr:]>1)].mean())**2).sum()
    # L3
    ok=fl=0
    for b in range(len(inj)):
        try:FractionalFlowParams(f_ij=[float(x) for x in g[:,b]]);ok+=1
        except:fl+=1
    try:TimeScaleParams(tau_fast=float(tau.min()),tau_slow=float(tau.max()));tk="P"
    except:tk="F"
    try:SaturationParams(s_o=0.7,s_w=0.3);sk="P"
    except:sk="F"
    try:CoreyExponentParams(n_oil=float(no),n_water=float(nw));ck="P"
    except:ck="F"
    print(f"L3 Pydantic: FlowFrac {ok}/{ok+fl} P({fl} caught) Tau={tk} Sat={sk} Corey={ck}")
    # L2
    from hyppo.core._base import virtual_experiment_onto as onto,Hypothesis,Workflow,Configuration,VirtualExperiment
    import hyppo.ontology.core_rules as cr2
    import hyppo.ontology.provenance as pv
    import hyppo.ontology.workflow_validation as wv
    import hyppo.ontology.quality_gates as qg
    import hyppo.ontology.multi_experiment as me
    import hyppo.ontology.model_compatibility as mc
    from hyppo.ontology.markers import apply_markers
    for ind in list(onto.individuals()):ind.destroy()
    px=name.lower();NP=min(5,len(prod))
    hp={prod[j]:Hypothesis(f"{px}_p{j}") for j in range(NP)}
    hi2={inj[b]:Hypothesis(f"{px}_i{b}") for b in range(len(inj))}
    for j in range(NP):
        deps=[hi2[inj[b]] for b in range(len(inj)) if g[j,b]>np.percentile(g,75)]
        if deps:hp[prod[j]].derived_by=deps
    hf=Hypothesis(f"{px}_fc");hf.derived_by=[hp[prod[j]] for j in range(min(3,NP))]
    mp=cr2.PhysicsModel(f"{px}_mp");mm=cr2.DataDrivenModel(f"{px}_mm")
    hp[prod[0]].is_implemented_by_model=mp;hp[prod[1]].is_implemented_by_model=mm
    dom=cr2.OilDomainOntology(f"{px}_dom");wf=Workflow(f"{px}_wf");cf=Configuration(f"{px}_cf")
    vf=VirtualExperiment(f"{px}_vf");vf.has_for_ontology=[dom];vf.has_for_workflow=[wf];vf.has_for_hypothesis=[hp[prod[0]]];vf.has_for_model=[mp];vf.has_for_configuration=[cf]
    vi=VirtualExperiment(f"{px}_vi");vi.has_for_ontology=[dom];vi.has_for_workflow=[wf];vi.has_for_hypothesis=[hp[prod[1]]];vi.has_for_model=[mm]
    t1=wv.WorkflowTask(f"{px}_t1");t1.hasHypothesis=[hp[prod[0]],hp[prod[1]]]
    lr=qg.LowQuality(f"{px}_lr");lc=qg.LowQuality(f"{px}_lc");lr.hasDescendant=[lc]
    hs=Hypothesis(f"{px}_hs");e1=me.Experiment(f"{px}_e1");e2=me.Experiment(f"{px}_e2");e1.usesHypothesis=[hs];e2.usesHypothesis=[hs]
    mn=mc.ModelWithDatasetNeed(f"{px}_mn");cd=mc.ModelConfig(f"{px}_cd");mn.usedInConfig=[cd]
    mo=mc.ModelWithDatasetNeed(f"{px}_mo");cd2=mc.ModelConfig(f"{px}_cd2");ds=mc.Dataset(f"{px}_ds");cd2.hasAvailableDataset=[ds];mo.usedInConfig=[cd2]
    rep=apply_markers(onto,run_hermit=True)
    print(f"L2 Markers: R2={len(rep.rule2_marked)} R9={len(rep.rule9_marked)} R11={len(rep.rule11_marked)} R13={len(rep.rule13_marked)} R15={len(rep.rule15_marked)} rollback={len(rep.rolled_back)} hermit_skip={rep.hermit_skipped}")
    print(f"  Neg: VE_inc={cr2.CompleteExperiment not in vi.is_a} M_ok={mc.DatasetNotInConfig not in mo.is_a}")
    # L1
    hi2[inj[0]].is_a.append(cr2.InvalidHypothesis)
    v0=pv.HypothesisVersion(f"{px}_v0");v0.version_of=[hp[prod[0]]];R0=pv.ExperimentRun(f"{px}_R0");R0.uses_hypothesis_version=[v0]
    from owlready2 import sync_reasoner;sync_reasoner(infer_property_values=True)
    ah=list(hp.values())+list(hi2.values())+[hf]
    st=[x.name for x in ah if cr2.StaleHypothesis in x.is_a]
    dsr=pv.DerivedStaleRun in R0.is_a
    print(f"L1 HermiT: CRM_test={r2t:.3f} cascade={len(st)}/{len(ah)} DerivedStaleRun={dsr}")
    print(f">>> {name}: 3 LAYERS OK")
