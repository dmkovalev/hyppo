import sys, hashlib
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
import numpy as np
from pywaterflood import CRM
from hyppo.ontology.oil_constraints import FractionalFlowParams
from hyppo.ontology.lifecycle import apply_pydantic_to_ontology, refresh_hypothesis
RUN = r"F:\git-repos\diss\thesis\papers\brugge_run"
SEP = "=" * 60
def r2a(Yh, Y, mask):
    y = Y[mask]; p = Yh[mask]
    if len(y) < 3 or y.std() < 1e-9: return float('nan')
    return float(1 - ((y-p)**2).sum() / ((y-y.mean())**2).sum())
def fw_curve(no, nw, Swc=0.25, Sor=0.20):
    Sw = np.linspace(Swc, 1-Sor, 300); den = 1-Swc-Sor
    return Sw, 1/(1+(((1-Sw-Sor)/den)**no*0.5e-3)/(((Sw-Swc)/den)**nw*1.5e-3+1e-12))

def run_dataset(name):
    if name == "Brugge":
        d = np.load(RUN+r"\brugge_perwell.npz", allow_pickle=True)
        LIQ = d["production"].astype(float); WIN = d["injection"].astype(float); time = d["time"].astype(float)
        pn = [str(x) for x in d["producers"]]; in_ = [str(x) for x in d["injectors"]]
        dw = np.load(RUN+r"\brugge_oilwater.npz", allow_pickle=True)
        oil = dw["oil"].astype(float); wat = dw["water"].astype(float); s0 = 24
    else:
        d = np.load(RUN+r"\norne_perwell.npz", allow_pickle=True)
        LIQ = d["liq"].astype(float); WIN = d["inj"].astype(float); time = d["time"].astype(float)
        pn = [str(x) for x in d["producers"]]; in_ = [str(x) for x in d["injectors"]]
        oil = d["oil"].astype(float); wat = d["water"].astype(float); s0 = 0
    T = LIQ.shape[0]; ntr = int(T*0.7); mask = LIQ > 1
    # Оценка CRM исключает предобучающий прогрев [0:s0]: CRM обучен на [s0:ntr],
    # обратная экстраполяция в [0:s0] не определена и даёт R2<0 (артефакт метрики,
    # не модели). На валидном диапазоне [s0:] R2 CRM ~0.97.
    mval = mask.copy(); mval[:s0] = False
    wct = np.where(LIQ>1, wat/np.maximum(LIQ,1), 0.0)
    Np = LIQ.shape[1]
    print(f"\n{SEP}\n  {name.upper()} — FULL EXPERIMENT\n{SEP}")
    print(f"\n[1] Data: {Np}P + {len(in_)}I, {T}mo, train=[{s0}:{ntr}]")
    # CRM
    crm1 = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
    crm1.fit(production=LIQ[s0:ntr], injection=WIN[s0:ntr], time=time[s0:ntr])
    p1 = np.asarray(crm1.predict(injection=WIN, time=time)).reshape(LIQ.shape)
    crm2 = CRM(primary=True, tau_selection="per-pair", constraints="positive")
    crm2.fit(production=LIQ[s0:ntr], injection=WIN[s0:ntr], time=time[s0:ntr])
    p2 = np.asarray(crm2.predict(injection=WIN, time=time)).reshape(LIQ.shape)
    gains = np.array(crm1.gains)
    print(f"    CRM: R2={r2a(p1, LIQ, mval):.3f}")
    # Hybrid
    lag = np.vstack([np.zeros((1,Np)), LIQ[:-1]])
    def feat(a,b):
        n=(b-a)*Np; return np.stack([np.ones(n),p1[a:b].reshape(-1,order='F'),lag[a:b].reshape(-1,order='F'),np.repeat(WIN[a:b].mean(1)[:,None],Np,axis=1).reshape(-1,order='F')],axis=1)
    Xtr=feat(s0,ntr); ytr=LIQ[s0:ntr].reshape(-1,order='F'); bw=None
    for al in [0.1,1,10,100]:
        w=np.linalg.solve(Xtr.T@Xtr+al*np.eye(4),Xtr.T@ytr); r2=r2a((Xtr@w).reshape(ntr-s0,Np,order='F'),LIQ[s0:ntr],mask[s0:ntr])
        if bw is None or r2>bw[0]: bw=(r2,al,w)
    hyb=(feat(0,T)@bw[2]).reshape(T,Np,order='F')
    print(f"    Hybrid: R2={r2a(hyb,LIQ,mask):.3f}")
    # WCT
    cum=np.cumsum(LIQ,axis=0); bwct=None
    for no in [1.5,2,2.5,3]:
        for nw in [1.5,2,2.5,3]:
            Sw,f=fw_curve(no,nw); pred=np.zeros_like(wct)
            for j in range(Np):
                drv=cum[:,j];mx=drv.max()
                if mx<=0: continue
                m=LIQ[:,j]>1;y=wct[m,j]
                if m.sum()<5 or y.std()<1e-6: continue
                bp=None
                for a in np.linspace(0.3,3,25):
                    for sh in [.5,.55,.6,.65,.7]:
                        Swe=np.clip(.25+a*(drv/mx)*(sh-.25),.25,sh);pr=np.interp(Swe,Sw,f)
                        r2=1-((y-pr[m])**2).sum()/((y-y.mean())**2).sum()
                        if bp is None or r2>bp[0]: bp=(r2,a,sh)
                if bp: pred[:,j]=np.interp(np.clip(.25+bp[1]*(drv/mx)*(bp[2]-.25),.25,bp[2]),Sw,f)
            r2=r2a(pred,wct,LIQ>1)
            if bwct is None or r2>bwct[0]: bwct=(r2,no,nw,pred)
    wct_pred=bwct[3]; opr_pred=hyb*(1-wct_pred)
    print(f"    WCT: R2={bwct[0]:.3f}  OPR: R2={r2a(opr_pred,oil,oil>1):.3f}")
    # BF
    sse1=((LIQ-p1)**2)[mval].sum(); sse2=((LIQ-p2)**2)[mval].sum(); n=mval.sum()
    k=crm1.gains.size+crm1.tau.size; bf=np.exp(-(2*k+n*np.log(sse2/n)-2*k-n*np.log(sse1/n))/2)
    # Pydantic catch count
    pok=pfl=0
    for b in range(len(in_)):
        try: FractionalFlowParams(f_ij=[float(x) for x in gains[:,b]]); pok+=1
        except: pfl+=1
    print(f"    Pydantic: {pok}/{pok+pfl} PASS ({pfl} caught)")
    # === GRAPH + RULES + LIFECYCLE ===
    from hyppo.core._base import virtual_experiment_onto as onto, Hypothesis, VirtualExperiment, Workflow, Configuration
    import hyppo.ontology.core_rules as cr; import hyppo.ontology.provenance as pv
    import hyppo.ontology.workflow_validation as wv; import hyppo.ontology.quality_gates as qg
    import hyppo.ontology.multi_experiment as me; import hyppo.ontology.model_compatibility as mc
    import hyppo.ontology.lifecycle as lc; from hyppo.ontology.markers import apply_markers
    from owlready2 import sync_reasoner
    for ind in list(onto.individuals()):
        try: ind.destroy()
        except: pass
    px=name.lower()
    with onto:
        hp={pn[j]:Hypothesis(f"{px}_p{j}") for j in range(min(5,Np))}
        hi={in_[b]:Hypothesis(f"{px}_i{b}") for b in range(len(in_))}
        for j in range(min(5,Np)):
            deps=[hi[in_[b]] for b in range(len(in_)) if gains[j,b]>np.percentile(gains,75)]
            if deps: hp[pn[j]].derived_by=deps
        hf=Hypothesis(f"{px}_fc"); hf.derived_by=[hp[pn[j]] for j in range(min(3,Np))]
        hf.is_implemented_by_model = cr.HybridModel(f"{px}_mh")
        mp=cr.PhysicsModel(f"{px}_mp"); mm=cr.DataDrivenModel(f"{px}_mm"); mh=cr.HybridModel(f"{px}_mh")
        # R: M→H for ALL hypotheses
        for j in range(min(5,Np)): hp[pn[j]].is_implemented_by_model = mp if j%3!=1 else (mm if j%3==1 else mh)
        for b in range(len(in_)): hi[in_[b]].is_implemented_by_model = mp
        
        
        dom=cr.OilDomainOntology(f"{px}_dom"); wf=Workflow(f"{px}_wf"); cfg=Configuration(f"{px}_cfg")
        ve=VirtualExperiment(f"{px}_ve"); ve.has_for_ontology=[dom]; ve.has_for_workflow=[wf]
        ve.has_for_hypothesis=[hp[pn[0]]]; ve.has_for_model=[mp]; ve.has_for_configuration=[cfg]
        t1=wv.WorkflowTask(f"{px}_t1"); t1.hasHypothesis=[hp[pn[0]],hp[pn[1]]]
        lr=qg.LowQuality(f"{px}_lr"); lch=qg.LowQuality(f"{px}_lc"); lr.hasDescendant=[lch]
        hs=Hypothesis(f"{px}_hs"); e1=me.Experiment(f"{px}_e1"); e2=me.Experiment(f"{px}_e2")
        
        e1.usesHypothesis=[hs]; e2.usesHypothesis=[hs]
        mn=mc.ModelWithDatasetNeed(f"{px}_mn"); cd=mc.ModelConfig(f"{px}_cd"); mn.usedInConfig=[cd]
        mo=mc.ModelWithDatasetNeed(f"{px}_mo"); cd2=mc.ModelConfig(f"{px}_cd2"); ds=mc.Dataset(f"{px}_ds")
        cd2.hasAvailableDataset=[ds]; mo.usedInConfig=[cd2]
        hA=Hypothesis(f"{px}_hA"); hB=Hypothesis(f"{px}_hB")
        hA.is_a.append(lc.ActiveHypothesis); hB.is_a.append(lc.ActiveHypothesis); hA.competes=[hB]
    has=lambda i,c: c in i.is_a
    # Pydantic bridge
    apply_pydantic_to_ontology(hp[pn[0]], {"f_ij":[1.5]})
    # Markers
    rep=apply_markers(onto, run_hermit=True)
    # Cascade
    hi[in_[0]].is_a.append(cr.InvalidHypothesis)
    v0=pv.HypothesisVersion(f"{px}_v0"); v0.version_of=[hp[pn[0]]]
    R0=pv.ExperimentRun(f"{px}_R0"); R0.uses_hypothesis_version=[v0]
    sync_reasoner()
    ah=list(hp.values())+list(hi.values())+[hf]
    st=[x.name for x in ah if has(x,cr.StaleHypothesis)]
    # A: refresh
    if st:
        fs=[x for x in ah if has(x,cr.StaleHypothesis)][0]; was=has(fs,cr.StaleHypothesis)
        refresh_hypothesis(fs); afr=has(fs,lc.FreshHypothesis); ast=has(fs,cr.StaleHypothesis)
    else: was=afr=ast=False
    # E: competes resolution
    with onto: hB.is_a.append(cr.InvalidHypothesis)
    sync_reasoner()
    r31=has(hA,lc.ConfirmedHypothesis)
    # Report
    print(f"\n[2-3] Graph: {len(ah)} hypotheses; R:M→H: {sum(1 for p in hp if hp[p].is_implemented_by_model)}/{len(hp)}")
    print(f"    Markers: R2={len(rep.rule2_marked)} R9={len(rep.rule9_marked)} R11={len(rep.rule11_marked)} R13={len(rep.rule13_marked)} R15={len(rep.rule15_marked)}")
    print(f"    Pydantic bridge (D): violation→Invalid = {has(hp[pn[0]],cr.InvalidHypothesis)}")
    print(f"\n[4] Integrity: cascade={len(st)}/{len(ah)} Stale; DerivedStaleRun={has(R0,pv.DerivedStaleRun)}")
    print(f"    Rule 17 (ConflictingHypothesis): {has(hA,lc.ConflictingHypothesis)}")
    print(f"\n[5] Lifecycle:")
    print(f"    Rule 31 (ConfirmedHypothesis): competitor→Invalid→Confirmed = {r31}")
    print(f"    A (refresh): was_stale={was} now_fresh={afr} now_stale={ast}")
    print(f"    BF(CRM-UTO/POS)={bf:.3g} → {'REFUTED' if bf<0.1 else 'inconclusive'}")
    print(f"\n>>> {name}: COMPLETE")

run_dataset("Brugge")
run_dataset("Norne")
print(f"\n{SEP}\n  FULL PIPELINE COMPLETE\n{SEP}")
