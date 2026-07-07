# Real Volve field-level fit + integrity-control demo (replaces restricted AC10 numbers).
import os,sys,hashlib; sys.path.insert(0,r"F:\git-repos\diss\hyppo-ref")
import numpy as np, csv
from pywaterflood import CRM
rows=list(csv.reader(open(r"F:\git-repos\diss\hyppo-ref\_volve.csv")))
hdr=rows[0]; d=np.array([[float(x) if x else 0.0 for x in r[1:]] for r in rows[1:]])  # p,Np,Gp,Wp,Gi,Wi,Rp
Np=d[:,1]; Wp=d[:,4]; Wi=d[:,5]
oil=np.diff(Np,prepend=Np[0]); wat=np.diff(Wp,prepend=Wp[0]); inj=np.diff(Wi,prepend=Wi[0])
liq=oil+wat
T=len(liq); time=np.arange(T)*30.4
print(f"Volve: {T} monthly steps (field-level). total oil={oil.sum():.0f} water_inj={inj.sum():.0f} STB")
inj_m=inj.reshape(-1,1); liq_m=liq.reshape(-1,1)
ntr=int(T*0.7)
def r2(y,p): m=y>1e-6; return 1-((y[m]-p[m])**2).sum()/((y[m]-y[m].mean())**2).sum() if m.sum()>1 and y[m].std()>0 else float('nan')
# h_CRM (physics): single-tank CRM, two parameterizations (baseline vs revision)
def fit(c): c.fit(production=liq_m[:ntr,:],injection=inj_m[:ntr,:],time=time[:ntr]); return np.asarray(c.predict(injection=inj_m,time=time)).reshape(-1)
c0=CRM(primary=True,tau_selection="per-pair",constraints="up-to one"); p0=fit(c0)
c1=CRM(primary=True,tau_selection="per-pair",constraints="positive");          p1=fit(c1)  # revision
crm_tr=r2(p0[:ntr],liq[:ntr]); crm_te=r2(p0[ntr:],liq[ntr:])
print(f"h_CRM (Volve): R2 train={crm_tr:.3f} test={crm_te:.3f}  (single-tank CRM, Wi->liquid)")
# h_LPR hybrid (CRM + ridge residual, one-step-ahead)
lag=np.concatenate([[0],liq[:-1]])
def feat(a,b): return np.stack([np.ones(b-a),p0[a:b],lag[a:b],inj[a:b]],axis=1)
Xtr=feat(6,ntr); ytr=liq[6:ntr]
w=np.linalg.solve(Xtr.T@Xtr+1.0*np.eye(4),Xtr.T@ytr)
hyb=(feat(0,T)@w)
hyb_tr=r2(hyb[6:ntr],liq[6:ntr]); hyb_te=r2(hyb[ntr:],liq[ntr:])
print(f"h_LPR (Volve hybrid): R2 train={hyb_tr:.3f} test={hyb_te:.3f}")
# material balance: cum water inj vs cum liquid prod
print(f"h_MB (Volve): cum water_inj={inj.sum():.0f}  cum liquid_prod={liq.sum():.0f} STB")
# hidden-staleness instance: revise h_CRM (p0->p1) ; h_LPR not recomputed
H=lambda a: hashlib.sha256(np.ascontiguousarray(a,np.float64).tobytes()).hexdigest()[:10]
print(f"\nHidden staleness on Volve:")
print(f"  h_CRM@v0 (baseline) hash = {H(p0)}")
print(f"  h_CRM@v1 (revision) hash = {H(p1)}   changed={H(p0)!=H(p1)}")
print(f"  h_LPR@v0 (hybrid, NOT recomputed) hash = {H(hyb)}   (immediate input to forecast -> file cache would SKIP)")
# ontology verdict
import hyppo.ontology.core_rules as cr, hyppo.ontology.provenance as pv
from hyppo.core._base import virtual_experiment_onto as onto, Hypothesis
from owlready2 import sync_reasoner
for ind in list(onto.individuals()): ind.destroy()
with onto:
    hC=Hypothesis("volve_h_CRM"); hL=Hypothesis("volve_h_LPR"); hL.derived_by=[hC]
    v=pv.HypothesisVersion("volve_vL_0"); v.version_of=[hL]; R=pv.ExperimentRun("volve_R0"); R.uses_hypothesis_version=[v]
    hC.is_a.append(cr.InvalidHypothesis)
sync_reasoner(infer_property_values=True)
print(f"  >>> rule 7: R0 in DerivedStaleRun = {pv.DerivedStaleRun in R.is_a}  (h_CRM revised -> h_LPR stale ancestor -> run flagged)")
