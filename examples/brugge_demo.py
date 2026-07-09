# Brugge integrity-control demo: real HermiT run of rules 4 (cascade) and 7 (hidden staleness).
import os, sys, time, hashlib
sys.path.insert(0, os.getcwd())
from owlready2 import sync_reasoner
from hyppo.core._base import virtual_experiment_onto, Hypothesis
from hyppo.ontology.core_rules import InvalidHypothesis, StaleHypothesis
from hyppo.ontology.provenance import (
    HypothesisVersion, ExperimentRun, version_of, uses_hypothesis_version,
    uses_hypothesis, run_depends_on_hypothesis, HypothesisWithStaleAncestor,
    DerivedStaleRun,
)

onto = virtual_experiment_onto
for ind in list(onto.individuals()):
    ind.destroy()

with onto:
    # Brugge 6-component hypothesis graph (mirrors norne_adapter.py:163-192)
    h_CRM = Hypothesis("brugge_h_CRM")   # CRM core (physics)
    h_ML  = Hypothesis("brugge_h_ML")    # ML correction
    h_LPR = Hypothesis("brugge_h_LPR")   # liquid production rate
    h_MB  = Hypothesis("brugge_h_MB")    # material balance
    h_BL  = Hypothesis("brugge_h_BL")    # baseline liquid
    h_WCT = Hypothesis("brugge_h_WCT")   # water cut
    OPR   = Hypothesis("brugge_OPR")     # oil production rate (merge)
    h_LPR.derived_by = [h_CRM, h_ML]
    h_MB.derived_by  = [h_LPR]
    h_BL.derived_by  = [h_MB]
    h_WCT.derived_by = [h_BL, h_ML]
    OPR.derived_by   = [h_BL, h_WCT]
    all_hyp = [h_CRM, h_ML, h_LPR, h_MB, h_BL, h_WCT, OPR]

    # versioning + a baseline run R0 that uses h_BL@v0 (immediate version)
    v_bl_0 = HypothesisVersion("brugge_v_BL_0"); v_bl_0.version_of = [h_BL]
    R0 = ExperimentRun("brugge_R0"); R0.uses_hypothesis_version = [v_bl_0]

    # "artifacts" (their hashes are what file-level caching inspects)
    art_CRM_v0 = b"h_CRM output: CRM connectivity matrix (Brugge 10->20)"
    art_BL_v0  = b"h_BL output: liquid prediction vector (Brugge 20 producers)"

def sha(b): return hashlib.sha256(b).hexdigest()[:12]
def stale_set():  return [h.name for h in all_hyp if StaleHypothesis in h.is_a]
def invalid_set():return [h.name for h in all_hyp if InvalidHypothesis in h.is_a]

print("="*70)
print("PHASE A — baseline (no revision)")
H_CRM_a, H_BL_a = sha(art_CRM_v0), sha(art_BL_v0)
t=time.time(); sync_reasoner(infer_property_values=True); dtA=time.time()-t
print(f"reasoning time: {dtA*1000:.0f} ms")
print("StaleHypothesis:   ", stale_set() or "(none)")
print("InvalidHypothesis: ", invalid_set() or "(none)")
print("R0 uses_hypothesis:      ", [o.name for o in R0.uses_hypothesis])
print("R0 run_depends_on_hyp.:  ", [o.name for o in R0.run_depends_on_hypothesis])
print("R0 is DerivedStaleRun:   ", DerivedStaleRun in R0.is_a)

print("="*70)
print("PHASE B — inject hidden staleness: revise h_CRM (mark prior invalid);")
print("          do NOT recompute h_BL -> its immediate artifact hash unchanged")
with onto:
    h_CRM.is_a.append(InvalidHypothesis)   # revision invalidates prior h_CRM results
art_CRM_v1 = b"h_CRM output: CRM connectivity matrix, Tikhonov-reg. (Brugge 10->20)"  # changed
H_CRM_b, H_BL_b = sha(art_CRM_v1), sha(art_BL_v0)   # h_BL artifact NOT recomputed
t=time.time(); sync_reasoner(infer_property_values=True); dtB=time.time()-t
print(f"reasoning time: {dtB*1000:.0f} ms")
print(f"h_CRM artifact hash: {H_CRM_a} -> {H_CRM_b}  (changed = revision)")
print(f"h_BL  artifact hash: {H_BL_a} -> {H_BL_b}  (UNCHANGED: file cache would skip)")
print("StaleHypothesis (rule 4 cascade):", sorted(stale_set()))
print("  expected descendants of h_CRM:  ['brugge_OPR','brugge_h_BL','brugge_h_LPR','brugge_h_MB','brugge_h_WCT']")
print("HypothesisWithStaleAncestor:     ", sorted(h.name for h in all_hyp if HypothesisWithStaleAncestor in h.is_a))
print("R0 uses_hypothesis:              ", [o.name for o in R0.uses_hypothesis])
print("R0 run_depends_on_hyp.:          ", sorted(o.name for o in R0.run_depends_on_hypothesis))
print(">>> R0 classified DerivedStaleRun (rule 7):", DerivedStaleRun in R0.is_a)
print("="*70)
print("FILE-CACHE VERDICT on R0 (keyed on immediate h_BL hash): UP-TO-DATE (skip)")
print("REASONER VERDICT  on R0:                          DERIVED-STALE (recompute)")
