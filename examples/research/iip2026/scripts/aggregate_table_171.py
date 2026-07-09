import json, statistics as st, os, math
here = os.path.dirname(__file__)
data = os.path.join(here, "..", "data")
sweep = json.load(open(os.path.join(data, "wfcommons_per_workflow_sweep.json")))
val = json.load(open(os.path.join(data, "wfcommons_validation_results.json")))
rgrid = sweep["__meta__"]["r_grid"]
idx = {round(r, 1): i for i, r in enumerate(rgrid)}
bw = sweep["by_workflow"]
vw = {v["label"]: v for v in val["per_workflow"]}

def vget(lab):
    for cand in (lab, lab.split("/", 1)[1] if "/" in lab else lab, lab.split("/")[-1]):
        if cand in vw:
            return vw[cand]
    return None

fams = {}
for lab, w in bw.items():
    fams.setdefault(w["family"], []).append(lab)  # уровень движка (как автореферат)
print("N workflows:", len(bw), "| groups:", {f: len(l) for f, l in fams.items()})

med = st.median

def agg(labels):
    n = [bw[l]["n"] for l in labels]
    dep = [vget(l)["depth"] for l in labels if vget(l)]
    def rr(r): return [bw[l]["rhos"][idx[r]] for l in labels]
    er = [vget(l)["rho_er"] for l in labels if vget(l) and not math.isnan(vget(l)["rho_er"])]
    ba = [vget(l)["rho_ba"] for l in labels if vget(l) and not math.isnan(vget(l)["rho_ba"])]
    return (len(labels), med(n), med(dep), med(rr(0.3)), med(rr(0.5)),
            med(rr(0.7)), med(rr(0.9)), med(er), med(ba))

hdr = ("class", "N", "med|H|", "medDep", "r0.3", "r0.5", "r0.7", "r0.9", "rhoER", "rhoBA")
print("%-26s" % hdr[0], " ".join("%7s" % h for h in hdr[1:]))
order = ["nextflow", "snakemake", "pegasus"]
for f in order:
    if f in fams:
        a = agg(fams[f])
        print("%-26s" % f, "%7d %7.1f %7.0f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % a)
a = agg(list(bw.keys()))
print("%-26s" % "ВЕСЬ КОРПУС (171)", "%7d %7.1f %7.0f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % a)
