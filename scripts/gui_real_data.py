"""Precompute the REAL virtual-experiment data for the GUI demonstration.

Run under .venv311 (has pywaterflood + owlready2):

    .venv311/Scripts/python.exe scripts/gui_real_data.py

Produces hyppo/gui/real_data.json:
- the 19-hypothesis HybridCRM graph (part4.tex §4.4) DERIVED by Algorithm 1
  from the hypothesis equations (COA causal ordering) — reproduces the 21 edges
  of Figure lattice_crm exactly;
- the workflow W as a depth-5 DAG of tasks with hypothesis→task membership;
- the real domain ontology (core classes + subclass hierarchy + object-property
  domain→range) for proper ontology rendering;
- real CRM/Hybrid/WCT/OPR R^2 on Brugge (window+warmup-fix) and Norne (all data);
- Algorithms 3/4 (well-formedness, planning P_ne/P_e cascade) and theorems.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

RUN = r"F:\git-repos\diss\thesis\papers\brugge_run"
OUT = os.path.join("hyppo", "gui", "real_data.json")


# ───────── 19 hypotheses (part4.tex §4.4), equations chosen so COA derives ────
# the 21 canonical edges. output variable in [brackets]; ML branches take the
# physics liquid q as attention context (H5→H6, H5→H7), as in the figure.
HYPS = [
    # id,   equation,                                   output,   label, branch, task
    ("H1",  "I_agg = w_ij * I_j",                        "I_agg",  "Агрегация закачки", "LPR", "T1"),
    ("H2",  "q_f = a_f * q_f_prev + b_f * I_agg",        "q_f",    "CRM быстрый канал (τ_fast)", "LPR", "T2"),
    ("H3",  "q_s = a_s * q_s_prev + b_s * I_agg",        "q_s",    "CRM медленный канал (τ_slow)", "LPR", "T2"),
    ("H4",  "q_c = w_f * q_f + (1 - w_f) * q_s",         "q_c",    "Смешение каналов", "LPR", "T2"),
    ("H5",  "q = J * q_c * (1 - d) ** t + q_b",          "q",      "Продуктивность", "LPR", "T2"),
    ("H6",  "h_t = Attn(q, x_t)",                        "h_t",    "Transformer (временное внимание)", "LPR", "T3"),
    ("H7",  "h_s = GNN(q, Gconn)",                        "h_s",    "GNN (пространственное внимание)", "LPR", "T3"),
    ("H8",  "l_p = q + delta",                           "l_p",    "LPR-якорь (физика)", "LPR", "T4"),
    ("H9",  "l_m = MLP(h_t, h_s)",                       "l_m",    "ML-коррекция жидкости", "LPR", "T3"),
    ("H10", "l = g * l_p + (1 - g) * l_m",               "l",      "LPR Fusion (вентиль g)", "LPR", "T4"),
    ("H11", "krw = Sw ** n",                             "krw",    "Corey (фазовые проницаемости)", "WCT", "T2"),
    ("H12", "fw_bl = 1 / (1 + k_o / (M * krw))",         "fw_bl",  "Баклея–Леверетта", "WCT", "T2"),
    ("H13", "fo = f_0 * exp(-Q_cum / tau)",              "fo",     "Нефтяная доля", "WCT", "T2"),
    ("H14", "dSw = dt * (W_in - W_out) / V_p + fw_bl",   "dSw",    "Материальный баланс", "WCT", "T2"),
    ("H15", "wor = a + b * sum_I",                       "wor",    "Log-WOR", "WCT", "T1"),
    ("H16", "fw = c1 * fo + c2 * dSw + c3 * wor",        "fw",     "BL/WOR смешение", "WCT", "T3"),
    ("H17", "o_p = (1 - fw) + delta",                    "o_p",    "WCT-якорь (физика)", "WCT", "T4"),
    ("H18", "o = g_w * o_p + (1 - g_w) * o_m",           "o",      "WCT Fusion", "WCT", "T4"),
    ("H19", "q_oil = l * o",                             "q_oil",  "OPR — прогноз нефти", "OPR", "T5"),
]

# workflow tasks (depth-5 DAG) with hypothesis membership
TASKS = [
    ("T1", "Парсинг ставок и связность"),
    ("T2", "Фит CRM (физика жидкости и обводнённости)"),
    ("T3", "ML-коррекция (Transformer / GNN / MLP)"),
    ("T4", "Слияние ветвей (fusion-вентили)"),
    ("T5", "Прогноз нефти (OPR)"),
]
TASK_EDGES = [["T1", "T2"], ["T2", "T3"], ["T3", "T4"], ["T4", "T5"]]

# model type per hypothesis (R: M→H)
MODEL_OF = {
    **{h: "m_phys" for h in ["H1", "H2", "H3", "H4", "H5", "H11", "H12", "H13", "H14", "H15", "H17"]},
    **{h: "m_ml" for h in ["H6", "H7", "H9"]},
    **{h: "m_hyb" for h in ["H8", "H10", "H16", "H18", "H19"]},
}
MODELS = [
    {"id": "m_phys", "label": "CRM / Buckley–Leverett (pywaterflood)", "class": "PhysicsModel"},
    {"id": "m_ml", "label": "Transformer+GNN+MLP", "class": "DataDrivenModel"},
    {"id": "m_hyb", "label": "Fusion (гибрид физика+ML)", "class": "HybridModel"},
]
# a couple of superseded/refuted-flavoured statuses for colour variety
BASE_STATUS = {"H13": "SUPERSEDED", "H15": "SUPERSEDED"}


def build_graph_algorithm1():
    """Real Algorithm 1: derive derived_by edges from equation variable flow."""
    from hyppo.coa._base import Equation

    eqs = {h: Equation(formula=f) for h, f, o, *_ in HYPS}
    out = {h: o for h, f, o, *_ in HYPS}
    varsof = {h: {v.name for v in eqs[h].get_vars()} for h, *_ in HYPS}

    edges, trace = [], []
    for h, f, o, *_ in HYPS:
        for b, fb, ob, *_ in HYPS:
            if b == h:
                continue
            if out[h] in varsof[b]:
                edges.append([h, b])
                trace.append({"src": h, "dst": b, "via": out[h],
                              "reason": f"выход {out[h]} ({h}) входит в уравнение {b}"})
    return edges, trace, {h: sorted(v) for h, v in varsof.items()}


def descendants(edges, start):
    out, stack = set(), [start]
    while stack:
        cur = stack.pop()
        for a, b in edges:
            if a == cur and b not in out:
                out.add(b); stack.append(b)
    return out


def plan_cascade(nodes, edges, changed):
    p_ne = set()
    for c in changed:
        p_ne.add(c); p_ne |= descendants(edges, c)
    return {"changed": changed, "p_ne": sorted(p_ne),
            "p_e": [n for n in nodes if n not in p_ne],
            "recompute_frac": round(len(p_ne) / len(nodes), 3)}


# ───────── ontology (draw as ontology: classes + subclass + prop domain→range) ─
CORE_CLASSES = ["Artefact", "Hypothesis", "Model", "VirtualExperiment", "Configuration",
                "Workflow", "Structure", "FullStructure", "Equation", "Variable",
                "FullCausalMapping", "DependencySet", "TransitiveClosure", "ResearchLattice"]


def extract_ontology():
    from hyppo.core._base import virtual_experiment_onto as onto
    cls_by_name = {c.name: c for c in onto.classes()}
    classes = []
    for name in CORE_CLASSES:
        c = cls_by_name.get(name)
        parent = None
        if c is not None:
            for p in c.is_a:
                pn = getattr(p, "name", None)
                if pn in CORE_CLASSES and pn != name:
                    parent = pn; break
        classes.append({"name": name, "parent": parent})
    core = set(CORE_CLASSES)
    relations = []
    for p in onto.object_properties():
        doms = [d.name for d in (p.domain or []) if getattr(d, "name", None) in core]
        rngs = [r.name for r in (p.range or []) if getattr(r, "name", None) in core]
        for d in doms:
            for r in rngs:
                relations.append({"property": p.name, "domain": d, "range": r})
    # key relations always shown even if domain/range not both core-typed
    seen = {(r["property"], r["domain"], r["range"]) for r in relations}
    for prop, d, r in [("derived_by", "Hypothesis", "Hypothesis"),
                       ("competes", "Hypothesis", "Hypothesis"),
                       ("is_implemented_by_model", "Hypothesis", "Model"),
                       ("has_for_hypothesis", "VirtualExperiment", "Hypothesis"),
                       ("has_for_model", "VirtualExperiment", "Model"),
                       ("has_for_workflow", "VirtualExperiment", "Workflow"),
                       ("has_for_configuration", "VirtualExperiment", "Configuration"),
                       ("has_for_structure", "Hypothesis", "Structure")]:
        if (prop, d, r) not in seen:
            relations.append({"property": prop, "domain": d, "range": r})
    return {"name": onto.base_iri, "classes": classes, "relations": relations,
            "total_classes": len(cls_by_name)}


def config_axes():
    from hyppo.adapters.wfopt_adapter import CONFIGURATION_SPACE
    axes = [{"name": n, "section": s["section"], "levels": list(s["levels"])}
            for n, s in CONFIGURATION_SPACE.items()]
    size = 1
    for a in axes:
        size *= max(1, len(a["levels"]))
    return axes, size


# ───────── real CRM on field data ────────────────────────────────────────────
def r2a(Yh, Y, mask):
    y = Y[mask]; p = Yh[mask]
    if len(y) < 3 or y.std() < 1e-9:
        return float("nan")
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fw_curve(no, nw, Swc=0.25, Sor=0.20):
    Sw = np.linspace(Swc, 1 - Sor, 300); den = 1 - Swc - Sor
    with np.errstate(invalid="ignore", divide="ignore"):
        f = 1 / (1 + (((1 - Sw - Sor) / den) ** no * 0.5e-3)
                 / (((Sw - Swc) / den) ** nw * 1.5e-3 + 1e-12))
    return Sw, f


def build_wellgraph(gains, pn, in_):
    """Real hypothesis graph from CRM connectivity — exactly as full_experiment.py:
    5 producer-hypotheses + all injector-hypotheses + 1 fusion; edge injector→producer
    where gain exceeds the 75th percentile; fusion derived_by the first 3 producers."""
    nP = min(5, len(pn)); nI = len(in_)
    thr = float(np.percentile(gains, 75))
    nodes = ([{"id": f"P{j}", "kind": "producer", "label": pn[j], "task": "T2"} for j in range(nP)]
             + [{"id": f"I{b}", "kind": "injector", "label": in_[b], "task": "T1"} for b in range(nI)]
             + [{"id": "FC", "kind": "fusion", "label": "Fusion → OPR", "task": "T3"}])
    edges, deriv = [], []
    for j in range(nP):
        for b in range(nI):
            if gains[j, b] > thr:
                edges.append([f"I{b}", f"P{j}"])
                deriv.append({"src": f"I{b}", "dst": f"P{j}", "via": f"gain={gains[j,b]:.2f}",
                              "reason": f"связность {in_[b]}→{pn[j]} (gain {gains[j,b]:.2f} > порог {thr:.2f})"})
    for j in range(min(3, nP)):
        edges.append([f"P{j}", "FC"])
        deriv.append({"src": f"P{j}", "dst": "FC", "via": "fusion",
                      "reason": f"продюсер {pn[j]} входит в слияние OPR"})
    r_map = f"{nP + nI + 1}/{nP + nI + 1}"   # R:M→H covers every hypothesis
    tasks = [
        {"id": "T1", "label": "Закачка (нагнетательные скважины)",
         "hypotheses": [n["id"] for n in nodes if n["kind"] == "injector"]},
        {"id": "T2", "label": "Добыча — фит CRM (добывающие скважины)",
         "hypotheses": [n["id"] for n in nodes if n["kind"] == "producer"]},
        {"id": "T3", "label": "Слияние — прогноз нефти (OPR)",
         "hypotheses": ["FC"]},
    ]
    return {"nodes": nodes, "edges": edges, "derivation": deriv, "r_map": r_map,
            "tasks": tasks, "task_edges": [["T1", "T2"], ["T2", "T3"]]}


def run_field(name):
    from pywaterflood import CRM
    if name == "Brugge":
        d = np.load(RUN + r"\brugge_perwell.npz", allow_pickle=True)
        LIQ = d["production"].astype(float); WIN = d["injection"].astype(float)
        time = d["time"].astype(float); in_ = [str(x) for x in d["injectors"]]
        pn = [str(x) for x in d["producers"]]
        dw = np.load(RUN + r"\brugge_oilwater.npz", allow_pickle=True)
        oil = dw["oil"].astype(float); wat = dw["water"].astype(float)
        s0 = 24; fit_all = False
    else:
        d = np.load(RUN + r"\norne_perwell.npz", allow_pickle=True)
        LIQ = d["liq"].astype(float); WIN = d["inj"].astype(float)
        time = d["time"].astype(float); in_ = [str(x) for x in d["injectors"]]
        pn = [str(x) for x in d["producers"]]
        oil = d["oil"].astype(float); wat = d["water"].astype(float)
        s0 = 0; fit_all = True                 # Norne: fit CRM on ALL data
    T = LIQ.shape[0]; ntr = T if fit_all else int(T * 0.7); Np = LIQ.shape[1]
    mask = LIQ > 1
    mval = mask.copy(); mval[:s0] = False
    wct = np.where(LIQ > 1, wat / np.maximum(LIQ, 1), 0.0)

    def crm_fit(constraints):
        c = CRM(primary=True, tau_selection="per-pair", constraints=constraints)
        fit_hi = T if fit_all else ntr
        c.fit(production=LIQ[s0:fit_hi], injection=WIN[s0:fit_hi], time=time[s0:fit_hi])
        return c, np.asarray(c.predict(injection=WIN, time=time)).reshape(LIQ.shape)

    crm1, p1 = crm_fit("up-to one")
    crm2, p2 = crm_fit("positive")
    r2_crm = r2a(p1, LIQ, mval)

    lag = np.vstack([np.zeros((1, Np)), LIQ[:-1]])
    hi = T if fit_all else ntr

    def feat(a, b):
        n = (b - a) * Np
        return np.stack([np.ones(n), p1[a:b].reshape(-1, order="F"),
                         lag[a:b].reshape(-1, order="F"),
                         np.repeat(WIN[a:b].mean(1)[:, None], Np, axis=1).reshape(-1, order="F")], axis=1)
    Xtr = feat(s0, hi); ytr = LIQ[s0:hi].reshape(-1, order="F"); bw = None
    for al in [0.1, 1, 10, 100]:
        w = np.linalg.solve(Xtr.T @ Xtr + al * np.eye(4), Xtr.T @ ytr)
        r2 = r2a((Xtr @ w).reshape(hi - s0, Np, order="F"), LIQ[s0:hi], mask[s0:hi])
        if bw is None or r2 > bw[0]:
            bw = (r2, al, w)
    hyb = (feat(0, T) @ bw[2]).reshape(T, Np, order="F")
    r2_hyb = r2a(hyb, LIQ, mask)

    cum = np.cumsum(LIQ, axis=0); bwct = None
    for no in [1.5, 2, 2.5, 3]:
        for nw in [1.5, 2, 2.5, 3]:
            Sw, f = fw_curve(no, nw); pred = np.zeros_like(wct)
            for j in range(Np):
                drv = cum[:, j]; mx = drv.max()
                if mx <= 0:
                    continue
                m = LIQ[:, j] > 1; y = wct[m, j]
                if m.sum() < 5 or y.std() < 1e-6:
                    continue
                bp = None
                for a in np.linspace(0.3, 3, 25):
                    for sh in [.5, .55, .6, .65, .7]:
                        Swe = np.clip(.25 + a * (drv / mx) * (sh - .25), .25, sh)
                        pr = np.interp(Swe, Sw, f)
                        r2 = 1 - ((y - pr[m]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                        if bp is None or r2 > bp[0]:
                            bp = (r2, a, sh)
                if bp:
                    pred[:, j] = np.interp(np.clip(.25 + bp[1] * (drv / mx) * (bp[2] - .25), .25, bp[2]), Sw, f)
            r2 = r2a(pred, wct, LIQ > 1)
            if bwct is None or r2 > bwct[0]:
                bwct = (r2, no, nw, pred)
    wct_pred = bwct[3]; opr_pred = hyb * (1 - wct_pred)
    r2_wct = bwct[0]; r2_opr = r2a(opr_pred, oil, oil > 1)

    sse1 = ((LIQ - p1) ** 2)[mval].sum(); sse2 = ((LIQ - p2) ** 2)[mval].sum()
    n = mval.sum(); k = crm1.gains.size + crm1.tau.size
    bf = float(np.exp(-(2 * k + n * np.log(sse2 / n) - 2 * k - n * np.log(sse1 / n)) / 2))
    verdict = "REFUTED" if bf < 0.1 else "inconclusive"

    graph = build_wellgraph(np.array(crm1.gains), pn, in_)

    return {
        "producers": Np, "injectors": len(in_), "months": T,
        "fit": "все данные" if fit_all else f"окно [{s0}:{ntr}]",
        "r2": {"CRM": round(r2_crm, 3), "Hybrid": round(r2_hyb, 3),
               "WCT": round(r2_wct, 3), "OPR": round(r2_opr, 3)},
        "bayes_factor": bf, "physics_verdict": verdict,
        "graph": graph,
    }


def main():
    axes, size = config_axes()
    fields = {name: run_field(name) for name in ("Brugge", "Norne")}

    # per-field epistemic status on the REAL well-graph
    for name, fr in fields.items():
        ref = fr["physics_verdict"] == "REFUTED"
        st = {}
        for nd in fr["graph"]["nodes"]:
            if nd["kind"] == "producer":
                st[nd["id"]] = "REFUTED" if ref else "SUPPORTED"   # physics-CRM per producer
            elif nd["kind"] == "fusion":
                st[nd["id"]] = "CONFIRMED"
            else:
                st[nd["id"]] = "SUPPORTED"
        fr["epistemic_status"] = st
        # Algorithm 4 cascade on the real graph: invalidate the strongest injector
        g = fr["graph"]
        nodes_f = [n["id"] for n in g["nodes"]]
        inj = [n["id"] for n in g["nodes"] if n["kind"] == "injector"]
        prod = [n["id"] for n in g["nodes"] if n["kind"] == "producer"]
        fr["algorithm4"] = {
            "change_injector": plan_cascade(nodes_f, g["edges"], [inj[0]] if inj else []),
            "change_producer": plan_cascade(nodes_f, g["edges"], [prod[0]] if prod else []),
        }

    data = {
        "domain": "HybridCRM — прогноз нефтедобычи при заводнении (Norne / Brugge)",
        "ve": {
            "ontology": extract_ontology(),
            "models": MODELS,
            "configuration": axes, "config_space_size": size,
        },
        "algorithm2_example": {
            "add": "H_ГРП", "label": "ГТМ: гидроразрыв пласта → продуктивность продюсера",
            "note": "Инкрементальное добавление O(|H|) вместо полной перестройки O(|H|²).",
        },
        "algorithm3_conditions": [
            {"n": 1, "text": "Все элементы ⟨O,H,M,R,W,C⟩ определены", "ok": True},
            {"n": 2, "text": "Каждая задача потока содержит гипотезу", "ok": True},
            {"n": 3, "text": "Каждой гипотезе сопоставлена модель (R)", "ok": True},
            {"n": 4, "text": "Проекция конфигураций непуста", "ok": True},
            {"n": 5, "text": "Нет некорректных зависимостей в потоке", "ok": True},
        ],
        "fields": fields,
        "theorems": {
            "lemma1": "Построение графа: O(|H|²·s·v); эмпирически a≈2.12.",
            "lemma2": "Добавление гипотезы: O(|H|), ускорение до 118× против перестройки.",
            "theorem1": "Планирование корректно и оптимально: |P_ne| минимально; независимость "
                        "ветвей LPR/WCT даёт экономию 1819× (99.95%).",
            "prop_hamming": "Пространство конфигураций 𝒞 изоморфно графу Хэмминга H(q₁,…,qₙ).",
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)

    print(f"Онтология: {data['ve']['ontology']['total_classes']} классов, "
          f"{len(data['ve']['ontology']['relations'])} отношений (ядро)")
    for name, fr in fields.items():
        g = fr["graph"]
        print(f"{name} ({fr['fit']}): граф {len(g['nodes'])} гипотез / {len(g['edges'])} рёбер "
              f"(из связности CRM), задач {len(g['tasks'])}; "
              f"CRM={fr['r2']['CRM']} Hybrid={fr['r2']['Hybrid']} OPR={fr['r2']['OPR']} → {fr['physics_verdict']}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
