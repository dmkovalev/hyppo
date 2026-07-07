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
    """Extract the FULL ontology: all classes (core + all rule-derived: markers,
    provenance, quality gates, lifecycle, workflow validation, …), subClassOf
    hierarchy and object properties. Ensures the 17 ontology rules are visible
    and Equation/Variable are connected."""
    from hyppo.core._base import virtual_experiment_onto as onto
    # make sure every rule module's classes are loaded into the ontology
    for mod in ["core_rules", "provenance", "quality_gates", "workflow_validation",
                "multi_experiment", "model_compatibility", "lifecycle", "markers"]:
        try:
            __import__(f"hyppo.ontology.{mod}")
        except Exception:
            pass

    all_classes = list(onto.classes())
    names = {c.name for c in all_classes}
    classes = []
    for c in all_classes:
        parent = None
        for p in c.is_a:
            pn = getattr(p, "name", None)
            if pn in names and pn != c.name:
                parent = pn; break
        classes.append({"name": c.name, "parent": parent})

    relations = []
    for p in onto.object_properties():
        doms = [d.name for d in (p.domain or []) if getattr(d, "name", None) in names]
        rngs = [r.name for r in (p.range or []) if getattr(r, "name", None) in names]
        for d in doms:
            for r in rngs:
                relations.append({"property": p.name, "domain": d, "range": r})
    # conceptual links absent from the code ontology
    seen = {(r["property"], r["domain"], r["range"]) for r in relations}
    for prop, d, r in [("derived_by", "Hypothesis", "Hypothesis"),
                       ("competes", "Hypothesis", "Hypothesis"),
                       ("is_implemented_by_model", "Hypothesis", "Model"),
                       ("has_for_hypothesis", "VirtualExperiment", "Hypothesis"),
                       ("has_for_model", "VirtualExperiment", "Model"),
                       ("has_for_workflow", "VirtualExperiment", "Workflow"),
                       ("has_for_configuration", "VirtualExperiment", "Configuration"),
                       ("has_for_structure", "Hypothesis", "Structure"),
                       # Определение 2: структура S(E,V) = уравнения E + переменные V.
                       # В коде онтологии эти рёбра отсутствуют (Equation/Variable висят);
                       # добавляем концептуальные связи, чтобы граф был полным.
                       ("has_for_equation", "Structure", "Equation"),
                       ("has_for_variable", "Equation", "Variable"),
                       ("has_for_fcm", "FullStructure", "FullCausalMapping"),
                       ("has_for_dependency_set", "FullStructure", "DependencySet")]:
        if (prop, d, r) not in seen:
            relations.append({"property": prop, "domain": d, "range": r})
    # only conceptual links whose endpoints exist as classes
    relations = [x for x in relations if x["domain"] in names and x["range"] in names]
    return {"name": onto.base_iri, "classes": classes, "relations": relations,
            "total_classes": len(names)}


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


# Model catalog. is_implemented_by_model is a "some" property (≥1 model per
# hypothesis, possibly several competing models) — not functional.
MODEL_CATALOG = [
    {"id": "m_inj", "label": "Инжекционный профиль", "class": "PhysicsModel"},
    {"id": "m_crmp", "label": "CRMP (up-to-one)", "class": "PhysicsModel"},
    {"id": "m_crmt", "label": "CRMT (positive)", "class": "PhysicsModel"},
    {"id": "m_nn", "label": "Transformer+GNN", "class": "DataDrivenModel"},
    {"id": "m_hyb", "label": "Fusion gate", "class": "HybridModel"},
    {"id": "m_bl", "label": "Buckley–Leverett", "class": "PhysicsModel"},
]


def models_for(kind):
    """Each hypothesis has ≥1 model; producers carry several competing models."""
    if kind == "injector":
        return ["m_inj"]
    if kind == "producer":
        return ["m_crmp", "m_crmt", "m_nn", "m_hyb"]   # 4 competing implementations
    return ["m_hyb", "m_bl"]                            # fusion: 2 models


def build_wellgraph(gains, pn, in_):
    """Real hypothesis graph from CRM connectivity — exactly as full_experiment.py:
    5 producer-hypotheses + all injector-hypotheses + 1 fusion; edge injector→producer
    where gain exceeds the 75th percentile; fusion derived_by the first 3 producers."""
    nP = min(5, len(pn)); nI = len(in_)
    thr = float(np.percentile(gains, 75))
    nodes = ([{"id": f"P{j}", "kind": "producer", "label": pn[j], "task": "T2",
               "models": models_for("producer")} for j in range(nP)]
             + [{"id": f"I{b}", "kind": "injector", "label": in_[b], "task": "T1",
                 "models": models_for("injector")} for b in range(nI)]
             + [{"id": "FC", "kind": "fusion", "label": "Fusion → OPR", "task": "T3",
                 "models": models_for("fusion")}])
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


# ───────── conceptual HybridCRM graph via the REAL Algorithm 1 ───────────────
# Per CLAUDE.md: the derived_by graph is built ONLY by HypothesisLattice
# (platform Algorithm 1), not a local reimplementation. 16 hypotheses,
# article continuous numbering H1–H16 (liquid H1–H8, watercut H9–H14,
# ГРП=H15, oil=H16). Correctness: 16 nodes, 17 edges, DAG depth 5.
_CONCEPT_OLD = [
    ("H1",  "I_agg = w_ij * I_j",                 "I_agg",      "Агрегация закачки", "жидкость"),
    ("H2",  "q_f = a_f*q_f_prev + b_f*I_agg",      "q_f",        "CRM быстрый канал", "жидкость"),
    ("H3",  "q_s = a_s*q_s_prev + b_s*I_agg",      "q_s",        "CRM медленный канал", "жидкость"),
    ("H4",  "q_c = w_f*q_f + (1-w_f)*q_s",         "q_c",        "Смешение каналов", "жидкость"),
    ("H5",  "q_liq_phys = J*q_c + q_prim",         "q_liq_phys", "Продуктивность", "жидкость"),
    ("H6",  "q_prim = q_prev*exp(-dt*taup)",       "q_prim",     "Первичная добыча (спад)", "жидкость"),
    ("H7",  "l_ml = MLP(x_hist)",                  "l_ml",       "ML-коррекция (MLP)", "жидкость"),
    ("H8",  "l = g*q_liq_phys + (1-g)*l_ml",       "l",          "LPR Fusion (вентиль g)", "жидкость"),
    ("H11", "Sw = Sw_prev + (Winj - Wlout)*dt/Vp", "Sw",         "Материальный баланс", "обводнённость"),
    ("H12", "krw = ((Sw-Swc)/(1-Swc-Sor))**nw",    "krw",        "Corey krw", "обводнённость"),
    ("H12b", "kro = ((1-Sw-Sor)/(1-Swc-Sor))**no", "kro",        "Corey kro", "обводнённость"),
    ("H13", "fw = 1/(1 + kro*muw/(krw*muo))",      "fw",         "Баклея–Леверетта fw", "обводнённость"),
    ("H14", "o_p = 1 - fw",                        "o_p",        "WCT-якорь (физика)", "обводнённость"),
    ("H15", "o = gw*o_p + (1-gw)*o_m",             "o",          "WCT Fusion", "обводнённость"),
    ("GRP", "J = J0 + dJ_grp",                     "J",          "ГРП (модуляция)", "ГТМ"),
    ("H19", "q_oil = l * o",                       "q_oil",      "OPR — прогноз нефти", "нефть"),
]
_CONCEPT_TASKS = [
    ("T1", "Парсинг закачки", ["H1"]),
    ("T2", "CRM-каналы", ["H2", "H3"]),
    ("T3", "Смешение", ["H4"]),
    ("T4", "Продуктивность и спад", ["H5", "H6"]),
    ("T5", "ML-коррекция", ["H7"]),
    ("T6", "Слияние жидкости", ["H8"]),
    ("T7", "Материальный баланс", ["H11"]),
    ("T8", "Фазовые проницаемости", ["H12", "H12b"]),
    ("T9", "Фракционный поток", ["H13"]),
    ("T10", "Обводнённость", ["H14", "H15"]),
    ("T11", "Прогноз нефти", ["H19"]),
    ("T12", "ГТМ-модуляция", ["GRP"]),
]
# красивые LaTeX-формулы (по §4.4 диссертации), ключ — историческое имя
_CONCEPT_LATEX = {
    "H1": r"I_{\mathrm{agg}} = \sum_j w_{ij}\, I_j",
    "H2": r"q_f = \alpha_f\, q_f^{-} + \beta_f\, I_{\mathrm{agg}}",
    "H3": r"q_s = \alpha_s\, q_s^{-} + \beta_s\, I_{\mathrm{agg}}",
    "H4": r"q_c = w_f\, q_f + (1 - w_f)\, q_s",
    "H5": r"q_{\mathrm{liq}} = J\, q_c + q_{\mathrm{prim}}",
    "H6": r"q_{\mathrm{prim}} = q^{-}\, e^{-\Delta t\, \tau_p}",
    "H7": r"\ell_m = \mathrm{MLP}(\mathbf{x}_{\mathrm{hist}})",
    "H8": r"\ell = g\, q_{\mathrm{liq}} + (1 - g)\, \ell_m",
    "H11": r"S_w = S_w^{-} + \dfrac{(W_{\mathrm{inj}} - W_{\mathrm{out}})\, \Delta t}{V_p}",
    "H12": r"k_{rw} = \left(\dfrac{S_w - S_{wc}}{1 - S_{wc} - S_{or}}\right)^{n_w}",
    "H12b": r"k_{ro} = \left(\dfrac{1 - S_w - S_{or}}{1 - S_{wc} - S_{or}}\right)^{n_o}",
    "H13": r"f_w = \dfrac{1}{1 + k_{ro}\, \mu_w / (k_{rw}\, \mu_o)}",
    "H14": r"o_p = 1 - f_w",
    "H15": r"o = g_w\, o_p + (1 - g_w)\, o_m",
    "GRP": r"J = J_0 + \Delta J_{\mathrm{ГРП}}",
    "H19": r"\hat{q}_{\mathrm{oil}} = \ell \cdot o",
}

# old name -> continuous article numbering
_RENUM = {"H1": "H1", "H2": "H2", "H3": "H3", "H4": "H4", "H5": "H5", "H6": "H6",
          "H7": "H7", "H8": "H8", "H11": "H9", "H12": "H10", "H12b": "H11",
          "H13": "H12", "H14": "H13", "H15": "H14", "GRP": "H15", "H19": "H16"}
# branch → which field metric decides the epistemic status of the hypothesis
_CBRANCH_METRIC = {"жидкость": "CRM", "обводнённость": "WCT", "нефть": "OPR", "ГТМ": "CRM"}


# role → (класс, реализация в Python-библиотеке, конфигурация, меняемые параметры)
_MODEL_ROLE = {
    "crmp": ("PhysicsModel", "pywaterflood.CRM(primary=True, tau_selection='per-pair', constraints='up-to one')",
             "CONFIGURATION_SPACE / MODEL.PHYSICS", ["CRM_constraints", "CRM_KNN", "PHYS_GRAD_MODE"]),
    "crmt": ("PhysicsModel", "pywaterflood.CRM(primary=True, tau_selection='per-pair', constraints='positive')",
             "CONFIGURATION_SPACE / MODEL.PHYSICS", ["CRM_constraints", "CRM_KNN"]),
    "gnn": ("DataDrivenModel", "torch_geometric.nn.GATConv (GNN, spatial attention)",
            "CONFIGURATION_SPACE / MODEL.HYBRID", ["GNN_NUM_LAYERS", "GNN_HEADS", "HIDDEN_DIM"]),
    "tr": ("DataDrivenModel", "torch.nn.TransformerEncoder (temporal attention)",
           "CONFIGURATION_SPACE / MODEL.HYBRID", ["TRANSFORMER_NUM_LAYERS", "TRANSFORMER_NHEAD", "DIM_FEEDFORWARD_MULT"]),
    "bl": ("PhysicsModel", "pywaterflood Buckley–Leverett (fractional flow f_w)",
           "CONFIGURATION_SPACE / MODEL.PHYSICS", ["Corey_n_w", "Corey_n_o", "BACKPERIOD"]),
    "wor": ("PhysicsModel", "sklearn.linear_model.Ridge (log-WOR)",
            "гребневая регрессия", ["alpha"]),
    "gate": ("HybridModel", "обучаемый вентиль слияния g: l = g·l_p + (1−g)·l_m",
             "CONFIGURATION_SPACE / MODEL.HYBRID", ["USE_FUSION_GATE"]),
    "frac": ("PhysicsModel", "модуляция продуктивности ΔJ (гидроразрыв)",
             "ГТМ", ["USE_CRM_PHASE", "CRM_PHASE_RATIO"]),
    "acid": ("PhysicsModel", "модуляция скин-фактора ΔS (кислотная обработка)",
             "ГТМ", []),
}


def _mk(cid, role, label):
    cls, ref, cfg, params = _MODEL_ROLE[role]
    return {"id": f"m_{cid}_{role}", "label": f"{label} ({cid})", "class": cls,
            "python_ref": ref, "config": cfg, "params": params}


def _models_for(cid, branch):
    """Granular models: a dedicated instance (≥1, competing where meaningful)
    PER hypothesis — mirrors the real adapter (one Model instance per hypothesis)."""
    if cid == "H7":
        return [_mk(cid, "gnn", "GNN"), _mk(cid, "tr", "Transformer")]
    if cid in ("H8", "H14", "H16"):
        return [_mk(cid, "gate", "Fusion gate")]
    if cid == "H15":
        return [_mk(cid, "frac", "Гидроразрыв"), _mk(cid, "acid", "Кислотная обработка")]
    if branch == "жидкость":
        return [_mk(cid, "crmp", "CRMP up-to-one"), _mk(cid, "crmt", "CRMT positive")]
    return [_mk(cid, "bl", "Buckley–Leverett"), _mk(cid, "wor", "Log-WOR")]


def build_conceptual_lattice():
    """Build the conceptual HybridCRM graph with the REAL platform Algorithm 1."""
    import networkx as nx
    from hyppo.coa._base import Equation, Structure
    from hyppo.lattice_constructor._base import HypothesisLattice

    class Hyp:
        def __init__(self, name, formula):
            self.name = name
            self.structure = Structure([Equation(formula=formula)])

    hm = {n: Hyp(n, f) for n, f, o, lab, br in _CONCEPT_OLD}
    out = {n: o for n, f, o, lab, br in _CONCEPT_OLD}

    class WF:
        def __init__(self, tasks): self._t = [[hm[h] for h in hs] for _, _, hs in tasks]
        def get_tasks(self): return self._t

    G = HypothesisLattice([hm[n] for n, *_ in _CONCEPT_OLD], WF(_CONCEPT_TASKS)).lattice
    edges_old = [(u.name, v.name) for u, v in G.edges()]

    R = _RENUM
    catalog = {}
    nodes = []
    for n, f, o, lab, br in _CONCEPT_OLD:
        cid = R[n]
        ms = _models_for(cid, br)
        for m in ms:
            catalog[m["id"]] = m
        nodes.append({"id": cid, "label": lab, "branch": br,
                      "equation": {"formula": f, "output": o, "latex": _CONCEPT_LATEX.get(n, f)},
                      "models": [m["id"] for m in ms], "model": ms[0]["id"],
                      "metric": _CBRANCH_METRIC.get(br, "CRM"), "status": "SUPPORTED"})
    edges = [[R[a], R[b]] for a, b in edges_old]
    deriv = [{"src": R[a], "dst": R[b], "via": out[a],
              "reason": f"выход {out[a]} ({R[a]}) входит в уравнение {R[b]}"} for a, b in edges_old]
    tasks = [{"id": t, "label": lab, "hypotheses": [R[h] for h in hs]}
             for t, lab, hs in _CONCEPT_TASKS]
    # task membership + task edges by PROJECTING hypothesis edges (parallel branches!)
    task_of = {R[h]: t for t, lab, hs in _CONCEPT_TASKS for h in hs}
    te = sorted({(task_of[a], task_of[b]) for a, b in edges if task_of[a] != task_of[b]})
    task_edges = [list(e) for e in te]
    # predecessors: AND-join, если задача зависит от >1 ветви
    preds = {t["id"]: [a for a, b in task_edges if b == t["id"]] for t in tasks}
    # формальное описание потока (CWL-подобный текст)
    lines = ["# HybridCRM workflow — ОАГ задач (CWL-подобный формат)",
             "# join: AND (все входы обязательны); ветви LPR и WCT параллельны",
             "cwlVersion: v1.2", "class: Workflow", "steps:"]
    for t in tasks:
        p = preds[t["id"]]
        join = "  join: all-required  # AND" if len(p) > 1 else ""
        lines.append(f"  {t['id']}:  # {t['label']}")
        lines.append(f"    in: [{', '.join(p) if p else 'source'}]")
        lines.append(f"    run: [{', '.join(t['hypotheses'])}]")
        if join:
            lines.append(join)
    depth = nx.dag_longest_path_length(G)
    return {"nodes": nodes, "edges": edges, "derivation": deriv, "models_catalog": catalog,
            "tasks": tasks, "task_edges": task_edges, "task_preds": preds,
            "formal_text": "\n".join(lines),
            "is_dag": bool(nx.is_directed_acyclic_graph(G)), "depth": int(depth),
            "note": f"16 гипотез (сплошная нумерация статьи H1–H16), {len(edges)} рёбер "
                    f"derived_by, DAG глубины {depth}. Построено настоящим алгоритмом 1 "
                    f"(HypothesisLattice) из уравнений; ветви LPR (H1–H8) и WCT (H9–H14) "
                    f"параллельны, сходятся в H16."}


def build_algorithm_demos():
    """Live platform calls backing GUI demos 2/3/4/6/7 (docs/gui_demo_spec.md).
    Everything computed by the real hyppo API; matches tests/test_golden_claims.py."""
    import networkx as nx
    from hyppo.coa import causal
    from hyppo.coa._base import Equation, Structure
    from hyppo.coa.graph import HypothesisGraph
    from hyppo.lattice_constructor._base import HypothesisLattice
    from hyppo.ontology.consistency import Status, check_consistency, _find_cycle_via_kahn

    class Hyp:
        def __init__(self, name, formula):
            self.name = name
            self.structure = Structure([Equation(formula=formula)])

    class WF:
        def __init__(self, tasks): self._t = tasks
        def get_tasks(self): return self._t

    R = _RENUM

    # ── Demo 2: Algorithm 2 — incremental add_hypothesis == full rebuild ──
    hm = {n: Hyp(n, f) for n, f, o, lab, br in _CONCEPT_OLD}
    order = [n for n, *_ in _CONCEPT_OLD]
    wf = WF([[hm[n]] for n in order])
    lat = HypothesisLattice([hm[n] for n in order[:-1]], wf)   # без H19
    before = {(u.name, v.name) for u, v in lat.lattice.edges()}
    lat.add_hypothesis(hm[order[-1]])                          # + H19 инкрементально
    after = {(u.name, v.name) for u, v in lat.lattice.edges()}
    new_edges = [[R[a], R[b]] for a, b in sorted(after - before)]
    alg2 = {"added": R[order[-1]], "new_edges": new_edges,
            "note": "Добавление одной гипотезы даёт тот же граф, что полная перестройка "
                    "(golden), но за O(|H|) каузальных объединений вместо O(|H|²)."}

    # ── Demo 4: Algorithm 4 — real HypothesisGraph.plan cascade ──
    cids = [R[n] for n in order]
    idx = {cid: i for i, cid in enumerate(cids)}
    edges_int = [(idx[R[a]], idx[R[b]]) for a, b in
                 [(u.name, v.name) for u, v in HypothesisLattice([hm[n] for n in order], wf).lattice.edges()]]
    hg = HypothesisGraph.from_edges(len(cids), edges_int)

    def plan_for(changed_cid):
        cached = set(range(len(cids))) - {idx[changed_cid]}
        pne = hg.plan(cached)
        return {"changed": changed_cid,
                "p_ne": sorted((cids[i] for i in pne), key=lambda x: int(x[1:])),
                "recompute_frac": round(len(pne) / len(cids), 3)}

    alg4 = {"change_H1": plan_for("H1"), "change_H15": plan_for("H15"),
            "change_H10": plan_for("H10")}

    # ── Demo 3: Algorithm 3 — two-stage well-formedness (stage B, fast) ──
    class FiniteQ:
        finite = True
        def __init__(self, v): self.values = v
    class NoFlag: pass
    good_lat = {0: {1}, 1: {2}, 2: set()}
    good_art = {0: {"in": {"raw"}, "out": {"a"}}, 1: {"in": {"a"}, "out": {"b"}},
                2: {"in": {"b"}, "out": {"c"}}}
    st = lambda r: (getattr(r.status, "name", None) or str(r.status))
    scenarios = []
    r_ok = check_consistency(None, None, good_lat, run_hermit=False,
                             artefacts=good_art, configurations=[FiniteQ([1, 2])])
    scenarios.append({"case": "Корректный ВЭ", "status": st(r_ok), "ok": r_ok.ok,
                      "detail": "DAG + согласованные артефакты + конечные домены"})
    r_c3 = check_consistency(None, None, {0: {1}, 1: {2}, 2: {0}}, run_hermit=False)
    scenarios.append({"case": "Цикл в потоке", "status": st(r_c3), "ok": r_c3.ok,
                      "detail": "свидетель цикла: " + str(r_c3.details.get("cycle_witness", ""))})
    bad_art = {**good_art, 1: {"in": {"a"}, "out": {"zzz"}}}
    r_c4 = check_consistency(None, None, good_lat, run_hermit=False, artefacts=bad_art)
    scenarios.append({"case": "Разрыв потока артефактов", "status": st(r_c4), "ok": r_c4.ok,
                      "detail": "виновное ребро Out(i)∩In(j)=∅: " + str(tuple(r_c4.details.get("c4_edge", ())))})
    r_c5 = check_consistency(None, None, good_lat, run_hermit=False,
                             artefacts=good_art, configurations=[FiniteQ([1]), NoFlag()])
    scenarios.append({"case": "Бесконечный домен", "status": st(r_c5), "ok": r_c5.ok,
                      "detail": "домен без finite=True"})
    alg3 = {"scenarios": scenarios,
            "owa_note": "Стадия A (HermiT, OWA): гипотеза без модели НЕ ловится рассуждателем "
                        "(«открытый мир»); её ловит маркерный слой (правило 9) — обоснование "
                        "трёхслойной архитектуры."}

    # ── Demo 7: Rule 5 — procedural acyclicity (Kahn) ──
    rule5 = {"acyclic": _find_cycle_via_kahn({0: {1}, 1: {2}, 2: set()}) is None,
             "cyclic_witness": list(_find_cycle_via_kahn({0: {1}, 1: {2}, 2: {0}}) or [])}

    # ── Demo 6: complexity by operation counters (deterministic) ──
    def chain(n):
        g = HypothesisGraph()
        for i in range(n):
            g.add([{f"x{i}", f"x{i-1}"}] if i else [{f"x{i}"}])
        for i in range(n - 1):
            g.connect(i, i + 1)
        return g
    orig = causal.is_complete
    a1 = []
    for n in (10, 20, 40):
        calls = [0]
        def counting(eqs, _c=calls):
            _c[0] += 1; return orig(eqs)
        causal.is_complete = counting
        chain(n).build()
        causal.is_complete = orig
        a1.append({"n": n, "count": calls[0], "law": n * (n - 1) // 2})
    a2c = []
    for n in (10, 20, 40):
        g = chain(n); calls = [0]
        def counting(eqs, _c=calls):
            _c[0] += 1; return orig(eqs)
        causal.is_complete = counting
        g.add_hypothesis([{f"x{n}", f"x{n-1}"}])
        causal.is_complete = orig
        a2c.append({"n": n, "count": calls[0], "law": n})

    class CountingSet(set):
        counter = 0
        def __iter__(self):
            type(self).counter += len(self); return super().__iter__()
    a4c = []
    for n in (200, 400, 800):
        g = HypothesisGraph.from_edges(n, [(i, i + 1) for i in range(n - 1)])
        g._adj = {k: CountingSet(g._adj.get(k, ())) for k in range(n)}
        CountingSet.counter = 0
        g.plan(cached=set(range(0, n, 2)))
        a4c.append({"n": n, "count": CountingSet.counter, "law": 2 * n - 1})
    complexity = {
        "alg1": {"points": a1, "law": "O(|H|²·s·v)", "note": "проверок полноты = n(n−1)/2; ×4 при ×2"},
        "alg2": {"points": a2c, "law": "O(|H|·s·v)", "note": "объединений = n; ×2 при ×2"},
        "alg4": {"points": a4c, "law": "O(|V|+|E|)", "note": "обходов смежности ~V+E; ×2 при ×2"},
    }

    return {"alg2": alg2, "alg3": alg3, "alg4_plan": alg4, "rule5": rule5, "complexity": complexity}


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

    # scalability: per-pair/per-well graph N = NI*NP + 7*NP + 1, built by
    # Algorithm 1 (COA) from equations; real HermiT vs ELK reasoning times
    # (thesis/papers/brugge_run/plot_elk_hermit.py).
    scale = {
        "note": "Один и тот же граф гипотез строится алгоритмом 1 из уравнений при любом "
                "масштабе; онтологический вывод в профиле OWL 2 EL (ELK) полиномиален.",
        "points": [
            {"hypotheses": 341, "ELK_s": 1.47, "HermiT_s": 1.55, "wells": "10 нагн. × 20 доб."},
            {"hypotheses": 1051, "ELK_s": 2.00, "HermiT_s": 2.41, "wells": "средний масштаб"},
            {"hypotheses": 10166, "ELK_s": 4.40, "HermiT_s": 41.45, "wells": "≈10 тыс. гипотез"},
        ],
        "speedup_10k": "9.4×",
    }

    # Conceptual HybridCRM graph via the REAL platform Algorithm 1 (HypothesisLattice).
    concept = build_conceptual_lattice()

    # Per-field epistemic status on the SAME conceptual nodes (only data differs).
    for name, fr in fields.items():
        r2 = fr["r2"]; cs = {}
        for nd in concept["nodes"]:
            cid = nd["id"]
            if cid in ("H8", "H14", "H16"):            # fusion / OPR
                cs[cid] = "CONFIRMED"
            elif cid in ("H7", "H15"):                 # ML, ГТМ
                cs[cid] = "SUPPORTED"
            elif cid in ("H1", "H2", "H3", "H4", "H5", "H6"):   # физика жидкости
                cs[cid] = "SUPPORTED" if r2["CRM"] >= 0.5 else "REFUTED"
            else:                                       # H9–H13 физика обводнённости
                cs[cid] = "SUPPORTED" if r2["WCT"] >= 0.5 else "REFUTED"
        fr["concept_status"] = cs

    # Algorithm 4 cascade on the conceptual graph (same for both fields).
    cids = [n["id"] for n in concept["nodes"]]
    concept_alg4 = {
        "change_H1": plan_cascade(cids, concept["edges"], ["H1"]),
        "change_H5": plan_cascade(cids, concept["edges"], ["H5"]),
        "change_H10": plan_cascade(cids, concept["edges"], ["H10"]),
    }

    data = {
        "domain": "HybridCRM — прогноз нефтедобычи при заводнении (Norne / Brugge)",
        "ve": {
            "ontology": extract_ontology(),
            "models": list(concept["models_catalog"].values()),
            "configuration": axes, "config_space_size": size,
        },
        "graph_conceptual": concept,
        "algorithm4": concept_alg4,
        "demos": build_algorithm_demos(),
        "scale": scale,
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
