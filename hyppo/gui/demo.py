import json

DEMO_NAME = "norne-brugge"

# ————— Полный граф HybridCRM (19 гипотез), part4.tex §4.4 —————
# Ветвь A (жидкость, LPR): H1–H10. Ветвь B (обводнённость, WCT): H11–H18.
# H19 (OPR) объединяет обе ветви. Топология идентична synthetic_honest.py.
_EDGES = [
    ["H1", "H2"], ["H1", "H3"], ["H2", "H4"], ["H3", "H4"], ["H4", "H5"],
    ["H5", "H6"], ["H5", "H7"], ["H5", "H8"], ["H6", "H9"], ["H7", "H9"],
    ["H8", "H10"], ["H9", "H10"],                        # ветвь A (LPR)
    ["H11", "H12"], ["H12", "H14"], ["H13", "H16"], ["H14", "H16"],
    ["H15", "H16"], ["H16", "H17"], ["H17", "H18"],      # ветвь B (WCT)
    ["H10", "H19"], ["H18", "H19"],                      # слияние → OPR
]

_LABELS = {
    "H1": "Агрегация закачки", "H2": "CRM-переток", "H3": "Первичная добыча",
    "H4": "CRM-жидкость", "H5": "Признаки ML", "H6": "ML-регрессия",
    "H7": "ML-прогноз жидкости", "H8": "LPR-якорь (физика)", "H9": "ML-ветвь жидкости",
    "H10": "LPR Fusion (вентиль)", "H11": "Насыщенность", "H12": "Прорыв воды (Buckley–Leverett)",
    "H13": "WOR-эмпирика", "H14": "BL-обводнённость", "H15": "Признаки WCT",
    "H16": "BL/WOR смешение", "H17": "WCT-регрессия", "H18": "WCT Fusion",
    "H19": "OPR — прогноз нефти",
}
_BRANCH = {**{f"H{i}": "LPR (жидкость)" for i in range(1, 11)},
           **{f"H{i}": "WCT (обводнённость)" for i in range(11, 19)},
           "H19": "OPR (слияние)"}

# 15 гиперпараметров (part4.tex, табл. C-пространства): 9 бинарных, 5 троичных,
# 1 четверичный → |C| = 2^9·3^5·4 = 497664 (до ограничения C1).
_PARAMS = {
    "H1": {"crm_constraint": ["bounded", "free"]},
    "H2": {"transmissibility": ["static", "dynamic"]},
    "H4": {"primary_regime": ["decline", "constant"]},
    "H5": {"features": ["base", "extended", "full"]},
    "H6": {"regressor": ["ridge", "gbm", "mlp"]},
    "H8": {"anchor": ["physical", "corrected"]},
    "H10": {"gate": ["learned", "fixed"]},
    "H11": {"saturation": ["corey", "loglinear", "spline"]},
    "H12": {"bl_form": ["classic", "extended"]},
    "H13": {"wor_fit": ["linear", "power", "log"]},
    "H16": {"blend": ["bl", "wor", "mixed", "adaptive"]},
    "H17": {"wct_reg": ["ridge", "gbm"]},
    "H18": {"wct_gate": ["learned", "fixed"]},
    "H19": {"combine": ["product", "weighted"]},
    "H7": {"ml_horizon": ["short", "long"]},
}

# Немного статусного разнообразия для наглядной раскраски графа.
_STATUS = {"H9": "SUPERSEDED", "H13": "SUPERSEDED"}


def _hypotheses():
    hs = []
    for i in range(1, 20):
        hid = f"H{i}"
        hs.append({
            "id": hid,
            "label": _LABELS[hid],
            "params": _PARAMS.get(hid, {}),
            "epistemic_status": _STATUS.get(hid, "SUPPORTED"),
            "description": f"Ветвь: {_BRANCH[hid]}.",
        })
    return hs


# Модели M и отображение R: M → H (каждая гипотеза реализована моделью).
_MODEL_OF = {
    **{f"H{i}": "m_crm" for i in [1, 2, 3, 4]},
    **{f"H{i}": "m_ml" for i in [5, 6, 7, 9]},
    **{f"H{i}": "m_hyb" for i in [8, 10]},
    **{f"H{i}": "m_bl" for i in [11, 12, 13, 14, 16]},
    **{f"H{i}": "m_wctml" for i in [15, 17]},
    "H18": "m_fusion", "H19": "m_opr",
}
_MODELS = [
    {"id": "m_crm", "label": "CRM (pywaterflood)", "kind": "аналитическая"},
    {"id": "m_ml", "label": "Gradient Boosting", "kind": "ML"},
    {"id": "m_hyb", "label": "Hybrid gate", "kind": "гибрид"},
    {"id": "m_bl", "label": "Buckley–Leverett", "kind": "аналитическая"},
    {"id": "m_wctml", "label": "Ridge WCT", "kind": "ML"},
    {"id": "m_fusion", "label": "Fusion gate", "kind": "гибрид"},
    {"id": "m_opr", "label": "OPR combiner", "kind": "композиция"},
]


def _build_ve():
    hyps = _hypotheses()
    for m in _MODELS:
        m["implements"] = next((h for h, mm in _MODEL_OF.items() if mm == m["id"]), "")
    mapping = [{"model": _MODEL_OF[h["id"]], "hypothesis": h["id"]} for h in hyps]
    configuration = [
        {"hypothesis": hid, "axis": ax, "levels": lv}
        for hid, pmap in _PARAMS.items() for ax, lv in pmap.items()
    ]
    return {
        "domain": "HybridCRM — прогноз нефтедобычи при заводнении (Norne / Brugge)",
        "ontology": {
            "name": "PetroReservoirOntology",
            "iri": "http://synthesis.ipi.ac.ru/petro.owl",
            "classes": [
                {"name": "Reservoir", "label": "Пласт-коллектор"},
                {"name": "Well", "label": "Скважина"},
                {"name": "Injector", "label": "Нагнетательная скважина", "subclass_of": "Well"},
                {"name": "Producer", "label": "Добывающая скважина", "subclass_of": "Well"},
                {"name": "Hypothesis", "label": "Гипотеза"},
                {"name": "Model", "label": "Модель, реализующая гипотезу"},
            ],
            "object_properties": [
                {"name": "injects_into", "domain": "Injector", "range": "Producer"},
                {"name": "is_implemented_by_model", "domain": "Hypothesis", "range": "Model"},
                {"name": "derived_by", "domain": "Hypothesis", "range": "Hypothesis",
                 "characteristics": ["Transitive", "Asymmetric"]},
                {"name": "competes", "domain": "Hypothesis", "range": "Hypothesis",
                 "characteristics": ["Symmetric"]},
            ],
            "data_properties": [
                {"name": "has_permeability", "range": "float"},
                {"name": "has_watercut", "range": "float"},
                {"name": "has_epistemic_status", "range": "EpistemicStatus"},
            ],
        },
        "hypotheses": hyps,
        "models": _MODELS,
        "mapping": mapping,
        "workflow_edges": _EDGES,
        "configuration": configuration,
        "constraint_note": "Ограничение C1 (делимость скрытого слоя на число голов) "
                           "сокращает |C| с 497664 до 359424.",
    }


_DEMO_VE = _build_ve()

# Две предзапущенные итерации (Обзор не пустой). Каскад по H12 (прорыв воды).
_DEMO_ITERATIONS = [
    {
        "results": {h: {"status": "SUCCESS", "metrics": {"r2": 0.71},
                        "epistemic_status": "SUPPORTED"} for h in [f"H{i}" for i in range(1, 20)]},
        "reused": 0, "best": {"hypothesis": "H19", "r2": 0.71},
        "note": "Базовая линия — полный расчёт всех 19 гипотез.",
    },
    {
        "results": {h: {"status": "SUCCESS", "metrics": {"r2": 0.86 if h == "H19" else 0.74},
                        "epistemic_status": "SUPPORTED"} for h in [f"H{i}" for i in range(1, 20)]},
        "reused": 14,
        "best": {"hypothesis": "H19", "r2": 0.86},
        "note": "Изменён прорыв воды (H12): пересчитаны 5 из 19, остальные из кэша.",
    },
]


def seed_demo(store) -> None:
    if any(p["name"] == DEMO_NAME for p in store.list()):
        return
    pid = store.create(name=DEMO_NAME,
                       description="HybridCRM · 19 гипотез · Norne/Brugge")
    store.save_ve(pid, json.dumps(_DEMO_VE))
    for it in _DEMO_ITERATIONS:
        store.add_iteration(pid, json.dumps(it))
