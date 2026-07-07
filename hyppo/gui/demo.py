import json

DEMO_NAME = "norne-brugge"

# ————— Канонический граф гибридной модели (16 гипотез H1–H16) —————
# Совпадает с real_data.json / golden-тестом / статьёй: ветвь жидкости H1–H8,
# ветвь обводнённости H9–H14, ГТМ H15 (модуляция), H16 (OPR — прогноз нефти).
# 18 рёбер derived_by, DAG глубины 10 (после связки H8→H9). Рёбра выведены из
# потока переменных уравнений Алгоритмом 1 (причинное упорядочение).
_EDGES = [
    ["H1", "H2"], ["H1", "H3"], ["H2", "H4"], ["H3", "H4"], ["H4", "H5"],
    ["H6", "H5"], ["H7", "H8"], ["H5", "H8"],            # ветвь жидкости
    ["H15", "H5"],                                        # ГТМ: модуляция продуктивности
    ["H8", "H9"],                                         # межветвевое: прогноз жидкости → мат. баланс
    ["H9", "H10"], ["H9", "H11"], ["H10", "H12"], ["H11", "H12"],
    ["H12", "H13"], ["H13", "H14"],                       # ветвь обводнённости
    ["H8", "H16"], ["H14", "H16"],                        # слияние → OPR
]

_LABELS = {
    "H1": "Агрегация закачки", "H2": "CRM быстрый канал", "H3": "CRM медленный канал",
    "H4": "Смешение каналов", "H5": "Продуктивность", "H6": "Первичная добыча (спад)",
    "H7": "ML-коррекция (MLP)", "H8": "LPR Fusion (вентиль g)", "H9": "Материальный баланс",
    "H10": "Corey krw", "H11": "Corey kro", "H12": "Баклея–Леверетта fw",
    "H13": "WCT-якорь (физика)", "H14": "WCT Fusion", "H15": "ГРП (модуляция)",
    "H16": "OPR — прогноз нефти",
}
_BRANCH = {**{f"H{i}": "LPR (жидкость)" for i in range(1, 9)},
           **{f"H{i}": "WCT (обводнённость)" for i in range(9, 15)},
           "H15": "ГТМ (модуляция)", "H16": "OPR (слияние)"}

# конкурирующие гипотезы: физика (H5) против ML (H7) для дебита жидкости
_COMPETES = {"H5": ["H7"], "H7": ["H5"]}

# Гиперпараметры C-пространства (16 осей): 13 бинарных + 3 троичных
# → |C| = 2^13 · 3^3 = 221184 (до ограничения C1 на совместимость моделей ветвей).
_PARAMS = {
    "H1": {"crm_constraint": ["bounded", "free"]},
    "H2": {"tau_fast": ["static", "dynamic"]},
    "H3": {"tau_slow": ["static", "dynamic"]},
    "H4": {"channel_mix": ["linear", "weighted"]},
    "H5": {"productivity": ["physical", "corrected"]},
    "H6": {"decline": ["exp", "harmonic"]},
    "H7": {"ml_model": ["gnn", "transformer", "mlp"]},
    "H8": {"lpr_gate": ["learned", "fixed"]},
    "H9": {"saturation": ["corey", "spline"]},
    "H10": {"corey_nw": ["2", "3", "4"]},
    "H11": {"corey_no": ["2", "3", "4"]},
    "H12": {"bl_form": ["classic", "extended"]},
    "H13": {"wct_anchor": ["physical", "empirical"]},
    "H14": {"wct_gate": ["learned", "fixed"]},
    "H15": {"gtm_type": ["frac", "acid"]},
    "H16": {"combine": ["product", "weighted"]},
}

# Немного статусного разнообразия для наглядной раскраски графа.
_STATUS = {"H15": "PROPOSED"}


def _hypotheses():
    hs = []
    for i in range(1, 17):
        hid = f"H{i}"
        hs.append({
            "id": hid,
            "label": _LABELS[hid],
            "params": _PARAMS.get(hid, {}),
            "epistemic_status": _STATUS.get(hid, "SUPPORTED"),
            "competes": _COMPETES.get(hid, []),
            "description": f"Ветвь: {_BRANCH[hid]}.",
        })
    return hs


# Модели M и отображение R: M → H (каждая гипотеза реализована моделью).
_MODEL_OF = {
    **{f"H{i}": "m_crm" for i in [1, 2, 3, 4, 5, 6]},
    "H7": "m_ml",
    **{f"H{i}": "m_hyb" for i in [8, 14]},
    **{f"H{i}": "m_bl" for i in [9, 10, 11, 12, 13]},
    "H15": "m_gtm", "H16": "m_opr",
}
_MODELS = [
    {"id": "m_crm", "label": "CRM (pywaterflood)", "kind": "аналитическая"},
    {"id": "m_ml", "label": "Transformer + GNN", "kind": "ML"},
    {"id": "m_bl", "label": "Buckley–Leverett", "kind": "аналитическая"},
    {"id": "m_hyb", "label": "Fusion gate (вентиль)", "kind": "гибрид"},
    {"id": "m_gtm", "label": "ГТМ-модуляция", "kind": "аналитическая"},
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
        "domain": "Гибридная ёмкостно-резистивная модель — прогноз нефтедобычи при заводнении (Norne / Brugge)",
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
        "constraint_note": "Пространство конфигураций |C| = произведение уровней осей "
                           "(16 осей: 13 бинарных + 3 троичных = 221184); ограничение C1 "
                           "(совместимость моделей ветвей) сокращает число допустимых сочетаний.",
    }


_DEMO_VE = _build_ve()

# Две предзапущенные итерации (Обзор не пустой). Каскад по H9 (материальный баланс):
# меняется H9 → пересчитываются H9 и его потомки вниз (7 из 16), остальные из кэша.
_ALL = [f"H{i}" for i in range(1, 17)]
_CASCADE_H9 = ["H9", "H10", "H11", "H12", "H13", "H14", "H16"]  # H9 + потомки
_DEMO_ITERATIONS = [
    {
        "results": {h: {"status": "SUCCESS", "metrics": {"r2": 0.71},
                        "epistemic_status": "SUPPORTED"} for h in _ALL},
        "reused": 0, "best": {"hypothesis": "H16", "r2": 0.71},
        "note": "Базовая линия — полный расчёт всех 16 гипотез.",
    },
    {
        "results": {h: {"status": "SUCCESS",
                        "metrics": {"r2": 0.86 if h == "H16" else 0.74},
                        "epistemic_status": "SUPPORTED"} for h in _ALL},
        "reused": len(_ALL) - len(_CASCADE_H9),   # 9 из кэша
        "best": {"hypothesis": "H16", "r2": 0.86},
        "note": "Изменён материальный баланс (H9): пересчитаны 7 из 16 "
                "(H9 и потомки), остальные 9 — из кэша.",
    },
]


def seed_demo(store) -> None:
    if any(p["name"] == DEMO_NAME for p in store.list()):
        return
    pid = store.create(name=DEMO_NAME,
                       description="Гибридная модель заводнения · 16 гипотез · Norne/Brugge")
    store.save_ve(pid, json.dumps(_DEMO_VE))
    for it in _DEMO_ITERATIONS:
        store.add_iteration(pid, json.dumps(it))
