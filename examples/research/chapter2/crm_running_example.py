"""Сквозной пример гл. 2: фрагменты гибридной модели CRM гл. 4 (16 гипотез).

Модуль задаёт уравнения гипотез H1--H16 в нумерации диссертации, поток работ W
(14 задач) и две иллюстративные модификации графа гл. 4:

* ``"base"``   --- граф гл. 4: алгоритм 1 даёт 19 рёбер при глубине 10;
* ``"rival"``  --- модификация 1: одноканальный соперник H4' (та же величина q_c,
  что у цепочки H2--H4) в задаче гипотезы H4; пару выявляет карта производителей;
* ``"split"``  --- модификация 2 (намеренная ошибка): материальный баланс H9*
  учитывает лишь отбор воды l f_w, а f_w вычисляется позже гипотезой H12; пара
  (H9*, H12) даёт конфликт порядка со свидетелем леммы о балансе (избыток 1);
* ``"merged"`` --- исправление модификации 2: H9*, H10, H11, H12 объединены в
  составную гипотезу H9-12 с нетривиальным блоком Далмеджа--Мендельсона.

Кроме того, функции модуля воспроизводят числа примеров разд. 2.3--2.5: каскад
правки показателя Кори (H10), каскад маркеров при нарушении инварианта tau_f < tau_s
(H3), а также сценарий вытеснения записи H4 из репозитория (N = {H4} != Cl(N)).

Запуск: ``PYTHONPATH=. python examples/research/chapter2/crm_running_example.py``.
"""
from __future__ import annotations

import networkx as nx

from hyppo.coa._base import Equation, Structure
from hyppo.coa.causal import block_decomposition
from hyppo.graph_constructor._base import GraphConstructor

# --- уравнения гипотез (нумерация гл. 2 и 4) --------------------------------
EQUATIONS: dict[str, list[str]] = {
    "H1": ["I_agg = w_ij*I_j"],                                   # агрегация закачки
    "H2": ["q_f = a_f*q_f_prev + b_f*I_agg"],                      # быстрый канал CRM
    "H3": ["q_s = a_s*q_s_prev + b_s*I_agg"],                      # медленный канал CRM
    "H4": ["q_c = w_f*q_f + (1 - w_f)*q_s"],                       # смешение каналов
    "H5": ["q_liq = J*q_c + q_prim"],                              # продуктивность
    "H6": ["q_prim = q_prev*exp(-dt*tau_p)"],                      # первичная добыча
    "H7": ["l_m = w_0 + w_1*q_c + w_2*l_prev + w_3*I_mean"],       # ML-коррекция
    "H8": ["l = g*q_liq + (1 - g)*l_m"],                           # слияние: жидкость
    "H9": ["S_w = S_w_prev + (W_inj - l)*dt/V_p"],                 # материальный баланс
    "H10": ["k_rw = ((S_w - S_wc)/(1 - S_wc - S_or))**n_w"],       # Кори k_rw
    "H11": ["k_ro = ((1 - S_w - S_or)/(1 - S_wc - S_or))**n_o"],   # Кори k_ro
    "H12": ["f_w = 1/(1 + k_ro*mu_w/(k_rw*mu_o))"],                # Бакли--Леверетт
    "H13": ["o_p = 1 - f_w"],                                      # WCT-якорь
    "H14": ["o = g_w*o_p + (1 - g_w)*o_m"],                        # слияние: вода
    "H15": ["J = J_0 + dJ_gtm"],                                   # ГТМ
    "H16": ["q_oil = l*o"],                                        # прогноз нефти
}
RIVAL = {"H4'": ["q_c = a_c*q_c_prev + b_c*I_agg"]}                 # одноканальная CRM
SPLIT_BALANCE = {"H9*": ["S_w = S_w_prev + (W_inj - l*f_w)*dt/V_p"]}
MERGED = {"H9-12": SPLIT_BALANCE["H9*"] + EQUATIONS["H10"] + EQUATIONS["H11"] + EQUATIONS["H12"]}

# --- поток работ: 14 задач, рёбра по индексам задач --------------------------
TASKS = [["H1"], ["H2", "H3"], ["H4"], ["H6"], ["H15"], ["H5"], ["H7"], ["H8"],
         ["H9"], ["H10", "H11"], ["H12"], ["H13"], ["H14"], ["H16"]]
TASK_EDGES = [(0, 1), (1, 2), (2, 5), (3, 5), (4, 5), (2, 6), (5, 7), (6, 7),
              (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]

CRM_EDGES_19 = {
    ("H1", "H2"), ("H1", "H3"), ("H2", "H4"), ("H3", "H4"), ("H4", "H5"),
    ("H6", "H5"), ("H7", "H8"), ("H5", "H8"), ("H8", "H9"), ("H9", "H10"),
    ("H9", "H11"), ("H10", "H12"), ("H11", "H12"), ("H12", "H13"), ("H13", "H14"),
    ("H15", "H5"), ("H8", "H16"), ("H14", "H16"), ("H4", "H7"),
}


class Hyp:
    def __init__(self, name: str, formulas: list[str]):
        self.name = name
        self.structure = Structure([Equation(formula=f) for f in formulas])

    def __repr__(self) -> str:
        return self.name


class Workflow:
    def __init__(self, tasks, edges):
        self._tasks, self._edges = tasks, edges

    def get_tasks(self):
        return self._tasks

    def get_task_edges(self):
        return self._edges


def spec(variant: str = "base"):
    """Гипотезы и поток работ варианта ``base | rival | split | merged``."""
    eqs = dict(EQUATIONS)
    tasks = [list(t) for t in TASKS]
    edges = list(TASK_EDGES)
    if variant == "rival":
        eqs.update(RIVAL)
        tasks[2].append("H4'")
    elif variant == "split":
        eqs.pop("H9"); eqs.update(SPLIT_BALANCE)
        tasks[8] = ["H9*"]
    elif variant == "merged":
        for h in ("H9", "H10", "H11", "H12"):
            eqs.pop(h)
        eqs.update(MERGED)
        tasks = [t for t in tasks if t not in (["H10", "H11"], ["H12"])]
        tasks[8] = ["H9-12"]
        edges = [(0, 1), (1, 2), (2, 5), (3, 5), (4, 5), (2, 6), (5, 7), (6, 7),
                 (7, 8), (8, 9), (9, 10), (10, 11)]
    elif variant != "base":
        raise ValueError(variant)
    hyps = {n: Hyp(n, f) for n, f in eqs.items()}
    wf = Workflow([[hyps[n] for n in t] for t in tasks], edges)
    return hyps, wf


def build(variant: str = "base"):
    """Алгоритм 1 на варианте: (GraphConstructor, граф с именами-строками)."""
    hyps, wf = spec(variant)
    gc = GraphConstructor(list(hyps.values()), wf)
    g = nx.DiGraph()
    g.add_nodes_from(hyps)
    g.add_edges_from((u.name, v.name) for u, v in gc.hyp_graph.edges())
    return gc, g


def signature(gc, name: str):
    h = next(x for x in gc.hypotheses if x.name == name)
    return set(gc._input_variables(h)), set(gc._output_variables(h))


def competes_pairs(gc) -> set[tuple[str, str]]:
    """Полное отношение соперничества по карте производителей."""
    return {tuple(sorted((h.name, c.name))) for h in gc.hypotheses for c in gc.competes(h)}


def order_conflicts(gc) -> set[tuple[str, str]]:
    return {(a.name, b.name) for a, b in gc.order_conflicts()}


def balance_excess(gc, a: str, b: str) -> int:
    """|E| - |V| экзогенизированного объединения пары (a, b), посчитанное по
    уравнениям: E_a, E_b и тривиальные уравнения X_ab = (In_a | In_b) - Out_a."""
    in_a, out_a = signature(gc, a)
    in_b, out_b = signature(gc, b)
    x_ab = (in_a | in_b) - out_a
    n_eq = len(out_a) + len(out_b) + len(x_ab)          # по У0 |E_h| = |Out(h)|
    n_var = len(in_a | out_a | in_b | out_b)
    return n_eq - n_var


def dm_blocks(gc, name: str) -> list[set[str]]:
    """Нетривиальные блоки Далмеджа--Мендельсона экзогенизированной структуры
    гипотезы: множества выходов, вычисляемых совместно."""
    ins, outs = signature(gc, name)
    h = next(x for x in gc.hypotheses if x.name == name)
    eq_vars = []
    for eq in h.structure.equations:
        lhs = eq.formula.split("=")[0].strip()
        eq_vars.append(frozenset({lhs} | (set(gc._input_variables(_single(eq))) )))
    eqs = eq_vars + [frozenset({x}) for x in sorted(ins)]
    blocks = block_decomposition(eqs)
    out = []
    for blk in blocks or []:
        if len(blk) > 1:
            out.append({e.formula.split("=")[0].strip() for i, e in enumerate(h.structure.equations) if i in blk})
    return out


def _single(eq):
    return Hyp("_", [eq.formula])


def cascade(g: nx.DiGraph, changed: str) -> set[str]:
    """Cl({changed}) = changed и его потомки (алгоритм 4)."""
    return {changed} | nx.descendants(g, changed)


def main() -> None:
    for variant in ("base", "rival", "split", "merged"):
        gc, g = build(variant)
        depth = nx.dag_longest_path_length(g) if nx.is_directed_acyclic_graph(g) else None
        print(f"== {variant}: |H|={g.number_of_nodes()} |E|={g.number_of_edges()} depth={depth}")
        print("   competes:", sorted(competes_pairs(gc)), " order conflicts:", sorted(order_conflicts(gc)))
        if variant == "split":
            print("   balance excess (H9*, H12):", balance_excess(gc, "H9*", "H12"))
        if variant == "merged":
            print("   DM blocks of H9-12:", dm_blocks(gc, "H9-12"), signature(gc, "H9-12"))
    _, g = build("base")
    print("Corey edit H10 -> P_ne =", sorted(cascade(g, "H10"), key=lambda s: int(s[1:])))
    print("tau_f<tau_s violated at H3 -> stale =", sorted(nx.descendants(g, "H3"), key=lambda s: int(s[1:])))
    print("evict H4: N = ['H4'], Cl(N) =", sorted(cascade(g, "H4"), key=lambda s: int(s[1:])))


if __name__ == "__main__":
    main()
