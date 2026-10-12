"""Минимальный по стоимости безопасный статический план (обобщение thm:safe).

Постановка. Граф гипотез L (ОАГ, ребро g -> h: g является непосредственным
источником h), множество актуальных гипотез A (N = V \\ A неактуальны),
стоимость пересчёта w(h) >= 0, стоимость чтения записи l(h) >= 0 (для h из A),
целевое множество T. План (P, Rd), P и Rd не пересекаются:

  (C1) T подмножество P | Rd                  -- цели получены;
  (C2) h в P  =>  pa(h) подмножество P | Rd   -- пересчёт требует всех входов;
  (C3) Rd подмножество A                      -- неактуальное читать нельзя.

Стоимость sum_{P} w + sum_{Rd} l. Задача является задачей о замыкании
минимального веса (project selection) и решается минимальным s-t разрезом.

Переменные: x_h = [h получена], r_h = [h пересчитана]. Стоимость
h: l(h) * x_h + (w(h) - l(h)) * r_h (для h из N: x_h = r_h, стоимость w(h) * r_h).
Импликации a => b кодируются дугами a -> b бесконечной пропускной способности:
  r_h => x_h;  r_h => x_g для g из pa(h);  x_h => r_h для h из N;  s => x_h для h из T.
Переменная с весом c >= 0 даёт дугу v -> t ёмкости c, с весом c < 0 -- дугу
s -> v ёмкости -c. Сторона источника минимального разреза -- оптимальный план.
"""

from __future__ import annotations

from typing import Hashable, Mapping, Set, Tuple

import networkx as nx

_S = ("__source__",)
_T = ("__sink__",)


def plan_min_cost(
    graph: nx.DiGraph,
    actual: Set[Hashable],
    w: Mapping[Hashable, float],
    l: Mapping[Hashable, float],
    targets: Set[Hashable],
) -> Tuple[Set[Hashable], Set[Hashable]]:
    """Вернуть (recompute, read) -- безопасный статический план минимальной стоимости.

    Args:
        graph: ОАГ гипотез, ребро g -> h означает, что g -- непосредственный источник h.
        actual: актуальные гипотезы (есть запись с текущим контекстом и нет маркера
            устарелости); остальные вершины образуют N.
        w: стоимость пересчёта, w[h] >= 0 для всех вершин.
        l: стоимость чтения записи, l[h] >= 0 для актуальных вершин
            (для неактуальных не используется).
        targets: целевое множество T, подмножество вершин графа.

    Returns:
        (P, Rd): P -- гипотезы для пересчёта, Rd -- актуальные записи для чтения.
        Среди оптимальных планов возвращается план с минимальным по включению
        множеством переменных на стороне источника (вершины, достижимые из s
        в остаточной сети максимального потока).
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("graph must be a DAG")
    nodes = set(graph.nodes)
    if not set(targets) <= nodes:
        raise ValueError("targets must be a subset of graph nodes")
    actual = set(actual) & nodes

    net = nx.DiGraph()
    net.add_node(_S)
    net.add_node(_T)

    def x(h):
        return ("x", h)

    def r(h):
        return ("r", h)

    def add_cost(v, c):
        c = float(c)
        if c < 0:
            raise ValueError("costs must be non-negative")
        net.add_edge(v, _T, capacity=c)

    def add_signed(v, c):
        c = float(c)
        if c >= 0:
            net.add_edge(v, _T, capacity=c)
        else:
            net.add_edge(_S, v, capacity=-c)

    for h in nodes:
        wh = float(w[h])
        if wh < 0:
            raise ValueError("w must be non-negative")
        net.add_node(x(h))
        net.add_node(r(h))
        # r_h => x_h (без атрибута capacity дуга имеет бесконечную ёмкость)
        net.add_edge(r(h), x(h))
        if h in actual:
            lh = float(l[h])
            add_cost(x(h), lh)
            add_signed(r(h), wh - lh)
        else:
            # x_h => r_h: неактуальную гипотезу можно получить только пересчётом
            net.add_edge(x(h), r(h))
            add_cost(r(h), wh)
        # r_h => x_g для каждого непосредственного источника g
        for g in graph.predecessors(h):
            net.add_edge(r(h), x(g))
        if h in targets:
            net.add_edge(_S, x(h))

    src_side = _min_source_side(net)
    recompute = {h for h in nodes if r(h) in src_side}
    obtained = {h for h in nodes if x(h) in src_side}
    read = obtained - recompute
    return recompute, read


def _min_source_side(net: nx.DiGraph) -> Set:
    """Сторона источника минимального разреза, минимальная по включению.

    nx.minimum_cut возвращает максимальную по включению сторону источника
    (дополнение множества вершин, из которых достижим сток); здесь берём
    множество вершин, достижимых из s в остаточной сети максимального потока.
    Минимальный разрез при этом тот же по ёмкости.
    """
    _, flow = nx.maximum_flow(net, _S, _T)
    inf = float("inf")
    seen = {_S}
    stack = [_S]
    while stack:
        u = stack.pop()
        for v, data in net[u].items():  # прямые дуги с остаточной ёмкостью
            if v not in seen and flow[u][v] < data.get("capacity", inf):
                seen.add(v)
                stack.append(v)
        for v in net.predecessors(u):  # обратные дуги, по которым идёт поток
            if v not in seen and flow[v][u] > 0:
                seen.add(v)
                stack.append(v)
    return seen


def plan_cost(recompute, read, w, l) -> float:
    """Стоимость плана: sum_{P} w + sum_{Rd} l."""
    return sum(float(w[h]) for h in recompute) + sum(float(l[h]) for h in read)


def is_feasible(graph: nx.DiGraph, actual, targets, recompute, read) -> bool:
    """Проверка условий (C1)--(C3) и дизъюнктности P и Rd."""
    recompute, read = set(recompute), set(read)
    if recompute & read:
        return False
    got = recompute | read
    if not set(targets) <= got:
        return False
    if not read <= set(actual):
        return False
    return all(set(graph.predecessors(h)) <= got for h in recompute)
