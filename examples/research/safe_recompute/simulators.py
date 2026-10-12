"""Правила инвалидации систем на графе задач/гипотез (сверяются с реальными запусками Э6).

Режимы систем (имена совпадают с ключами JSON Э6):
* snakemake -- `--rerun-triggers mtime params`: пересчёт при смене params, каскад
  по mtime без обрыва (заново записанный файл новее потребителей);
* dvc -- `dvc repro --no-run-cache`: стадия пересчитывается при смене md5
  зависимости/params, обрыв каскада по md5 выхода (early cutoff);
* dagster -- материализация по устареванию (`stale_assets_only` /
  `data_version_changed`, как в порте): актив STALE, если data version родителя
  или источника отличается от записанной в data provenance; обрыв каскада по
  data version. Сверено реальными запусками;
* dagster_eager -- `AutomationCondition.eager()`: `any_deps_updated` срабатывает
  на любую новую материализацию родителя, обрыва нет (каскад, как у Snakemake).
  Реальным запуском не сверялся;
* nextflow -- cache='standard' хэширует путь+размер+mtime входов, не содержимое:
  обрыва нет; nextflow_deep (cache 'deep') хэширует содержимое и даёт early
  cutoff; bazel -- early cutoff по дайджесту действий. Только модель, реальным
  запуском не сверялись;
* hyppo_cascade / hyppo_safe -- комплекс в режимах planner 'cascade' (Cl(N)) и
  'safe' (ровно N, теорема thm:safe; цепочечные ключи).

Допущения модели:
* evict, вариант «forced» (по умолчанию): выход узла пропал вместе с объектом
  кэша (в DVC -- удалены файл и объект `.dvc/cache`), пользователь явно требует
  восстановить его; в Dagster -- ручная rematerialization пропавшего актива, в
  Snakemake -- явная цель. run-cache DVC отключён (`--no-run-cache`); это
  консервативно: может лишь увеличить число запусков DVC, но не пропуски.
  Вариант «native» (`native_evict`): что делает система сама, без явного
  требования. DVC восстанавливает выход из кэша (0 запусков), Snakemake с целью
  `all` пропавший промежуточный файл не восстанавливает (0 запусков, файл
  отсутствует), а пропавший конечный выход (цель `all`) пересчитывает, Dagster удаления файла не замечает (актив FRESH, 0 запусков,
  файл отсутствует). Отсутствующий файл не считается «устаревшим результатом»
  (он не прочитан как верный), а учитывается отдельно (`absent`).
* revise, модель сущностей хранилища (`Store`, используется в Э6): сущность e
  читается узлами-потребителями C(e); пара (e, c) может быть объявлена
  инструменту. Ревизия e делает устаревшими выходы потребителей (и их
  потомков). Инструменты видят только объявленных потребителей и пересчитывают
  их с замыканием по своему правилу (каскад или обрыв); комплекс видит всех
  потребителей независимо от объявления (сущность входит в уравнение модели
  потребителя).
* revise, старая статусная семантика (`simulate`, `simulate_sequence` без
  `Store`): эпистемическая ревизия статуса узла; результаты, построенные на нём
  (потомки), устарели, сам узел остаётся валидным; файловые системы её не видят.
* `simulate` / `simulate_sequence`: события независимы, каждое -- от полностью
  свежего состояния. `run_sequence`: события применяются по порядку, состояние
  (множество неверных результатов) сохраняется -- так же, как в портах Э6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import networkx as nx

SYSTEMS = ("snakemake", "dvc", "dagster", "dagster_eager", "nextflow", "nextflow_deep", "bazel",
           "hyppo_cascade", "hyppo_safe")
EARLY_CUTOFF = {"dvc", "dagster", "nextflow_deep", "bazel"}

FILE_CASCADE = {"snakemake", "nextflow", "dagster_eager"}  # внешние системы без early cutoff
HYPPO = {"hyppo_cascade", "hyppo_safe"}

# сверены реальным запуском (Э6) / только модель
VALIDATED = ("snakemake", "dvc", "dagster", "hyppo_safe")
NOT_VALIDATED = ("dagster_eager", "nextflow", "nextflow_deep", "bazel", "hyppo_cascade")

# нативная реакция на удаление выхода (без явного требования пересчитать):
# "restored" -- восстановлен из кэша, "absent" -- файл остаётся отсутствующим;
# прочие системы -- как вариант forced
NATIVE_EVICT = {"dvc": "restored", "snakemake": "absent", "dagster": "absent",
                "dagster_eager": "absent"}


@dataclass(frozen=True)
class Event:
    kind: str            # "param" | "evict" | "revise"
    node: str            # узел графа; для revise со Store (и param-сущности) -- имя сущности
    output_changed: bool  # does the node's output change after recomputation?


@dataclass
class Result:
    recomputed: set = field(default_factory=set)
    missed: set = field(default_factory=set)  # stale results kept (вновь появившиеся неверные)
    wrong: set = field(default_factory=set)   # run_sequence: все неверные результаты после события
    absent: set = field(default_factory=set)  # native_evict: выходы, оставшиеся отсутствующими


@dataclass(frozen=True)
class Store:
    """Сущности хранилища для `run_sequence`.

    consumers -- сущность -> узлы-потребители (читают её в рантайме);
    declared  -- пары (сущность, узел), объявленные инструментам;
    params    -- сущности-параметры (объявлены всегда, событие kind="param").
    """
    consumers: Mapping[str, frozenset]
    declared: frozenset = frozenset()
    params: frozenset = frozenset()

    def visible(self, entity: str, system: str) -> set:
        cons = set(self.consumers[entity])
        if system in HYPPO or entity in self.params:
            return cons
        return {c for c in cons if (entity, c) in self.declared}


def _one(system: str, g: nx.DiGraph, ev: Event,
         output_changes: Callable[[str, Event], bool]) -> Result:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    if ev.kind not in ("param", "evict", "revise"):
        raise ValueError(f"unknown event kind: {ev.kind}")
    if ev.node not in g:
        raise ValueError(f"node {ev.node!r} is not in the graph")
    desc = nx.descendants(g, ev.node)
    if ev.kind == "revise":
        # статусная ревизия: файловые системы её не видят
        if system in EARLY_CUTOFF or system in FILE_CASCADE:
            return Result(set(), set(desc))
        return Result(set(desc), set())
    if system in FILE_CASCADE or system == "hyppo_cascade":
        return Result({ev.node} | desc, set())
    if system == "hyppo_safe":
        return Result({ev.node} | (desc if ev.kind == "param" else set()), set())
    # early cutoff: пересчёт потомка, если изменился выход хотя бы одного
    # пересчитанного родителя
    rec = {ev.node}
    changed = set()
    for n in nx.topological_sort(g):
        if n != ev.node and not any(p in changed for p in g.predecessors(n)):
            continue
        rec.add(n)
        if output_changes(n, ev):
            changed.add(n)
    return Result(rec, set())


def simulate(system: str, g: nx.DiGraph, ev: Event) -> Result:
    """Один пересчёт; output_changed события относится ко всем пересчитанным узлам."""
    return _one(system, g, ev, lambda n, e: e.output_changed)


def simulate_sequence(system: str, g: nx.DiGraph, events,
                      output_changes: Callable[[str, Event], bool]) -> list[Result]:
    """Результат по каждому событию; output_changes(node, event) -> изменился ли
    выход пересчитанного узла (определяет обрыв каскада в early-cutoff системах)."""
    return [_one(system, g, ev, output_changes) for ev in events]


# ---------- последовательная модель с состоянием (Э6) ----------
class _Graph:
    """Граф с топологическим индексом и кэшем потомков (для больших корпусов)."""

    def __init__(self, g: nx.DiGraph):
        self.g = g
        self.order = {n: i for i, n in enumerate(nx.topological_sort(g))}
        self._desc: dict = {}

    def desc(self, n) -> set:
        if n not in self._desc:
            self._desc[n] = nx.descendants(self.g, n)
        return self._desc[n]

    def closure(self, seeds: Iterable) -> set:
        out = set(seeds)
        for s in list(out):
            out |= self.desc(s)
        return out


def _as_graph(g) -> _Graph:
    return g if isinstance(g, _Graph) else _Graph(g)


def event_targets(g, ev: Event, store: Store | None, system: str) -> set:
    """Узлы, которые система пересчитывает непосредственно (до замыкания)."""
    gg = _as_graph(g).g
    if ev.kind == "param":
        if ev.node in gg:
            return {ev.node}
        if store is None or ev.node not in store.params:
            raise ValueError(f"param {ev.node!r}: not a node and not a store parameter")
        return set(store.consumers[ev.node])
    if ev.kind == "evict":
        if ev.node not in gg:
            raise ValueError(f"node {ev.node!r} is not in the graph")
        return {ev.node}
    if ev.kind == "revise":
        if store is None:  # старая статусная семантика: потребители -- дети узла
            if ev.node not in gg:
                raise ValueError(f"node {ev.node!r} is not in the graph")
            return set(gg.successors(ev.node)) if system in HYPPO else set()
        if ev.node not in store.consumers:
            raise ValueError(f"unknown store entity {ev.node!r}")
        return store.visible(ev.node, system)
    raise ValueError(f"unknown event kind: {ev.kind}")


def default_delta(g, ev: Event, store: Store | None = None) -> set:
    """Узлы, верный выход которых меняется событием, если эталона нет:
    param/revise с output_changed -- непосредственные цели и все их потомки
    (родовой случай), иначе пусто; evict мир не меняет."""
    gg = _as_graph(g)
    if ev.kind == "evict" or not ev.output_changed:
        return set()
    return gg.closure(event_targets(gg, ev, store, "hyppo_safe"))


def run_sequence(system: str, g, events, deltas=None, store: Store | None = None) -> list[Result]:
    """События по порядку, состояние сохраняется (как в портах Э6; evict -- forced).

    deltas[i] -- узлы, верный выход которых меняется i-м событием (эталон
    полным пересчётом); по умолчанию `default_delta`. Состояние W -- неверные
    результаты. Пересчитанный узел становится верным, если все его родители
    верны; непересчитанный неверен, если был неверен или его эталон сменился.
    Early cutoff: выход пересчитанного узла «изменился» (относительно записанного),
    если сменился его эталон, он был неверен или прочёл неверного родителя.
    Result.missed -- вновь появившиеся неверные (W' minus W), Result.wrong -- W'.
    """
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    gg = _as_graph(g)
    events = list(events)
    if deltas is None:
        deltas = [default_delta(gg, ev, store) for ev in events]
    if len(deltas) != len(events):
        raise ValueError("deltas and events differ in length")
    pred = gg.g.predecessors
    wrong: set = set()
    out = []
    for ev, delta in zip(events, deltas):
        delta = set(delta)
        seeds = event_targets(gg, ev, store, system)
        if system == "hyppo_safe" and ev.kind == "evict":
            cand = set(seeds)                      # ровно N: потомки актуальны
        else:
            cand = gg.closure(seeds)
        new_wrong = wrong | delta
        rec, changed = set(), set()
        for n in sorted(cand, key=gg.order.__getitem__):
            if system in EARLY_CUTOFF and n not in seeds and not any(p in changed for p in pred(n)):
                continue
            rec.add(n)
            bad = any(p in new_wrong for p in pred(n))
            if n in delta or n in wrong or bad:
                changed.add(n)
            if bad:
                new_wrong.add(n)
            else:
                new_wrong.discard(n)
        out.append(Result(rec, new_wrong - wrong, set(new_wrong)))
        wrong = new_wrong
    return out


def native_evict(system: str, node: str, g: nx.DiGraph | None = None) -> Result | None:
    """Нативная реакция на удаление выхода `node` (без явного требования).
    None -- нативный вариант совпадает с forced (`run_sequence`).
    Snakemake с целью `all` восстанавливает пропавший конечный выход (сток графа,
    он сам -- цель), но не промежуточный; без графа узел считается промежуточным."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    kind = NATIVE_EVICT.get(system)
    if kind is None:
        return None
    if system == "snakemake" and g is not None and g.out_degree(node) == 0:
        return Result({node}, set(), set(), set())
    return Result(set(), set(), set(), {node} if kind == "absent" else set())
