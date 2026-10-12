"""Правила инвалидации систем на графе задач/гипотез (сверяются с реальными запусками).

Допущения модели:
* evict: выход узла в рабочем каталоге удалён и должен быть пересчитан
  (run-cache/восстановление из кэша в DVC отключено; в Dagster это ручная
  rematerialization пропавшего актива).
* События независимы: каждое оценивается от полностью свежего состояния.
* revise: эпистемическая ревизия статуса узла; результаты, построенные на нём
  (потомки), устарели, сам узел остаётся валидным.
* nextflow: cache='standard' хэширует путь+размер+mtime входов, не содержимое,
  поэтому обрыва каскада нет (как у Snakemake); nextflow_deep (cache 'deep')
  хэширует содержимое и даёт early cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx

SYSTEMS = ("snakemake", "dvc", "dagster", "nextflow", "nextflow_deep", "bazel", "hyppo_cascade", "hyppo_safe")
EARLY_CUTOFF = {"dvc", "dagster", "nextflow_deep", "bazel"}

FILE_CASCADE = {"snakemake", "nextflow"}  # файловые системы без early cutoff


@dataclass(frozen=True)
class Event:
    kind: str            # "param" | "evict" | "revise"
    node: str
    output_changed: bool  # does the node's output change after recomputation?


@dataclass
class Result:
    recomputed: set = field(default_factory=set)
    missed: set = field(default_factory=set)  # stale results kept


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
