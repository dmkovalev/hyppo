# examples/research/baseline_filecache/dag.py
"""Задачи конвейера (идентификатор = путь выхода) и файловые зависимости."""
from __future__ import annotations


def jobs(grid: dict) -> list[str]:
    c, a, k = grid["cons"], grid["alphas"], grid["corey"]
    out = [f"p1/{x}.npy" for x in c]
    out += [f"hyb/{x}_{y}.npy" for x in c for y in a]
    out += [f"wct/{no}_{nw}.npy" for no in k for nw in k]
    out += [f"opr/{x}_{y}_{no}_{nw}.json" for x in c for y in a for no in k for nw in k]
    return out


def parents(grid: dict) -> dict[str, set[str]]:
    """Файловые (объявленные) входы каждой задачи."""
    par: dict[str, set[str]] = {j: set() for j in jobs(grid)}
    for x in grid["cons"]:
        for y in grid["alphas"]:
            par[f"hyb/{x}_{y}.npy"] = {f"p1/{x}.npy"}
            for no in grid["corey"]:
                for nw in grid["corey"]:
                    par[f"opr/{x}_{y}_{no}_{nw}.json"] = {f"hyb/{x}_{y}.npy", f"wct/{no}_{nw}.npy"}
    return par


def topo(grid: dict) -> list[str]:
    order = {"p1": 0, "wct": 0, "hyb": 1, "opr": 2}
    return sorted(jobs(grid), key=lambda j: order[j.split("/")[0]])


def wrong_set(grid: dict, stale: set[str], executed: set[str]) -> set[str]:
    """Задачи с неверным результатом после инкрементального прогона.

    Задача неверна, если её результат устарел (в эталоне S) и либо она не
    пересчитана, либо она пересчитана, но прочла неверный файл-родитель.
    Пересчитанная задача читает актуальное хранилище, поэтому зависимость
    через хранилище при пересчёте учитывается верно.
    """
    par = parents(grid)
    wrong: set[str] = set()
    for j in topo(grid):
        if j not in stale:
            continue
        if j not in executed or par[j] & wrong:
            wrong.add(j)
    return wrong
