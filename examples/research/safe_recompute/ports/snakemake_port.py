"""Тонкий адаптер Snakemake (для симметрии с портами DVC и Dagster).

Использует Snakefile и запуск из `baseline_filecache.experiment._snk`
(`snakemake --rerun-triggers mtime params --config meta=.. grid=.. declared=.. ridge=..`),
но реальным прогоном (`--cores 1`), а не `-n`. Выполненные задачи -- строки
`output:` вывода Snakemake (`experiment._dry_run_jobs`).

События:
  param  -- новый `ridge` в `--config` (params правила fit_liq);
  evict  -- удаление файла выхода (кэша вне рабочего каталога у Snakemake нет).
            С целью по умолчанию (`all`) Snakemake пропавший промежуточный файл
            НЕ восстанавливает, если конечные выходы на месте и свежи (как для
            temp()); это записывается в `res["native_evict"]` (сухой прогон
            `-n`). Сам шаг идёт с явными целями `all <job>`: так пользователь
            требует пересчитать удалённое; заново записанный файл новее
            потребителей, и mtime-триггер пересчитывает их без обрыва каскада;
  revise -- `revise.py`; видна только объявленная пара (живая версия в params).
"""
from __future__ import annotations

from pathlib import Path

from examples.research.baseline_filecache import experiment
from examples.research.baseline_filecache.config import GRID, HYPPO_PY, SMALL_GRID, SNAKEMAKE_PY

from . import check_event, declared_pairs, output_digests, revise, seed


def tools_available() -> bool:
    # Snakefile берёт интерпретатор воркеров из config.HYPPO_PY
    return SNAKEMAKE_PY.exists() and HYPPO_PY.exists()


def _grid_name(grid: dict) -> str:
    if grid == SMALL_GRID:
        return "small"
    if grid == GRID:
        return "full"
    raise ValueError("Snakefile знает только сетки small и full")


def run_workload(workdir: Path, grid: dict, events: list[tuple], declared) -> dict:
    for ev in events:
        check_event(ev, grid)
    gname, pairs = _grid_name(grid), sorted(declared_pairs(declared))
    root = Path(workdir)
    work, meta = root / "work", root / "meta.sqlite"
    work.mkdir(parents=True, exist_ok=False)
    seed(meta)
    ridge = "1"

    def run(extra=("--cores", "1")) -> set[str]:
        out = experiment._snk(work, meta, gname, pairs, ridge, list(extra))
        return experiment._dry_run_jobs(out, grid)

    res = {"initial": run(), "after": [], "native_evict": [], "outputs": [output_digests(work, grid)]}
    for ev in events:
        extra = ("--cores", "1")
        if ev[0] == "param":
            ridge = str(ev[2])
        elif ev[0] == "evict":
            (work / ev[1]).unlink()
            res["native_evict"].append(run(("-n", "--cores", "1")))
            extra = ("--cores", "1", "all", ev[1])
        else:
            revise(meta, ev[1])
        res["after"].append(run(extra))
        res["outputs"].append(output_digests(work, grid))
    return res
