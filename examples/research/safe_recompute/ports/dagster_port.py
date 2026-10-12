"""Конвейер Brugge в Dagster (1.13.26): один актив (asset) на задачу.

Актив задачи j (имя `job_name(j)`):
  deps         -- активы файлов-родителей из `dag.parents` и наблюдаемые
                  источники (observable_source_asset) объявленных входов;
  code_version -- "<правило>:<sha256(workers.py)[:12]>" (аналог зависимости
                  DVC от workers.py);
  тело         -- та же команда `workers.py`, что в правиле Snakefile; возвращает
                  MaterializeResult(data_version=DataVersion(sha256 файла выхода)).
Наблюдаемые источники: `store_<entity>[_<token>]` -- DataVersion = версия
сущности в хранилище (только для объявленных пар); `param_ridge` --
DataVersion = значение ridge из `params.json` (вход всех fit_liq, объявлен
всегда). Run config Dagster в устаревании не участвует, поэтому параметр
смоделирован наблюдаемым источником, а не конфигом.

Шаг прогона (`_dagster_runner.py` в интерпретаторе .venv-cache, постоянный
экземпляр DAGSTER_HOME=workdir/dagster_home):
  1) `observe(sources, instance)` -- записать текущие версии источников;
  2) `CachingStaleStatusResolver.get_status` для каждого актива;
  3) выбрать STALE и MISSING -- так же, как `resolve_stale_or_missing_assets`
     (режим `stale_assets_only` у RunRequest);
  4) `materialize(all_assets, selection=выбранные, instance)`;
  5) повторять 1-4 до пустого выбора (неподвижная точка; как последовательные
     тики сенсора). Один проход недостаточен: потомок становится STALE только
     после того, как родитель материализован с новой data version.
  Выполненные -- активы с событием материализации за все раунды шага
  (повторная материализация актива за шаг -- ошибка); статусы и выбор каждого
  раунда -- в `res["rounds"]`.
Устаревание по data version: актив STALE, если сменилась code_version или
текущая data version родителя/источника отличается от записанной в его
data provenance. Родитель, материализованный заново с той же data version,
потомка не делает STALE (обрыв каскада).

События:
  param  -- запись нового ridge в `params.json`;
  evict  -- удаление файла выхода. Dagster ведёт учёт по событиям
            материализации, а не по файлам: после удаления статус актива
            остаётся FRESH (виден в `res["rounds"]`). Как и в
            simulators.py, это моделируется ручной rematerialization пропавшего
            актива: он добавляется к выбору шага (`force`). Потомки
            пересчитываются, только если их статус стал STALE;
  revise -- `revise.py` правит хранилище; Dagster видит правку, только если
            пара объявлена (источник наблюдается и входит в deps потребителя).
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from examples.research.baseline_filecache import dag

from . import (WORKERS, check_event, declared_pairs, entity_inputs, file_sha256, job_command,
               job_name, output_digests, param_key, revise, rule_of, seed, tools_python, worker_env)

RUNNER = Path(__file__).resolve().parent / "_dagster_runner.py"


@lru_cache(maxsize=1)
def tools_available() -> bool:
    py = tools_python()
    if py is None or not py.exists():
        return False
    res = subprocess.run([str(py), "-c", "import dagster"], capture_output=True)
    return res.returncode == 0


def source_name(entity: str, token: str) -> str:
    return "store" + param_key(entity, token)[1:]      # store_<entity>[_<token>]


def spec(workdir: Path, grid: dict, meta: Path, params_file: Path, declared, ridge: str,
         force: list[str] = ()) -> dict:
    pairs = declared_pairs(declared)
    par = dag.parents(grid)
    code = file_sha256(WORKERS)[:12]
    assets, sources = [], {}
    for j in dag.jobs(grid):
        srcs = []
        for e, t in entity_inputs(j, pairs):
            n = source_name(e, t)
            sources[n] = {"name": n, "kind": "store", "meta": str(meta), "entity": e, "token": t}
            srcs.append(n)
        if rule_of(j) == "fit_liq":
            sources["param_ridge"] = {"name": "param_ridge", "kind": "param",
                                      "params_file": str(params_file), "key": "ridge"}
            srcs.append("param_ridge")
        assets.append({"name": job_name(j), "out": j, "cmd": job_command(j, meta, ridge),
                       "deps": sorted(job_name(p) for p in par[j]), "sources": srcs,
                       "code_version": f"{rule_of(j)}:{code}"})
    return {"workdir": str(workdir), "assets": assets, "sources": list(sources.values()),
            "force": list(force)}


def _step(root: Path, sp: dict, grid: dict) -> tuple[set[str], dict]:
    path = root / "spec.json"
    path.write_text(json.dumps(sp), encoding="utf-8")
    env = worker_env()
    env["DAGSTER_HOME"] = str(root / "dagster_home")
    res = subprocess.run([str(tools_python()), str(RUNNER), str(path)], capture_output=True,
                         text=True, env=env, cwd=root)
    if res.returncode != 0:
        raise RuntimeError(res.stdout + res.stderr)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    by_name = {job_name(j): j for j in dag.jobs(grid)}
    if len(out["executed"]) != len(set(out["executed"])):
        raise RuntimeError(f"актив материализован дважды за шаг: {out['executed']}")
    rounds = [{"status": {by_name[n]: s for n, s in r["status"].items()},
               "selected": {by_name[n] for n in r["selected"]}} for r in out["rounds"]]
    return {by_name[n] for n in out["executed"]}, rounds


def run_workload(workdir: Path, grid: dict, events: list[tuple], declared) -> dict:
    for ev in events:
        check_event(ev, grid)
    root = Path(workdir)
    work, meta, params_file = root / "work", root / "meta.sqlite", root / "params.json"
    if (root / "dagster_home").exists() or work.exists():
        raise FileExistsError(f"{root}: нужен пустой каталог")
    work.mkdir(parents=True)
    (root / "dagster_home").mkdir()
    seed(meta)
    ridge = "1"
    params_file.write_text(json.dumps({"ridge": ridge}))
    executed, rounds = _step(root, spec(work, grid, meta, params_file, declared, ridge), grid)
    res = {"initial": executed, "after": [], "rounds": [rounds], "outputs": [output_digests(work, grid)]}
    for ev in events:
        force = []
        if ev[0] == "param":
            ridge = str(ev[2])
            params_file.write_text(json.dumps({"ridge": ridge}))
        elif ev[0] == "evict":
            (work / ev[1]).unlink()
            force = [job_name(ev[1])]
        else:
            revise(meta, ev[1])
        executed, rounds = _step(root, spec(work, grid, meta, params_file, declared, ridge, force), grid)
        res["after"].append(executed)
        res["rounds"].append(rounds)
        res["outputs"].append(output_digests(work, grid))
    return res
