"""Конвейер Brugge в DVC (3.67.1): одна стадия `dvc.yaml` на задачу.

Стадия задачи j:
  cmd    -- та же команда `workers.py`, что в правиле Snakefile (интерпретатор
            venv комплекса); для fit_liq множитель штрафа подставляется DVC из
            `params.yaml` шаблоном `${ridge}`;
  deps   -- `workers.py` (абсолютный путь, внешняя зависимость) и файлы-родители
            из `dag.parents`;
  params -- `ridge` у fit_liq (объявлен всегда) и версии ОБЪЯВЛЕННЫХ сущностей
            хранилища (ключи `v_<entity>[_<token>]` в `params.yaml`);
  outs   -- [j] (с кэшем DVC по умолчанию).

Прогон: `dvc init --no-scm` в `workdir/repo`; перед каждым `dvc repro
--no-run-cache` порт переписывает `params.yaml` живыми версиями из хранилища
(аналог функции params в Snakefile, вычисляемой при каждом запуске).
`--no-run-cache` отключает восстановление выхода из run-cache.
Выполненные стадии -- строки `Running stage '<name>':` вывода `dvc repro`.

События:
  param  -- запись нового `ridge` в `params.yaml`;
  evict  -- удаление файла выхода И объекта кэша `.dvc/cache/files/md5/..` по
            md5 из `dvc.lock`; иначе `dvc repro` не пересчитывает стадию, а
            восстанавливает выход из кэша ("is cached - skipping run, checking
            out outputs");
  revise -- `revise.py` правит хранилище; DVC видит правку, только если пара
            объявлена (версия попадает в `params.yaml`).
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

from examples.research.baseline_filecache import dag

from . import (WORKERS, all_entity_tokens, check_event, declared_pairs, entity_inputs, job_command,
               job_name, output_digests, param_key, revise, rule_of, seed, store_version,
               tools_python, worker_env)

_RUN_RE = re.compile(r"^Running stage '([^']+)'", re.M)


def stage_name(job: str) -> str:
    return job_name(job)


def dvc_exe() -> Path | None:
    py = tools_python()
    if py is None:
        return None
    exe = py.parent / ("dvc.exe" if os.name == "nt" else "dvc")
    return exe if exe.exists() else None


def tools_available() -> bool:
    return dvc_exe() is not None


def _cmdline(args: list[str]) -> str:
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


def dvc_yaml(grid: dict, meta: Path | str, declared) -> dict:
    pairs = declared_pairs(declared)
    par = dag.parents(grid)
    stages = {}
    for j in dag.jobs(grid):
        st = {
            "cmd": _cmdline(job_command(j, Path(meta).as_posix(), "${ridge}")),
            "deps": [WORKERS.as_posix(), *sorted(par[j])],
        }
        params = (["ridge"] if rule_of(j) == "fit_liq" else [])
        params += [param_key(e, t) for e, t in entity_inputs(j, pairs)]
        if params:
            st["params"] = params
        st["outs"] = [j]
        stages[stage_name(j)] = st
    return {"stages": stages}


def params_yaml(grid: dict, meta: Path | str, ridge: str) -> dict:
    p = {"ridge": str(ridge)}
    for e, t in all_entity_tokens(grid):
        p[param_key(e, t)] = store_version(meta, e, t)
    return p


def _dump(path: Path, data: dict) -> None:
    import yaml
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _dvc(repo: Path, *args: str) -> str:
    env = worker_env()
    env["DVC_NO_ANALYTICS"] = "1"
    res = subprocess.run([str(dvc_exe()), *args], cwd=repo, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"dvc {' '.join(args)}:\n{res.stdout}\n{res.stderr}")
    return res.stdout + res.stderr


def _repro(repo: Path, grid: dict, meta: Path, ridge: str) -> set[str]:
    _dump(repo / "params.yaml", params_yaml(grid, meta, ridge))
    out = _dvc(repo, "repro", "--no-run-cache")
    by_name = {stage_name(j): j for j in dag.jobs(grid)}
    return {by_name[n] for n in _RUN_RE.findall(out)}


def _evict(repo: Path, job: str) -> None:
    import yaml
    lock = yaml.safe_load((repo / "dvc.lock").read_text(encoding="utf-8"))
    md5 = next(o["md5"] for o in lock["stages"][stage_name(job)]["outs"] if o["path"] == job)
    (repo / job).unlink()
    obj = repo / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]
    if obj.exists():
        os.chmod(obj, stat.S_IWRITE)  # объекты кэша DVC только для чтения
        obj.unlink()


def run_workload(workdir: Path, grid: dict, events: list[tuple], declared) -> dict:
    for ev in events:
        check_event(ev, grid)
    workdir = Path(workdir)
    repo, meta = workdir / "repo", workdir / "meta.sqlite"
    shutil.rmtree(repo, ignore_errors=True)
    repo.mkdir(parents=True)
    if meta.exists():
        meta.unlink()
    _dvc(repo, "init", "--no-scm", "-q")
    _dvc(repo, "config", "core.analytics", "false")
    _dump(repo / "dvc.yaml", dvc_yaml(grid, meta, declared))
    seed(meta)
    ridge = "1"
    res = {"initial": _repro(repo, grid, meta, ridge), "after": [], "outputs": [output_digests(repo, grid)]}
    for ev in events:
        if ev[0] == "param":
            ridge = str(ev[2])
        elif ev[0] == "evict":
            _evict(repo, ev[1])
        else:
            revise(meta, ev[1])
        res["after"].append(_repro(repo, grid, meta, ridge))
        res["outputs"].append(output_digests(repo, grid))
    return res
