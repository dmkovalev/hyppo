"""Порты конвейера Brugge (`baseline_filecache`) на реальные системы для Э6.

Общий интерфейс всех портов::

    run_workload(workdir, grid, events, declared) -> {
        "initial": set[job],            # задачи, выполненные при первом прогоне
        "after":   [set[job], ...],     # выполненные после каждого события
        "outputs": [{job: digest|None}, ...],  # дайджесты выходов: после первого
                                               # прогона и после каждого события
    }

Задача (job) -- путь выхода, как в `baseline_filecache.dag`.

События:
  ("param", "ridge", value) -- смена множителя штрафа fit_liq (объявленный
      CLI-параметр; объявлен всегда, независимо от `declared`);
  ("evict", job)            -- удаление выхода задачи в рабочем каталоге
      (что ещё делает порт, см. его docstring);
  ("revise", entity)        -- ревизия сущности хранилища через `revise.py`,
      ни один файл рабочего каталога не трогается.

`declared` -- множество пар «сущность:правило» (`"gains:fit_liq"`) или имён
сущностей (`"corey_ref"` = все пары этой сущности из `config.PAIRS`).
Объявленная пара делает версию сущности входом задачи-потребителя;
необъявленная системе не видна.

Версия сущности, ещё не созданной производителем (gains/tau до crm_fit),
читается как "v1" -- ровно ту версию, которую crm_fit запишет. Иначе первый
прогон дал бы ложное изменение «missing -> v1» во всех портах одинаково.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from examples.research.baseline_filecache import config, dag

BASE = Path(config.__file__).resolve().parent          # baseline_filecache/
WORKERS = BASE / "workers.py"
REVISE = BASE / "revise.py"
RULE_OF = {"p1": "crm_fit", "hyb": "fit_liq", "wct": "fit_wct", "opr": "fit_opr"}


# ---------- окружение ----------
def hyppo_python() -> Path:
    """Интерпретатор воркеров (venv комплекса)."""
    env = os.environ.get("HYPPO_PY")
    if env:
        return Path(env)
    return config.HYPPO_PY if config.HYPPO_PY.exists() else Path(sys.executable)


def tools_python() -> Path | None:
    """Интерпретатор .venv-cache (Snakemake, DVC, Dagster)."""
    env = os.environ.get("HYPPO_TOOLS_PY")
    if env:
        return Path(env)
    if config.SNAKEMAKE_PY.exists():
        return config.SNAKEMAKE_PY
    for p in BASE.parents:
        cand = p / ".venv-cache" / "Scripts" / "python.exe"
        if cand.exists():
            return cand
    return None


def brugge_run_dir() -> Path:
    env = os.environ.get("HYPPO_BRUGGE_RUN")
    if env:
        return Path(env)
    return WORKERS.parents[4] / "thesis" / "papers" / "brugge_run"


def worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env["HYPPO_BRUGGE_RUN"] = str(brugge_run_dir())
    return env


# ---------- задачи ----------
def rule_of(job: str) -> str:
    return RULE_OF[job.split("/")[0]]


def wildcards(job: str) -> list[str]:
    return Path(job).stem.split("_")


def job_name(job: str) -> str:
    """Имя задачи без точек и слешей: 'opr/uto_0.1_2_2.json' -> 'opr_uto_0p1_2_2'."""
    return job.split("/")[0] + "_" + "_".join(config.tok(w) for w in wildcards(job))


def job_command(job: str, meta: Path | str, ridge: str, py: Path | str | None = None) -> list[str]:
    """Команда воркера, как в соответствующем правиле Snakefile (пути -- относительно рабочего каталога)."""
    py = str(py or hyppo_python())
    w, rule, meta = wildcards(job), rule_of(job), str(meta)
    head = [py, str(WORKERS), rule]
    if rule == "crm_fit":
        return head + ["--constraints", w[0], "--p1", job, "--meta", meta]
    if rule == "fit_liq":
        return head + ["--constraints", w[0], "--alpha", w[1], "--ridge", str(ridge),
                       "--p1", f"p1/{w[0]}.npy", "--meta", meta, "--out", job]
    if rule == "fit_wct":
        return head + ["--no", w[0], "--nw", w[1], "--meta", meta, "--out", job]
    return head + ["--hyb", f"hyb/{w[0]}_{w[1]}.npy", "--wct", f"wct/{w[2]}_{w[3]}.npy",
                   "--meta", meta, "--out", job]


def declared_pairs(declared) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for d in declared:
        if ":" in d:
            pair = tuple(d.split(":", 1))
            if pair not in config.PAIRS:
                raise ValueError(f"неизвестная пара {d}")
            out.add(pair)
        else:
            hit = {p for p in config.PAIRS if p[0] == d}
            if not hit:
                raise ValueError(f"неизвестная сущность {d}")
            out |= hit
    return out


def entity_inputs(job: str, pairs: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Объявленные входы-сущности задачи: [(entity, token)] в порядке config.PAIRS."""
    rule = rule_of(job)
    out = []
    for e, r in config.PAIRS:
        if r == rule and (e, r) in pairs:
            out.append((e, wildcards(job)[0] if e in config.PER_TOKEN else config.GLOBAL_TOKEN))
    return out


def param_key(entity: str, token: str) -> str:
    return f"v_{entity}" if token == config.GLOBAL_TOKEN else f"v_{entity}_{token}"


def store_version(meta: Path | str, entity: str, token: str) -> str:
    """Версия сущности в хранилище (чистый sqlite3); не созданная ещё -- 'v1'."""
    if not Path(meta).exists():
        return "v1"
    con = sqlite3.connect(str(meta), timeout=60)
    try:
        row = con.execute("SELECT version FROM meta WHERE entity=? AND token=?", (entity, token)).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        con.close()
    return row[0] if row else "v1"


def all_entity_tokens(grid: dict) -> list[tuple[str, str]]:
    out = []
    for e in config.ENTITIES:
        if e in config.PER_TOKEN:
            out += [(e, c) for c in grid["cons"]]
        else:
            out.append((e, config.GLOBAL_TOKEN))
    return out


# ---------- события и выходы ----------
def seed(meta: Path) -> None:
    subprocess.run([str(hyppo_python()), str(WORKERS), "seed", "--meta", str(meta)],
                   check=True, env=worker_env(), capture_output=True)


def revise(meta: Path, entity: str) -> None:
    subprocess.run([str(hyppo_python()), str(REVISE), "--meta", str(meta), "--entity", entity],
                   check=True, env=worker_env(), capture_output=True)


def check_event(ev: tuple, grid: dict) -> None:
    kind = ev[0]
    if kind == "param":
        if ev[1] != "ridge":
            raise ValueError(f"поддерживается только параметр ridge: {ev}")
    elif kind == "evict":
        if ev[1] not in dag.jobs(grid):
            raise ValueError(f"нет такой задачи: {ev}")
    elif kind == "revise":
        if ev[1] not in config.ENTITIES:
            raise ValueError(f"нет такой сущности: {ev}")
    else:
        raise ValueError(f"неизвестное событие {ev}")


def output_digests(root: Path, grid: dict) -> dict[str, str | None]:
    """Дайджест каждого выхода (None -- файла нет). Как experiment._digests, но терпит пропуски."""
    import numpy as np
    d: dict[str, str | None] = {}
    for j in dag.jobs(grid):
        f = root / j
        if not f.exists():
            d[j] = None
        elif j.endswith(".json"):
            d[j] = json.loads(f.read_text())["digest"]
        else:
            d[j] = hashlib.sha256(np.round(np.load(f), 9).tobytes()).hexdigest()
    return d


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
