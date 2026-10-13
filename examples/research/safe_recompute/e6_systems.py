"""Э6: сравнение с существующими системами (Snakemake, DVC, Dagster) и комплексом.

1. `real_runs` -- реальные прогоны конвейера Brugge (`baseline_filecache`) в
   Snakemake, DVC, Dagster (порты `ports/`) и в комплексе (граф гипотез
   алгоритмом 1 из `baseline_filecache.hyppo_side`, менеджер в режиме
   planner 'safe', цепочечные ключи) на одной последовательности из 13 событий
   (`EVENTS`): 5 правок параметра ridge (3 из них не меняют выход: "10.0",
   "3.0", "3" дают тот же float, что предыдущее значение), 5 удалений выхода, 3 ревизии сущностей
   хранилища (файлы рабочего каталога не меняются). События применяются по
   порядку, состояние сохраняется. Эталон -- полный пересчёт с нуля для
   каждого состояния мира (ridge, применённые ревизии); устаревший
   результат -- выход, дайджест которого (округление до 1e-9, как в
   `baseline_filecache`) отличается от эталона.
2. `validate_simulators` -- те же события в `simulators.run_sequence` (состояние
   сохраняется, тот же порядок; изменения эталона берутся из реального полного
   пересчёта по точным sha256 файлов) и сравнение множеств выполненных и
   неверных задач; нативный evict сверяется с `simulators.native_evict`.
3. `simulate_corpora` -- симуляторы на CRM, 26 графах nf-core и 171 трассе
   WfCommons: случайные события, доля объявленных пар p in {0, 0.5, 1}.

Запуск:  python -m examples.research.safe_recompute.e6_systems real small|full
         python -m examples.research.safe_recompute.e6_systems scale 200
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx

from examples.research.baseline_filecache import config, dag, experiment, hyppo_side, metastore
from examples.research.safe_recompute import ports, simulators as sim
from examples.research.safe_recompute.common import (DATA, crm_graph, load_json, nfcore_graphs,
                                                     save_json, wfcommons_graphs)

SEED = 20261013
OUT = DATA / "e6_systems.json"

# Последовательность событий реальных прогонов. evict p1 -- до ревизий gains/tau:
# crm_fit перезаписывает gains/tau в хранилище (так ведёт себя сам конвейер).
# Значения ridge без выхода: только пары «целое / с точкой» ("10" -> "10.0"):
# Snakemake разбирает значения --config (1e0 и 1.0 -> один float 1.0, правки
# для него нет), а DVC и Dagster сравнивают строки. Возврат к прежнему
# значению ("1") не используется: комплекс по цепочечному ключу взял бы
# прежнюю запись (мемоизация), этого нет ни в симуляторе, ни в DVC без run-cache.
EVENTS = [
    ("param", "ridge", "10"),          # выход fit_liq меняется
    ("evict", "p1/uto.npy"),
    ("param", "ridge", "10.0"),        # тот же float: выход не меняется
    ("evict", "hyb/pos_1.npy"),
    ("param", "ridge", "3"),           # выход меняется
    ("evict", "wct/2_2.npy"),
    ("revise", "corey_ref"),           # объявлена пара corey_ref:fit_wct (fit_opr -- нет)
    ("param", "ridge", "3.0"),         # тот же float
    ("evict", "opr/uto_1_2_2.json"),
    ("revise", "tau"),                 # не объявлена: пропуск у инструментов
    ("param", "ridge", "3"),           # тот же float, но hyb устарел после tau
    ("evict", "hyb/uto_1.npy"),
    ("revise", "well_status"),         # объявлена пара well_status:fit_liq (fit_opr -- нет)
]
DECLARED = ("corey_ref:fit_wct", "well_status:fit_liq")
P_SENSITIVITY = (0.5, 0.7, 0.9, 1.0)
REAL_SYSTEMS = ("snakemake", "dvc", "dagster", "hyppo_safe")
PS = (0.0, 0.5, 1.0)
WEIGHTS = {"param": 6, "evict": 3, "revise": 1}
P_OUTPUT_CHANGED = 0.7


def grid_of(name: str) -> dict:
    return {"small": config.SMALL_GRID, "full": config.GRID}[name]


def _port(system: str):
    from examples.research.safe_recompute.ports import dagster_port, dvc_port, snakemake_port
    return {"snakemake": snakemake_port, "dvc": dvc_port, "dagster": dagster_port}[system]


# ---------- граф задач и хранилище Brugge для симулятора ----------
def task_graph(grid: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(dag.jobs(grid))
    g.add_edges_from((p, j) for j, ps in dag.parents(grid).items() for p in ps)
    return g


def brugge_store(grid: dict, declared=DECLARED) -> sim.Store:
    """Сущности хранилища -> задачи-потребители (по config.PAIRS); ridge -- параметр
    fit_liq. Ревизия gains/tau правит все токены, поэтому сущность -- целиком."""
    jobs = dag.jobs(grid)
    cons = {e: frozenset(j for j in jobs if (e, ports.rule_of(j)) in config.PAIRS)
            for e in config.ENTITIES}
    cons["ridge"] = frozenset(j for j in jobs if ports.rule_of(j) == "fit_liq")
    pairs = ports.declared_pairs(declared)
    decl = frozenset((e, j) for e in config.ENTITIES for j in cons[e] if (e, ports.rule_of(j)) in pairs)
    return sim.Store(consumers=cons, declared=decl, params=frozenset({"ridge"}))


def sim_events(events) -> list[sim.Event]:
    """События портов -> события симулятора (output_changed не используется:
    изменения задаёт эталон)."""
    return [sim.Event(ev[0], ev[1], True) for ev in events]


# ---------- эталон: полный пересчёт ----------
def _exact(root: Path, grid: dict) -> dict[str, str | None]:
    return {j: (ports.file_sha256(root / j) if (root / j).exists() else None) for j in dag.jobs(grid)}


def _run_jobs(work: Path, meta: Path, ridge: str, jobs, threads: int) -> None:
    def one(j):
        res = subprocess.run(ports.job_command(j, meta, ridge), cwd=work, capture_output=True,
                             text=True, env=ports.worker_env())
        if res.returncode != 0:
            raise RuntimeError(f"{j}: {res.stdout}\n{res.stderr}")
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, jobs))


def worlds(events) -> list[tuple]:
    """Состояние мира после каждого события (индекс 0 -- до событий):
    (float(ridge), применённые ревизии). evict мир не меняет."""
    ridge, revs = "1", ()
    out = [(float(ridge), revs)]
    for ev in events:
        if ev[0] == "param":
            ridge = ev[2]
        elif ev[0] == "revise":
            revs = revs + (ev[1],)
        out.append((float(ridge), revs))
    return out


def gold_runs(root: Path, grid: dict, events, threads: int = 4) -> dict:
    """Эталон: для каждого различного состояния мира -- полный пересчёт с нуля
    (crm_fit один раз, его p1/gains/tau общие: от ridge и ревизий они не зависят;
    ревизии применяются к хранилищу по порядку, как в портах)."""
    t0 = time.perf_counter()
    root = Path(root)
    base, bmeta = root / "base", root / "base_meta.sqlite"
    base.mkdir(parents=True)
    ports.seed(bmeta)
    _run_jobs(base, bmeta, "1", [j for j in dag.jobs(grid) if ports.rule_of(j) == "crm_fit"], threads)
    ws = worlds(events)
    cache: dict[tuple, tuple[dict, dict]] = {}
    for i, w in enumerate(ws):
        if w in cache:
            continue
        d = root / f"w{len(cache)}"
        work, meta = d / "work", d / "meta.sqlite"
        shutil.copytree(base, work)
        shutil.copy2(bmeta, meta)
        for e in w[1]:
            metastore.revise(meta, e)
        ridge = repr(w[0])
        level1 = [j for j in dag.jobs(grid) if ports.rule_of(j) in ("fit_liq", "fit_wct")]
        level2 = [j for j in dag.jobs(grid) if ports.rule_of(j) == "fit_opr"]
        _run_jobs(work, meta, ridge, level1, threads)
        _run_jobs(work, meta, ridge, level2, threads)
        cache[w] = (ports.output_digests(work, grid), _exact(work, grid))
    return {"digests": [cache[w][0] for w in ws], "exact": [cache[w][1] for w in ws],
            "worlds": len(cache), "seconds": time.perf_counter() - t0}


def deltas_of(gold: dict, key: str = "exact") -> list[set[str]]:
    seq = gold[key]
    return [{j for j in seq[i + 1] if seq[i + 1][j] != seq[i][j]} for i in range(len(seq) - 1)]


# ---------- комплекс ----------
def _store_node(name: str) -> tuple[str, str] | None:
    if name.startswith("store_gains_"):
        return "gains", name[len("store_gains_"):]
    if name.startswith("store_tau_"):
        return "tau", name[len("store_tau_"):]
    return {"store_corey": ("corey_ref", config.GLOBAL_TOKEN),
            "store_status": ("well_status", config.GLOBAL_TOKEN)}.get(name)


def run_hyppo(workdir: Path, grid: dict, events, declared=DECLARED) -> dict:
    """Конвейер Brugge под управлением комплекса (режим 'safe').

    Граф -- алгоритм 1 над гипотезами `hyppo_side` (гипотезы-хранилища store_*
    и параметр par_ridge -- отдельные узлы); конфигурация гипотезы-хранилища --
    версия сущности в хранилище (комплекс читает её всегда, `declared` не
    используется: сущность входит в уравнение модели потребителя), par_ridge --
    значение ridge. План -- `Manager(plan_mode="safe")` по цепочечным ключам;
    evict -- удаление файла выхода и записи репозитория по её ключу.
    Результат комплекса -- запись репозитория (sha256 выхода) и артефакт в
    хранилище артефактов; при переиспользовании записи рабочий файл
    приводится к её артефакту (`res["restored"]`, не считается выполнением).
    Интерфейс результата -- как у портов (initial/after/outputs/seconds)."""
    from hyppo.manager import Manager

    for ev in events:
        ports.check_event(ev, grid)
    root = Path(workdir)
    work, meta = root / "work", root / "meta.sqlite"
    work.mkdir(parents=True, exist_ok=False)
    ports.seed(meta)
    model = hyppo_side.build(grid)
    job_of = model.node_job
    node_of = {j: n for n, j in job_of.items()}
    code = ports.file_sha256(ports.WORKERS)[:12]
    model_ids = {n: (f"{ports.rule_of(job_of[n])}:{code}" if n in job_of else "store") for n in model.nodes}
    state = {"ridge": "1"}
    calls: list[str] = []
    restored: list[str] = []     # артефакты, восстановленные из хранилища комплекса
    arts = root / "artifacts"    # артефакт результата по sha256 (запись хранит sha256)
    arts.mkdir()

    def mk(n):
        if n not in job_of:
            return lambda cfg: {}
        j = job_of[n]

        def f(cfg):
            calls.append(j)
            res = subprocess.run(ports.job_command(j, meta, state["ridge"]), cwd=work,
                                 capture_output=True, text=True, env=ports.worker_env())
            if res.returncode != 0:
                raise RuntimeError(res.stdout + res.stderr)
            sha = ports.file_sha256(work / j)
            art = arts / sha
            if not art.exists():
                shutil.copy2(work / j, art)
            return {"sha256": sha}
        return f

    models = {n: mk(n) for n in model.nodes}

    def cfg():
        c = {n: {} for n in model.nodes}
        c["par_ridge"] = {"ridge": state["ridge"]}
        for n in model.nodes:
            st = _store_node(n)
            if st:
                c[n] = {"version": ports.store_version(meta, *st)}
        return c

    mgr = Manager(db_path=root / "hyppo.db", plan_mode="safe")
    hg = nx.DiGraph()
    hg.add_nodes_from(model.nodes)
    hg.add_edges_from(model.edges)

    def step() -> set[str]:
        calls.clear()
        res = mgr.orchestrate(hypotheses=model.nodes, workflow_edges=model.edges, models=models,
                              config=cfg(), model_ids=model_ids)
        bad = {h: r for h, r in res.items() if r["status"] != "SUCCESS"}
        if bad:
            raise RuntimeError(f"комплекс: неуспешные гипотезы {bad}")
        if len(calls) != len(set(calls)):
            raise RuntimeError(f"задача выполнена дважды: {calls}")
        # переиспользованная запись -> её артефакт в рабочем каталоге
        for n, j in job_of.items():
            if j in calls:
                continue
            sha = res[n]["metrics"]["sha256"]
            f = work / j
            if not f.exists() or ports.file_sha256(f) != sha:
                shutil.copy2(arts / sha, f)
                restored.append(j)
        return set(calls)

    t0 = time.perf_counter()
    res = {"initial": step(), "after": [], "outputs": [ports.output_digests(work, grid)],
           "graph_edges": len(model.edges), "graph_build_s": model.build_seconds, "restored": [list(restored)]}
    res["seconds"] = [time.perf_counter() - t0]
    try:
        for ev in events:
            if ev[0] == "param":
                state["ridge"] = str(ev[2])
            elif ev[0] == "evict":
                n = node_of[ev[1]]
                keys = mgr.cache_keys(hg, models, cfg(), model_ids)
                if not mgr.repository.evict(n, keys[n]):
                    raise RuntimeError(f"нет записи {n} в репозитории")
                (work / ev[1]).unlink()
            else:
                ports.revise(meta, ev[1])
            t0 = time.perf_counter()
            restored.clear()
            res["after"].append(step())
            res["seconds"].append(time.perf_counter() - t0)
            res["restored"].append(list(restored))
            res["outputs"].append(ports.output_digests(work, grid))
    finally:
        mgr.close()
    return res


# ---------- реальные прогоны ----------
def _snk_triggers_check(work: Path, meta: Path, grid_name: str, grid: dict, ridge: str) -> dict:
    """Snakemake: `--rerun-triggers mtime params` против триггеров по умолчанию
    (все: mtime, params, input, software-env, code) -- сухие прогоны в конечном
    состоянии: без правки и с новым ridge."""
    pairs = sorted(ports.declared_pairs(DECLARED))

    def dry(r, default):
        cmd = [str(config.SNAKEMAKE_PY), "-m", "snakemake", "-s", str(experiment.SNAKEFILE),
               "--directory", str(work)]
        if not default:
            cmd += ["--rerun-triggers", "mtime", "params"]
        cmd += ["--config", f"meta={meta}", f"grid={grid_name}",
                "declared=" + ",".join(config.pair_id(p) for p in pairs), f"ridge={r}", "-n", "--cores", "1"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stdout + out.stderr)
        return sorted(experiment._dry_run_jobs(out.stdout + out.stderr, grid))

    res = {}
    for label, r in (("no_change", ridge), ("ridge_changed", "7")):
        a, b = dry(r, False), dry(r, True)
        res[label] = {"mtime_params": len(a), "default": len(b), "same": a == b}
    return res


def _sets(x):
    return sorted(x)


def derive_fields(rec: dict) -> dict:
    """Производные поля прогона системы: wrong_at_end -- неверные результаты после
    последнего события; miss_spans -- для каждой группы пропущенных (событие
    появления, событие исправления): сколько событий результат оставался неверным
    (repaired_at_event=None -- не исправлен до конца последовательности)."""
    evs = rec["events"]
    rec["wrong_at_end"] = len(evs[-1]["wrong"]) if evs else 0
    spans: dict = {}
    for k, e in enumerate(evs):
        for j in e["missed"]:
            to = next((t for t in range(k + 1, len(evs)) if j not in evs[t]["wrong"]), None)
            spans[(k, to)] = spans.get((k, to), 0) + 1
    rec["miss_spans"] = [{"from_event": k, "repaired_at_event": to,
                          "events_wrong": (to - k) if to is not None else len(evs) - k, "jobs": n}
                         for (k, to), n in sorted(spans.items(), key=lambda x: (x[0][0], x[0][1] is None, x[0][1] or 0))]
    for nv in rec.get("native_evict", []):
        if nv["event"][1] in nv["executed"]:  # пересчёт в нативном плане -- не «отсутствует»
            nv["absent"] = []
        nv.setdefault("absent_source", "measured" if ("restored" in nv or nv.get("same_as_forced")) else "inferred")
    return rec


def _record(name: str, res: dict, gold: dict, grid: dict, events) -> dict:
    jobs = set(dag.jobs(grid))
    init_wrong = {j for j in jobs if res["outputs"][0][j] != gold["digests"][0][j]}
    rec = {"initial_executed": len(res["initial"]), "initial_all": res["initial"] == jobs,
           "initial_wrong": _sets(init_wrong), "seconds_initial": res["seconds"][0], "events": []}
    prev = init_wrong
    for k, ev in enumerate(events):
        out = res["outputs"][k + 1]
        wrong = {j for j in jobs if out[j] != gold["digests"][k + 1][j]}
        e = {"event": list(ev), "executed": _sets(res["after"][k]), "n_executed": len(res["after"][k]),
             "wrong": _sets(wrong), "missed": _sets(wrong - prev), "seconds": res["seconds"][k + 1]}
        prev = wrong
        rec["events"].append(e)
    ev_idx = [k for k, ev in enumerate(events) if ev[0] == "evict"]
    native = []
    if name == "snakemake":
        measured = res.get("native_absent") or [None] * len(ev_idx)
        for k, ex, ab in zip(ev_idx, res["native_evict"], measured):
            # сухой прогон файл не создаёт: отсутствует = нет на диске и не в плане
            if ab is None:  # старые прогоны: файл не проверялся
                ab, src = events[k][1] not in ex, "inferred"
            else:
                ab, src = ab and events[k][1] not in ex, "measured"
            native.append({"event": list(events[k]), "executed": _sets(ex),
                           "absent": [events[k][1]] if ab else [], "absent_source": src})
        if res.get("trigger_check"):
            rec["trigger_check"] = [{"event": list(ev), "mtime_params": _sets(t["mtime_params"]),
                                     "default": _sets(t["default"]),
                                     "same": set(t["mtime_params"]) == set(t["default"])}
                                    for ev, t in zip(events, res["trigger_check"])]
    elif name == "dvc":
        for k, nv in zip(ev_idx, res["native_evict"]):
            native.append({"event": list(events[k]), "executed": _sets(nv["executed"]),
                           "absent": [] if nv["restored"] else [events[k][1]], "restored": nv["restored"],
                           "absent_source": "measured"})
    elif name == "dagster" and res.get("native_evict"):
        for k, nv in zip(ev_idx, res["native_evict"]):
            native.append({"event": list(events[k]), "executed": _sets(nv["executed"]),
                           "status_evicted": nv["status_evicted"],
                           "absent": [events[k][1]] if nv["absent"] else [], "absent_source": "measured"})
    elif name == "dagster":  # без нативного шага: выбор первого раунда forced-шага без force
        for k in ev_idx:
            r0 = res["rounds"][k + 1][0]
            sel = {j for j, s in r0["status"].items() if s in ("STALE", "MISSING")}
            native.append({"event": list(events[k]), "executed": _sets(sel),
                           "status_evicted": r0["status"][events[k][1]], "absent": [events[k][1]],
                           "absent_source": "inferred"})
    else:  # комплекс: удаление записи и есть его нативный evict
        for k in ev_idx:
            native.append({"event": list(events[k]), "executed": rec["events"][k]["executed"],
                           "absent": [], "same_as_forced": True, "absent_source": "measured"})
    rec["native_evict"] = native
    derive_fields(rec)
    rec["seconds_events"] = sum(res["seconds"][1:])
    rec["executed_total"] = sum(e["n_executed"] for e in rec["events"])
    rec["missed_total"] = sum(len(e["missed"]) for e in rec["events"])
    if "graph_edges" in res:
        rec["restored"] = res["restored"]
        rec["graph_edges"] = res["graph_edges"]
        rec["graph_build_s"] = res["graph_build_s"]
    return rec


def _rmtree(path: Path) -> None:
    import os
    import stat

    def onerror(func, q, _):  # объекты кэша DVC только для чтения
        os.chmod(q, stat.S_IWRITE)
        func(q)
    shutil.rmtree(path, onerror=onerror)


def real_runs(grid_name: str = "small", out=None, root=None, systems=REAL_SYSTEMS, threads: int = 4,
              native_checks: bool = True) -> dict:
    """native_checks: Snakemake -- перед каждым шагом сухие прогоны с триггерами
    `mtime params` и по умолчанию; Dagster -- отдельный нативный шаг при evict
    (файл проверяется на диске). Временный каталог (root=None) удаляется."""
    grid = grid_of(grid_name)
    if not (ports.brugge_run_dir() / "brugge_perwell.npz").exists():
        raise RuntimeError("нет данных Brugge (HYPPO_BRUGGE_RUN)")
    tmp = root is None
    root = Path(root) if root else Path(tempfile.mkdtemp(prefix=f"e6_{grid_name}_"))
    try:
        return _real_runs(grid_name, grid, root, out, systems, threads, native_checks)
    finally:
        if tmp:
            _rmtree(root)


def _real_runs(grid_name, grid, root, out, systems, threads, native_checks) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    print(f"[e6] {grid_name}: каталог {root}", flush=True)
    gold = gold_runs(root / "gold", grid, EVENTS, threads)
    print(f"[e6] эталон: {gold['worlds']} состояний, {gold['seconds']:.1f} с", flush=True)
    d_exact, d_round = deltas_of(gold, "exact"), deltas_of(gold, "digests")
    result = {
        "grid": grid_name, "jobs": len(dag.jobs(grid)), "events": [list(e) for e in EVENTS],
        "declared": list(DECLARED),
        "gold": {"worlds": gold["worlds"], "seconds": gold["seconds"],
                 "delta_exact": [_sets(d) for d in d_exact], "delta_rounded": [_sets(d) for d in d_round],
                 "delta_exact_equals_rounded": d_exact == d_round},
        "systems": {},
    }
    raw = {}
    for name in systems:
        t0 = time.perf_counter()
        wd = root / name
        if name == "hyppo_safe":
            res = run_hyppo(wd, grid, EVENTS)
        else:
            mod = _port(name)
            if not mod.tools_available():
                raise RuntimeError(f"{name}: инструмент недоступен")
            kw = {}
            if native_checks and name == "snakemake":
                kw["trigger_check"] = True
            if native_checks and name == "dagster":
                kw["native_step"] = True
            res = mod.run_workload(wd, grid, EVENTS, set(DECLARED), **kw)
        raw[name] = res
        result["systems"][name] = _record(name, res, gold, grid, EVENTS)
        result["systems"][name]["seconds_total_wall"] = time.perf_counter() - t0
        print(f"[e6] {name}: {time.perf_counter() - t0:.1f} с, выполнено "
              f"{result['systems'][name]['executed_total']}, пропущено "
              f"{result['systems'][name]['missed_total']}", flush=True)
    if "snakemake" in systems:
        last_ridge = [e[2] for e in EVENTS if e[0] == "param"][-1]
        result["snakemake_triggers_check"] = _snk_triggers_check(
            root / "snakemake" / "work", root / "snakemake" / "meta.sqlite", grid_name, grid, last_ridge)
    if "snakemake" in systems:
        tc = result["systems"]["snakemake"].get("trigger_check")
        if tc:
            result["snakemake_triggers_per_event"] = {"checked": len(tc), "same": sum(t["same"] for t in tc)}
    if out is not None:
        save_json(result, Path(out))
    return result


# ---------- сверка симуляторов ----------
def validate_simulators(real: dict) -> dict:
    """Сравнить run_sequence (тот же порядок, состояние сохраняется) с реальными
    множествами выполненных и неверных задач; нативный evict -- с native_evict."""
    grid = grid_of(real["grid"])
    events = [tuple(e) for e in real["events"]]
    g = task_graph(grid)
    store = brugge_store(grid, real["declared"])
    deltas = [set(d) for d in real["gold"]["delta_exact"]]
    out = {}
    for name, rec in real["systems"].items():
        sims = sim.run_sequence(name, g, sim_events(events), deltas, store)
        rows = []
        for ev, s, r in zip(events, sims, rec["events"]):
            ex_ok = s.recomputed == set(r["executed"])
            wr_ok = s.wrong == set(r["wrong"])
            row = {"event": list(ev), "executed_match": ex_ok, "wrong_match": wr_ok}
            if not (ex_ok and wr_ok):
                row["sim_executed"] = _sets(s.recomputed)
                row["sim_wrong"] = _sets(s.wrong)
            rows.append(row)
        n_ok = sum(r["executed_match"] and r["wrong_match"] for r in rows)
        nat_rows = []
        ev_idx = [k for k, ev in enumerate(events) if ev[0] == "evict"]
        if len(ev_idx) != len(rec["native_evict"]):
            raise ValueError(f"{name}: число нативных evict не совпадает с событиями")
        for k, nv in zip(ev_idx, rec["native_evict"]):
            if tuple(nv["event"]) != events[k]:
                raise ValueError(f"{name}: нативный evict {nv['event']} не на месте события {k}")
            node = nv["event"][1]
            ns = sim.native_evict(name, node, g)
            if ns is None:  # нативный = forced
                ns = sims[k]
                ok = set(nv["executed"]) == ns.recomputed
            else:
                ok = set(nv["executed"]) == ns.recomputed and set(nv["absent"]) == ns.absent
            nat_rows.append({"event": nv["event"], "match": ok, "absent_source": nv.get("absent_source")})
        out[name] = {
            "forced": {"events": len(rows), "matched": n_ok, "share": n_ok / len(rows), "rows": rows},
            "native_evict": {"events": len(nat_rows), "matched": sum(r["match"] for r in nat_rows),
                             "share": (sum(r["match"] for r in nat_rows) / len(nat_rows)) if nat_rows else 1.0,
                             "rows": nat_rows},
            "initial_all": rec["initial_all"], "initial_wrong": rec["initial_wrong"],
        }
    return out


# ---------- корпуса ----------
def corpus_graphs(graphs_limit: int | None = None) -> dict[str, nx.DiGraph]:
    out = {"crm/brugge": crm_graph()}
    out.update({f"nfcore/{k}": v for k, v in nfcore_graphs().items()})
    if graphs_limit is not None and len(out) >= graphs_limit:
        return dict(list(out.items())[:graphs_limit])
    rest = None if graphs_limit is None else graphs_limit - len(out)
    out.update({f"wfcommons/{k}": v for k, v in wfcommons_graphs(rest).items()})
    return out


def gen_world(name: str, g: nx.DiGraph, n_events: int, seed: int = SEED, p_out: float = P_OUTPUT_CHANGED):
    """Генератор (детерминирован по seed и имени графа):
    сущности: m = ceil(|V|/4), у каждой 1..3 потребителя (равномерно, без повторов);
    пары (сущность, потребитель) перемешаны, объявлены первые round(p*|пар|)
    (вложенные множества по p); события: тип param:evict:revise = 6:3:1,
    param -- равномерный узел, выход меняется с вероятностью p_out (0.7; тогда
    меняются узел и все потомки, иначе ничего); evict -- равномерный узел (мир
    не меняется); revise -- равномерная сущность (меняются все потребители и их
    потомки)."""
    rng = random.Random(f"{seed}:{name}")
    nodes = sorted(g.nodes, key=str)
    m = max(1, math.ceil(len(nodes) / 4))
    consumers = {f"e{i}": frozenset(rng.sample(nodes, rng.randint(1, min(3, len(nodes))))) for i in range(m)}
    pairs = sorted((e, c) for e, cs in consumers.items() for c in cs)
    rng.shuffle(pairs)
    kinds = list(WEIGHTS)
    events = []
    for _ in range(n_events):
        k = rng.choices(kinds, weights=[WEIGHTS[x] for x in kinds])[0]
        if k == "param":
            events.append(sim.Event("param", rng.choice(nodes), rng.random() < p_out))
        elif k == "evict":
            events.append(sim.Event("evict", rng.choice(nodes), False))
        else:
            events.append(sim.Event("revise", rng.choice(sorted(consumers)), True))
    return consumers, pairs, events


def _declared_for(pairs, p: float) -> frozenset:
    return frozenset(pairs[:int(math.floor(p * len(pairs) + 0.5))])


def simulate_corpora(events_per_graph: int = 200, out=None, graphs_limit: int | None = None,
                     seed: int = SEED, ps=PS, systems=sim.SYSTEMS, p_out: float = P_OUTPUT_CHANGED) -> dict:
    """Корпуса; by_p[p].by_kind -- forced evict, by_kind_native -- нативный evict
    (param/revise совпадают с forced; evict: выполнено по native_evict, пропусков
    нет, отсутствующие файлы -- absent)."""
    t0 = time.perf_counter()
    graphs = corpus_graphs(graphs_limit)
    zero = lambda: {"recomputed_total": 0, "missed_total": 0, "events": 0,
                    "by_kind": {k: {"recomputed": 0, "missed": 0, "events": 0} for k in WEIGHTS},
                    "by_kind_native": {k: {"recomputed": 0, "missed": 0, "absent": 0, "events": 0}
                                       for k in WEIGHTS}}
    acc = {s: {str(p): zero() for p in ps} for s in systems}
    native = {s: {"evict_events": 0, "executed": 0, "absent": 0, "restored": 0}
              for s in systems if sim.NATIVE_EVICT.get(s)}
    cutoff = [s for s in systems if s in sim.EARLY_CUTOFF]
    fav = {str(p): {s: {"fewer_safe": 0, "fewer_with_miss": 0, "more": 0, "equal": 0,
                        "fewer_safe_by_kind": {k: 0 for k in WEIGHTS}} for s in cutoff} for p in ps}
    sizes = {"graphs": len(graphs), "nodes": 0, "edges": 0, "events": 0}
    for name, g in graphs.items():
        gg = sim._Graph(g)
        consumers, pairs, events = gen_world(name, g, events_per_graph, seed, p_out)
        sizes["nodes"] += g.number_of_nodes()
        sizes["edges"] += g.number_of_edges()
        sizes["events"] += len(events)
        for p in ps:
            store = sim.Store(consumers=consumers, declared=_declared_for(pairs, p))
            deltas = [sim.default_delta(gg, ev, store) for ev in events]
            res = {s: sim.run_sequence(s, gg, events, deltas, store) for s in systems}
            for s in systems:
                a = acc[s][str(p)]
                for ev, r in zip(events, res[s]):
                    a["recomputed_total"] += len(r.recomputed)
                    a["missed_total"] += len(r.missed)
                    a["events"] += 1
                    bk = a["by_kind"][ev.kind]
                    bk["recomputed"] += len(r.recomputed)
                    bk["missed"] += len(r.missed)
                    bk["events"] += 1
                    nr = sim.native_evict(s, ev.node, g) if ev.kind == "evict" else None
                    bn = a["by_kind_native"][ev.kind]
                    bn["recomputed"] += len((nr or r).recomputed)
                    bn["missed"] += len((nr or r).missed)
                    bn["absent"] += len(nr.absent) if nr else 0
                    bn["events"] += 1
            if "hyppo_safe" in res:
                for s in cutoff:
                    f = fav[str(p)][s]
                    for ev, rc, rh in zip(events, res[s], res["hyppo_safe"]):
                        a, b = len(rc.recomputed), len(rh.recomputed)
                        if a < b and not rc.wrong:
                            f["fewer_safe"] += 1
                            f["fewer_safe_by_kind"][ev.kind] += 1
                        elif a < b:
                            f["fewer_with_miss"] += 1
                        elif a > b:
                            f["more"] += 1
                        else:
                            f["equal"] += 1
        for s in native:
            for ev in events:
                if ev.kind == "evict":
                    nr = sim.native_evict(s, ev.node, g)
                    native[s]["evict_events"] += 1
                    native[s]["executed"] += len(nr.recomputed)
                    native[s]["absent"] += len(nr.absent)
                    native[s]["restored"] += int(not nr.absent and not nr.recomputed)
    headline = "0.5" if "0.5" in acc[systems[0]] else str(ps[0])
    per_system = {}
    for s in systems:
        h = acc[s][headline]
        per_system[s] = {"headline": f"p={headline}, evict=forced",
                         "recomputed_total": h["recomputed_total"], "missed_total": h["missed_total"],
                         "events": h["events"], "by_p": acc[s],
                         "validated_by_real_run": s in sim.VALIDATED}
        if s in native:
            per_system[s]["native_evict"] = native[s]
    res = {
        "meta": {
            "seed": seed, "events_per_graph": events_per_graph, "ps": list(ps), "corpus": sizes,
            "p_output_changed": p_out, "weights": dict(WEIGHTS), "revise_always_changes_output": True,
            "entities_per_node": 0.25, "consumers_per_entity": [1, 3],
            "delta_rule": "closure over descendants (no mid-chain cutoff in corpora; cutoff-tool counts "
                          "are upper bounds)",
            "generator": " ".join(gen_world.__doc__.split()),
            "state": "events per graph applied in order with persistent state (simulators.run_sequence)",
            "missed": "newly wrong results kept after the event (W' minus W), summed over events",
            "native_evict": "evaluated per evict event; state continues from the forced variant "
                            "(as in the real runs, where each native step is followed by the forced one)",
            "not_validated_by_real_run": list(sim.NOT_VALIDATED),
            "seconds": None,
        },
        "per_system": per_system,
        "favourable_worlds": {
            "note": "events where an early-cutoff system executes strictly fewer jobs than hyppo_safe: "
                    "fewer_safe -- and no wrong result after the event (output unchanged: favourable world), "
                    "fewer_with_miss -- with wrong results kept (undeclared revisions, now or earlier); more -- the cutoff "
                    "system executes more (repairs results it missed earlier). Consistent with theorem 2 "
                    "(minimax: hyppo_safe is minimal over all worlds consistent with the observations, "
                    "not in each world)",
            "by_p": fav,
        },
    }
    res["meta"]["seconds"] = time.perf_counter() - t0
    if out is not None:
        save_json(res, Path(out))
    return res


def sensitivity(events_per_graph: int = 200, graphs_limit: int | None = None, seed: int = SEED,
                p_values=P_SENSITIVITY, ps=PS) -> dict:
    """Чувствительность к P(output_changed): recomputed/missed по системам и p,
    плюс число благоприятных миров DVC (fewer_safe). Миры при разных P общие
    (те же случайные числа, меняется только порог)."""
    out = {"p_output_changed": list(p_values), "by_P": {}}
    for P in p_values:
        c = simulate_corpora(events_per_graph, graphs_limit=graphs_limit, seed=seed, ps=ps, p_out=P)
        out["by_P"][str(P)] = {
            "per_system": {s: {p: {"recomputed": x["recomputed_total"], "missed": x["missed_total"]}
                               for p, x in v["by_p"].items()} for s, v in c["per_system"].items()},
            "dvc_fewer_safe": {p: v["dvc"]["fewer_safe"] for p, v in c["favourable_worlds"]["by_p"].items()},
        }
    return out


# ---------- CLI ----------
META = {
    "modes": {
        "snakemake": "Snakemake 9.23.1, --rerun-triggers mtime params; evict forced = explicit target "
                     "'all <job>', native = target 'all' (dry run)",
        "dvc": "DVC 3.67.1, dvc repro --no-run-cache; evict forced = output file and cache object "
               "deleted ('cache object lost'), native = only the file deleted (restored from cache). "
               "run-cache off is conservative: can only add executions, never misses",
        "dagster": "Dagster 1.13.26, stale-based materialization (stale_assets_only / data_version_changed: "
                   "STALE|MISSING to a fixed point); evict forced = manual rematerialization, native = "
                   "deleted file not noticed (status FRESH)",
        "dagster_eager": "AutomationCondition.eager(): any_deps_updated on any new parent "
                         "materialization, no cutoff; simulator only",
        "hyppo_safe": "the complex: Algorithm-1 graph of baseline_filecache.hyppo_side, "
                      "Manager(plan_mode='safe'), chained keys; store revision visible regardless "
                      "of declaration",
        "nextflow": "cache='standard' (path+size+mtime); simulator only",
        "nextflow_deep": "cache='deep' (content hash, early cutoff); simulator only",
        "bazel": "action-digest early cutoff; simulator only",
        "hyppo_cascade": "planner 'cascade' (Cl(N)); simulator only",
    },
    "not_validated_by_real_run": list(sim.NOT_VALIDATED),
    "snakemake_triggers": "real runs use --rerun-triggers mtime params. Compared with the default (all "
                          "triggers) by dry runs: (a) both grids -- two dry runs in the final state, no edit "
                          "and ridge=7 (real.<grid>.snakemake_triggers_check); (b) small grid -- before each "
                          "of the 13 steps, dry runs with both settings and the step's targets "
                          "(real.small.systems.snakemake.trigger_check, summary "
                          "real.small.snakemake_triggers_per_event). The full grid was not checked per event",
    "native_evict_absent": "absent_source=measured: file checked on disk after the native step "
                           "(Snakemake dry run; Dagster separate native step without force; DVC restored "
                           "flag); inferred: not checked (full grid run before the check was added)",
    "gold": "full recompute from scratch for every world state; wrong = output digest (rounded 1e-9) "
            "differs from gold; simulator deltas = exact sha256 changes of the gold outputs",
    "order": "events applied sequentially with persistent state in both real runs and run_sequence",
    "scope": "validation covers one designed 13-event sequence per grid (small, full); mid-chain early "
             "cutoff (a recomputed node whose output is unchanged while its parent changed) is not "
             "exercised; a return to an earlier parameter value (memoisation by chained key in the "
             "complex) and Snakemake's --config parsing of equivalent numbers (1e0 = 1.0) are excluded "
             "by design; the completeness of the complex comes from the modeller's equations "
             "(baseline_filecache/hyppo_side.py: every store entity read by a model is in its equation)",
}


def refresh(data: dict) -> dict:
    """Пересчитать производные поля и сверку из записанных реальных прогонов."""
    for grid, real in data.get("real", {}).items():
        for rec in real["systems"].values():
            derive_fields(rec)
        val = validate_simulators(real)
        data.setdefault("validation", {})[grid] = {
            k: {"forced_share": v["forced"]["share"], "native_evict_share": v["native_evict"]["share"], **v}
            for k, v in val.items()}
    return data


def _load(out: Path) -> dict:
    return load_json(out) if Path(out).exists() else {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("grid", choices=["small", "full"])
    r.add_argument("--root")
    s = sub.add_parser("scale")
    s.add_argument("events", type=int, nargs="?", default=200)
    s.add_argument("--limit", type=int)
    sub.add_parser("derive")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    data = _load(out)
    data["meta"] = META
    if a.cmd == "real":
        real = real_runs(a.grid, root=a.root)
        val = validate_simulators(real)
        data.setdefault("real", {})[a.grid] = real
        for k, v in val.items():
            print(f"[e6] сверка {a.grid} {k}: forced {v['forced']['matched']}/{v['forced']['events']}, "
                  f"native {v['native_evict']['matched']}/{v['native_evict']['events']}")
    elif a.cmd == "scale":
        corp = simulate_corpora(a.events, graphs_limit=a.limit)
        data["corpus_meta"] = corp["meta"]
        data["per_system"] = corp["per_system"]
        data["favourable_worlds"] = corp["favourable_worlds"]
        data["sensitivity"] = sensitivity(a.events, graphs_limit=a.limit)
        for k, v in corp["per_system"].items():
            print(k, {p: (x["recomputed_total"], x["missed_total"]) for p, x in v["by_p"].items()})
    refresh(data)
    save_json(data, out)


if __name__ == "__main__":
    main(sys.argv[1:])
