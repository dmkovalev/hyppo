"""Порты конвейера Brugge на DVC и Dagster (Э6): чистые проверки и интеграция."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.research.baseline_filecache import dag
from examples.research.baseline_filecache.config import SMALL_GRID, GRID
from examples.research.safe_recompute import ports
from examples.research.safe_recompute.ports import dagster_port, dvc_port, snakemake_port


def _closure(grid, roots):
    """roots и все их потомки по файловым зависимостям dag.parents."""
    par = dag.parents(grid)
    out = set(roots)
    changed = True
    while changed:
        changed = False
        for j, ps in par.items():
            if j not in out and ps & out:
                out.add(j)
                changed = True
    return out


def _rule_jobs(grid, prefix):
    return {j for j in dag.jobs(grid) if j.startswith(prefix + "/")}


# ---------- чистые проверки ----------
def test_stage_names_unique_and_valid():
    names = [dvc_port.stage_name(j) for j in dag.jobs(GRID)]
    assert len(set(names)) == len(names)
    assert all(n.replace("_", "").isalnum() for n in names)


def test_declared_pairs_resolution():
    assert ports.declared_pairs({"corey_ref"}) == {("corey_ref", "fit_wct"), ("corey_ref", "fit_opr")}
    assert ports.declared_pairs({"gains:fit_liq"}) == {("gains", "fit_liq")}
    with pytest.raises(ValueError):
        ports.declared_pairs({"nonsense"})


@pytest.mark.parametrize("grid", [SMALL_GRID, GRID])
def test_dvc_yaml_one_stage_per_job_and_parent_deps(grid):
    y = dvc_port.dvc_yaml(grid, Path("meta.sqlite"), set())
    stages = y["stages"]
    jobs = dag.jobs(grid)
    assert set(stages) == {dvc_port.stage_name(j) for j in jobs}
    par = dag.parents(grid)
    for j in jobs:
        st = stages[dvc_port.stage_name(j)]
        assert set(st["deps"]) >= par[j]
        assert any(str(d).endswith("workers.py") for d in st["deps"])
        assert st["outs"] == [j]
        assert "workers.py" in st["cmd"]


def test_dvc_yaml_params_follow_declarations():
    g = SMALL_GRID
    none = dvc_port.dvc_yaml(g, Path("m.sqlite"), set())["stages"]
    some = dvc_port.dvc_yaml(g, Path("m.sqlite"), {"corey_ref", "gains:fit_liq"})["stages"]
    liq, wct, opr = (dvc_port.stage_name(j) for j in ("hyb/uto_1.npy", "wct/2_2.npy", "opr/uto_1_2_2.json"))
    # ridge -- всегда объявленный параметр fit_liq
    assert none[liq]["params"] == ["ridge"]
    assert "params" not in none[wct] and "params" not in none[opr]
    assert set(some[liq]["params"]) == {"ridge", "v_gains_uto"}
    assert some[wct]["params"] == ["v_corey_ref"]
    assert some[opr]["params"] == ["v_corey_ref"]


def test_params_yaml_keys_cover_stage_params(tmp_path):
    g = SMALL_GRID
    y = dvc_port.dvc_yaml(g, tmp_path / "m.sqlite", {"gains", "tau", "well_status", "corey_ref"})
    p = dvc_port.params_yaml(g, tmp_path / "m.sqlite", ridge="1")
    used = {k for st in y["stages"].values() for k in st.get("params", [])}
    assert used <= set(p)
    # сущность ещё не создана crm_fit -> версия, которую запишет производитель (v1)
    assert p["v_gains_uto"] == "v1"


# ---------- интеграция ----------
def _data_ok() -> bool:
    return (ports.brugge_run_dir() / "brugge_perwell.npz").exists()


EVENTS = [
    ("param", "ridge", "10"),
    # evict до ревизий: иначе пересчёт прочёл бы уже правленое хранилище
    # (необъявленная tau) и выход законно изменился бы
    ("evict", "hyb/uto_1.npy"),
    ("revise", "tau"),          # необъявлена -> пропуск
    ("revise", "corey_ref"),    # объявлена -> пересчёт потребителей
]
DECLARED = {"corey_ref"}


@pytest.fixture(scope="module", params=["dvc", "dagster", "snakemake"])
def workload(request, tmp_path_factory):
    mod = {"dvc": dvc_port, "dagster": dagster_port, "snakemake": snakemake_port}[request.param]
    if not mod.tools_available():
        pytest.skip(f"{request.param}: инструмент недоступен")
    if not _data_ok():
        pytest.skip("нет данных Brugge (HYPPO_BRUGGE_RUN)")
    wd = tmp_path_factory.mktemp(request.param)
    return request.param, mod.run_workload(wd, SMALL_GRID, EVENTS, DECLARED)


def test_initial_run_executes_all_jobs(workload):
    _, res = workload
    assert res["initial"] == set(dag.jobs(SMALL_GRID))


def test_param_ridge_reexecutes_fit_liq_and_dependants(workload):
    _, res = workload
    assert res["after"][0] == _closure(SMALL_GRID, _rule_jobs(SMALL_GRID, "hyb"))


def test_undeclared_revision_is_missed(workload):
    _, res = workload
    assert res["after"][2] == set()


def test_declared_revision_reexecutes_consumers(workload):
    _, res = workload
    consumers = _rule_jobs(SMALL_GRID, "wct") | _rule_jobs(SMALL_GRID, "opr")
    assert res["after"][3] == _closure(SMALL_GRID, consumers)


def test_evict_recomputes_evicted_job(workload):
    name, res = workload
    ev = res["after"][1]
    if name == "snakemake":
        # mtime-триггер: свежий файл новее потребителей -> каскад без обрыва
        assert ev == _closure(SMALL_GRID, {"hyb/uto_1.npy"})
        # с целью all пропавший промежуточный файл не восстанавливается
        assert res["native_evict"] == [set()]
    else:
        # хэш выхода тот же -> обрыв каскада (early cutoff)
        assert ev == {"hyb/uto_1.npy"}
    assert res["outputs"][2]["hyb/uto_1.npy"] == res["outputs"][1]["hyb/uto_1.npy"]
    if name == "dagster":
        # Dagster не видит удаления файла: актив FRESH, пересчёт -- ручной (force)
        assert res["rounds"][2][0]["status"]["hyb/uto_1.npy"] == "FRESH"
