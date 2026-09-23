# examples/research/baseline_filecache/experiment.py
"""Оркестратор: базовый прогон, эталоны полным пересчётом, испытания через
snakemake -n, сводка, самопроверки, график."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics as st
import subprocess
import time
from pathlib import Path

from . import dag, hyppo_side, metastore
from .config import (GRID, HYPPO_PY, PAIRS, REVISIONS, SMALL_GRID, SNAKEMAKE_PY,
                     declaration_subsets, pair_id)

HERE = Path(__file__).resolve().parent
SNAKEFILE = HERE / "Snakefile"
WORKERS = HERE / "workers.py"
_OUT_RE = re.compile(r"^\s*output:\s*(.+)$", re.M)


# ---------- Snakemake ----------
def _snk(workdir: Path, meta: Path, grid_name: str, declared, ridge: str, extra: list[str]) -> str:
    cmd = [str(SNAKEMAKE_PY), "-m", "snakemake", "-s", str(SNAKEFILE), "--directory", str(workdir),
           "--rerun-triggers", "mtime", "params", "--config", f"meta={meta}", f"grid={grid_name}",
           "declared=" + ",".join(pair_id(p) for p in declared), f"ridge={ridge}", *extra]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stdout + res.stderr)
    return res.stdout + res.stderr


def _dry_run_jobs(out: str, grid: dict) -> set[str]:
    jobs = set(dag.jobs(grid))
    found = set()
    for m in _OUT_RE.finditer(out):
        for p in m.group(1).split(","):
            p = p.strip().replace("\\", "/")
            if p in jobs:
                found.add(p)
    return found


def _digests(workdir: Path, grid: dict) -> dict[str, str]:
    import hashlib
    import numpy as np
    d = {}
    for j in dag.jobs(grid):
        f = workdir / j
        if j.endswith(".json"):
            d[j] = json.loads(f.read_text())["digest"]
        else:
            d[j] = hashlib.sha256(np.round(np.load(f), 9).tobytes()).hexdigest()
    return d


def base_run(root: Path, grid: dict, grid_name: str, cores: int) -> tuple[Path, Path]:
    work, meta = root / "base" / "work", root / "base" / "meta.sqlite"
    shutil.rmtree(root / "base", ignore_errors=True)
    subprocess.run([str(HYPPO_PY), str(WORKERS), "seed", "--meta", str(meta)], check=True)
    _snk(work, meta, grid_name, PAIRS, "1", ["--cores", str(cores)])
    return work, meta


def ground_truth(root: Path, grid: dict, grid_name: str, base_work: Path, base_meta: Path,
                 cores: int) -> dict[str, set[str]]:
    """Эталон S: полный пересчёт с нуля после правки; устаревшие — изменившие выход."""
    base = _digests(base_work, grid)
    truth = {}
    for rev in REVISIONS:
        d = root / f"truth_{rev}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        meta = d / "meta.sqlite"
        shutil.copy2(base_meta, meta)
        ridge = "10" if rev == "ridge" else "1"
        if rev != "ridge":
            metastore.revise(meta, rev)
        # crm_fit перезаписал бы gains/tau: эталон пересчитывает всё, кроме crm_fit
        (d / "work").mkdir()
        shutil.copytree(base_work / "p1", d / "work" / "p1")
        _snk(d / "work", meta, grid_name, PAIRS, ridge, ["--cores", str(cores), "--forcerun",
                                                          "fit_liq", "fit_wct", "fit_opr"])
        new = _digests(d / "work", grid)
        truth[rev] = {j for j in dag.jobs(grid) if new[j] != base[j]}
    return truth


def snakemake_e_factory(root: Path, grid: dict, grid_name: str, base_work: Path, base_meta: Path):
    timings: list[float] = []

    def e(declared, revision) -> set[str]:
        d = root / "trial"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(base_work, d / "work", copy_function=shutil.copy2)
        meta = d / "meta.sqlite"
        shutil.copy2(base_meta, meta)
        if revision != "ridge":
            metastore.revise(meta, revision)
        ridge = "10" if revision == "ridge" else "1"
        t0 = time.perf_counter()
        out = _snk(d / "work", meta, grid_name, declared, ridge, ["-n", "--cores", "1"])
        timings.append(time.perf_counter() - t0)
        return _dry_run_jobs(out, grid)

    e.timings = timings
    return e


# ---------- испытания и сводка ----------
def run_trials(grid, truth, model, e_fn, subsets=None) -> list[dict]:
    subsets = subsets if subsets is not None else declaration_subsets()
    trials = []
    for p, declared in subsets:
        for rev in REVISIONS:
            S = truth[rev]
            E = e_fn(declared, rev)
            P, t_plan = hyppo_side.timed_plan(model, rev)
            trials.append({
                "p": p, "declared": [pair_id(x) for x in declared], "revision": rev,
                "S": len(S),
                "snakemake_executed": len(E), "snakemake_missed": len(S - E),
                "snakemake_extra": len(E - S), "snakemake_wrong": len(dag.wrong_set(grid, S, E)),
                "hyppo_planned": len(P), "hyppo_missed": len(S - P), "hyppo_extra": len(P - S),
                "hyppo_wrong": len(dag.wrong_set(grid, S, P)),
                "declarations_snakemake": len(declared), "declarations_hyppo": model.extra_declarations,
                "t_plan_s": t_plan,
                "_sets": {"S": sorted(S), "E": sorted(E), "P": sorted(P)},
            })
    return trials


def self_checks(trials: list[dict]) -> None:
    for t in trials:
        sets = {k: set(v) for k, v in t["_sets"].items()}
        if t["revision"] == "ridge":
            assert sets["E"] == sets["P"] == sets["S"], f"контроль ridge: {t}"
        if t["p"] == 1.0:
            assert sets["E"] == sets["S"], f"p=1, {t['revision']}: E != S"
        assert t["hyppo_missed"] == 0, f"комплекс пропустил: {t}"
        assert t["S"] > 0, f"вырожденная правка {t['revision']}"
        assert len(t["declared"]) == t["declarations_snakemake"]


def summarize(trials: list[dict]) -> dict:
    out = {}
    for p in sorted({t["p"] for t in trials}):
        ts = [t for t in trials if t["p"] == p and t["revision"] != "ridge"]
        share = [t["snakemake_missed"] / t["S"] for t in ts]
        out[p] = {
            "declarations": st.median(t["declarations_snakemake"] for t in ts),
            "snakemake_missed_mean": st.mean(t["snakemake_missed"] for t in ts),
            "snakemake_missed_share_mean": st.mean(share),
            "snakemake_missed_share_min": min(share),
            "snakemake_missed_share_max": max(share),
            "snakemake_wrong_mean": st.mean(t["snakemake_wrong"] for t in ts),
            "snakemake_executed_mean": st.mean(t["snakemake_executed"] for t in ts),
            "hyppo_missed_max": max(t["hyppo_missed"] for t in ts),
            "hyppo_planned_mean": st.mean(t["hyppo_planned"] for t in ts),
            "S_mean": st.mean(t["S"] for t in ts),
        }
    return out


def plot(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = sorted(summary)
    mean = [summary[p]["snakemake_missed_share_mean"] for p in ps]
    lo = [summary[p]["snakemake_missed_share_min"] for p in ps]
    hi = [summary[p]["snakemake_missed_share_max"] for p in ps]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.fill_between(ps, lo, hi, alpha=0.25, label="Snakemake: мин–макс")
    ax.plot(ps, mean, "o-", label="Snakemake: среднее")
    ax.plot(ps, [0] * len(ps), "s-", label="комплекс")
    ax.set_xlabel("доля объявленных зависимостей p")
    ax.set_ylabel("доля пропущенных устаревших")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["full", "small"], default="full")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--root", default=str(HERE / "work"))
    a = ap.parse_args()
    grid = GRID if a.grid == "full" else SMALL_GRID
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    base_work, base_meta = base_run(root, grid, a.grid, a.cores)
    truth = ground_truth(root, grid, a.grid, base_work, base_meta, a.cores)
    model = hyppo_side.build(grid)
    e_fn = snakemake_e_factory(root, grid, a.grid, base_work, base_meta)
    trials = run_trials(grid, truth, model, e_fn)
    self_checks(trials)
    summary = summarize(trials)
    res = {
        "grid": a.grid, "jobs": len(dag.jobs(grid)), "pairs": [pair_id(p) for p in PAIRS],
        "truth_sizes": {k: len(v) for k, v in truth.items()},
        "lattice_build_s": model.build_seconds,
        "t_plan_median_s": st.median(t["t_plan_s"] for t in trials),
        "t_snakemake_dry_median_s": st.median(e_fn.timings),
        "summary": {str(k): v for k, v in summary.items()},
        "trials": [{k: v for k, v in t.items() if k != "_sets"} for t in trials],
    }
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"results_{a.grid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    plot(summary, out_dir / f"missed_vs_p_{a.grid}.pdf")
    print(json.dumps({k: res[k] for k in ("truth_sizes", "t_plan_median_s", "t_snakemake_dry_median_s")}, indent=1))
    for p, v in summary.items():
        print(p, v)


if __name__ == "__main__":
    main()
