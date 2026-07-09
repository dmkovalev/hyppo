"""Cascade-эксперимент на двух источниках hypothesis-graph:

(1) Hierarchy-aggregated DAG из 157 WfCommons-трасс на рекомендованных
    уровнях (nf-core: L3, snakemake/pegasus: L1) — этап-уровневая
    «гипотеза».
(2) EDAM native derived-by DAG (operation → operation через общие данные).

Для каждого графа: cascade-эффект при r ∈ {0.3, 0.5, 0.7, 0.9},
10 повторов на экземпляр, медиана.
"""
from __future__ import annotations
import io
import json
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Set, Tuple

from hyppo.coa import HypothesisGraph

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent   # examples/research/planning/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

CACHE_DIR = CACHE / "wfcommons"
EDAM_OWL = CACHE / "edam" / "EDAM.owl"
OUT_HYP = DATA / "cascade_hypothesis_results.json"
OUT_EDAM = DATA / "cascade_edam_results.json"

REC_LEVEL = {"nextflow": 3, "snakemake": 1, "pegasus": 1}
R_GRID = [0.3, 0.5, 0.7, 0.9]
N_REPS = 10


def strip_suffix(name: str) -> str:
    """Снимает суффиксы вида _ID0000XXX (Pegasus) и _001 / -001 (Snakemake)."""
    name = re.sub(r"_ID\d+$", "", name)
    name = re.sub(r"[-_]\d+$", "", name)
    return name


def aggregate_at_level(name: str, level: int, family: str) -> str:
    if family == "nextflow":
        parts = name.split(".")
        return ".".join(parts[:level]) if len(parts) >= level else name
    # snakemake, pegasus: уровень 1 = базовое имя после strip_suffix
    base = strip_suffix(name)
    if level == 1:
        return base
    # для будущего: можно camelCase split
    return base


def parse_workflow_to_hypothesis(path: Path, family: str, level: int) -> tuple[int, dict] | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    spec = d.get("workflow", {}).get("specification", {})
    tasks = spec.get("tasks", [])
    if not tasks:
        return None
    # task id → hypothesis (group name)
    id_to_hyp = {}
    hyp_names = []
    for t in tasks:
        name = t.get("name") or t.get("id", "")
        hyp = aggregate_at_level(name, level, family)
        if hyp not in hyp_names:
            hyp_names.append(hyp)
        id_to_hyp[t["id"]] = hyp
    hyp_idx = {h: i for i, h in enumerate(hyp_names)}
    n = len(hyp_names)
    # hypothesis-граф: ребро (h_i, h_j) если есть task u→v с u в h_i, v в h_j, h_i != h_j
    adj: Dict[int, Set[int]] = defaultdict(set)
    for t in tasks:
        h_src = hyp_idx[id_to_hyp[t["id"]]]
        for child_id in t.get("children", []):
            if child_id in id_to_hyp:
                h_tgt = hyp_idx[id_to_hyp[child_id]]
                if h_tgt != h_src:
                    adj[h_src].add(h_tgt)
    return n, {k: set(v) for k, v in adj.items()}


def algorithm_2(n: int, adj: dict, cached: set) -> set:
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    return HypothesisGraph.from_edges(n, edges).plan(cached)


def cascade_experiment(n: int, adj: dict, rng) -> dict:
    """Для каждого r ∈ R_GRID — медиана ρ по N_REPS повторам."""
    out = {}
    for r in R_GRID:
        n_cached = int(r * n)
        rhos = []
        for _ in range(N_REPS):
            if n_cached >= n:
                cached = set(range(n))
            else:
                cached = set(rng.sample(range(n), n_cached))
            pne = algorithm_2(n, adj, cached)
            rhos.append(len(pne) / n)
        rhos.sort()
        out[r] = {
            "median": rhos[N_REPS // 2],
            "p05": rhos[max(0, int(0.05 * N_REPS))],
            "p95": rhos[min(N_REPS - 1, int(0.95 * N_REPS))],
        }
    return out


# ===== EXPERIMENT 1: Hierarchy hypothesis graphs =====

def exp_hierarchy():
    items: list[tuple[str, Path]] = []
    for fam in ("nextflow", "snakemake", "pegasus"):
        d = CACHE_DIR / fam
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.json")):
            items.append((fam, p))
    print(f"[hierarchy] Found {len(items)} JSONs", flush=True)

    rng = random.Random(42)
    results = []
    skipped = 0
    for fam, path in items:
        level = REC_LEVEL.get(fam, 1)
        parsed = parse_workflow_to_hypothesis(path, fam, level)
        if parsed is None:
            skipped += 1
            continue
        n, adj = parsed
        if n < 3:
            skipped += 1
            continue
        edges = sum(len(v) for v in adj.values())
        casc = cascade_experiment(n, adj, rng)
        results.append({
            "family": fam,
            "name": path.stem,
            "n_hypotheses": n,
            "n_edges": edges,
            "level": level,
            "cascade": {str(r): casc[r] for r in R_GRID},
        })
    print(f"[hierarchy] Processed {len(results)}, skipped {skipped}",
          flush=True)

    # Сводка по семействам
    by_fam = defaultdict(list)
    for w in results:
        by_fam[w["family"]].append(w)

    summary = {}
    for fam, lst in by_fam.items():
        ns = np.array([w["n_hypotheses"] for w in lst])
        edges = np.array([w["n_edges"] for w in lst])
        per_r = {}
        for r in R_GRID:
            rhos = np.array([w["cascade"][str(r)]["median"] for w in lst])
            per_r[str(r)] = {
                "median_rho": float(np.median(rhos)),
                "p05_rho": float(np.percentile(rhos, 5)),
                "p95_rho": float(np.percentile(rhos, 95)),
                "naive_1mr": 1 - r,
            }
        summary[fam] = {
            "n_workflows": len(lst),
            "median_H": float(np.median(ns)),
            "median_E": float(np.median(edges)),
            "level": REC_LEVEL[fam],
            "rho_by_r": per_r,
        }
    # Общий корпус
    all_rhos = {}
    for r in R_GRID:
        rhos = np.array([w["cascade"][str(r)]["median"] for w in results])
        all_rhos[str(r)] = {
            "median_rho": float(np.median(rhos)),
            "p05_rho": float(np.percentile(rhos, 5)),
            "p95_rho": float(np.percentile(rhos, 95)),
            "naive_1mr": 1 - r,
            "excess_pct": float(100 * (np.median(rhos) - (1 - r)) / (1 - r)),
        }

    print("\n=== Hierarchy: median ρ per family per r ===")
    print(f"{'family':12s} {'|H|':>6s} {'|E|':>6s} {'r=0.3':>10s} {'r=0.5':>10s} {'r=0.7':>10s} {'r=0.9':>10s}")
    for fam in ("nextflow", "snakemake", "pegasus"):
        s = summary[fam]
        line = f"{fam:12s} {s['median_H']:>6.1f} {s['median_E']:>6.1f}"
        for r in R_GRID:
            line += f" {s['rho_by_r'][str(r)]['median_rho']:>9.3f}"
        print(line)
    print(f"\n=== ALL CORPUS: ρ vs 1-r ===")
    for r in R_GRID:
        a = all_rhos[str(r)]
        print(f"  r={r}: ρ={a['median_rho']:.3f}, наивно={a['naive_1mr']:.2f}, "
              f"превышение={a['excess_pct']:+.1f}%")

    out = {
        "n_workflows": len(results),
        "recommended_levels": REC_LEVEL,
        "R_GRID": R_GRID,
        "N_REPS": N_REPS,
        "by_family": summary,
        "all_corpus": all_rhos,
        "per_workflow": results,
    }
    OUT_HYP.write_text(json.dumps(out, indent=2))
    print(f"\n[hierarchy] Saved {OUT_HYP}")
    return out


# ===== EXPERIMENT 2: EDAM native derived-by =====

def build_edam_graph() -> tuple[int, dict]:
    import xml.etree.ElementTree as ET
    NS = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
    }
    tree = ET.parse(EDAM_OWL)
    root = tree.getroot()
    classes = root.findall(".//owl:Class", NS)

    op_outputs = defaultdict(set)
    op_inputs = defaultdict(set)
    for c in classes:
        uri = c.get(f"{{{NS['rdf']}}}about", "")
        sid = uri.replace("http://edamontology.org/", "")
        if not sid.startswith("operation_"):
            continue
        for sub in c.findall("rdfs:subClassOf", NS):
            r = sub.find("owl:Restriction", NS)
            if r is None:
                continue
            prop = r.find("owl:onProperty", NS)
            if prop is None:
                continue
            p_uri = prop.get(f"{{{NS['rdf']}}}resource", "")
            p_short = p_uri.replace("http://edamontology.org/", "")
            if p_short not in ("has_input", "has_output"):
                continue
            tgt = r.find("owl:someValuesFrom", NS)
            if tgt is None:
                continue
            t_uri = tgt.get(f"{{{NS['rdf']}}}resource", "")
            t_short = t_uri.replace("http://edamontology.org/", "")
            if p_short == "has_output":
                op_outputs[sid].add(t_short)
            else:
                op_inputs[sid].add(t_short)

    # derived-by: op_i → op_j если output(op_i) ∩ input(op_j) ≠ ∅
    edges = set()
    for op_i, outs in op_outputs.items():
        for op_j, ins in op_inputs.items():
            if op_i == op_j:
                continue
            if outs & ins:
                edges.add((op_i, op_j))

    nodes = {v for e in edges for v in e}
    op_idx = {op: i for i, op in enumerate(sorted(nodes))}
    adj = defaultdict(set)
    for s, t in edges:
        adj[op_idx[s]].add(op_idx[t])
    return len(nodes), {k: set(v) for k, v in adj.items()}


def exp_edam():
    n, adj = build_edam_graph()
    edges = sum(len(v) for v in adj.values())
    print(f"\n[EDAM] Graph: |V|={n}, |E|={edges}", flush=True)
    if n < 3:
        print("[EDAM] Graph too small, skipping")
        return None

    rng = random.Random(42)
    # Больше повторов т.к. граф один маленький — увеличиваем N_REPS
    n_reps_local = 200
    out_per_r = {}
    for r in R_GRID:
        n_cached = int(r * n)
        rhos = []
        for _ in range(n_reps_local):
            if n_cached >= n:
                cached = set(range(n))
            else:
                cached = set(rng.sample(range(n), n_cached))
            pne = algorithm_2(n, adj, cached)
            rhos.append(len(pne) / n)
        rhos.sort()
        out_per_r[str(r)] = {
            "median_rho": rhos[n_reps_local // 2],
            "p05_rho": rhos[int(0.05 * n_reps_local)],
            "p95_rho": rhos[int(0.95 * n_reps_local)],
            "naive_1mr": 1 - r,
            "excess_pct": float(100 * (rhos[n_reps_local // 2] - (1 - r)) / (1 - r)),
        }

    print("\n=== EDAM: ρ vs 1-r (n_reps=200) ===")
    for r in R_GRID:
        a = out_per_r[str(r)]
        print(f"  r={r}: ρ={a['median_rho']:.3f}, наивно={a['naive_1mr']:.2f}, "
              f"превышение={a['excess_pct']:+.1f}%")

    out = {
        "n_operations": n,
        "n_edges": edges,
        "R_GRID": R_GRID,
        "n_reps": n_reps_local,
        "rho_by_r": out_per_r,
    }
    OUT_EDAM.write_text(json.dumps(out, indent=2))
    print(f"\n[EDAM] Saved {OUT_EDAM}")
    return out


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Hierarchy-aggregated hypothesis graphs")
    print("=" * 60)
    hyp_out = exp_hierarchy()

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: EDAM native derived-by graph")
    print("=" * 60)
    edam_out = exp_edam()

    # Сравнение с task-DAG из существующего файла
    print("\n" + "=" * 60)
    print("COMPARISON: task-DAG vs hypothesis-DAG vs EDAM")
    print("=" * 60)
    task_results = DATA / "wfcommons_validation_results.json"
    if task_results.exists():
        td = json.loads(task_results.read_text())
        task_rho_07 = float(np.median([w["rho_real"] for w in td["per_workflow"]]))
        print(f"\nTask-DAG median ρ @ r=0.7:           {task_rho_07:.3f}")
    print(f"Hierarchy-DAG (all 157) median ρ @ r=0.7: "
          f"{hyp_out['all_corpus']['0.7']['median_rho']:.3f}")
    if edam_out:
        print(f"EDAM derived-by ρ @ r=0.7:            "
              f"{edam_out['rho_by_r']['0.7']['median_rho']:.3f}")


if __name__ == "__main__":
    main()
