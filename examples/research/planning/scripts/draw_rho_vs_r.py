"""Fig 2 (B-вариант): зависимость $\\rho(r)$ для пяти классов графов.

Показывает robustness двусторонней огибаемости ER ≤ real ≤ BA
во~всём диапазоне $r \\in [0.1, 0.9]$, а~не только при~$r=0.7$.
"""
from __future__ import annotations
import io
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

from hyppo.coa import HypothesisGraph

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT_FILE = OUT / "rho_vs_r_classes.pdf"
OUT_JSON = DATA / "rho_vs_r_sweep.json"

R_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def algorithm_2(n, adj, cached):
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    return HypothesisGraph.from_edges(n, edges).plan(cached)


def er_dag(n, p):
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i].add(j)
    return n, dict(adj)


def ba_dag(n, m):
    adj = defaultdict(set)
    degrees = [0] * n
    for i in range(min(m + 1, n)):
        for j in range(i + 1, min(m + 1, n)):
            adj[i].add(j)
            degrees[i] += 1
            degrees[j] += 1
    for new in range(m + 1, n):
        total = sum(degrees[:new])
        if total == 0:
            targets = list(range(min(m, new)))
        else:
            probs = [degrees[i] / total for i in range(new)]
            targets = list(np.random.choice(new, size=min(m, new),
                                             replace=False, p=probs))
        for t in targets:
            adj[t].add(new)
            degrees[t] += 1
            degrees[new] += 1
    return n, dict(adj)


def cascade_over_r(n, adj, n_reps=50, seed=42):
    rng = random.Random(seed)
    out = {}
    for r in R_GRID:
        n_cached = int(r * n)
        rhos = []
        for _ in range(n_reps):
            if n_cached >= n:
                cached = set(range(n))
            else:
                cached = set(rng.sample(range(n), n_cached))
            pne = algorithm_2(n, adj, cached)
            rhos.append(len(pne) / n)
        out[r] = float(np.median(rhos))
    return out


def sweep_synthetic(model_fn, model_args, n_graphs=50, seed=42):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    medians = {r: [] for r in R_GRID}
    for i in range(n_graphs):
        # для BA используем np.random.choice — нужен seed
        np.random.seed(seed + i)
        random.seed(seed + i)
        n, adj = model_fn(*model_args)
        out = cascade_over_r(n, adj, n_reps=20, seed=seed + i)
        for r in R_GRID:
            medians[r].append(out[r])
    return {r: float(np.median(v)) for r, v in medians.items()}


def parse_wf_to_adj(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = d["workflow"]["specification"]["tasks"]
    id_to_idx = {t["id"]: i for i, t in enumerate(tasks)}
    adj = defaultdict(set)
    for t in tasks:
        u = id_to_idx[t["id"]]
        for c in t.get("children", []):
            if c in id_to_idx:
                adj[u].add(id_to_idx[c])
    return len(tasks), dict(adj)


def sweep_real(jsons, n_reps_per_wf=20):
    medians = {r: [] for r in R_GRID}
    for path in jsons:
        try:
            n, adj = parse_wf_to_adj(path)
        except Exception:
            continue
        if n < 10:
            continue
        out = cascade_over_r(n, adj, n_reps=n_reps_per_wf, seed=42)
        for r in R_GRID:
            medians[r].append(out[r])
    return {r: float(np.median(v)) for r, v in medians.items()}


def sweep_curated():
    g = json.loads(
        (DATA / "hand_curated_hypothesis_graphs.json")
        .read_text(encoding="utf-8"))
    medians = {r: [] for r in R_GRID}
    individual = {}
    for key in ("rnaseq", "airrflow", "sarek", "atacseq", "chipseq"):
        if key not in g:
            continue
        gi = g[key]
        nodes = {h["id"]: i for i, h in enumerate(gi["hypotheses"])}
        n = len(nodes)
        adj = defaultdict(set)
        for e in gi["subsumption_edges"] + gi["inter_layer_edges"]:
            adj[nodes[e["src"]]].add(nodes[e["tgt"]])
        out = cascade_over_r(n, dict(adj), n_reps=100, seed=42)
        for r in R_GRID:
            medians[r].append(out[r])
        # individual: keep r ∈ {0.3, 0.5, 0.7, 0.9} for plot stars
        individual[key] = {f"{r:.1f}": out[r]
                           for r in (0.3, 0.5, 0.7, 0.9)}
    return {r: float(np.median(v)) for r, v in medians.items()}, individual


def main():
    print("Computing sweeps... (may take a few minutes)")
    n_synth = 200  # размер графа для synthetic
    # matched-density ER: p* = 2·d_bar/(n-1) с d_bar=1.5
    # (для i<j ER-DAG: E[d_bar_ER] = p(n-1)/2, R11/A04-G1 fix)
    # (медианная средняя степень в WfCommons, см. cascade_models таблицу)
    p_er = min(1.0, 2.0 * 1.5 / (n_synth - 1))
    er_med = sweep_synthetic(er_dag, (n_synth, p_er), n_graphs=20)
    ba_med = sweep_synthetic(ba_dag, (n_synth, 2), n_graphs=20)
    # real — полная выборка
    jsons = []
    for fam in ("nextflow", "snakemake", "pegasus"):
        d = CACHE / "wfcommons" / fam
        if d.exists():
            jsons.extend(sorted(d.rglob("*.json")))
    random.Random(42).shuffle(jsons)
    print(f"  real: {len(jsons)} workflows")
    real_med = sweep_real(jsons)
    cur_med, cur_individual = sweep_curated()

    # Subworkflow: используем cascade_hypothesis_results.json (median по всем workflows)
    sub_d = json.loads(
        (DATA / "cascade_hypothesis_results.json").read_text())
    sub_med = {}
    for r in R_GRID:
        # cascade_hypothesis_results имеет только {0.3, 0.5, 0.7, 0.9}
        key = f"{r:.1f}" if r in (0.3, 0.5, 0.7, 0.9) else None
        if key and key in sub_d.get("all_corpus", {}):
            sub_med[r] = sub_d["all_corpus"][key]["median_rho"]
        else:
            sub_med[r] = None  # будем интерполировать

    print(f"\nMedians:")
    for r in R_GRID:
        print(f"  r={r:.1f}: ER={er_med[r]:.3f}  real={real_med[r]:.3f}  "
              f"sub={sub_med[r] if sub_med[r] else '--':>5}  "
              f"curated={cur_med[r]:.3f}  BA={ba_med[r]:.3f}")

    # === Save sweep JSON for downstream composite figure ===
    sweep_data = {
        "r": R_GRID,
        "ER": [er_med[r] for r in R_GRID],
        "real": [real_med[r] for r in R_GRID],
        "sub": [sub_med[r] for r in R_GRID],
        "curated": [cur_med[r] for r in R_GRID],
        "BA": [ba_med[r] for r in R_GRID],
        "curated_individual": cur_individual,
        "__meta__": {
            "fix_note": "R11/A04-G1: ER uses p* = 2*d_bar/(n-1) with d_bar=1.5",
            "n_synth": n_synth,
            "p_er": p_er,
        },
    }
    OUT_JSON.write_text(json.dumps(sweep_data, indent=2),
                        encoding="utf-8")
    print(f"Saved {OUT_JSON}")

    # === Plot ===
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    rs = R_GRID
    # 1-r reference
    ax.plot(rs, [1 - r for r in rs], "k:", lw=1.0,
            label=r"наивно $1-r$")
    series = [
        ("BA ($m=2$)",                              ba_med,   "C2", "^"),
        ("Subworkflow-агрегация",                   sub_med,  "C3", "D"),
        ("Hand-curated (5 pipeline)",               cur_med,  "C4", "*"),
        ("WfCommons task-DAG",                      real_med, "C0", "o"),
        ("ER разрежённый",                          er_med,   "C1", "s"),
    ]
    for label, vals, color, marker in series:
        xs = [r for r in rs if vals.get(r) is not None]
        ys = [vals[r] for r in xs]
        ax.plot(xs, ys, color=color, marker=marker, ms=6, lw=1.6,
                label=label, alpha=0.85)

    ax.set_xlabel(r"$r = |\mathrm{Cache}|/|H|$ — доля кэшированных гипотез")
    ax.set_ylabel(r"$\rho = |P_{ne}|/|H|$ (медиана)")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FILE)
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
