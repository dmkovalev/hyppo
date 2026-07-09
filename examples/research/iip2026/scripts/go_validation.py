"""
Эксперимент на реальной онтологии Gene Ontology (GO).

Цель: показать что асимптотики алгоритма планирования и величина каскадного
эффекта на реальной онтологической DAG-структуре согласуются с результатами
на синтетических ER (sparse, d_bar≈2) и BA (m=2) графах.

Загружает go-basic.obo (https://purl.obolibrary.org/obo/go/go-basic.obo),
строит is_a DAG, извлекает связные подграфы целевых размеров,
прогоняет планировщик при разных cache_rate, сравнивает с ER/BA.
"""
from __future__ import annotations
import io
import json
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from hyppo.coa import HypothesisGraph

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Принудительно UTF-8 на stdout для Windows cp1251 консолей.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent   # examples/research/iip2026/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

OBO_PATH = CACHE / "go-basic.obo"


def parse_go_obo(path: Path) -> tuple[list[str], list[tuple[str, str]], dict[str, str]]:
    """Простой парсер go-basic.obo.
    Возвращает (nodes, edges, namespace).
    Edges направлены parent → child (родитель → потомок по is_a).
    """
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    namespace: dict[str, str] = {}
    cur_id: str | None = None
    cur_ns: str | None = None
    cur_parents: list[str] = []
    in_term = False
    obsolete = False

    def flush():
        if cur_id and not obsolete:
            nodes.append(cur_id)
            if cur_ns:
                namespace[cur_id] = cur_ns
            for p in cur_parents:
                edges.append((p, cur_id))

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "[Term]":
                if in_term:
                    flush()
                in_term = True
                cur_id = None
                cur_ns = None
                cur_parents = []
                obsolete = False
            elif line.startswith("[") and line.endswith("]"):
                if in_term:
                    flush()
                in_term = False
            elif in_term:
                if line.startswith("id: "):
                    cur_id = line[4:].strip()
                elif line.startswith("namespace: "):
                    cur_ns = line[11:].strip()
                elif line.startswith("is_a: "):
                    tail = line[6:].strip()
                    parent = tail.split(" ", 1)[0].split("!", 1)[0].strip()
                    if parent:
                        cur_parents.append(parent)
                elif line.startswith("is_obsolete: true"):
                    obsolete = True
    if in_term:
        flush()
    return nodes, edges, namespace


def build_adj(nodes: list[str], edges: list[tuple[str, str]]):
    succ: dict[str, set[str]] = defaultdict(set)
    pred: dict[str, set[str]] = defaultdict(set)
    nset = set(nodes)
    for u, v in edges:
        if u in nset and v in nset:
            succ[u].add(v)
            pred[v].add(u)
    return succ, pred


def weak_bfs_subgraph(nodes: list[str], succ, pred, start: str,
                      target_size: int) -> set[str]:
    """BFS по неориентированному графу из start до достижения target_size."""
    visited = {start}
    queue: deque[str] = deque([start])
    while queue and len(visited) < target_size:
        node = queue.popleft()
        for nb in list(succ.get(node, ())) + list(pred.get(node, ())):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= target_size:
                    break
    return visited


def downward_bfs_subgraph(succ, start: str, target_size: int) -> set[str]:
    """BFS только по succ (is_a-children) — извлекает глубокий поддерева
    из корневого узла GO namespace."""
    visited = {start}
    queue: deque[str] = deque([start])
    while queue and len(visited) < target_size:
        node = queue.popleft()
        for nb in succ.get(node, ()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= target_size:
                    break
    return visited


def induced_edges(sub: set[str], succ) -> list[tuple[str, str]]:
    out = []
    for u in sub:
        for v in succ.get(u, ()):
            if v in sub:
                out.append((u, v))
    return out


def plan(n_idx: int, adj: dict[int, set[int]], cached: set[int]) -> tuple[int, int]:
    """Алгоритм планирования: вернуть (|Pne|, |Pe|).

    Делегирует каскад библиотеке (hyppo.coa.HypothesisGraph.plan); P_e~--- остальные
    вершины DAG (в~ациклическом графе P_ne ∪ P_e покрывает все вершины)."""
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    p_ne = HypothesisGraph.from_edges(n_idx, edges).plan(cached)
    return len(p_ne), n_idx - len(p_ne)


def relabel(sub_nodes: list[str], sub_edges: list[tuple[str, str]]):
    idx = {n: i for i, n in enumerate(sub_nodes)}
    adj = defaultdict(set)
    for u, v in sub_edges:
        adj[idx[u]].add(idx[v])
    return adj, len(sub_nodes)


# --- Синтетические генераторы (для сравнения) ---

def gen_er_sparse(n: int, d_bar: float, rng: random.Random):
    # WfCommons d_bar = |E|/n; для ER-DAG с i<j: E[d_bar_ER] = p(n-1)/2.
    # Чтобы совпасть с real d_bar нужно p* = 2·d_bar/(n-1) (R11/A04-G1 fix).
    p = min(1.0, 2.0 * d_bar / max(n - 1, 1))
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j)
    return adj


def gen_ba(n: int, m: int, rng: random.Random, np_rng: np.random.Generator):
    adj = defaultdict(set)
    deg = np.zeros(n, dtype=int)
    init = min(m + 1, n)
    for i in range(init):
        for j in range(i + 1, init):
            adj[i].add(j)
            deg[i] += 1
            deg[j] += 1
    for new in range(init, n):
        if deg[:new].sum() == 0:
            targets = list(range(min(m, new)))
        else:
            probs = deg[:new] / deg[:new].sum()
            targets = list(np_rng.choice(new, size=min(m, new),
                                          replace=False, p=probs))
        for t in targets:
            adj[t].add(new)
            deg[t] += 1
            deg[new] += 1
    return adj


def cascade_curve(adj, n_nodes: int, cache_rates: list[float], n_reps: int,
                  np_rng: np.random.Generator) -> dict[float, tuple[float, float, float]]:
    """Для каждого r вернуть (median, p05, p95) от rho = |Pne|/n."""
    out: dict[float, tuple[float, float, float]] = {}
    arr = np.arange(n_nodes)
    for r in cache_rates:
        vals = []
        n_cached = int(r * n_nodes)
        for rep in range(n_reps):
            cached = set(np_rng.choice(arr, size=n_cached, replace=False).tolist())
            ne, _ = plan(n_nodes, adj, cached)
            vals.append(ne / max(n_nodes, 1))
        vals.sort()
        out[r] = (float(np.median(vals)),
                  float(np.percentile(vals, 5)),
                  float(np.percentile(vals, 95)))
    return out


def main():
    print(f"Reading {OBO_PATH}...")
    nodes_all, edges_all, ns = parse_go_obo(OBO_PATH)
    print(f"GO: {len(nodes_all)} terms, {len(edges_all)} is_a edges")
    # ограничимся одним namespace для связности
    target_ns = "biological_process"
    nodes_bp = [n for n in nodes_all if ns.get(n) == target_ns]
    nset_bp = set(nodes_bp)
    edges_bp = [(u, v) for u, v in edges_all if u in nset_bp and v in nset_bp]
    print(f"  namespace={target_ns}: {len(nodes_bp)} terms, {len(edges_bp)} edges")
    succ, pred = build_adj(nodes_bp, edges_bp)

    # выбираем стартовые узлы среди тех, у кого много is_a-потомков
    # (это даёт глубокие поддеревья содержательных понятий).
    out_deg = {n: len(succ.get(n, set())) for n in nodes_bp}
    hubs = sorted([n for n in nodes_bp if out_deg[n] >= 5],
                  key=lambda n: -out_deg[n])[:200]
    print(f"  Selected {len(hubs)} hubs with out_deg>=5 "
          f"(max out_deg={out_deg[hubs[0]]})")

    target_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    cache_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    summary = {}
    go_curves = {}
    er_curves = {}
    ba_curves = {}

    for tsize in target_sizes:
        # репликаций меньше для больших размеров, чтобы уложиться в разумное время
        n_reps = 30 if tsize <= 500 else (10 if tsize <= 2000 else 5)
        n_subgraphs = 5 if tsize <= 2000 else 3
        print(f"\n=== Size {tsize} (n_reps={n_reps}, n_sub={n_subgraphs}) ===")
        # GO: подграфы
        go_vals_at_r = {r: [] for r in cache_rates}
        depths = []
        edge_counts = []
        for sg_seed in range(n_subgraphs):
            start = hubs[sg_seed * 7 % len(hubs)]
            sub = downward_bfs_subgraph(succ, start, tsize)
            sub_list = list(sub)
            sub_edges = induced_edges(sub, succ)
            adj, n = relabel(sub_list, sub_edges)
            edge_counts.append(len(sub_edges))
            # глубина = длина самого длинного пути
            in_deg = defaultdict(int)
            for u, vs in adj.items():
                for v in vs:
                    in_deg[v] += 1
            dist = [0] * n
            q2 = deque(i for i in range(n) if in_deg[i] == 0)
            in_d2 = dict(in_deg)
            while q2:
                x = q2.popleft()
                for v in adj.get(x, ()):
                    if dist[v] < dist[x] + 1:
                        dist[v] = dist[x] + 1
                    in_d2[v] -= 1
                    if in_d2[v] == 0:
                        q2.append(v)
            depths.append(max(dist) if dist else 0)
            arr_n = np.arange(n)
            for r in cache_rates:
                n_cached = int(r * n)
                for rep in range(n_reps):
                    cached = set(np_rng.choice(arr_n, size=n_cached,
                                                replace=False).tolist())
                    ne, _ = plan(n, adj, cached)
                    go_vals_at_r[r].append(ne / max(n, 1))
        go_curves[tsize] = {
            r: (float(np.median(vs)),
                float(np.percentile(vs, 5)),
                float(np.percentile(vs, 95)))
            for r, vs in go_vals_at_r.items()
        }
        d_bar_real = sum(edge_counts) / max(n_subgraphs, 1) / tsize
        print(f"  GO: depth median={np.median(depths):.0f}, "
              f"d_bar={d_bar_real:.2f}, edges median={np.median(edge_counts):.0f}")

        # ER sparse: подграфы с d_bar матчащим GO
        er_vals = {r: [] for r in cache_rates}
        for sg_seed in range(n_subgraphs):
            adj_er = gen_er_sparse(tsize, d_bar_real, rng)
            er_vals_local = cascade_curve(adj_er, tsize, cache_rates, n_reps, np_rng)
            for r, (med, _, _) in er_vals_local.items():
                er_vals[r].append(med)
        er_curves[tsize] = {
            r: (float(np.median(vs)),
                float(np.percentile(vs, 5)),
                float(np.percentile(vs, 95)))
            for r, vs in er_vals.items()
        }

        # BA: подграфы с m=2
        ba_vals = {r: [] for r in cache_rates}
        for sg_seed in range(n_subgraphs):
            adj_ba = gen_ba(tsize, 2, rng, np_rng)
            ba_vals_local = cascade_curve(adj_ba, tsize, cache_rates, n_reps, np_rng)
            for r, (med, _, _) in ba_vals_local.items():
                ba_vals[r].append(med)
        ba_curves[tsize] = {
            r: (float(np.median(vs)),
                float(np.percentile(vs, 5)),
                float(np.percentile(vs, 95)))
            for r, vs in ba_vals.items()
        }

        # ключевая точка: r=0.7 (как в статье)
        r_key = 0.7
        go_m = go_curves[tsize][r_key][0]
        er_m = er_curves[tsize][r_key][0]
        ba_m = ba_curves[tsize][r_key][0]
        print(f"  r=0.7: GO rho={go_m:.3f}  ER rho={er_m:.3f}  BA rho={ba_m:.3f}  "
              f"(naive=0.300)")
        summary[tsize] = {
            "d_bar_go": d_bar_real,
            "depth_go": float(np.median(depths)),
            "rho_at_r07": {"GO": go_m, "ER_sparse": er_m, "BA": ba_m},
        }

    # сохранить результаты
    out_json = DATA / "go_validation_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "summary": summary,
            "go": go_curves,
            "er_sparse": er_curves,
            "ba": ba_curves,
        }, f, indent=2)
    print(f"\nSaved JSON to {out_json}")

    # график: rho vs r при tsize=500 для GO/ER_sparse/BA + naive
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    tsize_plot = 500
    rs = cache_rates
    for label, curves, style in [("GO (real)", go_curves, "-o"),
                                  ("ER sparse (d_bar≈2)", er_curves, "--s"),
                                  ("BA (m=2)", ba_curves, ":^"),
                                  ("naive 1-r", None, "k:")]:
        if curves is None:
            ax.plot(rs, [1 - r for r in rs], style, lw=1, label=label)
        else:
            meds = [curves[tsize_plot][r][0] for r in rs]
            los = [curves[tsize_plot][r][1] for r in rs]
            his = [curves[tsize_plot][r][2] for r in rs]
            ax.plot(rs, meds, style, ms=4, label=label)
            ax.fill_between(rs, los, his, alpha=0.15)
    ax.set_xlabel(r"Cache rate $r = |\mathrm{Cache}|/|H|$")
    ax.set_ylabel(r"Recompute share $\rho = |P_{ne}|/|H|$")
    ax.set_title(f"Каскадный эффект на GO vs синтетика (|H|={tsize_plot})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    out_png = OUT / "go_vs_synthetic_cascade.pdf"
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    main()
