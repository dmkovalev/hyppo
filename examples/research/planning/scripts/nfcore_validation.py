"""
Валидация каскадного эффекта на реальных workflow-графах nf-core.

Парсит главный workflow main.nf для каждого пайплайна:
- Узлы = вызываемые процессы и подworkflow (CAPITAL_CASE NAME)
- Рёбра = data flow через `X.out[.something]` references

Сравнивает с разреженным ER (d̄ совпадает с конкретным пайплайном)
и BA (m=2).
"""
from __future__ import annotations
import io
import json
import random
import re
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from go_validation import (  # type: ignore
    plan, gen_er_sparse, gen_ba,
)

ROOT = Path(__file__).resolve().parent.parent   # examples/research/planning/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

CACHE_DIR = CACHE / "nfcore_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PIPELINES = [
    "rnaseq", "sarek", "scrnaseq", "mag", "ampliseq", "chipseq",
    "nanoseq", "atacseq", "eager", "methylseq", "taxprofiler",
    "rnafusion", "viralrecon", "raredisease", "cutandrun",
    "hic", "funcscan", "oncoanalyser", "smrnaseq",
    "differentialabundance", "bacass", "airrflow", "hlatyping",
    "bactmap", "rnavar", "rnasplice", "fetchngs", "epitopeprediction",
]


def fetch_pipeline(name: str) -> Path | None:
    """Скачать tarball пайплайна, вернуть путь к распакованной директории."""
    out_dir = CACHE_DIR / name
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    url = f"https://api.github.com/repos/nf-core/{name}/tarball"
    try:
        print(f"  fetching nf-core/{name}...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "curl"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except Exception as e:
        print(f"    ERROR fetching {name}: {e}", flush=True)
        return None
    out_dir.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(out_dir)
    return out_dir


def find_main_workflow(pipe_dir: Path) -> Path | None:
    """Найти главный workflow main.nf пайплайна."""
    # nf-core convention: workflows/<name>/main.nf или workflows/<name>.nf
    candidates = list(pipe_dir.rglob("workflows/*/main.nf")) + \
                 list(pipe_dir.rglob("workflows/*.nf"))
    if not candidates:
        return None
    # Берём самый большой по размеру (главный workflow)
    return max(candidates, key=lambda p: p.stat().st_size)


def extract_balanced_args(text: str, open_idx: int) -> tuple[str, int] | None:
    """Извлечь содержимое скобок от open_idx (символ '('). Вернуть (содержимое, индекс_закрывающей_скобки)."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i], i
        i += 1
    return None


def parse_workflow_dag(nf_path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    """Парсер workflow main.nf: возвращает (nodes, edges)."""
    text = nf_path.read_text(encoding="utf-8", errors="replace")
    # выкинуть комментарии // и /* */
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Выделить workflow {...}-блоки (берём весь файл — главный workflow)
    # Ищем "workflow X { ... }" или "workflow { ... }" и берём ВСЕ
    workflow_blocks = []
    i = 0
    while i < len(text):
        m = re.search(r"\bworkflow\s+\w*\s*\{", text[i:])
        if not m:
            break
        start = i + m.end() - 1  # позиция '{'
        depth = 0
        j = start
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j < len(text):
            workflow_blocks.append(text[start + 1:j])
            i = j + 1
        else:
            break

    if not workflow_blocks:
        # Если нет явного workflow {} — берём весь файл (fallback для старых DSL1)
        workflow_blocks = [text]

    # имя процесса: PROC_NAME, может содержать буквы, цифры, _
    # вызов: NAME( ... )
    call_re = re.compile(r"\b([A-Z][A-Z0-9_]+)\s*\(")
    out_ref_re = re.compile(r"\b([A-Z][A-Z0-9_]+)\.out\b")

    nodes: set[str] = set()
    edges: list[tuple[str, str]] = []

    for block in workflow_blocks:
        # найти все вызовы
        for m in call_re.finditer(block):
            callee = m.group(1)
            if callee in ("INFO", "WARN", "ERROR"):  # logger calls
                continue
            paren_pos = m.end() - 1
            extracted = extract_balanced_args(block, paren_pos)
            if extracted is None:
                continue
            args, _ = extracted
            nodes.add(callee)
            # найти все .out references → рёбра X → callee
            for m2 in out_ref_re.finditer(args):
                source = m2.group(1)
                if source != callee:
                    nodes.add(source)
                    edges.append((source, callee))

    # дедупликация рёбер
    edges_set = set(edges)
    return nodes, list(edges_set)


def relabel_dag(nodes: set[str], edges: list[tuple[str, str]]):
    idx = {n: i for i, n in enumerate(sorted(nodes))}
    adj = defaultdict(set)
    for u, v in edges:
        if u in idx and v in idx and u != v:
            adj[idx[u]].add(idx[v])
    return adj, len(idx)


def make_acyclic(adj: dict[int, set[int]], n: int) -> dict[int, set[int]]:
    """Удалить рёбра, образующие циклы (по топологическому DFS).
    nf-core workflows иногда содержат feedback-вызовы (например, MULTIQC после
    всех других модулей через .collect()), которые парсер видит как рёбра в обе стороны."""
    color = [0] * n  # 0=white, 1=gray, 2=black
    new_adj = defaultdict(set)

    def visit(u: int):
        color[u] = 1
        for v in list(adj.get(u, ())):
            if color[v] == 0:
                new_adj[u].add(v)
                visit(v)
            elif color[v] == 2:
                new_adj[u].add(v)
            # серый → ребро образует цикл, пропускаем
        color[u] = 2

    for u in range(n):
        if color[u] == 0:
            visit(u)
    return new_adj


def graph_depth(adj: dict[int, set[int]], n: int) -> int:
    in_deg = defaultdict(int)
    for u, vs in adj.items():
        for v in vs:
            in_deg[v] += 1
    dist = [0] * n
    q = deque(i for i in range(n) if in_deg[i] == 0)
    in_d = dict(in_deg)
    while q:
        x = q.popleft()
        for v in adj.get(x, ()):
            if dist[v] < dist[x] + 1:
                dist[v] = dist[x] + 1
            in_d[v] -= 1
            if in_d[v] == 0:
                q.append(v)
    return max(dist) if dist else 0


def cascade_at_r(adj, n: int, r: float, n_reps: int, np_rng) -> float:
    arr = np.arange(n)
    n_cached = int(r * n)
    vals = []
    for _ in range(n_reps):
        cached = set(np_rng.choice(arr, size=n_cached, replace=False).tolist())
        ne, _ = plan(n, adj, cached)
        vals.append(ne / max(n, 1))
    return float(np.median(vals))


def cascade_curve(adj, n: int, rates: list[float], n_reps: int, np_rng) -> dict:
    out = {}
    arr = np.arange(n)
    for r in rates:
        n_cached = int(r * n)
        vals = []
        for _ in range(n_reps):
            cached = set(np_rng.choice(arr, size=n_cached, replace=False).tolist())
            ne, _ = plan(n, adj, cached)
            vals.append(ne / max(n, 1))
        vals.sort()
        out[float(r)] = (float(np.median(vals)),
                         float(np.percentile(vals, 5)) if len(vals) > 1 else float(vals[0]),
                         float(np.percentile(vals, 95)) if len(vals) > 1 else float(vals[0]))
    return out


def main():
    cache_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_reps = 30
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    results = []
    successful = []

    for name in TARGET_PIPELINES:
        pipe_dir = fetch_pipeline(name)
        if pipe_dir is None:
            continue
        wf = find_main_workflow(pipe_dir)
        if wf is None:
            print(f"  [{name}] no workflow found")
            continue
        nodes, edges = parse_workflow_dag(wf)
        if len(nodes) < 10:
            print(f"  [{name}] too small ({len(nodes)} nodes), skip")
            continue
        adj_raw, n = relabel_dag(nodes, edges)
        adj = make_acyclic(adj_raw, n)
        depth = graph_depth(adj, n)
        edge_count = sum(len(vs) for vs in adj.values())
        d_bar = edge_count / max(n, 1)
        rho_07 = cascade_at_r(adj, n, 0.7, n_reps, np_rng)
        print(f"  [{name:25s}] n={n:3d}  edges={edge_count:4d}  d_bar={d_bar:.2f}  "
              f"depth={depth:2d}  rho(0.7)={rho_07:.3f}", flush=True)
        results.append({
            "name": name,
            "n": n,
            "edges": edge_count,
            "d_bar": d_bar,
            "depth": depth,
            "rho_07_nfcore": rho_07,
            "rho_curve_nfcore": cascade_curve(adj, n, cache_rates, n_reps, np_rng),
        })
        successful.append((name, n, d_bar, depth, adj))

    print(f"\n{len(successful)} pipelines parsed")

    # Параллельные синтетические замеры для каждого пайплайна
    print("\nComparing with synthetic ER/BA matched to each pipeline...")
    for r_entry, (name, n, d_bar, depth, adj) in zip(results, successful):
        # ER sparse — d_bar матчинг
        adj_er = gen_er_sparse(n, d_bar, rng)
        r_entry["rho_07_er"] = cascade_at_r(adj_er, n, 0.7, n_reps, np_rng)
        # BA m=2
        adj_ba = gen_ba(n, 2, rng, np_rng)
        r_entry["rho_07_ba"] = cascade_at_r(adj_ba, n, 0.7, n_reps, np_rng)
        # Кривые для усреднения
        r_entry["rho_curve_er"] = cascade_curve(adj_er, n, cache_rates, n_reps, np_rng)
        r_entry["rho_curve_ba"] = cascade_curve(adj_ba, n, cache_rates, n_reps, np_rng)

    # Сохранить
    out_json = DATA / "nfcore_validation_results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")

    # Summary stats
    rhos_nf = [r["rho_07_nfcore"] for r in results]
    rhos_er = [r["rho_07_er"] for r in results]
    rhos_ba = [r["rho_07_ba"] for r in results]
    depths = [r["depth"] for r in results]
    ns = [r["n"] for r in results]
    d_bars = [r["d_bar"] for r in results]
    print(f"\nSummary (n={len(results)} pipelines):")
    print(f"  n:       median={int(np.median(ns)):4d}  range=[{min(ns)}, {max(ns)}]")
    print(f"  d_bar:   median={np.median(d_bars):.2f}  range=[{min(d_bars):.2f}, {max(d_bars):.2f}]")
    print(f"  depth:   median={int(np.median(depths)):2d}  range=[{min(depths)}, {max(depths)}]")
    print(f"  rho(0.7) nf-core: median={np.median(rhos_nf):.3f}  range=[{min(rhos_nf):.3f}, {max(rhos_nf):.3f}]")
    print(f"  rho(0.7) ER:      median={np.median(rhos_er):.3f}")
    print(f"  rho(0.7) BA:      median={np.median(rhos_ba):.3f}")

    # График: scatter (n, rho_07) для трёх семейств
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ns, rhos_nf, c="C0", marker="o", s=40, label="nf-core (real workflows)")
    ax.scatter(ns, rhos_er, c="C1", marker="s", s=30, label="ER sparse (matched d_bar)", alpha=0.7)
    ax.scatter(ns, rhos_ba, c="C2", marker="^", s=30, label="BA (m=2)", alpha=0.7)
    ax.axhline(0.30, color="k", linestyle=":", lw=0.8, label="naive 1−r=0.30")
    ax.set_xlabel(r"|H| (graph size)")
    ax.set_ylabel(r"$\rho = |P_{ne}|/|H|$ at r=0.7")
    ax.set_title("Cascade effect on real nf-core workflows vs synthetic")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    out_pdf = OUT / "nfcore_vs_synthetic_cascade.pdf"
    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
