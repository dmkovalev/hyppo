"""
Каскадный эффект на реальных workflow-DAG из WfCommons WfInstances.

WfCommons (Coleman et al., FGCS 2022) публикует real-execution traces в
едином формате WfFormat. Каждый JSON содержит полный DAG с явными
parent/children ссылками — не нужно парсить DSL никаких WMS. Это
семантически точное приближение derived_by из VE-модели.

Скачивает workflows из:
- wfcommons/wfinstances: nextflow, snakemake, pegasus, makeflow
- wfcommons/pegasus-instances: 1000genome, cycles, montage и др.

Для каждого DAG рассчитывает cascade ρ при разных r и сравнивает с
sparse ER (matched d_bar) и BA (m=2).
"""
from __future__ import annotations
import io
import json
import random
import subprocess
import sys
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
except Exception:
    pass


ROOT = Path(__file__).resolve().parent.parent  # experiments/iip2026/
CACHE = ROOT / "cache" / "wfcommons"
OUT_DIR = ROOT / "out"
PAPERS = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPERS.mkdir(parents=True, exist_ok=True)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    print(f"  fetch {dest.name}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())
    return dest


def list_dir(repo: str, path: str):
    url = f"https://api.github.com/repos/wfcommons/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def gather_workflow_jsons() -> list[tuple[str, str, Path]]:
    """Возвращает [(family, label, json_path), ...]."""
    items: list[tuple[str, str, Path]] = []
    # WfInstances: nextflow, snakemake, makeflow (плоские)
    for fam in ("nextflow", "snakemake", "pegasus"):
        try:
            entries = list_dir("wfinstances", fam)
        except Exception as e:
            print(f"  list {fam}: {e}", flush=True)
            continue
        for e in entries:
            if e["type"] == "file" and e["name"].endswith(".json"):
                local = CACHE / fam / e["name"]
                local.parent.mkdir(exist_ok=True)
                fetch(e["download_url"], local)
                items.append((fam, e["name"].replace(".json", ""), local))
            elif e["type"] == "dir":
                # Pegasus подкаталоги: 1000genome/, cycles/...
                try:
                    sub = list_dir("wfinstances", f"{fam}/{e['name']}")
                except Exception:
                    continue
                for s in sub:
                    if s["type"] == "file" and s["name"].endswith(".json"):
                        local = CACHE / fam / e["name"] / s["name"]
                        local.parent.mkdir(exist_ok=True, parents=True)
                        fetch(s["download_url"], local)
                        items.append((f"{fam}/{e['name']}",
                                      s["name"].replace(".json", ""), local))
    return items


def parse_wfformat(path: Path) -> tuple[int, dict[int, set[int]], int]:
    """JSON → (n, adj, depth)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data["workflow"]["specification"]["tasks"]
    id_to_idx = {t["id"]: i for i, t in enumerate(tasks)}
    n = len(tasks)
    adj: dict[int, set[int]] = defaultdict(set)
    for t in tasks:
        u = id_to_idx[t["id"]]
        for c in t.get("children", []):
            if c in id_to_idx:
                adj[u].add(id_to_idx[c])
    # depth
    in_deg = defaultdict(int)
    for vs in adj.values():
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
    depth = max(dist) if dist else 0
    return n, dict(adj), depth


def plan(n: int, adj: dict[int, set[int]], cached: set[int]) -> int:
    in_deg = defaultdict(int)
    for vs in adj.values():
        for v in vs:
            in_deg[v] += 1
    q = deque(i for i in range(n) if in_deg[i] == 0)
    topo = []
    in_d = dict(in_deg)
    while q:
        x = q.popleft()
        topo.append(x)
        for v in adj.get(x, ()):
            in_d[v] -= 1
            if in_d[v] == 0:
                q.append(v)
    p_ne: set[int] = set()
    for h in topo:
        if h in p_ne:
            continue
        if h not in cached:
            st = [h]
            while st:
                u = st.pop()
                if u in p_ne:
                    continue
                p_ne.add(u)
                for v in adj.get(u, ()):
                    if v not in p_ne:
                        st.append(v)
    return len(p_ne)


def cascade(adj, n: int, r: float, n_reps: int, np_rng) -> float:
    arr = np.arange(n)
    n_cached = int(r * n)
    vals = []
    for _ in range(n_reps):
        cached = set(np_rng.choice(arr, size=n_cached, replace=False).tolist())
        vals.append(plan(n, adj, cached) / max(n, 1))
    return float(np.median(vals))


def gen_er(n: int, d_bar: float, rng) -> dict[int, set[int]]:
    p = d_bar / max(n - 1, 1)
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j)
    return dict(adj)


def gen_ba(n: int, m: int, rng, np_rng) -> dict[int, set[int]]:
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
            tg = list(range(min(m, new)))
        else:
            probs = deg[:new] / deg[:new].sum()
            tg = list(np_rng.choice(new, size=min(m, new),
                                     replace=False, p=probs))
        for t in tg:
            adj[t].add(new)
            deg[t] += 1
            deg[new] += 1
    return dict(adj)


def main():
    print("Gathering WfCommons workflow JSONs...", flush=True)
    items = gather_workflow_jsons()
    print(f"  total {len(items)} workflows\n", flush=True)

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    n_reps = 30
    r_test = 0.7

    out_partial = PAPERS / "wfcommons_validation_results.json"
    results = []
    N_SYNTH_CAP = 1500  # выше — не запускать ER/BA (слишком медленно)

    for family, label, path in items:
        try:
            n, adj, depth = parse_wfformat(path)
        except Exception as e:
            print(f"  [{family}/{label}] parse error: {e}", flush=True)
            continue
        if n < 10:
            continue
        edge_count = sum(len(vs) for vs in adj.values())
        d_bar = edge_count / max(n, 1)
        try:
            rho_real = cascade(adj, n, r_test, n_reps, np_rng)
        except Exception as e:
            print(f"  [{family}/{label}] cascade real error: {e}", flush=True)
            continue
        rho_er = float("nan")
        rho_ba = float("nan")
        if n <= N_SYNTH_CAP:
            try:
                adj_er = gen_er(n, d_bar, rng)
                rho_er = cascade(adj_er, n, r_test, n_reps, np_rng)
            except Exception as e:
                print(f"  [{family}/{label}] ER error: {e}", flush=True)
            try:
                adj_ba = gen_ba(n, 2, rng, np_rng)
                rho_ba = cascade(adj_ba, n, r_test, n_reps, np_rng)
            except Exception as e:
                print(f"  [{family}/{label}] BA error: {e}", flush=True)
        results.append({
            "family": family,
            "label": label,
            "n": n,
            "edges": edge_count,
            "d_bar": d_bar,
            "depth": depth,
            "rho_real": rho_real,
            "rho_er": rho_er,
            "rho_ba": rho_ba,
        })
        print(f"  [{family:18s} | {label[:30]:30s}] n={n:4d}  E={edge_count:5d}  "
              f"d_bar={d_bar:.2f}  depth={depth:3d}  "
              f"ρ_real={rho_real:.3f}  ρ_ER={rho_er:.3f}  ρ_BA={rho_ba:.3f}",
              flush=True)
        # сохранять partial после каждой строки на случай крашa
        out_partial.write_text(json.dumps({"per_workflow": results}, indent=2))

    # Сводка
    by_family: dict[str, list] = defaultdict(list)
    for r in results:
        by_family[r["family"].split("/")[0]].append(r)
    print(f"\n=== Summary by family (r={r_test}) ===")
    summary = {}
    for fam, lst in by_family.items():
        if not lst:
            continue
        ns = [r["n"] for r in lst]
        dbars = [r["d_bar"] for r in lst]
        depths = [r["depth"] for r in lst]
        rhos_r = [r["rho_real"] for r in lst]
        rhos_e = [r["rho_er"] for r in lst]
        rhos_b = [r["rho_ba"] for r in lst]
        summary[fam] = {
            "n_count": len(lst),
            "n_median": int(np.median(ns)),
            "n_range": [int(min(ns)), int(max(ns))],
            "d_bar_median": float(np.median(dbars)),
            "depth_median": int(np.median(depths)),
            "depth_range": [int(min(depths)), int(max(depths))],
            "rho_real_median": float(np.median(rhos_r)),
            "rho_real_p05": float(np.percentile(rhos_r, 5)),
            "rho_real_p95": float(np.percentile(rhos_r, 95)),
            "rho_er_median": float(np.median(rhos_e)),
            "rho_ba_median": float(np.median(rhos_b)),
        }
        print(f"  {fam:18s} count={len(lst):3d}  n_med={int(np.median(ns)):4d}  "
              f"depth_med={int(np.median(depths)):3d}  "
              f"ρ_real_med={np.median(rhos_r):.3f}  ρ_ER={np.median(rhos_e):.3f}  "
              f"ρ_BA={np.median(rhos_b):.3f}")

    out_path = PAPERS / "wfcommons_validation_results.json"
    out_path.write_text(json.dumps({"per_workflow": results, "summary": summary},
                                    indent=2))
    print(f"\nSaved {out_path}")

    # График: scatter (n, rho) для трёх семейств
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [r["n"] for r in results]
    ax.scatter(ns, [r["rho_real"] for r in results], c="C0", marker="o",
               s=30, alpha=0.7, label="WfCommons (real)")
    ax.scatter(ns, [r["rho_er"] for r in results], c="C1", marker="s",
               s=22, alpha=0.6, label="ER sparse (matched d_bar)")
    ax.scatter(ns, [r["rho_ba"] for r in results], c="C2", marker="^",
               s=22, alpha=0.6, label="BA (m=2)")
    ax.axhline(0.30, color="k", linestyle=":", lw=0.8, label="naive 1-r=0.30")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|H|$ (workflow size)")
    ax.set_ylabel(r"$\rho = |P_{ne}|/|H|$ at r=0.7")
    ax.set_title("Cascade effect: WfCommons real workflows vs synthetic")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out_pdf = OUT_DIR / "wfcommons_vs_synthetic_cascade.pdf"
    fig.savefig(out_pdf)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
