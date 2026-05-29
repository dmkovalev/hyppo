"""Композитная fig 2 (2×2):
(а) каскад nf-core/airrflow (15 hyp, биоинформатика);
(б) каскад BGM (10 hyp, астроинформатика);
(в) каскад HybridCRM (19 hyp, петроинформатика);
(г) sensitivity ρ(r) + звёзды BGM/CRM.
"""
from __future__ import annotations
import json, random, shutil, subprocess, sys
from collections import defaultdict, deque
from pathlib import Path

from hyppo.coa import HypothesisGraph

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent   # experiments/iip2026/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT / "fig2_composite.pdf"
SENSITIVITY_JSON = DATA / "rho_vs_r_sweep.json"
HAND_CURATED = DATA / "hand_curated_hypothesis_graphs.json"
WFCOMMONS_VALIDATION = DATA / "wfcommons_validation_results.json"
WFCOMMONS_CACHE_DIR = CACHE / "wfcommons"
WFCOMMONS_SWEEP_CACHE = DATA / "wfcommons_per_workflow_sweep.json"
DOT_BIN = shutil.which("dot") or "C:/Program Files/Graphviz/bin/dot.exe"
R_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ── BGM (10 nodes, 12 edges, from BGM_lattice_standalone.tex) ────────
BGM_NODES = [
    "Закон\nплотн.", "Возр.\nячейки", "SFR", "IMF",
    "Лок. объём.\nплотность", "Скорость\nзвёздообр.",
    "Эволюц.\nтреки", "Лок.\nсветим.",
    "Дисп.\nскор.", "Дин.\nсамосогл.",
]
BGM_EDGES = [
    (1, 4), (0, 4),          # Age,Dens → LVD
    (2, 5), (3, 5), (1, 5),  # SFR,IMF,Age → SBR
    (4, 6), (5, 7), (4, 7),  # LVD→ET, SBR→LL, LVD→LL
    (6, 8), (7, 9), (6, 9), (8, 9),  # ET→AVD, LL→DSC, ET→DSC, AVD→DSC
]
BGM_SEED = [3]  # IMF → cascade SBR, LL, DSC = 4/10

# ── HCP (3 nodes, 2 edges, from part4.tex: atlas→conn→group) ────────
HCP_NODES = ["Atlas", "Conn", "Group"]
HCP_EDGES = [(0, 1), (1, 2)]  # atlas→conn→group
HCP_SEED = [0]  # atlas → cascade conn + group = 2/3

# ── Oil/HybridCRM (19 nodes, from part4.tex) ────────────────────────
OIL_NODES = [
    "$H_1$", "$H_2$", "$H_3$", "$H_4$", "$H_5$",
    "$H_6$", "$H_7$", "$H_8$", "$H_9$", "$H_{10}$",
    "$H_{11}$", "$H_{12}$", "$H_{13}$", "$H_{14}$",
    "$H_{15}$", "$H_{16}$", "$H_{17}$", "$H_{18}$",
    "$H_{19}$",
]
OIL_EDGES = [
    # Branch A (LPR)
    (0, 1), (0, 2),       # H1→H2, H1→H3
    (1, 3), (2, 3),       # H2→H4, H3→H4
    (3, 4),               # H4→H5
    (4, 5), (4, 6), (4, 7),  # H5→H6, H5→H7, H5→H8
    (5, 8), (6, 8),       # H6→H9, H7→H9
    (7, 9), (8, 9),       # H8→H10, H9→H10
    # Branch B (WCT)
    (10, 11),             # H11→H12
    (11, 13),             # H12→H14
    (12, 15), (13, 15), (14, 15),  # H13,H14,H15→H16
    (15, 16),             # H16→H17
    (16, 17),             # H17→H18
    # Merge
    (9, 18), (17, 18),    # H10,H18→H19
]
OIL_SEED = [0]  # H1 → cascade through LPR branch + OPR


def algorithm_2(adj_d, cached, vertices):
    vlist = list(vertices)
    v_to_idx = {v: i for i, v in enumerate(vlist)}
    n = len(vlist)
    edges = [(v_to_idx[u], v_to_idx[v])
             for u, vs in adj_d.items() if u in v_to_idx
             for v in vs if v in v_to_idx]
    cached_idx = {v_to_idx[v] for v in cached if v in v_to_idx}
    p_ne_idx = HypothesisGraph.from_edges(n, edges).plan(cached_idx)
    return {vlist[i] for i in p_ne_idx}


def graphviz_layout(n, edges, rankdir="TB"):
    lines = ["digraph G {",
             f"  rankdir={rankdir};",
             "  nodesep=0.22; ranksep=0.40;",
             "  node [shape=box, width=0.45, height=0.25, fixedsize=true];"]
    for i in range(n):
        lines.append(f"  n{i};")
    for u, v in edges:
        lines.append(f"  n{u} -> n{v};")
    lines.append("}")
    proc = subprocess.run([DOT_BIN, "-Tplain"], input="\n".join(lines),
                          capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"dot failed: {proc.stderr}")
    pos = {}
    for line in proc.stdout.splitlines():
        toks = line.split()
        if toks and toks[0] == "node" and toks[1].startswith("n"):
            pos[int(toks[1][1:])] = (float(toks[2]), float(toks[3]))
    return pos


def load_hand_curated(key, dirty_seeds_ids):
    data = json.loads(HAND_CURATED.read_text(encoding="utf-8"))
    g = data[key]
    id_to_idx = {h["id"]: i for i, h in enumerate(g["hypotheses"])}
    n = len(g["hypotheses"])
    edges = []
    for e in g["subsumption_edges"] + g["inter_layer_edges"]:
        edges.append((id_to_idx[e["src"]], id_to_idx[e["tgt"]]))
    labels = {i: h["id"] for i, h in enumerate(g["hypotheses"])}
    seeds_set = {id_to_idx[s] for s in dirty_seeds_ids if s in id_to_idx}
    cached = set(range(n)) - seeds_set
    return n, edges, labels, cached, seeds_set


def _draw_dag(ax, n, edges, labels, cached, seeds, title,
              node_size=420, font_size=4.5):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    vertices = list(range(n))
    p_ne = algorithm_2(dict(adj), cached, vertices)
    propagated = p_ne - seeds
    pos = graphviz_layout(n, edges)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    cmap = {}
    for v in range(n):
        if v in seeds:
            cmap[v] = "#e74c3c"
        elif v in propagated:
            cmap[v] = "#f39c12"
        else:
            cmap[v] = "#2ecc71"
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=[cmap[v] for v in range(n)],
                           node_size=node_size, edgecolors="black",
                           linewidths=0.5, node_shape="o")
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                            font_size=font_size)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=6,
                           edge_color="#444", alpha=0.65,
                           connectionstyle="arc3,rad=0.04",
                           width=0.5, node_size=node_size)
    r = len(cached)
    rho = len(p_ne) / n
    ax.set_title(f"{title}\n$|H|={n}$, $r{{=}}{r}/{n}$, "
                 f"$\\rho{{=}}{rho:.2f}$", fontsize=11)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")
    print(f"[{title}] |H|={n}, seeds={[labels[s] for s in seeds]}, "
          f"|P_ne|={len(p_ne)}, ρ={rho:.2f}")


def draw_dag_nfcore(ax, key, dirty_seeds_ids, title,
                    node_size=600, font_size=5.5):
    n, edges, labels, cached, seeds = load_hand_curated(key, dirty_seeds_ids)
    _draw_dag(ax, n, edges, labels, cached, seeds, title,
              node_size=node_size, font_size=font_size)


def draw_dag_inline(ax, nodes, edges, seed_indices, title,
                    node_size=380, font_size=4.0):
    n = len(nodes)
    labels = {i: nodes[i] for i in range(n)}
    seeds = set(seed_indices)
    cached = set(range(n)) - seeds
    _draw_dag(ax, n, edges, labels, cached, seeds, title,
              node_size=node_size, font_size=font_size)


def _parse_wfformat(path):
    """Парсит WfFormat-трассу → (n, adj-dict, depth) — рёбра по children."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data["workflow"]["specification"]["tasks"]
    id_to_idx = {t["id"]: i for i, t in enumerate(tasks)}
    n = len(tasks)
    adj = defaultdict(set)
    for t in tasks:
        u = id_to_idx[t["id"]]
        for c in t.get("children", []):
            if c in id_to_idx:
                adj[u].add(id_to_idx[c])
    return n, dict(adj)


def _compute_wfcommons_sweep(r_grid, n_runs=30):
    """Для каждой из 157 трасс WfCommons считает ρ при каждом r из r_grid.
    Результат: {label: {family, rhos: [ρ_r1, ρ_r2, …]}}. Кешируется в JSON."""
    if WFCOMMONS_SWEEP_CACHE.exists():
        cached = json.loads(WFCOMMONS_SWEEP_CACHE.read_text(encoding="utf-8"))
        if cached.get("__meta__", {}).get("r_grid") == list(r_grid):
            print(f"[WfCommons sweep] cache hit: {WFCOMMONS_SWEEP_CACHE.name}")
            return cached["by_workflow"]
    print(f"[WfCommons sweep] computing for r={r_grid} …")
    out = {}
    families = ["nextflow", "snakemake", "pegasus"]
    for fam in families:
        fam_dir = WFCOMMONS_CACHE_DIR / fam
        if not fam_dir.exists():
            continue
        for path in sorted(fam_dir.rglob("*.json")):
            label = f"{path.parent.name}/{path.stem}" if path.parent != fam_dir else path.stem
            n, adj = _parse_wfformat(path)
            edges = [(u, v) for u, vs in adj.items() for v in vs]
            rho_d = _sweep_rho(edges, n, r_grid, n_runs=n_runs)
            out[label] = {"family": fam, "n": n,
                          "rhos": [rho_d[f"{r:.1f}"] for r in r_grid]}
            print(f"  {fam}/{label}: |H|={n}, "
                  f"ρ@r=0.7={rho_d['0.7']:.3f}")
    payload = {"__meta__": {"r_grid": list(r_grid), "n_runs": n_runs,
                            "n_workflows": len(out)},
               "by_workflow": out}
    WFCOMMONS_SWEEP_CACHE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[WfCommons sweep] saved {len(out)} workflows → "
          f"{WFCOMMONS_SWEEP_CACHE.name}")
    return out


def _sweep_rho(edges, n, r_grid, n_runs=300):
    rng = random.Random(42)
    vertices = list(range(n))
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    results = {}
    for r in r_grid:
        k = max(0, min(n, round(r * n)))
        rhos = []
        for _ in range(n_runs):
            cached = set(rng.sample(vertices, k))
            p_ne = algorithm_2(dict(adj), cached, vertices)
            rhos.append(len(p_ne) / n)
        results[f"{r:.1f}"] = float(np.median(rhos))
    return results


def draw_sensitivity(ax, data):
    rs = data["r"]
    ax.plot(rs, [1 - r for r in rs], "k:", lw=1.4, label=r"наивно $1-r$")
    # ЭР и БА — выбор синтетической модели-ориентира для оценки каскада:
    #   БА — графы с хабами (верхняя оценка),
    #   ЭР — равномерные связи той же плотности (двусторонняя оценка).
    line_series = [
        ("БА ($m{=}2$, хабы) — верхняя оценка",
         "BA", "C2", "-",  "верхняя оценка (хабы)"),
        ("ЭР (равномерные связи) — двусторонняя",
         "ER", "C1", "--", "двусторонняя оценка"),
    ]
    annot_targets = {}
    for label, key, color, ls, _anno in line_series:
        vals = data[key]
        xs_y = [(r, v) for r, v in zip(rs, vals) if v is not None]
        if not xs_y:
            continue
        xs, ys = zip(*xs_y)
        ax.plot(xs, ys, color=color, linestyle=ls, lw=2.4,
                label=label, alpha=0.95, zorder=4)
        annot_targets[key] = (xs, ys, color)
    # Текстовые подписи возле кривых — подчёркивают роль выбора модели
    if "BA" in annot_targets:
        _, _, color = annot_targets["BA"]
        # под полкой БА, у самого правого края (правый край текста на x=1.04)
        ax.text(1.04, 0.92, "БА (хабы) — верхняя оценка",
                color=color, fontsize=11, weight="bold",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="none", alpha=0.75),
                zorder=10)
    if "ER" in annot_targets:
        _, _, color = annot_targets["ER"]
        # ниже ЭР в правой части (на спаде, между кривой и осью X)
        ax.text(0.78, 0.46, "ЭР (равном. связи) — двусторонняя",
                color=color, fontsize=11, weight="bold",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="none", alpha=0.75),
                zorder=10)
    # 157 трасс WfCommons: ρ при каждом r из R_GRID (сетка по всему диапазону)
    wf_sweep = _compute_wfcommons_sweep(R_GRID, n_runs=30)
    wf_xs, wf_ys = [], []
    rng_jitter = np.random.default_rng(42)
    for label, entry in wf_sweep.items():
        rhos = entry["rhos"]
        for r, rho in zip(R_GRID, rhos):
            x = r + rng_jitter.uniform(-0.012, 0.012)
            wf_xs.append(x)
            wf_ys.append(rho)
    ax.scatter(wf_xs, wf_ys, marker="o", s=14, c="C0",
               edgecolors="none", alpha=0.40, zorder=5,
               label=f"WfCommons ({len(wf_sweep)}~трасс)")
    # 20 nf-core ручных конвейеров: индивидуальные траектории по R_GRID
    hc = json.loads(HAND_CURATED.read_text(encoding="utf-8"))
    nfcore_xs, nfcore_ys = [], []
    for name, g in hc.items():
        id_to_idx = {h["id"]: i for i, h in enumerate(g["hypotheses"])}
        n = len(g["hypotheses"])
        edges = []
        for e in g["subsumption_edges"] + g["inter_layer_edges"]:
            edges.append((id_to_idx[e["src"]], id_to_idx[e["tgt"]]))
        rho_d = _sweep_rho(edges, n, R_GRID, n_runs=50)
        for r_str, rho in rho_d.items():
            nfcore_xs.append(float(r_str))
            nfcore_ys.append(rho)
    ax.scatter(nfcore_xs, nfcore_ys, marker="*", s=45, c="C4",
               edgecolors="black", linewidths=0.3,
               alpha=0.75, zorder=6,
               label=f"nf-core ручные ({len(hc)}~конв.)")
    # BGM + HCP + Oil sweeps
    bgm_rho = _sweep_rho(BGM_EDGES, len(BGM_NODES), R_GRID)
    hcp_rho = _sweep_rho(HCP_EDGES, len(HCP_NODES), R_GRID)
    oil_rho = _sweep_rho(OIL_EDGES, len(OIL_NODES), R_GRID)
    for name, rho_d, color, yoff in [
        ("Астро", bgm_rho, "#8B0000", 8),
        ("Нейро", hcp_rho, "#006400", 0),
        ("Петро", oil_rho, "#00008B", -10),
    ]:
        xs = sorted(float(r) for r in rho_d)
        ys = [rho_d[f"{x:.1f}"] for x in xs]
        ax.scatter(xs, ys, marker="p", s=95, c=color,
                   edgecolors="black", linewidths=0.5,
                   alpha=0.9, zorder=7)
        ax.annotate(name, (xs[-1], ys[-1]),
                    xytext=(8, yoff), textcoords="offset points",
                    fontsize=9, color=color, weight="bold",
                    va="center", alpha=0.95)
    ax.set_xlabel(r"$r = |\mathrm{Cache}|/|H|$", fontsize=11)
    ax.set_ylabel(r"$\rho = |P_{ne}|/|H|$ (медиана)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9.5,
              framealpha=0.85, borderpad=0.5)
    ax.set_title("(д) Доля пересчёта $\\rho(r)$: БА — верхняя оценка",
                 fontsize=12)


def main():
    sens_data = json.loads(SENSITIVITY_JSON.read_text())
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 12))
    # Первая строка: 4 DAG-графа равной ширины.
    # Вторая строка: панель (д) ρ(r) на всю ширину, увеличенная по высоте.
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.0],
                          height_ratios=[1.0, 1.6],
                          wspace=0.18, hspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, :])

    draw_dag_inline(ax_a, BGM_NODES, BGM_EDGES, BGM_SEED,
                    "(а) Астроинформатика", node_size=750, font_size=5.5)
    draw_dag_inline(ax_b, HCP_NODES, HCP_EDGES, HCP_SEED,
                    "(б) Нейроинформатика", node_size=850, font_size=7.0)
    draw_dag_inline(ax_c, OIL_NODES, OIL_EDGES, OIL_SEED,
                    "(в) Петроинформатика", node_size=520, font_size=6.5)
    draw_dag_nfcore(ax_d, "airrflow", ["vdj01"],
                    "(г) Биоинформатика")
    draw_sensitivity(ax_e, sens_data)

    legend_handles = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="black",
                       label="Кэш ($P_e$)"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black",
                       label="Изменённая гипотеза ($P_{ne}$)"),
        mpatches.Patch(facecolor="#f39c12", edgecolor="black",
                       label="Каскадно устаревшие ($P_{ne}$)"),
    ]
    # Под DAG-графами, перед панелью (д). y в figure-coords между rows.
    fig.legend(handles=legend_handles, loc="center",
               ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.605))
    fig.savefig(OUT_FILE, bbox_inches="tight", dpi=200)
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
