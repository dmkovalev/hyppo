"""Общие загрузчики графов гипотез для экспериментов статьи о безопасном пересчёте."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PLANNING_DATA = ROOT.parent / "planning" / "data"
PLANNING_CACHE = ROOT.parent / "planning" / "cache"

CRM_EDGES = [
    ("H1", "H2"), ("H1", "H3"), ("H2", "H4"), ("H3", "H4"), ("H4", "H5"),
    ("H6", "H5"), ("H7", "H8"), ("H5", "H8"), ("H8", "H9"), ("H9", "H10"),
    ("H9", "H11"), ("H10", "H12"), ("H11", "H12"), ("H12", "H13"), ("H13", "H14"),
    ("H15", "H5"), ("H8", "H16"), ("H14", "H16"),
]


def crm_graph() -> nx.DiGraph:
    """Граф CRM (Brugge) из гл. 4 диссертации: 16 гипотез, 18 рёбер, глубина 10 (выведен алгоритмом 1)."""
    g = nx.DiGraph()
    g.add_nodes_from(f"H{i}" for i in range(1, 17))
    g.add_edges_from(CRM_EDGES)
    return g


def nfcore_graphs() -> dict[str, nx.DiGraph]:
    """26 ручных графов nf-core (`hand_curated_hypothesis_graphs_26.json`)."""
    d = json.loads((PLANNING_DATA / "hand_curated_hypothesis_graphs_26.json").read_text(encoding="utf-8"))
    out = {}
    for name, g in d.items():
        dag = nx.DiGraph()
        dag.add_nodes_from(h["id"] for h in g["hypotheses"])
        dag.add_edges_from((e["src"], e["tgt"]) for e in g["subsumption_edges"] + g["inter_layer_edges"])
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(f"nf-core graph {name} is cyclic")
        out[name] = dag
    return out


def wfcommons_graphs(limit: int | None = None) -> dict[str, nx.DiGraph]:
    """Корпус Э2 диссертации: 171 трасса WfCommons на уровне задач (nextflow/snakemake/pegasus).
    Разбор совпадает с `_parse_wfformat` из `planning/scripts/draw_fig2_composite.py`:
    узлы -- `t["id"]`, рёбра -- `children`. Ключ: «семейство/имя файла»."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    out: dict[str, nx.DiGraph] = {}
    for fam in ("nextflow", "snakemake", "pegasus"):
        for path in sorted((PLANNING_CACHE / "wfcommons" / fam).rglob("*.json")):
            tasks = json.loads(path.read_text(encoding="utf-8"))["workflow"]["specification"]["tasks"]
            ids = {t["id"] for t in tasks}
            dag = nx.DiGraph()
            dag.add_nodes_from(ids)
            dag.add_edges_from((t["id"], ch) for t in tasks for ch in t.get("children", []) if ch in ids)
            out[f"{fam}/{path.stem}"] = dag
            if limit is not None and len(out) == limit:
                return out
    return out


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
