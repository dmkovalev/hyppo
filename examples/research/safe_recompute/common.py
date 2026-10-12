"""Общие загрузчики графов гипотез для экспериментов статьи о безопасном пересчёте."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PLANNING_DATA = ROOT.parent / "planning" / "data"

CRM_EDGES = [
    ("H1", "H2"), ("H1", "H3"), ("H2", "H4"), ("H3", "H4"), ("H4", "H5"),
    ("H6", "H5"), ("H7", "H8"), ("H5", "H8"), ("H8", "H9"), ("H9", "H10"),
    ("H10", "H11"), ("H11", "H12"), ("H12", "H13"), ("H13", "H14"),
    ("H15", "H5"), ("H8", "H16"), ("H14", "H16"),
]


def crm_graph() -> nx.DiGraph:
    """Граф CRM (Brugge) из гл. 4 диссертации: 16 гипотез, 17 рёбер, глубина 11."""
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
        if nx.is_directed_acyclic_graph(dag):
            out[name] = dag
    return out


def wfcommons_graphs(limit: int | None = None) -> dict[str, nx.DiGraph]:
    """Трассы WfCommons как графы гипотез тем же разбором, что в Э2 диссертации
    (`planning/scripts/cascade_on_hypothesis_graphs.py`, exp_hierarchy): семейства
    nextflow/snakemake/pegasus, уровень агрегации REC_LEVEL[family], n >= 3.
    Ключ: «семейство/имя файла»."""
    from examples.research.planning.scripts import cascade_on_hypothesis_graphs as c

    out: dict[str, nx.DiGraph] = {}
    for fam in ("nextflow", "snakemake", "pegasus"):
        d = c.CACHE_DIR / fam
        if not d.exists():
            continue
        for path in sorted(d.rglob("*.json")):
            parsed = c.parse_workflow_to_hypothesis(path, fam, c.REC_LEVEL.get(fam, 1))
            if parsed is None:
                continue
            n, adj = parsed
            if n < 3:
                continue
            dag = nx.DiGraph()
            dag.add_nodes_from(range(n))
            dag.add_edges_from((u, v) for u, vs in adj.items() for v in vs)
            if nx.is_directed_acyclic_graph(dag):
                out[f"{fam}/{path.stem}"] = dag
            if limit and len(out) >= limit:
                return out
    return out


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
