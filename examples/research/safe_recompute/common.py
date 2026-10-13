"""Общие загрузчики графов гипотез для экспериментов статьи о безопасном пересчёте."""
from __future__ import annotations

import hashlib
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


WFCOMMONS_MANIFEST = DATA / "wfcommons_manifest.json"
WFCOMMONS_RAW = "https://raw.githubusercontent.com/wfcommons/wfinstances/{ref}/{path}"


def _wfcommons_corpus() -> set[tuple[str, str]]:
    """Корпус Э2 диссертации: 171 трасса из `planning/data/wfcommons_validation_results.json`."""
    rows = json.loads((PLANNING_DATA / "wfcommons_validation_results.json").read_text(encoding="utf-8"))
    return {(w["family"].split("/")[0], w["label"].split("/")[-1]) for w in rows["per_workflow"]}


def wfcommons_manifest() -> dict[str, dict]:
    """Содержимое корпуса, зафиксированное по sha256: ключ «семейство/имя файла» ->
    {path, sha256, bytes, git_blob_sha1, n_nodes, n_edges}; порядок ключей -- порядок обхода."""
    return json.loads(WFCOMMONS_MANIFEST.read_text(encoding="utf-8"))["files"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_wfformat(path: Path) -> nx.DiGraph:
    """Разбор совпадает с `_parse_wfformat` из `planning/scripts/draw_fig2_composite.py`:
    узлы -- `t["id"]`, рёбра -- `children`."""
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))["workflow"]["specification"]["tasks"]
    ids = {t["id"] for t in tasks}
    dag = nx.DiGraph()
    dag.add_nodes_from(ids)
    dag.add_edges_from((t["id"], ch) for t in tasks for ch in t.get("children", []) if ch in ids)
    return dag


class CorpusMismatch(ValueError):
    """Файл трассы отсутствует или его содержимое не совпадает с манифестом."""


def wfcommons_graphs(limit: int | None = None, root: Path | None = None,
                     verify: bool = True) -> dict[str, nx.DiGraph]:
    """Корпус Э2 диссертации: 171 трасса WfCommons на уровне задач (nextflow/snakemake/pegasus).
    Файлы читаются из `root` (по умолчанию `planning/cache/wfcommons`, загрузка --
    `python -m examples.research.safe_recompute.fetch_wfcommons`) и сверяются с
    `data/wfcommons_manifest.json` (sha256, число узлов и рёбер); при расхождении --
    `CorpusMismatch`. Ключ: «семейство/имя файла»."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    root = Path(root) if root is not None else PLANNING_CACHE / "wfcommons"
    out: dict[str, nx.DiGraph] = {}
    for key, m in wfcommons_manifest().items():
        path = root / m["path"]
        if not path.exists():
            raise CorpusMismatch(f"WfCommons trace {key} missing at {path}; run "
                                 "python -m examples.research.safe_recompute.fetch_wfcommons")
        if verify and sha256_of(path) != m["sha256"]:
            raise CorpusMismatch(f"WfCommons trace {key} ({path}) differs from the pinned corpus "
                                 f"(sha256 {m['sha256']}); re-fetch it")
        dag = parse_wfformat(path)
        if (dag.number_of_nodes(), dag.number_of_edges()) != (m["n_nodes"], m["n_edges"]):
            raise CorpusMismatch(f"WfCommons trace {key}: parsed {dag.number_of_nodes()} nodes/"
                                 f"{dag.number_of_edges()} edges, manifest {m['n_nodes']}/{m['n_edges']}")
        out[key] = dag
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
