from hyppo.coa.graph import plan_cascade
from hyppo.manager import Manager
from hyppo.metadata_repository import MetadataRepository


def build_graph(ve: dict) -> dict:
    nodes = [h["id"] for h in ve["hypotheses"]]
    edges = [list(e) for e in ve.get("workflow_edges", [])]
    return {"nodes": nodes, "edges": edges}


def plan_preview(ve: dict, db_path: str) -> dict:
    """Reproduce Algorithm 4 partitioning against repository cache state.

    A hypothesis with no cached result (has_result False for empty config)
    forces itself and all descendants into P_ne; others go to P_e. The cascade
    itself is the single canonical two-way core via ``plan_cascade``.
    """
    repo = MetadataRepository(db_path=db_path)
    try:
        nodes = [h["id"] for h in ve["hypotheses"]]
        edges = [tuple(e) for e in ve.get("workflow_edges", [])]
        cached = {h for h in nodes if repo.has_result(h, {})}
        p_ne = plan_cascade(nodes, edges, cached)
        p_e = [h for h in nodes if h not in p_ne]
        return {"p_ne": sorted(p_ne), "p_e": p_e}
    finally:
        repo.close()


def _stub_model(hid: str):
    # Deterministic placeholder model: real models are registered by the
    # researcher; the GUI ships stubs so the lifecycle is exercisable on demo.
    def model(config: dict) -> dict:
        seed = sum(ord(ch) for ch in hid)
        return {"r2": 0.5 + (seed % 40) / 100.0}

    return model


def run_iteration(ve: dict, db_path: str) -> dict:
    from hyppo.metadata_repository import MetadataRepository

    nodes = [h["id"] for h in ve["hypotheses"]]
    edges = [tuple(e) for e in ve.get("workflow_edges", [])]

    repo = MetadataRepository(db_path=db_path)
    try:
        reused = sum(1 for h in nodes if repo.has_result(h, {}))
    finally:
        repo.close()

    mgr = Manager(db_path=db_path)
    try:
        results = mgr.orchestrate(
            hypotheses=nodes,
            workflow_edges=edges,
            models={h: _stub_model(h) for h in nodes},
        )
    finally:
        mgr.close()

    best: tuple[str | None, dict] = max(
        results.items(),
        key=lambda kv: kv[1].get("metrics", {}).get("r2", 0.0),
        default=(None, {}),
    )
    return {
        "results": results,
        "reused": reused,
        "best": {"hypothesis": best[0], "r2": best[1].get("metrics", {}).get("r2")},
    }


def iteration_history(store, pid: str) -> list[dict]:
    return store.list_iterations(pid)
