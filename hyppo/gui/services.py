from hyppo.metadata_repository import MetadataRepository


def build_graph(ve: dict) -> dict:
    nodes = [h["id"] for h in ve["hypotheses"]]
    edges = [list(e) for e in ve.get("workflow_edges", [])]
    return {"nodes": nodes, "edges": edges}


def _descendants(edges: list[list[str]], start: str) -> set[str]:
    out, stack = set(), [start]
    while stack:
        n = stack.pop()
        for a, b in edges:
            if a == n and b not in out:
                out.add(b)
                stack.append(b)
    return out


def plan_preview(ve: dict, db_path: str) -> dict:
    """Reproduce Algorithm 4 partitioning against repository cache state.

    A hypothesis with no cached result (has_result False for empty config)
    forces itself and all descendants into P_ne; others go to P_e.
    """
    repo = MetadataRepository(db_path=db_path)
    try:
        nodes = [h["id"] for h in ve["hypotheses"]]
        edges = [list(e) for e in ve.get("workflow_edges", [])]
        p_ne: set[str] = set()
        for h in nodes:
            if not repo.has_result(h, {}):
                p_ne.add(h)
                p_ne |= _descendants(edges, h)
        p_e = [h for h in nodes if h not in p_ne]
        return {"p_ne": sorted(p_ne), "p_e": p_e}
    finally:
        repo.close()
