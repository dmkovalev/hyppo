"""Tests for Manager orchestration."""


def test_orchestrate_end_to_end(tmp_path):
    from hyppo.manager import Manager

    manager = Manager(db_path=tmp_path / "test.db")
    results = manager.orchestrate(
        hypotheses=["h1", "h2"],
        workflow_edges=[("h1", "h2")],
        models={"h1": lambda c: {"r2": 0.9}, "h2": lambda c: {"r2": 0.8}},
        config={"h1": {"param": 1}, "h2": {"param": 2}},
    )
    assert results["h1"]["status"] == "SUCCESS"
    assert results["h2"]["status"] == "SUCCESS"
    assert results["h1"]["metrics"]["r2"] == 0.9
    manager.close()


def test_orchestrate_builds_lattice_from_structures(tmp_path):
    """When causal structures are supplied, Stage 2 runs Algorithm 1
    (HypothesisGraph.build) instead of using the workflow edges verbatim."""
    from hyppo.manager import Manager

    manager = Manager(db_path=tmp_path / "test.db")
    # two variable-disjoint complete structures -> their union is complete,
    # so build() yields the reachable lattice edge and execution succeeds.
    structures = {
        "h1": [frozenset({"a", "b"}), frozenset({"b"})],
        "h2": [frozenset({"c", "d"}), frozenset({"d"})],
    }
    results = manager.orchestrate(
        hypotheses=["h1", "h2"],
        workflow_edges=[("h1", "h2")],
        models={"h1": lambda c: {"r2": 0.9}, "h2": lambda c: {"r2": 0.8}},
        structures=structures,
    )
    assert results["h1"]["status"] == "SUCCESS"
    assert results["h2"]["status"] == "SUCCESS"
    manager.close()


def test_orchestrate_low_r2_prunes_descendants(tmp_path):
    """A cached hypothesis with R2 below threshold prunes itself AND its
    descendants from BOTH P_ne and P_e (planner's three-way semantics).

    Regression guard: the previous manager `continue` pruned only the node
    itself, so descendants of a low-R2 node still reached the runner. Here h2
    (R2=0.4) and its descendant h3 must be absent from results entirely."""
    from hyppo.manager import Manager

    manager = Manager(db_path=tmp_path / "test.db", r2_threshold=0.7)
    manager.repository.save_result("h1", {}, {"r2": 0.9})
    manager.repository.save_result("h2", {}, {"r2": 0.4})
    results = manager.orchestrate(
        hypotheses=["h1", "h2", "h3"],
        workflow_edges=[("h1", "h2"), ("h2", "h3")],
        models={h: (lambda c: {"r2": 0.9}) for h in ("h1", "h2", "h3")},
    )
    assert results["h1"]["status"] == "SUCCESS"  # cached, good R2 -> reused
    assert "h2" not in results  # pruned (R2 < threshold)
    assert "h3" not in results  # pruned as descendant of h2 (cascade)
    manager.close()


def test_orchestrate_cache_miss_cascades_descendants(tmp_path):
    """A cache miss forces the node AND all descendants into P_ne, even if a
    descendant has its own good cached result (cache-miss cascade)."""
    from hyppo.manager import Manager

    manager = Manager(db_path=tmp_path / "test.db", r2_threshold=0.7)
    # h3 is cached-good, but its ancestor h2 misses -> h3 recomputed anyway.
    manager.repository.save_result("h3", {}, {"r2": 0.95})
    results = manager.orchestrate(
        hypotheses=["h1", "h2", "h3"],
        workflow_edges=[("h1", "h2"), ("h2", "h3")],
        models={h: (lambda c: {"r2": 0.8}) for h in ("h1", "h2", "h3")},
    )
    # all execute (h1 miss cascades to h2, h3); h3's stale cache is not reused
    assert results["h1"]["status"] == "SUCCESS"
    assert results["h2"]["status"] == "SUCCESS"
    assert results["h3"]["metrics"]["r2"] == 0.8  # recomputed, not cached 0.95
    manager.close()


def _failing_model(c):
    raise RuntimeError("fail")


def test_orchestrate_with_failure(tmp_path):
    from hyppo.manager import Manager

    manager = Manager(db_path=tmp_path / "test.db", max_retries=1)
    results = manager.orchestrate(
        hypotheses=["h1", "h2"],
        workflow_edges=[("h1", "h2")],
        models={"h1": _failing_model, "h2": lambda c: {"r2": 0.8}},
    )
    assert results["h1"]["status"] == "FAILED"
    assert results["h2"]["status"] == "SKIPPED"
    manager.close()
