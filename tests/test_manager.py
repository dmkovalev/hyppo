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
