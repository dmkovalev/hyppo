"""Tests for VirtualExperimentRunner."""

from hyppo.metadata_repository import MetadataRepository
from hyppo.runner import Runner


def test_runner_success(tmp_path):
    repo = MetadataRepository(db_path=tmp_path / "test.db")
    runner = Runner(repo, max_retries=3)
    results = runner.execute(
        plan={"p_ne": ["h1"], "p_e": set()},
        models={"h1": lambda c: {"r2": 0.85}},
        configs={"h1": {"param": 1}},
    )
    assert results["h1"]["status"] == "SUCCESS"
    assert results["h1"]["metrics"]["r2"] == 0.85
    repo.close()


def test_runner_retry(tmp_path):
    call_count = 0

    def flaky(config):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient failure")
        return {"r2": 0.7}

    repo = MetadataRepository(db_path=tmp_path / "test.db")
    runner = Runner(repo, max_retries=3)
    results = runner.execute(
        plan={"p_ne": ["h1"], "p_e": set()},
        models={"h1": flaky},
    )
    assert results["h1"]["status"] == "SUCCESS"
    assert call_count == 3
    repo.close()


def test_runner_cascade_skip(tmp_path):
    def fail_always(config):
        raise RuntimeError("permanent failure")

    repo = MetadataRepository(db_path=tmp_path / "test.db")
    runner = Runner(repo, max_retries=2)
    results = runner.execute(
        plan={"p_ne": ["h1", "h2"], "p_e": set()},
        models={"h1": fail_always, "h2": lambda c: {"r2": 0.9}},
        lattice_edges=[("h1", "h2")],
    )
    assert results["h1"]["status"] == "FAILED"
    assert results["h2"]["status"] == "SKIPPED"
    repo.close()


def test_runner_no_model(tmp_path):
    """Hypothesis without model function should be marked FAILED."""
    repo = MetadataRepository(db_path=tmp_path / "test.db")
    runner = Runner(repo, max_retries=2)
    results = runner.execute(
        plan={"p_ne": ["h1"], "p_e": set()},
        models={},
    )
    assert results["h1"]["status"] == "FAILED"
    assert results["h1"]["error"] == "No model function"
    repo.close()


def test_runner_cached_p_e(tmp_path):
    """P_e hypotheses should be loaded as SUCCESS from cache."""
    repo = MetadataRepository(db_path=tmp_path / "test.db")
    repo.save_result("h1", {"param": 1}, {"r2": 0.9}, "SUCCESS")
    runner = Runner(repo, max_retries=3)
    results = runner.execute(
        plan={"p_ne": set(), "p_e": ["h1"]},
        models={},
        configs={"h1": {"param": 1}},
    )
    assert results["h1"]["status"] == "SUCCESS"
    assert results["h1"]["metrics"]["r2"] == 0.9
    repo.close()
