"""Tests for hyppo.core module."""
import math
import pytest


def test_workflow_dag():
    from hyppo.core._workflow import Workflow
    w = Workflow(tasks=["t1", "t2", "t3"], edges=[("t1", "t2"), ("t2", "t3")])
    assert w.is_dag()
    assert w.topological_order() == ["t1", "t2", "t3"]


def test_workflow_rejects_cycle():
    from hyppo.core._workflow import Workflow
    with pytest.raises(ValueError, match="DAG"):
        Workflow(tasks=["a", "b"], edges=[("a", "b"), ("b", "a")])


def test_workflow_reachable():
    from hyppo.core._workflow import Workflow
    w = Workflow(tasks=["a", "b", "c"], edges=[("a", "b"), ("b", "c")])
    assert w.reachable_from("a") == {"b", "c"}


def test_get_aic():
    from hyppo.core._hypothesis import get_aic
    assert get_aic(n_params=2, log_likelihood=0.0) == 4.0


def test_get_bic():
    from hyppo.core._hypothesis import get_bic
    result = get_bic(n_params=2, n_observations=100, log_likelihood=0.0)
    assert result == pytest.approx(2 * math.log(100))


def test_range_models():
    from hyppo.core._hypothesis import range_models
    results = range_models(scores={"m1": 0.8, "m2": 0.6, "m3": 0.9}, threshold=0.7)
    assert results[0] == ("m3", 0.9)
    assert len(results) == 2  # m2 отсечена


def test_range_models_empty():
    from hyppo.core._hypothesis import range_models
    results = range_models(scores={"m1": 0.3}, threshold=0.7)
    assert results == []
