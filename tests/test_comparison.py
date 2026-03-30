"""Tests for hypothesis comparison methods."""
import math
import pytest


def test_sign_test_significant():
    from hyppo.comparison import sign_test
    errors_a = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.13, 0.14, 0.16, 0.17]
    errors_b = [0.5, 0.6, 0.55, 0.52, 0.58, 0.51, 0.53, 0.54, 0.56, 0.57]
    p = sign_test(errors_a, errors_b)
    assert p < 0.05


def test_sign_test_not_significant():
    from hyppo.comparison import sign_test
    errors_a = [0.5, 0.1, 0.5, 0.1, 0.5]
    errors_b = [0.1, 0.5, 0.1, 0.5, 0.1]
    p = sign_test(errors_a, errors_b)
    assert p > 0.05


def test_compute_aic():
    from hyppo.comparison import compute_aic
    assert compute_aic(n_params=2, log_likelihood=0.0) == 4.0


def test_compute_bic():
    from hyppo.comparison import compute_bic
    result = compute_bic(n_params=2, n_observations=100, log_likelihood=0.0)
    assert result == pytest.approx(2 * math.log(100))


def test_combined_ranking():
    from hyppo.comparison import combined_ranking
    scores = {
        "h1": {"aic": -100, "r2": 0.9},
        "h2": {"aic": -80, "r2": 0.7},
        "h3": {"aic": -120, "r2": 0.6},
    }
    ranking = combined_ranking(scores, weights={"aic": 0.5, "r2": 0.5})
    assert ranking[0][0] == "h1"  # best combined


def test_combined_ranking_equal_weights():
    from hyppo.comparison import combined_ranking
    scores = {"a": {"x": 1.0}, "b": {"x": 0.5}}
    ranking = combined_ranking(scores)
    assert ranking[0][0] == "a"
