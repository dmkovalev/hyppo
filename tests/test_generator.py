"""Tests for HypothesisGenerator."""

import numpy as np


def test_generate_linear():
    from hyppo.generator import generate_hypothesis

    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    results = generate_hypothesis(X, y, r2_min=0.7)
    assert len(results) >= 1
    assert results[0][1] >= 0.7


def test_generate_low_r2():
    from hyppo.generator import generate_hypothesis

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.randn(50)  # random noise, R² should be ~0
    results = generate_hypothesis(X, y, r2_min=0.7)
    assert len(results) == 0
