"""Hypothesis comparison methods for virtual experiments.

Implements ranking functions from Definitions 9-11 (Chapter 2):
- Sign test (Definition 11)
- Wilcoxon signed-rank test (Definition 11)
- AIC/BIC (Definition 10)
- Combined ranking rho_comb with min-max normalization
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def sign_test(errors_a: list[float], errors_b: list[float]) -> float:
    """Sign test for paired comparison of two models (Definition 11).

    Tests H0: P(|e_a| > |e_b|) = P(|e_b| > |e_a|) = 0.5.
    Returns p-value.
    """
    deltas = [abs(a) - abs(b) for a, b in zip(errors_a, errors_b)]
    n_positive = sum(1 for d in deltas if d > 0)
    n_nonzero = sum(1 for d in deltas if d != 0)
    if n_nonzero == 0:
        return 1.0
    return stats.binomtest(n_positive, n_nonzero, 0.5).pvalue


def wilcoxon_test(errors_a: list[float], errors_b: list[float]) -> float:
    """Wilcoxon signed-rank test for paired comparison (Definition 11).

    More powerful than sign test --- accounts for magnitude of differences.
    Returns p-value.
    """
    deltas = [abs(a) - abs(b) for a, b in zip(errors_a, errors_b)]
    if all(d == 0 for d in deltas):
        return 1.0
    _, p_value = stats.wilcoxon(deltas, alternative="two-sided")
    return p_value


def compute_aic(n_params: int, log_likelihood: float) -> float:
    """Akaike Information Criterion (Definition 10): AIC = 2k - 2ln(L)."""
    return 2 * n_params - 2 * log_likelihood


def compute_bic(n_params: int, n_observations: int, log_likelihood: float) -> float:
    """Bayesian Information Criterion (Definition 10): BIC = k*ln(n) - 2ln(L)."""
    return n_params * math.log(n_observations) - 2 * log_likelihood


def combined_ranking(
    scores: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Combined ranking rho_comb with min-max normalization (Section 2.6.4).

    Args:
        scores: {hypothesis_name: {criterion_name: value}}
        weights: {criterion_name: weight}, must sum to 1. Equal weights if None.

    Returns:
        Sorted list of (hypothesis_name, combined_score), best first.
    """
    if not scores:
        return []
    criteria = list(next(iter(scores.values())).keys())
    if weights is None:
        weights = {c: 1.0 / len(criteria) for c in criteria}

    # Min-max normalization per criterion
    normalized: dict[str, dict[str, float]] = {}
    for criterion in criteria:
        values = [s[criterion] for s in scores.values()]
        vmin, vmax = min(values), max(values)
        for name, s in scores.items():
            normalized.setdefault(name, {})[criterion] = (
                (s[criterion] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            )

    # Weighted sum
    combined = []
    for name, norms in normalized.items():
        total = sum(weights[c] * norms[c] for c in criteria)
        combined.append((name, total))

    return sorted(combined, key=lambda x: x[1], reverse=True)
