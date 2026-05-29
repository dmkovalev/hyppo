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


def benjamini_yekutieli(
    p_values: list[float], q: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Benjamini-Yekutieli FDR control for dependent tests (part2.tex:514).

    Controls the false discovery rate at level ``q`` under *arbitrary* dependence
    between the tests -- appropriate here because all pairwise comparisons share
    the same dataset D -- at the cost of the harmonic penalty
    ``c(m) = sum_{j=1}^m 1/j`` relative to Benjamini-Hochberg. A test of rank ``i``
    (ascending p) is a discovery iff ``p_(i) <= i*q / (m*c(m))`` (step-up).

    Returns ``(rejected, adjusted)`` aligned with the input order: ``rejected[i]``
    is True iff test ``i`` is a discovery; ``adjusted[i]`` is its BY-adjusted
    p-value.
    """
    m = len(p_values)
    if m == 0:
        return [], []
    c_m = sum(1.0 / j for j in range(1, m + 1))
    order = sorted(range(m), key=lambda i: p_values[i])  # indices by ascending p
    threshold_rank = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= rank * q / (m * c_m):
            threshold_rank = rank
    rejected = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= threshold_rank:
            rejected[idx] = True
    # BY-adjusted p-values, made monotone non-decreasing in rank (largest -> smallest)
    adjusted = [1.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running_min = min(running_min, min(1.0, p_values[idx] * m * c_m / rank))
        adjusted[idx] = running_min
    return rejected, adjusted


def pairwise_wilcoxon_by(
    errors: dict[str, list[float]], q: float = 0.05
) -> list[tuple[str, str, float, bool]]:
    """All C(k,2) pairwise Wilcoxon comparisons with Benjamini-Yekutieli FDR
    control (the procedure of part2.tex:514). ``errors`` maps each hypothesis to
    its per-sample errors. Returns ``(name_a, name_b, p_by, significant)`` for
    every unordered pair, where ``p_by`` is the BY-adjusted p-value."""
    names = list(errors)
    pairs = [(names[i], names[j])
             for i in range(len(names)) for j in range(i + 1, len(names))]
    raw = [wilcoxon_test(errors[a], errors[b]) for a, b in pairs]
    rejected, adjusted = benjamini_yekutieli(raw, q)
    return [(a, b, adjusted[k], rejected[k]) for k, (a, b) in enumerate(pairs)]


def compute_aic(n_params: int, log_likelihood: float) -> float:
    """Akaike Information Criterion (Definition 10): AIC = 2k - 2ln(L)."""
    return 2 * n_params - 2 * log_likelihood


def compute_bic(n_params: int, n_observations: int, log_likelihood: float) -> float:
    """Bayesian Information Criterion (Definition 10): BIC = k*ln(n) - 2ln(L)."""
    return n_params * math.log(n_observations) - 2 * log_likelihood


def gaussian_log_likelihood(y_true, y_pred) -> float:
    """Gaussian log-likelihood of predictions at the MLE residual variance:
    ``ln L = -n/2 (ln(2*pi*sigma^2) + 1)`` with ``sigma^2 = mean((y-yhat)^2)``.
    Lets AIC/BIC (Definition 10) and the Bayesian posterior be computed directly
    from data. Returns ``+inf`` for an exact fit (sigma^2 = 0)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = yt.size
    if n == 0:
        return 0.0
    sigma2 = float(np.mean((yt - yp) ** 2))
    if sigma2 <= 0.0:
        return float("inf")
    return -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)


def bayesian_posterior(
    bic_by_hypothesis: dict[str, float],
    prior: dict[str, float] | None = None,
) -> dict[str, float]:
    """Posterior probability over competing hypotheses via the Schwarz/BIC
    approximation to the marginal likelihood: ``P(h|D) ∝ exp(-BIC_h/2)·prior_h``,
    normalised over the competitors (Bayesian ranking, Chapter 2). ``bic_by_hypothesis``
    maps a hypothesis name to its BIC; lower BIC -> higher posterior. Uniform prior
    if none given. The shift by the minimum BIC is numerical only and cancels in
    the normalisation."""
    names = list(bic_by_hypothesis)
    if not names:
        return {}
    if prior is None:
        prior = {nm: 1.0 / len(names) for nm in names}
    bmin = min(bic_by_hypothesis.values())
    weights = {nm: prior[nm] * math.exp(-0.5 * (bic_by_hypothesis[nm] - bmin))
               for nm in names}
    z = sum(weights.values())
    if z <= 0.0:
        return {nm: 1.0 / len(names) for nm in names}
    return {nm: weights[nm] / z for nm in names}


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
