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


# --- Benjamini-Yekutieli FDR control (part2.tex:514) ---


def test_by_empty():
    from hyppo.comparison import benjamini_yekutieli

    assert benjamini_yekutieli([]) == ([], [])


def test_by_adjusted_matches_scipy_oracle():
    from scipy.stats import false_discovery_control

    from hyppo.comparison import benjamini_yekutieli

    ps = [0.01, 0.02, 0.2, 0.5, 0.001]
    _, adjusted = benjamini_yekutieli(ps)
    oracle = list(false_discovery_control(ps, method="by"))
    for a, o in zip(adjusted, oracle):
        assert abs(a - o) < 1e-9


def test_by_harmonic_factor_more_conservative():
    from hyppo.comparison import benjamini_yekutieli

    # a single p just under the naive 0.05 is NOT a discovery once c(m) penalises it
    rejected, _ = benjamini_yekutieli([0.04, 0.6, 0.7, 0.8], q=0.05)
    assert rejected == [False, False, False, False]


def test_by_rejects_clear_signal():
    from hyppo.comparison import benjamini_yekutieli

    rejected, _ = benjamini_yekutieli([0.0001, 0.0002, 0.9, 0.95], q=0.05)
    assert rejected[0] and rejected[1]
    assert not rejected[2] and not rejected[3]


def test_by_adjusted_monotone_in_p():
    from hyppo.comparison import benjamini_yekutieli

    ps = [0.001, 0.3, 0.02, 0.04, 0.5]
    _, adjusted = benjamini_yekutieli(ps)
    adj_sorted = [a for _, a in sorted(zip(ps, adjusted))]
    assert adj_sorted == sorted(adj_sorted)  # non-decreasing with p


def test_pairwise_wilcoxon_by_shape():
    from hyppo.comparison import pairwise_wilcoxon_by

    errors = {
        "h1": [0.1, 0.12, 0.11, 0.13, 0.1],
        "h2": [0.9, 0.8, 0.85, 0.95, 0.9],
        "h3": [0.5, 0.55, 0.5, 0.45, 0.5],
    }
    res = pairwise_wilcoxon_by(errors, q=0.05)
    assert len(res) == 3  # C(3,2)
    for a, b, p_by, sig in res:
        assert 0.0 <= p_by <= 1.0
        assert isinstance(sig, bool)


# --- Gaussian log-likelihood + Bayesian posterior (Chapter 2) ---


def test_gaussian_log_likelihood_perfect_fit():
    from hyppo.comparison import gaussian_log_likelihood

    assert gaussian_log_likelihood([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == float("inf")


def test_gaussian_log_likelihood_better_fit_higher():
    from hyppo.comparison import gaussian_log_likelihood

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    good = gaussian_log_likelihood(y, [1.1, 2.0, 2.9, 4.1, 5.0])
    bad = gaussian_log_likelihood(y, [3.0, 3.0, 3.0, 3.0, 3.0])
    assert good > bad  # smaller residuals -> higher log-likelihood


def test_gaussian_ll_consistent_with_aic():
    # AIC computed from the data LL matches compute_aic
    from hyppo.comparison import compute_aic, gaussian_log_likelihood

    y = [1.0, 2.0, 3.0, 4.0]
    yhat = [1.2, 1.9, 3.1, 3.8]
    ll = gaussian_log_likelihood(y, yhat)
    assert compute_aic(n_params=2, log_likelihood=ll) == 4 - 2 * ll


def test_bayesian_posterior_normalised_and_favours_low_bic():
    from hyppo.comparison import bayesian_posterior

    post = bayesian_posterior({"a": 100.0, "b": 110.0, "c": 130.0})
    assert abs(sum(post.values()) - 1.0) < 1e-12
    assert post["a"] > post["b"] > post["c"]  # lower BIC -> higher posterior


def test_bayesian_posterior_uniform_on_ties():
    from hyppo.comparison import bayesian_posterior

    post = bayesian_posterior({"a": 50.0, "b": 50.0})
    assert abs(post["a"] - 0.5) < 1e-12 and abs(post["b"] - 0.5) < 1e-12


def test_bayesian_posterior_empty():
    from hyppo.comparison import bayesian_posterior

    assert bayesian_posterior({}) == {}
