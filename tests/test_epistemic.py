"""Tests for the epistemic-status state machine (part2.tex:570-588).

Covers the pure transition function (all four states + thresholds) and the
Runner integration that assigns the status from R^2 / Delta AIC and the
``competes`` relation.
"""

from hyppo.core._epistemic import EpistemicStatus, evaluate_status
from hyppo.runner import Runner

# --------------------------------------------------------------------------
# pure transition function
# --------------------------------------------------------------------------


def test_proposed_when_not_evaluated():
    assert evaluate_status(None) is EpistemicStatus.PROPOSED


def test_supported_above_threshold():
    assert evaluate_status(0.85) is EpistemicStatus.SUPPORTED
    assert evaluate_status(0.7) is EpistemicStatus.SUPPORTED  # boundary: >= theta


def test_refuted_below_threshold():
    assert evaluate_status(0.69) is EpistemicStatus.REFUTED
    assert evaluate_status(0.0) is EpistemicStatus.REFUTED


def test_superseded_when_competitor_dominates():
    # own R^2 is high (would be SUPPORTED) but a competitor beats it by >10 AIC
    assert (
        evaluate_status(0.9, own_aic=120.0, best_competitor_aic=100.0)
        is EpistemicStatus.SUPERSEDED
    )


def test_not_superseded_within_aic_margin():
    # competitor better, but only by 10 (not strictly > theta_aic) -> stays SUPPORTED
    assert (
        evaluate_status(0.9, own_aic=110.0, best_competitor_aic=100.0)
        is EpistemicStatus.SUPPORTED
    )


def test_superseded_dominates_even_when_refuted():
    # low R^2 AND a far-better competitor -> SUPERSEDED takes precedence over REFUTED
    assert (
        evaluate_status(0.4, own_aic=200.0, best_competitor_aic=100.0)
        is EpistemicStatus.SUPERSEDED
    )


def test_custom_thresholds():
    assert evaluate_status(0.85, theta_sup=0.9) is EpistemicStatus.REFUTED
    assert (
        evaluate_status(0.95, own_aic=105.0, best_competitor_aic=100.0, theta_aic=2.0)
        is EpistemicStatus.SUPERSEDED
    )


# --------------------------------------------------------------------------
# Runner integration
# --------------------------------------------------------------------------


def test_runner_assigns_supported_and_refuted():
    runner = Runner(max_retries=1)
    results = runner.execute(
        plan={"p_ne": ["good", "bad"], "p_e": set()},
        models={
            "good": lambda c: {"r2": 0.9, "aic": 100.0},
            "bad": lambda c: {"r2": 0.5, "aic": 200.0},
        },
    )
    assert results["good"]["epistemic_status"] == "SUPPORTED"
    assert results["bad"]["epistemic_status"] == "REFUTED"


def test_runner_assigns_superseded_via_competes():
    runner = Runner(max_retries=1)
    results = runner.execute(
        plan={"p_ne": ["h", "rival"], "p_e": set()},
        models={
            "h": lambda c: {"r2": 0.85, "aic": 150.0},
            "rival": lambda c: {"r2": 0.95, "aic": 100.0},
        },
        competes={"h": {"rival"}, "rival": {"h"}},
    )
    # h is beaten by rival (150 - 100 = 50 > 10) -> SUPERSEDED despite R^2 >= 0.7
    assert results["h"]["epistemic_status"] == "SUPERSEDED"
    # rival has the best AIC -> SUPPORTED
    assert results["rival"]["epistemic_status"] == "SUPPORTED"


def test_runner_failed_stays_proposed():
    def boom(c):
        raise RuntimeError("nope")

    runner = Runner(max_retries=1)
    results = runner.execute(
        plan={"p_ne": ["h"], "p_e": set()},
        models={"h": boom},
    )
    assert results["h"]["status"] == "FAILED"
    assert results["h"]["epistemic_status"] == "PROPOSED"
