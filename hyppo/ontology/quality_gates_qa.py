"""QA-01..QA-05: Quality assessment rules from TING VI (2024).

Pure functions — no OWL dependency. Called by experiment evaluation
pipelines to produce quality markers.
"""

from typing import Literal


def check_training_data(
    n_points: int,
    min_points: int = 180,
) -> Literal["insufficient", "sufficient"]:
    """QA-01: Minimum training data check."""
    return "insufficient" if n_points < min_points else "sufficient"


def check_coverage(
    active_ratio: float,
    min_coverage: float = 0.70,
) -> Literal["low_coverage", "ok"]:
    """QA-02: Active connection coverage check."""
    return "low_coverage" if active_ratio < min_coverage else "ok"


def check_stability(
    delta_oil_pct: float,
    threshold: float = 0.02,
) -> Literal["stable", "unstable"]:
    """QA-03: Optimization stability after ±10% perturbation."""
    return "stable" if abs(delta_oil_pct) < threshold else "unstable"


def classify_stability(
    time_in_optimal_pct: float,
) -> Literal["stable", "medium", "unstable"]:
    """QA-04: Stability classification by time in optimal mode."""
    if time_in_optimal_pct > 0.66:
        return "stable"
    if time_in_optimal_pct > 0.33:
        return "medium"
    return "unstable"


def apply_realization_coefficient(
    potential: float,
    coefficient: float = 0.60,
) -> float:
    """QA-05: Discount potential by manual implementation factor."""
    return round(potential * coefficient, 4)
