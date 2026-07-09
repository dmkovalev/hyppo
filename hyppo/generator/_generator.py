"""HypothesisGenerator — generates hypotheses from data (Section 3.1.3).

Implements a simplified version of the 5-stage pipeline:
1. GLM — extract significant variable combinations
2. DCM — extract derivatives (skipped in minimal implementation)
3. GP — symbolic regression (optional, requires deap)
4. NeuralODE — (skipped in minimal implementation)
5. Ranking by R²
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def generate_hypothesis(
    X: np.ndarray,
    y: np.ndarray,
    r2_min: float = 0.7,
) -> list[tuple[str, float]]:
    """Generate hypotheses from data using linear regression.

    Returns list of (description, r2_score) sorted by R² descending,
    filtered by r2_min threshold.
    """
    hypotheses = []

    # Stage 1: GLM — fit linear model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    coefs = model.coef_
    significant = [i for i, c in enumerate(coefs) if abs(c) > 1e-6]
    desc = f"LinearModel(features={significant}, R²={r2:.4f})"
    hypotheses.append((desc, r2))

    # Stage 3: GP — symbolic regression (optional)
    try:
        from deap import base as _  # noqa: F401 -- check if deap available
        from gplearn.genetic import SymbolicRegressor

        sr = SymbolicRegressor(
            population_size=200,
            generations=20,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            random_state=42,
        )
        sr.fit(X, y)
        y_pred_gp = sr.predict(X)
        r2_gp = r2_score(y, y_pred_gp)
        hypotheses.append((f"GP({sr._program})", r2_gp))
    except ImportError:
        pass  # GP not available without deap/gplearn

    # Stage 5: Filter and rank
    filtered = [(desc, score) for desc, score in hypotheses if score >= r2_min]
    return sorted(filtered, key=lambda x: x[1], reverse=True)
